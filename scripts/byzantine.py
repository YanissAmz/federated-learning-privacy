"""Byzantine attack × robust aggregator matrix runner.

Trains a small FedAvg federation with K malicious clients out of N total,
sweeping over (attack_type, aggregator). For each combination, measures
final accuracy and the cosine similarity between the global model's update
direction and the *target* honest client's update direction (the
deanonymization metric).

Usage:
    python -m scripts.byzantine --quick           # smoke (3 rounds, subset matrix)
    python -m scripts.byzantine                   # full matrix (5 rounds)
    python -m scripts.byzantine --rounds 10       # custom round count
    python -m scripts.byzantine --attacks sign_flip suppression \\
                                --aggregators mean median krum

Outputs:
    results/byzantine_<attack>_<aggregator>_K<k>_metrics.json — one per cell
    results/byzantine_summary.json                            — full grid
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import yaml

from src.attacks.byzantine import ByzantineAttack, build_attack
from src.defenses.dp import state_dict_diff
from src.fl.client import FLClient
from src.fl.data import get_dataset, partition_iid
from src.fl.model import build_model
from src.fl.server import FLServer

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CHECKPOINTS = RESULTS / "checkpoints"

ALL_ATTACKS = ["sign_flip", "constant", "suppression", "stealth"]
ALL_AGGREGATORS = ["mean", "median", "trimmed_mean", "krum"]
QUICK_ATTACKS = ["sign_flip", "suppression"]
QUICK_AGGREGATORS = ["mean", "median"]


def _flat_norm(state: dict[str, torch.Tensor]) -> float:
    return float(torch.cat([v.flatten().float() for v in state.values()]).norm(2).item())


def _cosine_similarity(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    """L2-cosine similarity between two flattened state-dict diffs."""
    fa = torch.cat([v.flatten().float() for v in a.values()])
    fb = torch.cat([v.flatten().float() for v in b.values()])
    denom = fa.norm(2) * fb.norm(2) + 1e-12
    return float((fa @ fb / denom).item())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--rounds", type=int, default=5, help="FL rounds per cell")
    p.add_argument("--clients", type=int, default=5)
    p.add_argument(
        "--n-malicious", type=int, default=2, help="K malicious clients (target = client 0)"
    )
    p.add_argument(
        "--target-index", type=int, default=0, help="honest client whose update we measure"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--quick", action="store_true", help="smaller matrix + 3 rounds")
    p.add_argument("--attacks", nargs="+", default=None, help=f"subset of {ALL_ATTACKS}")
    p.add_argument("--aggregators", nargs="+", default=None, help=f"subset of {ALL_AGGREGATORS}")
    p.add_argument("--model", default=None, help="override model.name (cnn | tiny_mlp)")
    return p.parse_args()


def _build_attacks_for_round(
    attack_name: str, n_malicious: int, magnitude: float, target_norm: float
) -> dict[int, ByzantineAttack]:
    """Return {client_idx: ByzantineAttack} for indices [1..n_malicious] (target is 0)."""
    if attack_name == "constant":
        return {i + 1: build_attack("constant", magnitude=magnitude) for i in range(n_malicious)}
    out = {}
    for i in range(n_malicious):
        out[i + 1] = build_attack(
            attack_name,
            n_malicious=n_malicious,
            index=i,
            magnitude=magnitude,
            target_norm=target_norm,
        )
    return out


def run_cell(
    attack: str,
    aggregator: str,
    *,
    cfg: dict,
    rounds: int,
    n_clients: int,
    n_malicious: int,
    target_idx: int,
    device: str,
    seed: int,
    model_name: str,
    trainset,
    testset,
    partitions,
) -> dict:
    """Run one (attack × aggregator) cell. Returns metrics dict."""
    torch.manual_seed(seed)

    model = build_model(model_name, num_classes=cfg["model"]["num_classes"], in_channels=3)
    server = FLServer(model, device=device)
    clients = [FLClient(i, p, device=device) for i, p in enumerate(partitions)]

    init_acc, _ = server.evaluate(testset)

    # Pick magnitude for the attack: large enough to corrupt mean, but only
    # the "stealth" attack needs to be small.
    magnitude = 100.0
    target_norm = 0.5

    aggregator_kwargs = {"trim_ratio": 0.2, "n_byzantine": n_malicious}

    target_diffs_per_round = []
    global_diffs_per_round = []
    history = []

    t0 = time.time()
    for r in range(1, rounds + 1):
        # State at round start (target's diff is computed against this).
        global_at_start = {
            k: v.detach().clone() for k, v in server.global_model.state_dict().items()
        }

        # Capture target's *honest* diff by training a clone alone.
        clone = copy.deepcopy(server.global_model)
        target_honest_state = clients[target_idx].train(
            clone,
            epochs=cfg["fl"]["local_epochs"],
            batch_size=cfg["fl"]["batch_size"],
            lr=cfg["fl"]["learning_rate"],
        )
        target_diff = state_dict_diff(global_at_start, target_honest_state)
        target_diffs_per_round.append(target_diff)

        # Now do the real round with malicious injection.
        byzantine_attacks = _build_attacks_for_round(attack, n_malicious, magnitude, target_norm)
        m = server.train_round(
            clients,
            eval_dataset=testset,
            epochs=cfg["fl"]["local_epochs"],
            batch_size=cfg["fl"]["batch_size"],
            lr=cfg["fl"]["learning_rate"],
            byzantine_attacks=byzantine_attacks,
            aggregator=aggregator,
            aggregator_kwargs=aggregator_kwargs,
            round_idx=r,
        )
        global_after = {k: v.detach().clone() for k, v in server.global_model.state_dict().items()}
        global_diff = state_dict_diff(global_at_start, global_after)
        global_diffs_per_round.append(global_diff)

        elapsed = time.time() - t0
        history.append(
            {
                "round": r,
                "accuracy": m.test_accuracy,
                "loss": m.test_loss,
                "elapsed_s": round(elapsed, 1),
            }
        )
        print(
            f"  [r{r:2d}/{rounds}] acc={m.test_accuracy:.3f} "
            f"loss={m.test_loss:.3f} ({elapsed:.0f}s)"
        )

    # Aggregate cosine sim across rounds: how much of the target's update
    # leaked into the global update direction.
    cos_sims = []
    for tgt_d, glb_d in zip(target_diffs_per_round, global_diffs_per_round, strict=True):
        cos_sims.append(_cosine_similarity(tgt_d, glb_d))
    avg_cos_sim = sum(cos_sims) / len(cos_sims)

    final_acc = history[-1]["accuracy"]
    return {
        "attack": attack,
        "aggregator": aggregator,
        "n_clients": n_clients,
        "n_malicious": n_malicious,
        "target_idx": target_idx,
        "rounds": rounds,
        "model": model_name,
        "init_accuracy": init_acc,
        "final_accuracy": final_acc,
        "history": history,
        "cosine_target_to_global_per_round": cos_sims,
        "avg_cosine_target_to_global": avg_cos_sim,
        "wall_clock_s": round(time.time() - t0, 2),
    }


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    rounds = 3 if args.quick else args.rounds
    attacks = args.attacks or (QUICK_ATTACKS if args.quick else ALL_ATTACKS)
    aggregators = args.aggregators or (QUICK_AGGREGATORS if args.quick else ALL_AGGREGATORS)

    # Sanity: target = an honest client 0..N, malicious = 1..K (target=0 default).
    if args.target_index >= args.clients:
        raise ValueError("target index out of range")

    print(
        f"[matrix] attacks={attacks}  aggregators={aggregators}  "
        f"K={args.n_malicious}/{args.clients}  rounds={rounds}  device={args.device}"
    )

    print(f"[data] loading {cfg['dataset']['name']} train + test …")
    trainset = get_dataset(cfg["dataset"]["name"], train=True)
    testset = get_dataset(cfg["dataset"]["name"], train=False)
    partitions = partition_iid(trainset, args.clients)

    model_name = args.model or cfg["model"]["name"]

    summary = {"cells": [], "matrix": {}, "args": vars(args), "model": model_name}
    for attack in attacks:
        summary["matrix"][attack] = {}
        for agg in aggregators:
            tag = f"byzantine_{attack}_{agg}_K{args.n_malicious}"
            print(f"\n=== {attack} × {agg} ===")
            cell = run_cell(
                attack,
                agg,
                cfg=cfg,
                rounds=rounds,
                n_clients=args.clients,
                n_malicious=args.n_malicious,
                target_idx=args.target_index,
                device=args.device,
                seed=args.seed,
                model_name=model_name,
                trainset=trainset,
                testset=testset,
                partitions=partitions,
            )
            (RESULTS / f"{tag}_metrics.json").write_text(json.dumps(cell, indent=2))
            summary["cells"].append(
                {
                    "tag": tag,
                    **{
                        k: cell[k]
                        for k in (
                            "attack",
                            "aggregator",
                            "final_accuracy",
                            "avg_cosine_target_to_global",
                        )
                    },
                }
            )
            summary["matrix"][attack][agg] = {
                "final_accuracy": cell["final_accuracy"],
                "avg_cosine_target_to_global": cell["avg_cosine_target_to_global"],
            }
            print(
                f"  [done] acc={cell['final_accuracy']:.3f}  "
                f"cos(target→global)={cell['avg_cosine_target_to_global']:+.3f}"
            )

    out = RESULTS / "byzantine_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n[save] {out}")
    print("\n=== matrix (final accuracy / cosine target→global) ===")
    for attack in attacks:
        for agg in aggregators:
            cell = summary["matrix"][attack][agg]
            print(
                f"  {attack:>15} × {agg:>14} : acc={cell['final_accuracy']:.3f}  "
                f"cos={cell['avg_cosine_target_to_global']:+.3f}"
            )


if __name__ == "__main__":
    main()
