"""CLI: run one full FL training run, optionally with Central DP.

Usage:
    python -m scripts.train --rounds 20 --clients 5
    python -m scripts.train --rounds 20 --clients 5 --epsilon 8 --delta 1e-5
    python -m scripts.train --rounds 20 --clients 5 --epsilon 1 --delta 1e-5 --tag dp_eps1

Outputs:
    results/<tag>_metrics.json     per-round accuracy / loss + final eval
    results/checkpoints/<tag>.pt   final global model state_dict
"""

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from src.defenses.dp import noise_multiplier_from_epsilon
from src.fl.client import FLClient
from src.fl.data import get_dataset, partition_iid, partition_non_iid
from src.fl.model import build_model
from src.fl.server import DPConfig, FLServer

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CHECKPOINTS = RESULTS / "checkpoints"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--rounds", type=int, default=None, help="override fl.num_rounds")
    p.add_argument("--clients", type=int, default=None, help="override fl.num_clients")
    p.add_argument("--epochs", type=int, default=None, help="override fl.local_epochs")
    p.add_argument("--lr", type=float, default=None, help="override fl.learning_rate")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epsilon", type=float, default=None, help="enable DP with target ε")
    p.add_argument("--delta", type=float, default=None, help="DP δ (default cfg)")
    p.add_argument("--max-norm", type=float, default=None, help="DP clip threshold")
    p.add_argument("--tag", default="no_def", help="run identifier (filename prefix)")
    p.add_argument("--non-iid", action="store_true", help="use Dirichlet partition")
    p.add_argument(
        "--model",
        default=None,
        help="override model.name (cnn | tiny_mlp). Default: read from config.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    torch.manual_seed(args.seed)

    rounds = args.rounds or cfg["fl"]["num_rounds"]
    n_clients = args.clients or cfg["fl"]["num_clients"]
    epochs = args.epochs or cfg["fl"]["local_epochs"]
    lr = args.lr or cfg["fl"]["learning_rate"]
    delta = args.delta or cfg["defense"]["target_delta"]
    max_norm = args.max_norm or cfg["defense"]["max_grad_norm"]

    # ----- DP config from epsilon (or none) ----------------------------------
    dp = None
    if args.epsilon is not None:
        sigma = noise_multiplier_from_epsilon(args.epsilon, delta, rounds)
        dp = DPConfig(max_norm=max_norm, noise_multiplier=sigma)
        print(
            f"[dp] target ε={args.epsilon}, δ={delta}, T={rounds}  →  "
            f"clip={max_norm}, σ={sigma:.4f}"
        )
    else:
        print("[dp] disabled (no_def)")

    # ----- data --------------------------------------------------------------
    print(f"[data] loading {cfg['dataset']['name']} train + test …")
    trainset = get_dataset(cfg["dataset"]["name"], train=True)
    testset = get_dataset(cfg["dataset"]["name"], train=False)
    if args.non_iid:
        partitions = partition_non_iid(trainset, n_clients, alpha=cfg["dataset"]["alpha"])
    else:
        partitions = partition_iid(trainset, n_clients)
    print(f"[data] {n_clients} clients, ~{len(partitions[0])} samples each")

    # ----- model + clients + server ------------------------------------------
    model_name = args.model or cfg["model"]["name"]
    model = build_model(model_name, num_classes=cfg["model"]["num_classes"], in_channels=3)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {model_name} ({n_params:,} params)")
    server = FLServer(model, device=args.device)
    clients = [FLClient(i, p, device=args.device) for i, p in enumerate(partitions)]

    # ----- baseline accuracy --------------------------------------------------
    acc0, loss0 = server.evaluate(testset)
    print(f"[round 0] init acc={acc0:.3f}  loss={loss0:.3f}")

    # ----- training rounds ----------------------------------------------------
    t0 = time.time()
    for r in range(1, rounds + 1):
        m = server.train_round(
            clients,
            eval_dataset=testset,
            epochs=epochs,
            batch_size=cfg["fl"]["batch_size"],
            lr=lr,
            dp=dp,
            round_idx=r,
        )
        elapsed = time.time() - t0
        print(
            f"[round {r:2d}/{rounds}] acc={m.test_accuracy:.3f}  loss={m.test_loss:.3f}  "
            f"elapsed={elapsed:.1f}s"
        )

    # ----- save --------------------------------------------------------------
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINTS / f"{args.tag}.pt"
    torch.save(server.global_model.state_dict(), ckpt_path)

    metrics = {
        "tag": args.tag,
        "config": cfg,
        "args": {
            "rounds": rounds,
            "clients": n_clients,
            "epochs": epochs,
            "lr": lr,
            "epsilon": args.epsilon,
            "delta": delta if args.epsilon is not None else None,
            "max_norm": max_norm if args.epsilon is not None else None,
            "noise_multiplier": dp.noise_multiplier if dp is not None else None,
            "non_iid": args.non_iid,
            "seed": args.seed,
            "model": model_name,
            "n_params": n_params,
        },
        "init_accuracy": acc0,
        "init_loss": loss0,
        "final_accuracy": server.history[-1].test_accuracy,
        "final_loss": server.history[-1].test_loss,
        "rounds": [
            {"round": m.round, "accuracy": m.test_accuracy, "loss": m.test_loss}
            for m in server.history
        ],
        "wall_clock_s": round(time.time() - t0, 2),
    }
    out_path = RESULTS / f"{args.tag}_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"[save] {ckpt_path}")
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
