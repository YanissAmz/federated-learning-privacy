"""Run the full privacy/utility tradeoff matrix used in the README.

Trains four FL configurations:

    no_def        — vanilla FedAvg, no DP
    dp_eps8       — Central DP at target ε=8 (loose privacy)
    dp_eps4       — Central DP at target ε=4
    dp_eps1       — Central DP at target ε=1 (strong privacy)

Then picks 3 representative CIFAR-10 train samples and runs the iDLG attack on
each one **at round 1's gradients** (worst case for DP — defenses must
already protect a freshly-released gradient). Defenses are wrapped in DP for
the dp_* configs at the same noise level used during training.

Usage:
    python -m scripts.evaluate                # full matrix (~25 min on RTX 3090)
    python -m scripts.evaluate --quick        # fewer rounds, smoke test (~3 min)

Outputs:
    results/<tag>_metrics.json          per FL config
    results/figures/<tag>.png/.gif      per attack
    results/summary.json                aggregated table for the README
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

CONFIGS = [
    {"tag": "no_def", "args": []},
    {"tag": "dp_eps64", "args": ["--epsilon", "64.0", "--delta", "1e-5"]},
    {"tag": "dp_eps16", "args": ["--epsilon", "16.0", "--delta", "1e-5"]},
    {"tag": "dp_eps4", "args": ["--epsilon", "4.0", "--delta", "1e-5"]},
    {"tag": "dp_eps1", "args": ["--epsilon", "1.0", "--delta", "1e-5"]},
]

ATTACK_SAMPLES = [7, 42, 100]


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}\n")
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        sys.exit(f"command failed: {' '.join(cmd)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="fewer rounds for a smoke test")
    p.add_argument("--skip-train", action="store_true", help="reuse existing checkpoints")
    p.add_argument("--skip-attack", action="store_true", help="only train, no attacks")
    p.add_argument(
        "--model",
        default=None,
        help="override model.name (cnn | tiny_mlp). Tags are auto-suffixed with _<model>.",
    )
    args = p.parse_args()

    suffix = f"_{args.model}" if args.model else ""
    rounds = 5 if args.quick else 20

    # Tag rewriting for the suffix
    configs = [{"tag": cfg["tag"] + suffix, "args": cfg["args"]} for cfg in CONFIGS]
    if args.model:
        for cfg in configs:
            cfg["args"] = [*list(cfg["args"]), "--model", args.model]

    # ----- 1. train each config (skip if metrics already exist) --------------
    if not args.skip_train:
        for cfg in configs:
            metrics_path = RESULTS / f"{cfg['tag']}_metrics.json"
            if metrics_path.exists():
                print(f"[skip] {cfg['tag']}: {metrics_path.name} already exists")
                continue
            run(
                [
                    sys.executable,
                    "-m",
                    "scripts.train",
                    "--rounds",
                    str(rounds),
                    "--clients",
                    "5",
                    "--tag",
                    cfg["tag"],
                    *cfg["args"],
                ]
            )

    # ----- 2. attacks on a fresh model (round 1 gradients, no checkpoint) ----
    if not args.skip_attack:
        attack_extra = ["--model", args.model] if args.model else []
        # No-defense attacks: 3 samples, all using random init (worst case for victim,
        # this is what an attacker sees from a fresh client in round 1).
        for s in ATTACK_SAMPLES:
            tag = f"attack_no_def{suffix}_s{s}"
            if (RESULTS / f"{tag}_metrics.json").exists():
                print(f"[skip] {tag}: already exists")
                continue
            run(
                [
                    sys.executable,
                    "-m",
                    "scripts.attack",
                    "--tag",
                    tag,
                    "--sample",
                    str(s),
                    "--defense",
                    "none",
                    *attack_extra,
                ]
            )

        # DP defense attacks: re-derive σ at each ε so the defense in the attack
        # matches what training would do.
        from src.defenses.dp import noise_multiplier_from_epsilon

        for eps in [64.0, 16.0, 4.0, 1.0]:
            sigma = noise_multiplier_from_epsilon(eps, 1e-5, rounds)
            for s in ATTACK_SAMPLES:
                tag = f"attack_dp_eps{int(eps)}{suffix}_s{s}"
                if (RESULTS / f"{tag}_metrics.json").exists():
                    print(f"[skip] {tag}: already exists")
                    continue
                run(
                    [
                        sys.executable,
                        "-m",
                        "scripts.attack",
                        "--tag",
                        tag,
                        "--sample",
                        str(s),
                        "--defense",
                        "dp",
                        "--noise-multiplier",
                        f"{sigma:.4f}",
                        "--max-norm",
                        "1.0",
                        *attack_extra,
                    ]
                )

    # ----- 3. aggregate ------------------------------------------------------
    summary: dict = {"fl_runs": [], "attacks": [], "model": args.model or "cnn"}
    for cfg in configs:
        path = RESULTS / f"{cfg['tag']}_metrics.json"
        if path.exists():
            d = json.loads(path.read_text())
            summary["fl_runs"].append(
                {
                    "tag": d["tag"],
                    "epsilon": d["args"]["epsilon"],
                    "noise_multiplier": d["args"]["noise_multiplier"],
                    "init_accuracy": d["init_accuracy"],
                    "final_accuracy": d["final_accuracy"],
                    "rounds": len(d["rounds"]),
                }
            )

    for tag_pat_base in [
        "attack_no_def",
        "attack_dp_eps64",
        "attack_dp_eps16",
        "attack_dp_eps4",
        "attack_dp_eps1",
    ]:
        tag_pat = tag_pat_base + suffix
        per_pat = []
        for s in ATTACK_SAMPLES:
            path = RESULTS / f"{tag_pat}_s{s}_metrics.json"
            if path.exists():
                d = json.loads(path.read_text())
                per_pat.append(
                    {
                        "sample": s,
                        "psnr_db": d["psnr_db"],
                        "ssim": d["ssim_simple"],
                        "label_correct": d["label_correct"],
                    }
                )
        if per_pat:
            avg_psnr = sum(x["psnr_db"] for x in per_pat) / len(per_pat)
            avg_ssim = sum(x["ssim"] for x in per_pat) / len(per_pat)
            summary["attacks"].append(
                {
                    "tag_pattern": tag_pat,
                    "tag_pattern_base": tag_pat_base,
                    "samples": per_pat,
                    "avg_psnr_db": avg_psnr,
                    "avg_ssim": avg_ssim,
                }
            )

    out = RESULTS / f"summary{suffix}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n[save] {out}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
