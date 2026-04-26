"""Rebuild results/byzantine_summary.json from all per-cell *_metrics.json on disk.

Useful when scripts.byzantine has been run incrementally (different --attacks
subsets at different times) and the saved summary only reflects the last
invocation.

Usage:
    python -m scripts.rebuild_byzantine_summary
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> None:
    summary: dict = {"cells": [], "matrix": {}, "args": {}, "model": None}

    for path in sorted(RESULTS.glob("byzantine_*_K*_metrics.json")):
        d = json.loads(path.read_text())
        attack = d["attack"]
        agg = d["aggregator"]

        summary["matrix"].setdefault(attack, {})[agg] = {
            "final_accuracy": d["final_accuracy"],
            "avg_cosine_target_to_global": d["avg_cosine_target_to_global"],
        }
        summary["cells"].append(
            {
                "tag": path.stem.replace("_metrics", ""),
                "attack": attack,
                "aggregator": agg,
                "final_accuracy": d["final_accuracy"],
                "avg_cosine_target_to_global": d["avg_cosine_target_to_global"],
            }
        )
        # Pick up args from any one cell (they're all the same).
        if not summary["args"]:
            summary["args"] = {
                "n_malicious": d["n_malicious"],
                "clients": d["n_clients"],
                "rounds": d["rounds"],
            }
            summary["model"] = d["model"]

    out = RESULTS / "byzantine_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"[save] {out}")
    print(f"  cells: {len(summary['cells'])}")
    print(f"  attacks: {sorted(summary['matrix'].keys())}")
    print(f"  aggregators: {sorted({a for at in summary['matrix'].values() for a in at})}")


if __name__ == "__main__":
    main()
