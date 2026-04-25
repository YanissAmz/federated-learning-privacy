"""Patch README.md with the real numbers from results/.

Reads results/<tag>_metrics.json + results/summary.json and replaces the two
placeholder tables in the README with values:

  - "FL training under Central DP" table — final accuracy + Δ + σ
  - "iDLG attack on a fresh round-1 gradient" table — avg PSNR + SSIM

Idempotent: if results aren't there yet, the README stays untouched.

Usage:
    python -m scripts.update_readme
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
README = ROOT / "README.md"


def _load_metrics(tag: str) -> dict | None:
    p = RESULTS / f"{tag}_metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def build_fl_table() -> str | None:
    rows = []
    no_def = _load_metrics("no_def")
    if no_def is None:
        return None
    base_acc = no_def["final_accuracy"]
    rows.append(f"| no defense (vanilla FedAvg) | **{base_acc * 100:.1f}%** | — | 0 |")
    for tag, eps in [("dp_eps64", 64), ("dp_eps16", 16), ("dp_eps4", 4), ("dp_eps1", 1)]:
        d = _load_metrics(tag)
        if d is None:
            rows.append(f"| DP target ε={eps}, δ=1e-5 | _missing_ | — | — |")
            continue
        acc = d["final_accuracy"]
        sigma = d["args"]["noise_multiplier"]
        delta = (acc - base_acc) * 100
        delta_str = f"{delta:+.1f} pts"
        rows.append(
            f"| DP target ε={eps}, δ=1e-5 | **{acc * 100:.1f}%** | {delta_str} | σ ≈ {sigma:.3f} |"
        )
    return "\n".join(rows)


def build_attack_table() -> str | None:
    summary_path = RESULTS / "summary.json"
    if not summary_path.exists():
        return None
    s = json.loads(summary_path.read_text())
    by_pat = {a["tag_pattern"]: a for a in s["attacks"]}

    rows = []
    for pat, label in [
        ("attack_no_def", "no defense (raw gradient)"),
        ("attack_dp_eps64", "DP applied at ε=64"),
        ("attack_dp_eps16", "DP applied at ε=16"),
        ("attack_dp_eps4", "DP applied at ε=4"),
        ("attack_dp_eps1", "DP applied at ε=1"),
    ]:
        a = by_pat.get(pat)
        if a is None:
            rows.append(f"| {label} | _missing_ | — | — |")
            continue
        all_label_ok = all(s["label_correct"] for s in a["samples"])
        label_status = "✓" if all_label_ok else "✗"
        rows.append(
            f"| {label} | **{a['avg_psnr_db']:.1f}** | {a['avg_ssim']:.3f} | "
            f"{label_status} ({sum(s['label_correct'] for s in a['samples'])}/3 correct) |"
        )
    return "\n".join(rows)


def patch_table(content: str, header_re: str, new_rows: str) -> tuple[str, bool]:
    """Replace the body of a markdown table whose header matches `header_re`.

    A markdown table here is:
      | col1 | col2 |
      |---|---|
      | row | row |
      | row | row |

    The header line and separator line are kept; only data rows below
    (until the first blank line) are replaced with `new_rows`.
    """
    pattern = re.compile(
        r"(" + header_re + r"\n\|[-| ]+\|\n)((?:\|.*\|\n)+)",
        re.MULTILINE,
    )
    m = pattern.search(content)
    if m is None:
        return content, False
    return content[: m.start(2)] + new_rows + "\n" + content[m.end(2) :], True


def main() -> None:
    content = README.read_text()
    changed = False

    fl_rows = build_fl_table()
    if fl_rows is not None:
        new, ok = patch_table(
            content,
            r"\| Configuration \| Final test accuracy \| Δ vs no defense \| σ \(per-round noise multiplier\) \|",
            fl_rows,
        )
        if ok:
            content = new
            changed = True
            print("[ok] FL training table patched")
        else:
            print("[skip] FL training table — header not found")
    else:
        print("[skip] FL training table — no_def_metrics.json missing")

    attack_rows = build_attack_table()
    if attack_rows is not None:
        new, ok = patch_table(
            content,
            r"\| Configuration \| Avg PSNR \(dB\) ↓ better defense \| Avg SSIM \| Label inference \|",
            attack_rows,
        )
        if ok:
            content = new
            changed = True
            print("[ok] iDLG attack table patched")
        else:
            print("[skip] iDLG attack table — header not found")
    else:
        print("[skip] iDLG attack table — summary.json missing")

    if changed:
        README.write_text(content)
        print(f"[save] {README}")
    else:
        print("README unchanged")


if __name__ == "__main__":
    main()
