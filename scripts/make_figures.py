"""Build the README figures from existing results/ JSONs.

Outputs:
    results/figures/accuracy_curves.png        4 FL configs, accuracy per round
    results/figures/reconstructions_grid.png   3 samples × 4 defenses grid

Usage:
    python -m scripts.make_figures
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
SAMPLES = [7, 42, 100]
TAGS = [
    ("attack_no_def", "no defense", "#dc2626"),
    ("attack_dp_eps64", "DP ε=64", "#d97706"),
    ("attack_dp_eps16", "DP ε=16", "#059669"),
    ("attack_dp_eps4", "DP ε=4", "#2563eb"),
    ("attack_dp_eps1", "DP ε=1", "#7c3aed"),
]


def make_accuracy_curves() -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = 0
    for tag, color, label in [
        ("no_def", "#2563eb", "no defense (ε = ∞)"),
        ("dp_eps64", "#059669", "DP ε=64 (loose)"),
        ("dp_eps16", "#d97706", "DP ε=16"),
        ("dp_eps4", "#dc2626", "DP ε=4"),
        ("dp_eps1", "#7c3aed", "DP ε=1 (tight)"),
    ]:
        path = RESULTS / f"{tag}_metrics.json"
        if not path.exists():
            print(f"[skip] {path} missing")
            continue
        d = json.loads(path.read_text())
        rounds = [0] + [r["round"] for r in d["rounds"]]
        accs = [d["init_accuracy"]] + [r["accuracy"] for r in d["rounds"]]
        ax.plot(rounds, accs, marker="o", color=color, label=label, linewidth=2)
        plotted += 1
    if plotted == 0:
        print("[abort] no accuracy data — run scripts.evaluate first")
        return
    ax.set_xlabel("Round")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Federated training under Central DP\n(CIFAR-10, SimpleCNN, 5 clients, 20 rounds)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = FIGURES / "accuracy_curves.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[save] {out}")
    plt.close(fig)


def _denormalize_to_uint8(t: torch.Tensor) -> np.ndarray:
    arr = (t * CIFAR_STD + CIFAR_MEAN).clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (arr * 255).clip(0, 255).astype(np.uint8)


def make_reconstructions_grid() -> None:
    """Grid: 3 samples × 5 columns (target + 4 defense levels)."""
    from src.fl.data import get_dataset

    trainset = get_dataset("cifar10", train=True)

    fig, axes = plt.subplots(
        len(SAMPLES), len(TAGS) + 1, figsize=(2.0 * (len(TAGS) + 1), 2.0 * len(SAMPLES))
    )
    if len(SAMPLES) == 1:
        axes = axes[None, :]

    for row, sample in enumerate(SAMPLES):
        # target
        img, _label = trainset[sample]
        img_t = img.unsqueeze(0)
        target_im = _denormalize_to_uint8(img_t)
        axes[row, 0].imshow(target_im)
        axes[row, 0].axis("off")
        if row == 0:
            axes[row, 0].set_title("target", fontsize=11)
        axes[row, 0].set_ylabel(f"sample {sample}", fontsize=10)

        # reconstructions
        for col, (tag_pat, _label, _color) in enumerate(TAGS, start=1):
            png = FIGURES / f"{tag_pat}_s{sample}.png"
            json_path = RESULTS / f"{tag_pat}_s{sample}_metrics.json"
            if png.exists():
                # extract just the right (reconstruction) half from saved 2-up png
                im = np.array(Image.open(png))
                # the saved figure has the recon on the right; recover by cropping
                # right half of the figure. Approximate: split horizontally at 50%.
                w = im.shape[1]
                recon = im[:, w // 2 :, :]
                axes[row, col].imshow(recon)
            else:
                axes[row, col].text(
                    0.5,
                    0.5,
                    "missing",
                    ha="center",
                    va="center",
                    transform=axes[row, col].transAxes,
                )
            axes[row, col].axis("off")
            if row == 0:
                title = TAGS[col - 1][1]
                if json_path.exists():
                    d = json.loads(json_path.read_text())
                    title = f"{title}\nPSNR={d['psnr_db']:.1f} dB"
                axes[row, col].set_title(title, fontsize=10)

    fig.suptitle("iDLG reconstruction with and without DP", fontsize=12)
    fig.tight_layout()
    out = FIGURES / "reconstructions_grid.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[save] {out}")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    make_accuracy_curves()
    make_reconstructions_grid()


if __name__ == "__main__":
    main()
