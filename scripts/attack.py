"""CLI: run an iDLG attack on a single CIFAR-10 sample.

Two modes:
  --no-defense     compute raw gradients on a single sample  →  iDLG attacks them
  --defense dp     wrap the gradients in clip+noise          →  iDLG fails

Usage:
    python -m scripts.attack --tag dlg_no_def --sample 7
    python -m scripts.attack --tag dlg_dp_eps8 --sample 7 --defense dp \\
        --noise-multiplier 0.5 --max-norm 1.0

Outputs:
    results/figures/<tag>.png            final reconstruction (target vs recon)
    results/figures/<tag>.gif            optimization trajectory
    results/<tag>_metrics.json           PSNR/SSIM/MSE + history
"""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image

from src.attacks.dlg import DLGAttack, infer_label_from_gradients
from src.defenses.dp import apply_dp
from src.fl.client import FLClient
from src.fl.data import get_dataset
from src.fl.model import build_model

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

# CIFAR-10 normalization (matches src/fl/data.py)
CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Undo CIFAR-10 normalization for visualization, clamp to [0, 1]."""
    return (x * CIFAR_STD + CIFAR_MEAN).clamp(0, 1)


def to_uint8(t: torch.Tensor) -> np.ndarray:
    """[1, 3, H, W] in [0,1] -> [H, W, 3] uint8."""
    arr = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (arr * 255).clip(0, 255).astype(np.uint8)


def upscale(im: np.ndarray, factor: int = 6) -> np.ndarray:
    """Upscale a (H, W, 3) uint8 image with nearest-neighbour for crisper GIFs."""
    pil = Image.fromarray(im)
    return np.array(pil.resize((im.shape[1] * factor, im.shape[0] * factor), Image.NEAREST))


def psnr(a: torch.Tensor, b: torch.Tensor, max_val: float = 1.0) -> float:
    mse = (a - b).pow(2).mean().item()
    if mse < 1e-10:
        return 100.0
    return 20.0 * np.log10(max_val) - 10.0 * np.log10(mse)


def ssim_simple(a: torch.Tensor, b: torch.Tensor) -> float:
    """A simplified SSIM (channel-mean) — full SSIM would need pytorch-msssim."""
    mu_a = a.mean()
    mu_b = b.mean()
    sigma_a = a.var()
    sigma_b = b.var()
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = 0.01**2, 0.03**2
    num = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (sigma_a + sigma_b + c2)
    return float((num / den).item())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--tag", required=True, help="run identifier (filename prefix)")
    p.add_argument("--sample", type=int, default=7, help="CIFAR-10 train sample index")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument(
        "--snapshot-every", type=int, default=5, help="GIF frame interval (smaller = smoother)"
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="optional global model checkpoint (default: random init)",
    )
    p.add_argument(
        "--defense",
        choices=["none", "dp"],
        default="none",
        help="apply DP to gradients before the attack sees them",
    )
    p.add_argument("--noise-multiplier", type=float, default=0.5)
    p.add_argument("--max-norm", type=float, default=1.0)
    p.add_argument("--model", default=None, help="override model.name (cnn | tiny_mlp)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    torch.manual_seed(args.seed)

    iters = args.iterations or cfg["attack"]["iterations"]

    # ----- model + optional checkpoint ---------------------------------------
    model_name = args.model or cfg["model"]["name"]
    model = build_model(model_name, num_classes=cfg["model"]["num_classes"], in_channels=3)
    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model = model.to(args.device).eval()

    # ----- pick a sample -----------------------------------------------------
    trainset = get_dataset(cfg["dataset"]["name"], train=True)
    img, label = trainset[args.sample]
    img = img.unsqueeze(0).to(args.device)
    label_t = torch.tensor([label], device=args.device)
    print(f"[target] sample idx={args.sample}, true label={label}")

    # ----- compute gradients (the client's would-be release) -----------------
    client = FLClient(client_id=0, dataset=trainset, device=args.device)
    target_grads = client.compute_gradients(model, img, label_t)

    if args.defense == "dp":
        target_grads = apply_dp(
            target_grads, max_norm=args.max_norm, noise_multiplier=args.noise_multiplier
        )
        print(
            f"[defense] DP applied to gradients: clip={args.max_norm}, "
            f"σ={args.noise_multiplier:.3f}"
        )

    # ----- iDLG attack -------------------------------------------------------
    attack = DLGAttack(
        model=model,
        input_shape=(3, 32, 32),
        num_classes=cfg["model"]["num_classes"],
        lr=cfg["attack"]["learning_rate"],
        iterations=iters,
        tv_weight=cfg["attack"]["tv_weight"],
        device=args.device,
        variant="idlg",
        snapshot_every=args.snapshot_every,
    )
    inferred = infer_label_from_gradients(target_grads, cfg["model"]["num_classes"])
    print(f"[idlg] inferred label = {inferred} (true = {label})")

    out = attack.attack(target_grads)
    recon = out["image"]  # [1, 3, 32, 32], normalized space (we'll denormalize)
    snapshots = out["snapshots"]
    history = out["history"]

    # ----- metrics on denormalized images ------------------------------------
    target_denorm = denormalize(img.cpu())
    recon_denorm = denormalize(recon)
    p = psnr(recon_denorm, target_denorm)
    s = ssim_simple(recon_denorm, target_denorm)
    mse = (recon_denorm - target_denorm).pow(2).mean().item()

    print(f"[metrics] PSNR={p:.2f} dB  SSIM≈{s:.3f}  MSE={mse:.5f}  final_loss={history[-1]:.4f}")

    # ----- save figure: target | reconstruction ------------------------------
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(to_uint8(target_denorm))
    axes[0].set_title(f"target (label={label})")
    axes[0].axis("off")
    axes[1].imshow(to_uint8(recon_denorm))
    axes[1].set_title(
        f"iDLG (PSNR={p:.1f} dB)" if args.defense == "none" else f"iDLG vs DP (PSNR={p:.1f})"
    )
    axes[1].axis("off")
    fig.tight_layout()
    fig_path = FIGURES / f"{args.tag}.png"
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {fig_path}")

    # ----- save GIF of optimization trajectory -------------------------------
    if snapshots:
        target_frame = upscale(to_uint8(target_denorm))
        gif_frames = []
        for snap in snapshots:
            recon_frame = upscale(to_uint8(denormalize(snap)))
            # side-by-side
            combo = np.concatenate([target_frame, recon_frame], axis=1)
            gif_frames.append(combo)
        # hold last frame for ~1s (~10 frames at 10fps)
        gif_frames.extend([gif_frames[-1]] * 10)
        gif_path = FIGURES / f"{args.tag}.gif"
        imageio.mimsave(gif_path, gif_frames, duration=0.08)
        print(f"[save] {gif_path}")

    # ----- metrics JSON ------------------------------------------------------
    metrics_out = {
        "tag": args.tag,
        "true_label": int(label),
        "inferred_label": inferred,
        "label_correct": inferred == label,
        "defense": args.defense,
        "noise_multiplier": args.noise_multiplier if args.defense == "dp" else None,
        "max_norm": args.max_norm if args.defense == "dp" else None,
        "psnr_db": p,
        "ssim_simple": s,
        "mse": mse,
        "iterations": iters,
        "loss_history": history,
    }
    json_path = RESULTS / f"{args.tag}_metrics.json"
    json_path.write_text(json.dumps(metrics_out, indent=2))
    print(f"[save] {json_path}")


if __name__ == "__main__":
    main()
