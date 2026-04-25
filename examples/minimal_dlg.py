"""Minimal stand-alone reproduction of the iDLG attack and DP defense.

This script does not depend on the FL plumbing — it just shows the core
gradient-leakage idea on a single CIFAR-10 sample. Useful as a 50-line
mental model of the repo.

    python examples/minimal_dlg.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attacks.dlg import DLGAttack, infer_label_from_gradients
from src.defenses.dp import apply_dp
from src.fl.client import FLClient
from src.fl.data import get_dataset
from src.fl.model import SimpleCNN


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    import math

    mse = (a - b).pow(2).mean().item()
    if mse < 1e-10:
        return 100.0
    return 20.0 * math.log10(1.0) - 10.0 * math.log10(mse)


def main() -> None:
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Pick a CIFAR-10 sample and a fresh randomly-initialized model.
    trainset = get_dataset("cifar10", train=True)
    img, label = trainset[7]  # one airplane (or w/e index 7 maps to)
    img = img.unsqueeze(0).to(device)
    model = SimpleCNN(num_classes=10, in_channels=3).to(device)

    # 2) Compute the gradient the client would share.
    client = FLClient(client_id=0, dataset=trainset, device=device)
    grads = client.compute_gradients(model, img, torch.tensor([label], device=device))

    # 3) Run iDLG with no defense → reconstruction succeeds.
    attack = DLGAttack(
        model=model,
        input_shape=(3, 32, 32),
        num_classes=10,
        iterations=200,
        device=device,
        variant="idlg",
    )
    inferred = infer_label_from_gradients(grads, 10)
    out_clear = attack.attack(grads)
    psnr_clear = psnr(out_clear["image"], img.cpu())
    print(f"[no defense] inferred label={inferred} (true={label})  PSNR={psnr_clear:.2f} dB")

    # 4) Wrap the gradient in DP → reconstruction collapses.
    grads_dp = apply_dp(grads, max_norm=1.0, noise_multiplier=1.0)
    out_dp = attack.attack(grads_dp)
    psnr_dp = psnr(out_dp["image"], img.cpu())
    print(f"[DP defense] PSNR={psnr_dp:.2f} dB  (drop = {psnr_clear - psnr_dp:.1f} dB)")

    if psnr_clear - psnr_dp >= 3.0:
        print("\n✓ DP visibly defeats the iDLG reconstruction on this sample.")
    else:
        print("\n✗ Defense gap too small — try increasing noise_multiplier.")


if __name__ == "__main__":
    main()
