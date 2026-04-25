"""Gradio demo: FL training curves, live iDLG attack, DP defense, tradeoff plot.

Run with:
    python -m src.demo.app

Requires `results/*.json` from `python -m scripts.evaluate` for the curves and
the privacy/utility plot. The "live iDLG attack" tab works without any prior
results — it runs the attack on a fresh random model in real time.
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.attacks.dlg import infer_label_from_gradients
from src.defenses.dp import apply_dp, noise_multiplier_from_epsilon
from src.fl.client import FLClient
from src.fl.data import get_dataset
from src.fl.model import SimpleCNN

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 10
INPUT_SHAPE = (3, 32, 32)
CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
CIFAR_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
DEMO_ITERATIONS = 100  # smaller than full attack (300) for snappy UI


# ----- shared resources, loaded once at startup ------------------------------


def _load_resources():
    print("[demo] loading CIFAR-10 (train) …")
    trainset = get_dataset("cifar10", train=True)
    print("[demo] CIFAR-10 ready")
    return trainset


_TRAINSET = _load_resources()


def _denormalize(t: torch.Tensor) -> torch.Tensor:
    return (t * CIFAR_STD + CIFAR_MEAN).clamp(0, 1)


def _to_image(t: torch.Tensor) -> np.ndarray:
    arr = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (arr * 255).clip(0, 255).astype(np.uint8)


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a - b).pow(2).mean().item()
    if mse < 1e-10:
        return 100.0
    return 20.0 * np.log10(1.0) - 10.0 * np.log10(mse)


# ----- TAB 1 — Federated Training curves -------------------------------------


def plot_fl_curves(
    *, include_no_def: bool, include_eps8: bool, include_eps4: bool, include_eps1: bool
):
    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = 0
    for tag, color, included in [
        ("no_def", "#2563eb", include_no_def),
        ("dp_eps64", "#059669", include_eps8),
        ("dp_eps16", "#d97706", include_eps4),
        ("dp_eps4", "#dc2626", include_eps1),
    ]:
        if not included:
            continue
        path = RESULTS / f"{tag}_metrics.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        rounds = [0] + [r["round"] for r in d["rounds"]]
        accs = [d["init_accuracy"]] + [r["accuracy"] for r in d["rounds"]]
        label = tag.replace("_", " ")
        if d["args"]["epsilon"] is not None:
            label = f"DP ε={d['args']['epsilon']:g}"
        else:
            label = "no defense"
        ax.plot(rounds, accs, marker="o", color=color, label=label, linewidth=2)
        plotted += 1
    if plotted == 0:
        ax.text(
            0.5,
            0.5,
            "No results yet — run:\n  python -m scripts.evaluate",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
        )
    ax.set_xlabel("Round")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Federated training under Central DP (CIFAR-10, SimpleCNN, 5 clients)")
    ax.grid(alpha=0.3)
    if plotted > 0:
        ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


# ----- TAB 2 / 3 — Live iDLG attack ------------------------------------------


def _build_target(sample_idx: int):
    img, label = _TRAINSET[int(sample_idx)]
    img = img.unsqueeze(0).to(DEVICE)
    label_t = torch.tensor([label], device=DEVICE)
    return img, label, label_t


def _build_model_and_grads(img, label_t, defense: str, sigma: float, max_norm: float):
    """Returns (model, gradients) — gradients optionally DP-protected."""
    model = SimpleCNN(num_classes=NUM_CLASSES, in_channels=3).to(DEVICE)
    client = FLClient(client_id=0, dataset=_TRAINSET, device=DEVICE)
    grads = client.compute_gradients(model, img, label_t)
    if defense == "dp":
        grads = apply_dp(grads, max_norm=max_norm, noise_multiplier=sigma)
    return model, grads


def run_attack_streaming(sample_idx: int, defense: str, eps_target: float):
    """Yield (target_img, current_recon, info_text) tuples as the attack progresses."""
    img, label, label_t = _build_target(sample_idx)
    target_im = _to_image(_denormalize(img.cpu()))
    yield (
        target_im,
        target_im * 0,
        f"Initializing iDLG attack on sample {sample_idx} (label = {CIFAR_CLASSES[label]})…",
    )

    sigma = noise_multiplier_from_epsilon(float(eps_target), 1e-5, 20) if defense == "dp" else 0.0
    max_norm = 1.0
    model, grads = _build_model_and_grads(img, label_t, defense, sigma, max_norm)

    inferred = infer_label_from_gradients(grads, NUM_CLASSES)
    label_correct = inferred == label
    label_msg = f"iDLG inferred label = {CIFAR_CLASSES[inferred]} ({'✓' if label_correct else '✗'})"

    # Custom streaming attack: re-implement the L-BFGS loop here so we can
    # yield intermediate frames to Gradio (the canonical DLGAttack returns the
    # full reconstruction in one shot).
    label_t_attack = torch.tensor([inferred], device=DEVICE)
    dummy = torch.randn(1, *INPUT_SHAPE, device=DEVICE, requires_grad=True)
    optimizer = torch.optim.LBFGS([dummy], lr=1.0)
    criterion = torch.nn.CrossEntropyLoss()

    target_denorm = _denormalize(img.cpu())
    last_yield = -1
    for it in range(DEMO_ITERATIONS):

        def closure():
            optimizer.zero_grad()
            pred = model(dummy)
            loss_fn = criterion(pred, label_t_attack)
            dummy_grads = torch.autograd.grad(loss_fn, model.parameters(), create_graph=True)
            grad_loss = sum(
                (dg - tg).pow(2).sum() for dg, tg in zip(dummy_grads, grads, strict=True)
            )
            tv = 0.001 * (
                (dummy[:, :, 1:, :] - dummy[:, :, :-1, :]).pow(2).sum()
                + (dummy[:, :, :, 1:] - dummy[:, :, :, :-1]).pow(2).sum()
            )
            total = grad_loss + tv
            total.backward()
            return total

        loss = optimizer.step(closure)

        if it % 5 == 0 or it == DEMO_ITERATIONS - 1:
            recon = dummy.detach().clamp(0, 1).cpu()
            recon_denorm = _denormalize(recon)
            psnr_val = _psnr(recon_denorm, target_denorm)
            recon_im = _to_image(recon_denorm)
            info = (
                f"iter {it + 1}/{DEMO_ITERATIONS}  ·  loss = {loss.item():.4f}  ·  "
                f"PSNR = {psnr_val:.2f} dB  ·  {label_msg}"
            )
            yield target_im, recon_im, info
            last_yield = it

    if last_yield != DEMO_ITERATIONS - 1:
        recon = dummy.detach().clamp(0, 1).cpu()
        psnr_val = _psnr(_denormalize(recon), target_denorm)
        info = f"done  ·  PSNR = {psnr_val:.2f} dB  ·  {label_msg}\n" + (
            "✓ Attack succeeded — data leaks in plain FedAvg."
            if defense != "dp" and psnr_val > 15
            else "✗ Attack defeated — DP noise destroyed the gradient signal."
            if defense == "dp"
            else "Attack inconclusive — try another sample."
        )
        yield target_im, _to_image(_denormalize(recon)), info


# ----- TAB 4 — Privacy/Utility tradeoff --------------------------------------


def plot_tradeoff():
    summary_path = RESULTS / "summary.json"
    if not summary_path.exists():
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(
            0.5,
            0.5,
            "No summary yet — run:\n  python -m scripts.evaluate",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
        )
        ax.axis("off")
        return fig

    s = json.loads(summary_path.read_text())

    eps_vals = []
    accs = []
    for run in s["fl_runs"]:
        eps = run["epsilon"]
        # use ε = inf for no_def, plot at right of the axis
        eps_vals.append(eps if eps is not None else float("inf"))
        accs.append(run["final_accuracy"])

    pat_to_eps = {
        "attack_no_def": float("inf"),
        "attack_dp_eps64": 64.0,
        "attack_dp_eps16": 16.0,
        "attack_dp_eps4": 4.0,
        "attack_dp_eps1": 1.0,
    }
    # Map either the full or base tag pattern.
    psnr_by_eps = {}
    for a in s["attacks"]:
        key = a.get("tag_pattern_base", a["tag_pattern"])
        if key in pat_to_eps:
            psnr_by_eps[pat_to_eps[key]] = a["avg_psnr_db"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left: accuracy vs epsilon
    finite = [(e, a) for e, a in zip(eps_vals, accs, strict=True) if e != float("inf")]
    if finite:
        ex, ay = zip(*sorted(finite), strict=True)
        ax1.plot(ex, ay, marker="o", color="#2563eb", linewidth=2, label="DP")
    no_def_acc = [a for e, a in zip(eps_vals, accs, strict=True) if e == float("inf")]
    if no_def_acc:
        ax1.axhline(no_def_acc[0], linestyle="--", color="#9ca3af", label="no defense")
    ax1.set_xscale("log")
    ax1.set_xlabel("ε (lower = more private)")
    ax1.set_ylabel("Final test accuracy")
    ax1.set_title("Utility cost of privacy")
    ax1.grid(alpha=0.3)
    ax1.legend()

    # Right: DLG reconstruction PSNR vs epsilon
    eps_attack_finite = sorted([e for e in psnr_by_eps if e != float("inf")])
    psnr_vals_finite = [psnr_by_eps[e] for e in eps_attack_finite]
    if eps_attack_finite:
        ax2.plot(eps_attack_finite, psnr_vals_finite, marker="s", color="#dc2626", linewidth=2)
    if float("inf") in psnr_by_eps:
        ax2.axhline(psnr_by_eps[float("inf")], linestyle="--", color="#9ca3af", label="no defense")
        ax2.legend()
    ax2.set_xscale("log")
    ax2.set_xlabel("ε (lower = more private)")
    ax2.set_ylabel("Avg DLG reconstruction PSNR (dB)")
    ax2.set_title("Privacy gain (lower PSNR = better defense)")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    return fig


# ----- Build UI --------------------------------------------------------------


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Federated Learning + Privacy Demo") as demo:
        gr.Markdown(
            "# Federated Learning + Privacy Attack Demo\n"
            "Federated training of a CNN on CIFAR-10. iDLG gradient-leakage "
            "attack reconstructs private samples. Differential Privacy "
            "(clip + Gaussian noise on the FedAvg state-dict diff) blocks the "
            "attack at a measurable accuracy cost.\n"
            "\n"
            "ℹ️ Tabs 1 and 4 require running `python -m scripts.evaluate` first. "
            "Tabs 2 and 3 work without prior training."
        )

        with gr.Tab("1. Federated training curves"):
            gr.Markdown("Test accuracy per round under different DP budgets.")
            with gr.Row():
                cb_no = gr.Checkbox(value=True, label="no defense")
                cb_8 = gr.Checkbox(value=True, label="DP ε=64")
                cb_4 = gr.Checkbox(value=True, label="DP ε=16")
                cb_1 = gr.Checkbox(value=True, label="DP ε=4")
            plot1 = gr.Plot()
            refresh1 = gr.Button("Refresh curves")
            refresh1.click(
                fn=lambda a, b, c, d: plot_fl_curves(
                    include_no_def=a, include_eps8=b, include_eps4=c, include_eps1=d
                ),
                inputs=[cb_no, cb_8, cb_4, cb_1],
                outputs=plot1,
            )
            demo.load(
                fn=lambda a, b, c, d: plot_fl_curves(
                    include_no_def=a, include_eps8=b, include_eps4=c, include_eps1=d
                ),
                inputs=[cb_no, cb_8, cb_4, cb_1],
                outputs=plot1,
            )

        with gr.Tab("2. iDLG attack — no defense"):
            gr.Markdown(
                "Pick a CIFAR-10 sample. The model receives normal gradients. "
                "The attacker reconstructs the input from those gradients in "
                f"{DEMO_ITERATIONS} L-BFGS steps. Watch the reconstruction emerge."
            )
            with gr.Row():
                sample_a = gr.Slider(0, 100, value=7, step=1, label="Sample index")
                btn_a = gr.Button("Run attack", variant="primary")
            with gr.Row():
                target_a = gr.Image(label="Target (private)", height=200, width=200)
                recon_a = gr.Image(label="Reconstruction", height=200, width=200)
            info_a = gr.Markdown("Click **Run attack**.")
            btn_a.click(
                fn=lambda s: run_attack_streaming(s, "none", 8.0),
                inputs=sample_a,
                outputs=[target_a, recon_a, info_a],
            )

        with gr.Tab("3. iDLG attack — DP defense"):
            gr.Markdown(
                "Same attack, but the gradients are wrapped in clip + Gaussian "
                "noise calibrated to a target ε. Reconstruction degrades with "
                "tighter privacy."
            )
            with gr.Row():
                sample_b = gr.Slider(0, 100, value=7, step=1, label="Sample index")
                eps_b = gr.Slider(
                    0.5, 16, value=4.0, step=0.5, label="Target ε (lower = more private)"
                )
                btn_b = gr.Button("Run attack with DP", variant="primary")
            with gr.Row():
                target_b = gr.Image(label="Target (private)", height=200, width=200)
                recon_b = gr.Image(label="Reconstruction (DP-blocked)", height=200, width=200)
            info_b = gr.Markdown("Click **Run attack with DP**.")
            btn_b.click(
                fn=lambda s, e: run_attack_streaming(s, "dp", e),
                inputs=[sample_b, eps_b],
                outputs=[target_b, recon_b, info_b],
            )

        with gr.Tab("4. Privacy/utility tradeoff"):
            gr.Markdown(
                "Lower ε = stronger privacy = lower reconstruction PSNR (good) "
                "+ lower model accuracy (bad). The two effects are visible in "
                "the same experiment."
            )
            plot4 = gr.Plot()
            demo.load(fn=plot_tradeoff, outputs=plot4)
            refresh4 = gr.Button("Refresh")
            refresh4.click(fn=plot_tradeoff, outputs=plot4)

        gr.Markdown(
            "---\n"
            "Repo: [github.com/YanissAmz/federated-learning-privacy]"
            "(https://github.com/YanissAmz/federated-learning-privacy) · "
            "References: [Zhu+ NeurIPS 2019](https://arxiv.org/abs/1906.08935) (DLG), "
            "[Zhao+ 2020](https://arxiv.org/abs/2001.02610) (iDLG), "
            "[McMahan+ 2018](https://arxiv.org/abs/1710.06963) (DP-FedAvg)."
        )
    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.queue()
    ui.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_api=False,
        theme=gr.themes.Soft(),
    )
