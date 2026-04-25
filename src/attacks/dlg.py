"""Gradient leakage attacks: DLG (Zhu+ 2019) and iDLG (Zhao+ 2020).

iDLG infers the true label from the sign of the last-layer's logits-bias
gradient before reconstruction, then optimizes only the input image. This
produces visibly sharper reconstructions than soft-label DLG, especially on
small images (CIFAR-10 32×32). Default attack here is iDLG.

References:
    Zhu, Liu, Han — "Deep Leakage from Gradients" (NeurIPS 2019).
    Zhao, Mopuri, Bilen — "iDLG: Improved Deep Leakage from Gradients" (2020).
"""

from collections.abc import Callable

import torch
import torch.nn as nn


def infer_label_from_gradients(
    gradients: list[torch.Tensor],
    num_classes: int,
) -> int:
    """iDLG label inference.

    For cross-entropy loss with a final linear layer, the gradient of the
    pre-softmax logit for the true class is `softmax(z)[true] - 1 < 0`, while
    for other classes it is `softmax(z)[c] > 0`. The same sign pattern shows
    on the **bias** of the final fully-connected layer (assuming there is one).

    We locate the final linear-layer bias gradient as the last 1-D tensor whose
    leading dimension equals `num_classes`, then return `argmin` of its sign.
    """
    for grad in reversed(gradients):
        if grad.dim() == 1 and grad.shape[0] == num_classes:
            return int(grad.argmin().item())
    raise ValueError(
        "Could not locate a last-layer bias gradient of length num_classes; "
        "iDLG label inference assumes the model ends in nn.Linear(_, num_classes)."
    )


class DLGAttack:
    """Reconstruct training data from shared gradients (DLG / iDLG variants)."""

    def __init__(
        self,
        model: nn.Module,
        input_shape: tuple[int, ...] = (3, 32, 32),
        num_classes: int = 10,
        lr: float = 1.0,
        iterations: int = 300,
        tv_weight: float = 0.001,
        device: str = "cpu",
        variant: str = "idlg",  # "idlg" | "dlg"
        snapshot_every: int = 0,
    ):
        if variant not in {"idlg", "dlg"}:
            raise ValueError(f"variant must be 'idlg' or 'dlg', got {variant!r}")
        self.model = model.to(device)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lr = lr
        self.iterations = iterations
        self.tv_weight = tv_weight
        self.device = device
        self.variant = variant
        self.snapshot_every = snapshot_every

    def _total_variation(self, x: torch.Tensor) -> torch.Tensor:
        diff_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).pow(2).sum()
        diff_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).pow(2).sum()
        return diff_h + diff_w

    def attack(
        self,
        target_gradients: list[torch.Tensor],
        true_label: int | None = None,
        progress_cb: Callable[[int, torch.Tensor, float], None] | None = None,
    ) -> dict:
        """Reconstruct input from gradients.

        Args:
            target_gradients: gradients that the client would share.
            true_label: optionally provided label (skips iDLG inference).
            progress_cb: optional callback ``cb(iteration, current_image, loss)``
                fired every `snapshot_every` iterations. Useful to record GIFs.

        Returns:
            dict with keys: image (Tensor [1,C,H,W] in [0,1]), inferred_label,
            history (list[float]), snapshots (list[Tensor], may be empty).
        """
        # ----- label handling --------------------------------------------------
        if self.variant == "idlg":
            inferred = (
                infer_label_from_gradients(target_gradients, self.num_classes)
                if true_label is None
                else true_label
            )
            label_t = torch.tensor([inferred], device=self.device)

            def label_for_loss() -> torch.Tensor:
                return label_t  # one-hot via int label, fixed throughout

            optimize_label = False
        else:
            inferred = -1  # not inferred
            dummy_label = torch.randn(1, self.num_classes, device=self.device, requires_grad=True)

            def label_for_loss() -> torch.Tensor:
                return dummy_label.softmax(dim=-1)

            optimize_label = True

        # ----- input init -----------------------------------------------------
        dummy_data = torch.randn(1, *self.input_shape, device=self.device, requires_grad=True)

        params = [dummy_data] + ([dummy_label] if optimize_label else [])
        optimizer = torch.optim.LBFGS(params, lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        history: list[float] = []
        snapshots: list[torch.Tensor] = []

        for it in range(self.iterations):

            def closure():
                optimizer.zero_grad()
                pred = self.model(dummy_data)
                if optimize_label:
                    loss_fn = criterion(pred, label_for_loss())
                else:
                    loss_fn = criterion(pred, label_for_loss())
                dummy_grads = torch.autograd.grad(
                    loss_fn, self.model.parameters(), create_graph=True
                )
                grad_loss = sum(
                    (dg - tg).pow(2).sum()
                    for dg, tg in zip(dummy_grads, target_gradients, strict=True)
                )
                tv_loss = self.tv_weight * self._total_variation(dummy_data)
                total = grad_loss + tv_loss
                total.backward()
                return total

            loss = optimizer.step(closure)
            loss_v = loss.item()
            history.append(loss_v)

            if self.snapshot_every > 0 and (
                it % self.snapshot_every == 0 or it == self.iterations - 1
            ):
                snap = dummy_data.detach().clamp(0, 1).cpu().clone()
                snapshots.append(snap)
                if progress_cb is not None:
                    progress_cb(it, snap, loss_v)

        final = dummy_data.detach().clamp(0, 1).cpu()
        return {
            "image": final,
            "inferred_label": inferred,
            "history": history,
            "snapshots": snapshots,
        }
