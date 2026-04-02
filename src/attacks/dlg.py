"""Deep Leakage from Gradients (DLG) attack implementation.

Reference: Zhu et al., "Deep Leakage from Gradients" (NeurIPS 2019).
"""

import torch
import torch.nn as nn


class DLGAttack:
    """Reconstruct training data from shared gradients."""

    def __init__(
        self,
        model: nn.Module,
        input_shape: tuple[int, ...] = (3, 32, 32),
        num_classes: int = 10,
        lr: float = 1.0,
        iterations: int = 300,
        tv_weight: float = 0.001,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lr = lr
        self.iterations = iterations
        self.tv_weight = tv_weight
        self.device = device

    def _total_variation(self, x: torch.Tensor) -> torch.Tensor:
        """Total variation regularization for smoother reconstructions."""
        diff_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).pow(2).sum()
        diff_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).pow(2).sum()
        return diff_h + diff_w

    def attack(
        self,
        target_gradients: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[float]]:
        """Reconstruct input from gradients.

        Args:
            target_gradients: gradients shared by a client

        Returns:
            (reconstructed_image, loss_history)
        """
        dummy_data = torch.randn(1, *self.input_shape, device=self.device, requires_grad=True)
        dummy_label = torch.randn(1, self.num_classes, device=self.device, requires_grad=True)
        optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        history: list[float] = []

        for _ in range(self.iterations):

            def closure():
                optimizer.zero_grad()
                pred = self.model(dummy_data)
                loss_fn = criterion(pred, dummy_label.softmax(dim=-1))
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
            history.append(loss.item())

        return dummy_data.detach().clamp(0, 1), history
