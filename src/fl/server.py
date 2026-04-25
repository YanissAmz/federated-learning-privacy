"""Federated learning server: FedAvg aggregation, optional Central DP, real eval."""

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.defenses.dp import state_dict_diff
from src.fl.client import FLClient


@dataclass
class DPConfig:
    """Central Differential Privacy parameters for FedAvg.

    Each client's state-dict diff (post-train minus pre-train) is L2-clipped to
    `max_norm`, then has Gaussian noise of std `noise_multiplier * max_norm`
    added before being averaged into the global model.
    """

    max_norm: float
    noise_multiplier: float


@dataclass
class RoundMetrics:
    """Per-round evaluation snapshot."""

    round: int
    test_accuracy: float
    test_loss: float


class FLServer:
    """Central server for federated averaging with optional Central DP."""

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.global_model = model.to(device)
        self.device = device
        self.history: list[RoundMetrics] = []

    # -------- aggregation -----------------------------------------------------

    def aggregate(
        self,
        client_states: list[dict[str, torch.Tensor]],
        dp: DPConfig | None = None,
    ) -> None:
        """FedAvg: average client diffs and apply to the global model.

        With `dp != None`, each client's diff is **clipped** to L2 ≤ max_norm,
        averaged, and then a **single** Gaussian noise term of std
        `max_norm * noise_multiplier / N` is added to the average. This is the
        canonical Central-DP-FedAvg pattern (McMahan+ 2018): noise once on the
        averaged update, not N times on each client's diff. With per-client
        independent noise the variance would scale as σ²/N after averaging,
        which over-noises by √N relative to the formal guarantee — fixed here.
        """
        from src.defenses.dp import (
            add_noise_to_diff,
            clip_state_diff,
        )

        global_state = self.global_model.state_dict()
        n_clients = len(client_states)

        # 1. Per-client diff + clip (no noise yet).
        clipped_diffs: list[dict[str, torch.Tensor]] = []
        for cs in client_states:
            diff = state_dict_diff(global_state, cs)
            if dp is not None:
                diff = clip_state_diff(diff, max_norm=dp.max_norm)
            clipped_diffs.append(diff)

        # 2. Average the clipped diffs.
        avg_diff: dict[str, torch.Tensor] = {}
        for key in global_state:
            avg_diff[key] = torch.stack([d[key] for d in clipped_diffs]).mean(dim=0)

        # 3. Add Gaussian noise ONCE to the average. Variance scales 1/N because
        #    sum sensitivity is max_norm and the average divides by N.
        if dp is not None:
            sigma_avg = dp.max_norm * dp.noise_multiplier / n_clients
            avg_diff = add_noise_to_diff(avg_diff, sigma=sigma_avg)

        # 4. Apply the (possibly noisy) average to the global model.
        #    Guard against NaN/Inf: at very small ε the noise can cause
        #    activations to explode, producing NaN parameters. We zero such
        #    keys so subsequent rounds don't propagate NaN. The accuracy will
        #    stay flat — this collapse is the message of the experiment.
        new_state: dict[str, torch.Tensor] = {}
        for key, gv in global_state.items():
            updated = gv.float() + avg_diff[key]
            mask_bad = ~torch.isfinite(updated)
            if mask_bad.any():
                updated = torch.where(mask_bad, gv.float(), updated)
            new_state[key] = updated.to(gv.dtype)
        self.global_model.load_state_dict(new_state)

    # -------- training round --------------------------------------------------

    def train_round(
        self,
        clients: list[FLClient],
        eval_dataset: Dataset | None = None,
        epochs: int = 1,
        batch_size: int = 64,
        lr: float = 0.01,
        dp: DPConfig | None = None,
        round_idx: int = 0,
    ) -> RoundMetrics:
        """One FL round: distribute, train locally, aggregate, (optionally) eval.

        Returns a `RoundMetrics` snapshot. If `eval_dataset` is None, accuracy
        and loss are reported as 0.0 / inf (used in unit tests where eval is
        intentionally skipped).
        """
        client_states = []
        for client in clients:
            local_model = copy.deepcopy(self.global_model)
            state = client.train(local_model, epochs=epochs, batch_size=batch_size, lr=lr)
            client_states.append(state)
        self.aggregate(client_states, dp=dp)

        if eval_dataset is not None:
            acc, loss = self.evaluate(eval_dataset, batch_size=batch_size)
        else:
            acc, loss = 0.0, float("inf")

        metrics = RoundMetrics(round=round_idx, test_accuracy=acc, test_loss=loss)
        self.history.append(metrics)
        return metrics

    # -------- evaluation ------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, dataset: Dataset, batch_size: int = 256) -> tuple[float, float]:
        """Compute (accuracy, mean cross-entropy loss) on a held-out dataset."""
        self.global_model.eval()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        criterion = nn.CrossEntropyLoss(reduction="sum")
        total_loss = 0.0
        total_correct = 0
        total = 0
        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            logits = self.global_model(images)
            total_loss += criterion(logits, labels).item()
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        return total_correct / total, total_loss / total
