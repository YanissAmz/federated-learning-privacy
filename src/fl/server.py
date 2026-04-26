"""Federated learning server: FedAvg aggregation, optional Central DP, real eval."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.defenses.dp import state_dict_diff
from src.fl.client import FLClient

if TYPE_CHECKING:
    from src.attacks.byzantine import ByzantineAttack


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
        aggregator: str = "mean",
        aggregator_kwargs: dict | None = None,
    ) -> None:
        """Aggregate client states into a new global model.

        Pipeline (DP and robust aggregator are independent and compose):
            1. Compute per-client diffs `Δ_i = client_state_i - global_state`.
            2. Optionally clip each `Δ_i` to L2 ≤ `dp.max_norm` (DP only).
            3. Combine the (possibly clipped) `Δ_i` via `aggregator`:
                - "mean"          — coordinate-wise mean (vanilla FedAvg)
                - "median"        — coordinate-wise median (Yin+ 2018)
                - "trimmed_mean"  — trimmed mean (Yin+ 2018)
                - "krum"          — single closest-to-peers client (Blanchard+ 2017)
            4. Optionally add Gaussian noise (DP only) to the combined diff,
               with σ = `dp.max_norm * dp.noise_multiplier / N`.
            5. Apply the combined+noised diff to the global model.

        Notes:
            - Krum selects exactly one client; with Krum the "averaged" diff
              is that client's clipped diff. DP still applies on it.
            - NaN/Inf guard at the end keeps a parameter pinned to its
              previous value if the noise blows it up — happens at very tight
              ε, see the v0.2 README.
        """
        from src.defenses.dp import add_noise_to_diff, clip_state_diff
        from src.defenses.robust import build_aggregator

        global_state = self.global_model.state_dict()
        n_clients = len(client_states)
        aggregator_kwargs = aggregator_kwargs or {}

        # 1+2. Per-client diffs (with DP clip if enabled).
        diffs: list[dict[str, torch.Tensor]] = []
        for cs in client_states:
            diff = state_dict_diff(global_state, cs)
            if dp is not None:
                diff = clip_state_diff(diff, max_norm=dp.max_norm)
            diffs.append(diff)

        # 3. Robust combination of diffs (mean is the FedAvg baseline).
        agg_fn = build_aggregator(aggregator, **aggregator_kwargs)
        combined_diff = agg_fn(diffs)

        # 4. DP noise on the combined diff (independent of aggregator).
        if dp is not None:
            sigma_avg = dp.max_norm * dp.noise_multiplier / n_clients
            combined_diff = add_noise_to_diff(combined_diff, sigma=sigma_avg)

        # 5. Apply, with NaN/Inf guard.
        new_state: dict[str, torch.Tensor] = {}
        for key, gv in global_state.items():
            updated = gv.float() + combined_diff[key]
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
        byzantine_attacks: dict[int, ByzantineAttack] | None = None,
        aggregator: str = "mean",
        aggregator_kwargs: dict | None = None,
    ) -> RoundMetrics:
        """One FL round: distribute, train locally, aggregate, (optionally) eval.

        Args:
            byzantine_attacks: optional `{client_index: ByzantineAttack}` map.
                When set, those clients return a poisoned state-dict instead
                of their honestly-trained one.
            aggregator: "mean" | "median" | "trimmed_mean" | "krum"
                — passed through to `aggregate()`.
        """
        global_state_at_round_start = {
            k: v.detach().clone() for k, v in self.global_model.state_dict().items()
        }
        client_states = []
        for i, client in enumerate(clients):
            local_model = copy.deepcopy(self.global_model)
            honest_state = client.train(local_model, epochs=epochs, batch_size=batch_size, lr=lr)
            if byzantine_attacks and i in byzantine_attacks:
                attack = byzantine_attacks[i]
                state = attack.craft_state(global_state_at_round_start, honest_state)
            else:
                state = honest_state
            client_states.append(state)
        self.aggregate(
            client_states,
            dp=dp,
            aggregator=aggregator,
            aggregator_kwargs=aggregator_kwargs,
        )

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
