"""Federated learning server: FedAvg aggregation."""

import copy

import torch
import torch.nn as nn

from src.fl.client import FLClient


class FLServer:
    """Central server for federated averaging."""

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.global_model = model.to(device)
        self.device = device
        self.history: list[dict] = []

    def aggregate(self, client_states: list[dict[str, torch.Tensor]]) -> None:
        """FedAvg: average client model weights."""
        global_state = self.global_model.state_dict()
        for key in global_state:
            stacked = torch.stack([s[key].float() for s in client_states])
            global_state[key] = stacked.mean(dim=0)
        self.global_model.load_state_dict(global_state)

    def train_round(
        self,
        clients: list[FLClient],
        epochs: int = 1,
        batch_size: int = 64,
        lr: float = 0.01,
    ) -> float:
        """Run one FL round: distribute, train locally, aggregate."""
        client_states = []
        for client in clients:
            local_model = copy.deepcopy(self.global_model)
            state = client.train(local_model, epochs=epochs, batch_size=batch_size, lr=lr)
            client_states.append(state)
        self.aggregate(client_states)
        return self._evaluate_placeholder()

    def _evaluate_placeholder(self) -> float:
        """Placeholder for evaluation --- returns 0.0 until eval dataset is wired."""
        return 0.0
