"""Federated learning client: local training and gradient computation."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class FLClient:
    """A single federated learning client with local data."""

    def __init__(self, client_id: int, dataset: torch.utils.data.Dataset, device: str = "cpu"):
        self.client_id = client_id
        self.dataset = dataset
        self.device = device

    def train(
        self,
        model: nn.Module,
        epochs: int = 1,
        batch_size: int = 64,
        lr: float = 0.01,
    ) -> dict[str, torch.Tensor]:
        """Train locally and return updated state_dict."""
        model = model.to(self.device)
        model.train()
        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for _ in range(epochs):
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()

        return model.state_dict()

    def compute_gradients(
        self,
        model: nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Compute gradients for a single batch (used by DLG attack)."""
        model = model.to(self.device)
        model.eval()
        images, labels = images.to(self.device), labels.to(self.device)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(model(images), labels)
        gradients = torch.autograd.grad(loss, model.parameters())
        return [g.detach().clone() for g in gradients]
