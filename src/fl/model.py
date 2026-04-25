"""Model architectures for federated learning experiments.

Two models are exposed because the Central-DP-FedAvg primitive is sensitive
to the model's parameter count `D`:

- `SimpleCNN`  — ~1.1 M params, the realistic case (Central DP collapses).
- `TinyMLP`    — ~31 K params (linear), the case where naive Central DP
                 still gives a usable privacy/utility curve.

Use `--model tiny_mlp` on the CLI to swap.
"""

import torch.nn as nn


class SimpleCNN(nn.Module):
    """Simple CNN for CIFAR-10/MNIST classification (~1.1 M params)."""

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class TinyMLP(nn.Module):
    """Tiny logistic-regression-like classifier on flattened pixels.

    For CIFAR-10 (3×32×32 = 3072 inputs, 10 classes): 30 730 params.
    For MNIST   (1×28×28 =   784 inputs, 10 classes):  7 850 params.

    Used to demonstrate that naive Central DP *can* give a meaningful
    privacy/utility curve when D is small relative to the noise budget.
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3, image_size: int = 32):
        super().__init__()
        self.flatten = nn.Flatten()
        in_dim = in_channels * image_size * image_size
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(self.flatten(x))


def build_model(name: str, num_classes: int = 10, in_channels: int = 3) -> nn.Module:
    """Factory used by `scripts/train.py` and friends."""
    if name in {"cnn", "simple_cnn"}:
        return SimpleCNN(num_classes=num_classes, in_channels=in_channels)
    if name == "tiny_mlp":
        return TinyMLP(num_classes=num_classes, in_channels=in_channels)
    raise ValueError(f"unknown model name: {name!r} (expected 'cnn' or 'tiny_mlp')")
