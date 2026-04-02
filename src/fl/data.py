"""Data loading and partitioning for federated learning."""

import numpy as np
from torch.utils.data import Subset
from torchvision import datasets, transforms


def get_dataset(name: str = "cifar10", train: bool = True):
    """Load a standard dataset with appropriate transforms."""
    if name == "cifar10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        return datasets.CIFAR10(root="./data", train=train, download=True, transform=transform)
    elif name == "mnist":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        return datasets.MNIST(root="./data", train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def partition_iid(dataset, num_clients: int) -> list[Subset]:
    """Split dataset into IID partitions for each client."""
    indices = np.random.permutation(len(dataset))
    splits = np.array_split(indices, num_clients)
    return [Subset(dataset, s.tolist()) for s in splits]


def partition_non_iid(
    dataset,
    num_clients: int,
    alpha: float = 0.5,
) -> list[Subset]:
    """Split dataset using Dirichlet distribution (non-IID)."""
    targets = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = len(np.unique(targets))
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_indices = np.where(targets == c)[0]
        np.random.shuffle(class_indices)
        proportions = np.random.dirichlet([alpha] * num_clients)
        splits = (proportions * len(class_indices)).astype(int)
        splits[-1] = len(class_indices) - splits[:-1].sum()
        start = 0
        for i, count in enumerate(splits):
            client_indices[i].extend(class_indices[start : start + count].tolist())
            start += count

    return [Subset(dataset, indices) for indices in client_indices]
