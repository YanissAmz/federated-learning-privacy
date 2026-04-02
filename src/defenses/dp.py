"""Differential Privacy defense for federated learning.

Implements gradient clipping and Gaussian noise addition.
"""

import torch


def clip_gradients(
    gradients: list[torch.Tensor],
    max_norm: float,
) -> list[torch.Tensor]:
    """Clip per-sample gradients to a maximum L2 norm.

    Args:
        gradients: list of gradient tensors
        max_norm: clipping threshold

    Returns:
        clipped gradients
    """
    flat = torch.cat([g.flatten() for g in gradients])
    total_norm = flat.norm(2)
    clip_factor = min(1.0, max_norm / (total_norm + 1e-8))
    return [g * clip_factor for g in gradients]


def add_noise(
    gradients: list[torch.Tensor],
    noise_multiplier: float,
    max_norm: float,
) -> list[torch.Tensor]:
    """Add calibrated Gaussian noise to gradients.

    The noise scale is: sigma = noise_multiplier * max_norm

    Args:
        gradients: list of gradient tensors (already clipped)
        noise_multiplier: noise scaling factor
        max_norm: clipping threshold (used to calibrate noise)

    Returns:
        noisy gradients
    """
    sigma = noise_multiplier * max_norm
    return [g + torch.randn_like(g) * sigma for g in gradients]


def apply_dp(
    gradients: list[torch.Tensor],
    max_norm: float = 1.0,
    noise_multiplier: float = 1.0,
) -> list[torch.Tensor]:
    """Apply differential privacy: clip then add noise.

    Args:
        gradients: raw gradients from a client
        max_norm: clipping threshold
        noise_multiplier: noise scaling factor

    Returns:
        differentially private gradients
    """
    clipped = clip_gradients(gradients, max_norm)
    return add_noise(clipped, noise_multiplier, max_norm)
