"""Differential Privacy defenses for federated learning.

Two modes are exposed:

1. **Gradient-level DP** (functions on `list[Tensor]`) — used in the DLG attack
   demo, where DP is applied to the gradients a client is about to share.
   This is what makes the attack reconstruction visibly fail.

2. **Central DP on state-dict diffs** (functions on `dict[str, Tensor]`) — used
   in FedAvg training. Each client's update (post-local-train state minus
   pre-round state) is L2-clipped then has Gaussian noise added by the server
   before averaging. Pedagogically simpler than DP-SGD, and matches the
   FedAvg-with-DP pattern from McMahan et al. 2018.

Privacy accounting uses **basic Gaussian composition** over T rounds: noise per
round is calibrated to (ε/T, δ/T), composing linearly to a total budget of
(ε, δ). This is loose vs. RDP/Mironov accounting but is exact, simple, and
appropriate for a pedagogical demo. Documented as such.
"""

import math

import torch

# ----- gradient-level DP (DLG attack defense) --------------------------------


def clip_gradients(
    gradients: list[torch.Tensor],
    max_norm: float,
) -> list[torch.Tensor]:
    """Clip a flat list of gradient tensors to a maximum global L2 norm."""
    flat = torch.cat([g.flatten() for g in gradients])
    total_norm = flat.norm(2)
    clip_factor = min(1.0, max_norm / (total_norm + 1e-8))
    return [g * clip_factor for g in gradients]


def add_noise(
    gradients: list[torch.Tensor],
    noise_multiplier: float,
    max_norm: float,
) -> list[torch.Tensor]:
    """Add calibrated Gaussian noise: sigma = noise_multiplier * max_norm."""
    sigma = noise_multiplier * max_norm
    return [g + torch.randn_like(g) * sigma for g in gradients]


def apply_dp(
    gradients: list[torch.Tensor],
    max_norm: float = 1.0,
    noise_multiplier: float = 1.0,
) -> list[torch.Tensor]:
    """Clip then add noise to a single gradient release."""
    clipped = clip_gradients(gradients, max_norm)
    return add_noise(clipped, noise_multiplier, max_norm)


# ----- state-dict-diff DP (Central DP for FedAvg) ----------------------------


def state_dict_diff(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Compute the per-parameter delta (after - before) for one client round."""
    return {k: after[k].float() - before[k].float() for k in before}


def clip_state_diff(
    diff: dict[str, torch.Tensor],
    max_norm: float,
) -> dict[str, torch.Tensor]:
    """Clip the global L2 norm of a flattened state-dict diff."""
    flat = torch.cat([v.flatten() for v in diff.values()])
    total_norm = flat.norm(2)
    clip_factor = min(1.0, max_norm / (total_norm + 1e-8))
    return {k: v * clip_factor for k, v in diff.items()}


def add_noise_to_diff(
    diff: dict[str, torch.Tensor],
    sigma: float,
) -> dict[str, torch.Tensor]:
    """Add fresh i.i.d. Gaussian noise of std `sigma` to every diff tensor."""
    return {k: v + torch.randn_like(v) * sigma for k, v in diff.items()}


def apply_dp_to_state_diff(
    diff: dict[str, torch.Tensor],
    max_norm: float,
    noise_multiplier: float,
) -> dict[str, torch.Tensor]:
    """Clip a state-dict diff to `max_norm` then add Gaussian noise.

    sigma = noise_multiplier * max_norm.
    """
    clipped = clip_state_diff(diff, max_norm)
    sigma = noise_multiplier * max_norm
    return add_noise_to_diff(clipped, sigma)


def noise_multiplier_from_epsilon(
    epsilon: float,
    delta: float,
    num_rounds: int,
) -> float:
    """zCDP-tight σ to achieve (ε, δ)-DP over T compositions of Gaussian.

    For T independent Gaussian-mechanism releases of sensitivity 1 with noise
    multiplier σ, the total privacy is (ρ_total = T/(2σ²))-zCDP. Converting
    zCDP → (ε,δ)-DP via the standard inequality:

        ε(ρ) = ρ + 2 √(ρ ln(1/δ))

    Inverting for ρ given target ε, δ:

        let u = √ρ. Then ε = u² + 2u √(ln(1/δ))
        u = -√(ln(1/δ)) + √(ln(1/δ) + ε)
        σ = √(T / (2u²))

    This is **much tighter than basic composition** (σ ∝ T/ε vs √T/ε), and
    matches what production DP libraries (Opacus, TF Privacy) use as the
    moments-accountant approximation for full-batch Gaussian.

    Reference: Bun & Steinke (2016), "Concentrated Differential Privacy".
    """
    if num_rounds <= 0:
        raise ValueError("num_rounds must be > 0")
    if epsilon <= 0 or delta <= 0:
        raise ValueError("epsilon and delta must be > 0")

    log_inv_delta = math.log(1.0 / delta)
    u = -math.sqrt(log_inv_delta) + math.sqrt(log_inv_delta + epsilon)
    if u <= 0:
        # Numerical edge case at very small epsilon — fall back to simple comp.
        eps_round = epsilon / num_rounds
        delta_round = delta / num_rounds
        return math.sqrt(2.0 * math.log(1.25 / delta_round)) / eps_round
    rho = u * u
    return math.sqrt(num_rounds / (2.0 * rho))
