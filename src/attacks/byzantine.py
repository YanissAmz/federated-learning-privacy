"""Byzantine / poisoning attacks against FedAvg.

Each attack returns a *poisoned client state-dict*: the same shape as a real
client's post-train state, but crafted to bias the global aggregation.

Threat model assumed throughout: the malicious clients control their own
local training and the resulting state-dict they send to the server. They
cannot tamper with other clients or the server itself.

Attacks:

- `SignFlipAttack`        — Δ → -Δ on the client's update relative to the
                            received global state. Pure utility-degradation
                            attack, classic Byzantine baseline.
- `ConstantAttack`        — Δ → c · 1 (a fixed value applied to every
                            parameter). Visible to a norm-bound defense.
- `GradientSuppressionAttack`
                          — N-1 malicious clients submit very-large-magnitude
                            updates of one sign so the FedAvg sum is
                            dominated by them. The signal of any single
                            honest client is drowned out, except for the
                            one *target* honest client whose update is
                            reflected in the residual after the malicious
                            updates cancel pairwise. Used to deanonymize
                            the target client (privacy attack via
                            poisoning), not just to break utility.
- `BackdoorAttack` (stub) — placeholder for Bagdasaryan+ 2020 model-
                            replacement; full implementation deferred.

`StealthSuppression` (in this module too) is the **defense-aware**
adversary: it estimates the honest-client norm and crafts updates that fit
inside that envelope, designed to slip past coordinate-wise median /
trimmed-mean / Krum.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from src.defenses.dp import state_dict_diff

# ----- base -----------------------------------------------------------------


class ByzantineAttack(ABC):
    """A poisoning attack that produces a single client's malicious state-dict."""

    @abstractmethod
    def craft_state(
        self,
        global_state: dict[str, torch.Tensor],
        honest_state: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return the malicious post-train state-dict this client will report.

        Args:
            global_state: the server's broadcast state at the start of the round.
            honest_state: what this client's state would have been *if* it had
                trained honestly (when available). Some attacks (sign-flip)
                need this; others (constant / suppression) ignore it.
        """
        ...


# ----- 1. sign-flip ---------------------------------------------------------


class SignFlipAttack(ByzantineAttack):
    """Return -Δ instead of +Δ on the client's update."""

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def craft_state(
        self,
        global_state: dict[str, torch.Tensor],
        honest_state: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        if honest_state is None:
            raise ValueError("SignFlipAttack needs the honest state to flip.")
        diff = state_dict_diff(global_state, honest_state)
        return {k: global_state[k].float() - self.scale * diff[k] for k in global_state}


# ----- 2. constant attack ---------------------------------------------------


class ConstantAttack(ByzantineAttack):
    """Submit a state-dict that pushes every parameter by a fixed constant.

    This is the textbook Byzantine attack — ignores local training entirely
    and broadcasts a deterministic poisoned update. Loud and easily caught
    by a norm bound, but useful as a baseline.
    """

    def __init__(self, value: float = 1e2):
        self.value = value

    def craft_state(
        self,
        global_state: dict[str, torch.Tensor],
        honest_state: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        return {k: global_state[k].float() + self.value for k in global_state}


# ----- 3. gradient-suppression (deanonymization via collusion) --------------


class GradientSuppressionAttack(ByzantineAttack):
    """Saturating-magnitude attack used to deanonymize one target honest client.

    All malicious clients submit very large updates of magnitude `magnitude`.
    Half of them push positive (`+m`), half negative (`-m`). Their sum at
    the server cancels out. Once divided by N (the FedAvg average), what's
    left is dominated by whichever client is asymmetric — and if N-1 of N
    clients are colluding malicious + 1 honest target, the only asymmetry
    in the average is the target client's update.

    Result: the global state after round = (target's Δ) / N, scaled and
    visible. An external observer of the global state can recover an
    estimate of the target's update — i.e., a privacy attack via poisoning.

    The construction here puts half the malicious clients at `+magnitude`
    and the other half at `-magnitude`, with one client absorbing any odd
    parity to preserve the cancellation property. Pass the malicious-client
    index `index_in_malicious_group` (0-based) to choose which malicious
    persona this instance plays.
    """

    def __init__(
        self,
        magnitude: float = 1e6,
        index_in_malicious_group: int = 0,
        n_malicious: int = 4,
    ):
        if n_malicious < 2:
            raise ValueError("Gradient suppression requires ≥ 2 malicious clients.")
        if not (0 <= index_in_malicious_group < n_malicious):
            raise ValueError("index_in_malicious_group out of range.")
        self.magnitude = magnitude
        self.index = index_in_malicious_group
        self.n_malicious = n_malicious

    def _sign(self) -> float:
        # Cancel pairwise. With odd n_malicious, the last one absorbs the
        # parity (so sum = 0 exactly when n is even, and = ±magnitude when
        # n is odd — fine, the asymmetry is small relative to magnitude).
        return 1.0 if self.index < self.n_malicious // 2 else -1.0

    def craft_state(
        self,
        global_state: dict[str, torch.Tensor],
        honest_state: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        s = self._sign()
        return {k: global_state[k].float() + s * self.magnitude for k in global_state}


# ----- 4. backdoor / model-replacement (stub) -------------------------------


class BackdoorAttack(ByzantineAttack):
    """Stub: scale a backdoor-trained delta to overwrite the global model.

    Full implementation deferred — backdoor training requires a poisoned
    sub-dataset and a target trigger pattern; not in scope for v0.3 ship.
    Raises NotImplementedError on use so the slot is reserved.
    """

    def craft_state(
        self,
        global_state: dict[str, torch.Tensor],
        honest_state: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError(
            "BackdoorAttack is reserved as a v0.4 placeholder; see docs/ROADMAP.md."
        )


# ----- 5. StealthSuppression (defense-aware adversary) ----------------------


class StealthSuppression(ByzantineAttack):
    """Defense-aware version of GradientSuppression.

    Designed to slip past coordinate-wise median, trimmed-mean and Krum by
    staying within an *estimated* honest-client norm envelope. The attacker:

    1. Receives the global state at round start.
    2. Estimates an honest-client update norm `target_norm` (in practice, by
       observing prior rounds' broadcast aggregates and deriving a typical
       per-client norm; here we accept it as a hyperparameter).
    3. Builds a unit-norm direction shared across the malicious group, with
       half of them along `+u` and half along `-u`. The malicious updates
       cancel like in `GradientSuppression` but each individual update has
       L2 norm `target_norm`, so it stays inside any norm bound a defense
       might enforce.
    4. The unit direction `u` can be `bias_toward_target_direction` if the
       attacker wants to bias the residual; left zero in the default.

    The defense is harder to detect because each malicious update *looks*
    like a plausible honest update (same norm, no obviously aberrant
    coordinates). Robust aggregators that operate per-coordinate (median,
    trimmed mean) are *not* fooled by this if the cancellation creates a
    visible per-coordinate bias; Krum is harder to fool when malicious
    clients are far from honest in direction space. The experiments matrix
    quantifies which defenses break.
    """

    def __init__(
        self,
        target_norm: float = 1.0,
        index_in_malicious_group: int = 0,
        n_malicious: int = 4,
        seed: int = 0,
    ):
        if n_malicious < 2:
            raise ValueError("StealthSuppression requires ≥ 2 malicious clients.")
        if not (0 <= index_in_malicious_group < n_malicious):
            raise ValueError("index_in_malicious_group out of range.")
        self.target_norm = target_norm
        self.index = index_in_malicious_group
        self.n_malicious = n_malicious
        self.seed = seed

    def _sign(self) -> float:
        return 1.0 if self.index < self.n_malicious // 2 else -1.0

    def craft_state(
        self,
        global_state: dict[str, torch.Tensor],
        honest_state: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        # Build a deterministic shared unit-direction across malicious group.
        # Generate on CPU then move to the param's device — torch.Generator can
        # only be CPU- or CUDA-bound at construction, and we want one source.
        cpu_gen = torch.Generator(device="cpu").manual_seed(self.seed)
        direction = {
            k: torch.randn(v.shape, generator=cpu_gen, dtype=torch.float32).to(v.device)
            for k, v in global_state.items()
        }
        # Normalize globally to L2 = target_norm
        flat = torch.cat([d.flatten() for d in direction.values()])
        norm = flat.norm(2)
        scale = self.target_norm / (norm + 1e-8)
        s = self._sign()
        return {k: global_state[k].float() + s * scale * direction[k] for k in global_state}


# ----- factory --------------------------------------------------------------


def build_attack(
    name: str,
    *,
    n_malicious: int = 4,
    index: int = 0,
    magnitude: float = 1e6,
    target_norm: float = 1.0,
) -> ByzantineAttack:
    """Build an attack instance by name. Used by `scripts/byzantine.py`."""
    if name == "sign_flip":
        return SignFlipAttack()
    if name == "constant":
        return ConstantAttack(value=magnitude)
    if name == "suppression":
        return GradientSuppressionAttack(
            magnitude=magnitude, index_in_malicious_group=index, n_malicious=n_malicious
        )
    if name == "stealth":
        return StealthSuppression(
            target_norm=target_norm, index_in_malicious_group=index, n_malicious=n_malicious
        )
    if name == "backdoor":
        return BackdoorAttack()
    raise ValueError(f"unknown attack: {name!r}")
