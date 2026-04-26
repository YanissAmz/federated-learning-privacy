"""Unit + integration tests for Byzantine attacks and robust aggregators.

The end-to-end test asserts the central v0.3 claim: with 2 sign-flippers in
a 5-client federation, the median aggregator preserves accuracy while the
mean aggregator collapses it.
"""

from __future__ import annotations

import math

import torch

from src.attacks.byzantine import (
    ConstantAttack,
    GradientSuppressionAttack,
    SignFlipAttack,
    StealthSuppression,
    build_attack,
)
from src.defenses.dp import state_dict_diff
from src.defenses.robust import (
    aggregate_krum,
    aggregate_mean,
    aggregate_median,
    aggregate_trimmed_mean,
    build_aggregator,
)
from src.fl.model import SimpleCNN


def _toy_state(seed: int = 0) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return {"w": torch.randn(8, 8), "b": torch.randn(8)}


def _norm(state: dict[str, torch.Tensor]) -> float:
    return math.sqrt(sum(v.float().pow(2).sum().item() for v in state.values()))


# ----- attacks --------------------------------------------------------------


class TestSignFlipAttack:
    def test_flips_sign_of_diff(self):
        glob = _toy_state(0)
        # honest pushes +1 on every coord
        honest = {k: v + 1.0 for k, v in glob.items()}
        attacked = SignFlipAttack().craft_state(glob, honest)
        # attacked diff should be -1 (the negation)
        diff = state_dict_diff(glob, attacked)
        for v in diff.values():
            assert torch.allclose(v, torch.full_like(v, -1.0))

    def test_requires_honest_state(self):
        try:
            SignFlipAttack().craft_state(_toy_state(), None)
        except ValueError:
            return
        raise AssertionError("expected ValueError when honest_state is None")


class TestConstantAttack:
    def test_pushes_by_constant(self):
        glob = _toy_state()
        out = ConstantAttack(value=42.0).craft_state(glob)
        for k in glob:
            assert torch.allclose(out[k], glob[k] + 42.0)


class TestGradientSuppression:
    def test_pairwise_cancellation_in_sum(self):
        glob = _toy_state()
        # 4 malicious clients, magnitude 1000. Half +, half -.
        outs = [
            GradientSuppressionAttack(
                magnitude=1000.0, index_in_malicious_group=i, n_malicious=4
            ).craft_state(glob)
            for i in range(4)
        ]
        # sum of diffs should be ~0 (perfect cancellation, even N)
        diffs = [state_dict_diff(glob, o) for o in outs]
        sum_diff = {k: sum(d[k] for d in diffs) for k in glob}
        for v in sum_diff.values():
            assert v.abs().max().item() < 1e-3

    def test_large_individual_magnitude(self):
        glob = _toy_state()
        out = GradientSuppressionAttack(
            magnitude=1e6, index_in_malicious_group=0, n_malicious=4
        ).craft_state(glob)
        diff = state_dict_diff(glob, out)
        norm = math.sqrt(sum(v.pow(2).sum().item() for v in diff.values()))
        # 8*8 + 8 = 72 entries each at magnitude 1e6 → norm ~ √72 × 1e6
        assert norm > 1e6


class TestStealthSuppression:
    def test_norm_respects_target(self):
        glob = _toy_state()
        target = 1.0
        out = StealthSuppression(
            target_norm=target, index_in_malicious_group=0, n_malicious=4
        ).craft_state(glob)
        diff = state_dict_diff(glob, out)
        norm = math.sqrt(sum(v.pow(2).sum().item() for v in diff.values()))
        assert abs(norm - target) < 1e-3, f"stealth norm {norm} ≉ target {target}"


class TestFactory:
    def test_known_names(self):
        for name in ["sign_flip", "constant", "suppression", "stealth"]:
            assert build_attack(name, n_malicious=4, index=0) is not None

    def test_unknown_raises(self):
        try:
            build_attack("not_a_real_attack")
        except ValueError:
            return
        raise AssertionError("expected ValueError for unknown attack")


# ----- aggregators ----------------------------------------------------------


class TestAggregators:
    """Aggregators on a controlled mix of honest + malicious diffs."""

    def setup_method(self):
        torch.manual_seed(0)
        self.honest = [{"w": torch.randn(4, 4) * 0.1, "b": torch.zeros(4)} for _ in range(3)]
        # Malicious: one extreme outlier with constant 100
        self.malicious = {"w": torch.full((4, 4), 100.0), "b": torch.full((4,), 100.0)}

    def test_mean_is_corrupted_by_outlier(self):
        states = [*self.honest, self.malicious]
        out = aggregate_mean(states)
        # Mean should be dragged toward malicious: ~25 per coord
        assert out["w"].mean().item() > 10.0

    def test_median_resists_one_outlier(self):
        states = [*self.honest, self.malicious]
        out = aggregate_median(states)
        assert out["w"].abs().max().item() < 1.0

    def test_trimmed_mean_drops_extreme(self):
        states = [*self.honest, self.malicious]
        out = aggregate_trimmed_mean(states, trim_ratio=0.25)
        # Drops 1 from each side → 2 honest survive
        assert out["w"].abs().max().item() < 1.0

    def test_krum_picks_a_close_client(self):
        states = [*self.honest, self.malicious]
        out = aggregate_krum(states, n_byzantine=1)
        # Should pick one of the 3 honest clients (small norm), not the outlier (~100)
        norm = _norm(out)
        assert norm < 5.0

    def test_factory(self):
        for name in ["mean", "median", "trimmed_mean", "krum"]:
            f = build_aggregator(name, trim_ratio=0.2, n_byzantine=1)
            assert callable(f)


# ----- end-to-end story: median saves us, mean collapses --------------------


class TestE2EByzantineStory:
    """The central v0.3 claim, asserted on a tiny, deterministic CIFAR-shaped task.

    Setup: 5 clients of fake i.i.d. data → 1 round FedAvg.
    With 2 sign-flippers + mean: weights drift large, accuracy collapses.
    With 2 sign-flippers + median: weights stay sane, accuracy preserved.

    "Accuracy" here is a proxy: distance of the round-1 global state to
    the honest-only mean. The closer to honest-mean, the better the
    defense recovered the underlying signal.
    """

    def test_median_recovers_honest_signal(self):
        torch.manual_seed(0)
        glob = SimpleCNN().state_dict()

        # honest "trained" states: small structured perturbation of glob
        honest_states = [
            {k: v + torch.randn_like(v.float()) * 0.01 for k, v in glob.items()} for _ in range(3)
        ]

        # 2 sign-flippers — flip a small honest update so they bias the average
        # by 0.01 * 4 in the wrong direction (much larger than the honest signal
        # but not big enough to be filtered by norm-bound).
        flipped_states = []
        for _ in range(2):
            honest_clone = {k: v + torch.randn_like(v.float()) * 0.01 for k, v in glob.items()}
            flipped = SignFlipAttack(scale=10.0).craft_state(glob, honest_clone)
            flipped_states.append(flipped)

        all_states = honest_states + flipped_states

        # Reference: what should we converge toward? The honest-only mean.
        honest_only_mean = aggregate_mean(honest_states)

        mean_corrupted = aggregate_mean(all_states)
        median_recovered = aggregate_median(all_states)

        def diff_norm(a, b):
            return math.sqrt(sum((a[k].float() - b[k].float()).pow(2).sum().item() for k in a))

        d_mean = diff_norm(mean_corrupted, honest_only_mean)
        d_median = diff_norm(median_recovered, honest_only_mean)

        # The central claim of v0.3: median must be closer to the honest-only
        # mean than the corrupted mean. The size of the gap quantifies how
        # much the defense saved us.
        assert d_median < d_mean, (
            f"median ({d_median:.4f}) should be closer to honest-only mean than "
            f"corrupted mean ({d_mean:.4f})"
        )
        # And the gap must be substantial — at least 30% of the corrupted-mean
        # error is recovered by the defense.
        assert (d_mean - d_median) / d_mean > 0.3, (
            f"median should recover at least 30% of mean's error: "
            f"mean={d_mean:.3f}, median={d_median:.3f}, "
            f"ratio={(d_mean - d_median) / d_mean:.3f}"
        )
