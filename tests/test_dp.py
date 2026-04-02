import torch

from src.defenses.dp import add_noise, apply_dp, clip_gradients


class TestClipGradients:
    def test_clips_large_gradients(self):
        grads = [torch.ones(10) * 100]
        clipped = clip_gradients(grads, max_norm=1.0)
        flat = torch.cat([g.flatten() for g in clipped])
        assert flat.norm(2).item() <= 1.0 + 1e-6

    def test_preserves_small_gradients(self):
        grads = [torch.ones(3) * 0.1]
        clipped = clip_gradients(grads, max_norm=10.0)
        assert torch.allclose(clipped[0], grads[0])


class TestAddNoise:
    def test_changes_gradients(self):
        grads = [torch.zeros(100)]
        noisy = add_noise(grads, noise_multiplier=1.0, max_norm=1.0)
        assert not torch.allclose(noisy[0], grads[0])

    def test_noise_scale(self):
        torch.manual_seed(42)
        grads = [torch.zeros(10000)]
        noisy = add_noise(grads, noise_multiplier=2.0, max_norm=1.0)
        # Noise should have std ~ 2.0
        assert 1.5 < noisy[0].std().item() < 2.5


class TestApplyDP:
    def test_full_pipeline(self):
        grads = [torch.ones(10) * 100]
        private = apply_dp(grads, max_norm=1.0, noise_multiplier=0.5)
        assert len(private) == 1
        assert private[0].shape == grads[0].shape
