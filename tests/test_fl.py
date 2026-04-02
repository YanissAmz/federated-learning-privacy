import torch

from src.fl.model import SimpleCNN


class TestSimpleCNN:
    def test_output_shape(self):
        model = SimpleCNN(num_classes=10, in_channels=3)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)

    def test_mnist_channels(self):
        model = SimpleCNN(num_classes=10, in_channels=1)
        # MNIST is 28x28, need to adjust or pad
        x = torch.randn(2, 1, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)
