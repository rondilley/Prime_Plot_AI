"""Tests for ML models."""

import numpy as np
import pytest
import torch

from prime_plot.ml.models import SimpleUNet, PrimeClassifier, create_model


class TestSimpleUNet:
    """Tests for SimpleUNet model."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        model = SimpleUNet(in_channels=1, out_channels=1)
        x = torch.randn(2, 1, 256, 256)

        output = model(x)

        assert output.shape == (2, 1, 256, 256)

    def test_custom_features(self):
        """Test with custom feature dimensions."""
        model = SimpleUNet(
            in_channels=1,
            out_channels=1,
            features=[32, 64, 128],
        )
        x = torch.randn(1, 1, 128, 128)

        output = model(x)

        assert output.shape == (1, 1, 128, 128)

    def test_different_input_size(self):
        """Test with different input sizes."""
        model = SimpleUNet()

        for size in [64, 128, 256]:
            x = torch.randn(1, 1, size, size)
            output = model(x)
            assert output.shape == (1, 1, size, size)


class TestPrimeClassifier:
    """Tests for PrimeClassifier model."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        model = PrimeClassifier(in_channels=1, num_classes=2)
        x = torch.randn(4, 1, 64, 64)

        output = model(x)

        assert output.shape == (4, 2)

    def test_single_class(self):
        """Test with single output class."""
        model = PrimeClassifier(in_channels=1, num_classes=1)
        x = torch.randn(2, 1, 32, 32)

        output = model(x)

        assert output.shape == (2, 1)


class TestCreateModel:
    """Tests for model factory function."""

    def test_create_simple_unet(self):
        """Test creating simple U-Net."""
        model = create_model("simple_unet")
        assert isinstance(model, SimpleUNet)

    def test_create_classifier(self):
        """Test creating classifier."""
        model = create_model("classifier")
        assert isinstance(model, PrimeClassifier)

    def test_invalid_model_type(self):
        """Test that invalid model type raises error."""
        with pytest.raises(ValueError):
            create_model("invalid_model")

    def test_pass_kwargs(self):
        """Test passing kwargs to model."""
        model = create_model("classifier", num_classes=5)
        x = torch.randn(1, 1, 32, 32)
        output = model(x)
        assert output.shape == (1, 5)


class TestGradients:
    """Tests for gradient flow."""

    def test_unet_gradients(self):
        """Test that gradients flow through U-Net."""
        model = SimpleUNet(features=[16, 32])
        x = torch.randn(1, 1, 64, 64, requires_grad=True)

        output = model(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_classifier_gradients(self):
        """Test that gradients flow through classifier."""
        model = PrimeClassifier()
        x = torch.randn(1, 1, 32, 32, requires_grad=True)

        output = model(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
