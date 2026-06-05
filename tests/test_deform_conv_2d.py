"""Tests for 2D deformable convolution."""

import pytest
import torch
from deformable_conv_2d import ConvOffset2D


class TestDeformConv2D:
    """Test suite for 2D deformable convolution."""

    def test_conv_offset_2d_creation(self):
        """Test that ConvOffset2D layer can be created."""
        layer = ConvOffset2D(filters=16)
        assert layer is not None

    def test_conv_offset_2d_forward(self):
        """Test forward pass with 2D input."""
        layer = ConvOffset2D(filters=16)
        x = torch.rand(2, 16, 32, 32)  # (batch, channels, height, width)

        output = layer(x)

        # Output should have same shape as input (with padding=1)
        assert output.shape == x.shape

    def test_conv_offset_2d_gradient_flow(self):
        """Test that gradients flow through the layer."""
        layer = ConvOffset2D(filters=16)
        x = torch.rand(1, 16, 32, 32, requires_grad=True)

        # Forward pass
        output = layer(x)
        loss = output.sum()

        # Backward pass
        loss.backward()

        # Check that gradients exist
        assert x.grad is not None
        for param in layer.parameters():
            assert param.grad is not None

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("channels", [3, 16, 32])
    def test_different_batch_and_channel_sizes(self, batch_size, channels):
        """Test with different batch sizes and channel counts."""
        layer = ConvOffset2D(filters=channels)
        x = torch.rand(batch_size, channels, 32, 32)

        output = layer(x)

        assert output.shape == (batch_size, channels, 32, 32)
