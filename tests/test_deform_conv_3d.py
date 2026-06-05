"""Tests for 3D deformable convolution."""

import pytest
import torch
from deformable_conv_3d import ConvOffset3D, deform_conv3d


class TestDeformConv3D:
    """Test suite for 3D deformable convolution."""

    def test_deform_conv3d_creation(self):
        """Test that deform_conv3d layer can be created."""
        layer = deform_conv3d(in_c=16, out_c=32, kernel_size=(3, 3, 3))
        assert layer is not None
        assert len(layer) == 2  # Conv3d + ConvOffset3D

    def test_deform_conv3d_forward(self):
        """Test forward pass with 3D input."""
        # Create layer and input
        layer = deform_conv3d(in_c=16, out_c=32, kernel_size=(3, 3, 3))
        x = torch.rand(2, 16, 8, 8, 8)  # (batch, channels, depth, height, width)

        # Forward pass
        output = layer(x)

        # Check output shape
        assert output.shape[0] == 2  # batch size
        assert output.shape[1] == 32  # output channels
        # Spatial dimensions reduced by convolution without padding
        assert output.shape[2] == 6  # depth
        assert output.shape[3] == 6  # height
        assert output.shape[4] == 6  # width

    def test_conv_offset_3d_forward(self):
        """Test ConvOffset3D layer forward pass."""
        layer = ConvOffset3D(in_channels=16)
        x = torch.rand(2, 16, 8, 8, 8)

        output = layer(x)

        # Output should have same shape as input
        assert output.shape == x.shape

    def test_deform_conv3d_with_padding(self):
        """Test deform_conv3d with padding to preserve spatial dimensions."""
        layer = deform_conv3d(in_c=16, out_c=32, kernel_size=(3, 3, 3), padding=1)
        x = torch.rand(2, 16, 8, 8, 8)

        output = layer(x)

        # With padding=1 and kernel=3, spatial dimensions should be preserved
        assert output.shape == (2, 32, 8, 8, 8)

    def test_deform_conv3d_gradient_flow(self):
        """Test that gradients flow through the layer."""
        layer = deform_conv3d(in_c=16, out_c=32, kernel_size=(3, 3, 3))
        x = torch.rand(1, 16, 8, 8, 8, requires_grad=True)

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
    @pytest.mark.parametrize("in_channels", [8, 16, 32])
    def test_different_batch_and_channel_sizes(self, batch_size, in_channels):
        """Test with different batch sizes and channel counts."""
        layer = deform_conv3d(in_c=in_channels, out_c=32, kernel_size=(3, 3, 3))
        x = torch.rand(batch_size, in_channels, 8, 8, 8)

        output = layer(x)

        assert output.shape[0] == batch_size
        assert output.shape[1] == 32
