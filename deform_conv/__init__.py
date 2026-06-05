"""Deformable convolution implementation in 2D and 3D for PyTorch.

This package provides PyTorch implementations of deformable convolution layers
for both 2D and 3D inputs, with learnable offset fields.
"""

__version__ = "0.1.0"

# Import from parent directory modules for backward compatibility
import sys
from pathlib import Path

# Add parent directory to path to import the modules
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from deformable_conv_2d import ConvOffset2D
from deformable_conv_3d import ConvOffset3D, deform_conv3d

__all__ = [
    "ConvOffset2D",
    "ConvOffset3D",
    "deform_conv3d",
    "__version__",
]
