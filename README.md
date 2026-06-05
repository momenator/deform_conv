# Deformable Convolution in 2D and 3D

PyTorch implementation of deformable convolution layers for 2D and 3D inputs with learnable offset fields.

## Installation

### From source

```bash
# Clone the repository
git clone https://github.com/momenator/deform_conv.git
cd deform_conv

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Requirements

- Python >= 3.8
- PyTorch >= 1.7.0
- NumPy >= 1.19.0

## Usage

### 3D Deformable Convolution

```python
import torch
from deformable_conv_3d import deform_conv3d

# Create input tensor (batch_size, channels, depth, height, width)
image = torch.rand(1, 16, 64, 64, 64)

# Create 3D deformable convolution layer
def_conv3 = deform_conv3d(in_c=16, out_c=32, kernel_size=(3, 3, 3))

# Forward pass
result = def_conv3(image)
print(result.shape)  # torch.Size([1, 32, 62, 62, 62])
```

### 2D Deformable Convolution

```python
import torch
from deformable_conv_2d import ConvOffset2D

# Create input tensor (batch_size, channels, height, width)
image = torch.rand(1, 3, 32, 32)

# Create 2D deformable convolution layer
def_conv2 = ConvOffset2D(filters=3)

# Forward pass
result = def_conv2(image)
print(result.shape)  # torch.Size([1, 3, 32, 32])
```

### Using the package import

```python
import torch
from deform_conv import deform_conv3d, ConvOffset2D, ConvOffset3D

# Use as shown above
```

## Development

### Setup development environment

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Run code formatter
black .

# Run linter
ruff check .

# Run type checker
mypy deformable_conv_2d.py deformable_conv_3d.py
```

### Running tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=deform_conv --cov-report=html
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

Adapted from [pytorch-deform-conv](https://github.com/oeway/pytorch-deform-conv/)

