import torch
from deformable_conv_3d import deform_conv3d

print("Testing 3D deformable convolution...")

# Test with the same example from the README
image = torch.rand(1, 16, 64, 64, 64)
def_conv3 = deform_conv3d(in_c=16, out_c=32, kernel_size=(3,3,3))

print(f"Input shape: {image.shape}")

try:
    result = def_conv3(image)
    print(f"Output shape: {result.shape}")
    print("SUCCESS: No errors!")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
