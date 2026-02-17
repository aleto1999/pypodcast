#!/usr/bin/env python3
"""check why is_available is false when device_count is 1"""

import torch

print("=== PyTorch CUDA Status ===")
print(f"torch.__version__: {torch.__version__}")
print(f"torch.version.cuda: {torch.version.cuda}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
print()

# check internal cuda state
print("=== Internal CUDA State ===")
try:
    print(f"torch.cuda.is_initialized(): {torch.cuda.is_initialized()}")
    print(f"torch.cuda._is_compiled(): {torch.cuda._is_compiled()}")
except Exception as e:
    print(f"Error checking internal state: {e}")

print()

# try to initialize
print("=== Attempting CUDA Initialization ===")
try:
    torch.cuda.init()
    print("✓ torch.cuda.init() succeeded")
except Exception as e:
    print(f"✗ torch.cuda.init() failed: {e}")

print()

# try to get device properties
print("=== Device Properties ===")
try:
    if torch.cuda.device_count() > 0:
        props = torch.cuda.get_device_properties(0)
        print(f"✓ Device 0: {props.name}")
        print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"  Compute: {props.major}.{props.minor}")
except Exception as e:
    print(f"✗ Error: {e}")

print()

# try creating a tensor
print("=== Tensor Creation Test ===")
try:
    t = torch.tensor([1.0])
    print(f"CPU tensor: {t}")
    
    t_cuda = t.cuda()
    print(f"✓ CUDA tensor created: {t_cuda}")
    print(f"✓ CUDA WORKS!")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
