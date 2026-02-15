#!/usr/bin/env python3
"""test nvidia driver and cuda runtime compatibility."""

import os
import sys

# setup cuda paths first
from pathlib import Path

nvidia_packages = ["nvidia.cuda_runtime", "nvidia.cublas", "nvidia.cudnn"]
lib_dirs = []

for pkg_name in nvidia_packages:
    try:
        parts = pkg_name.split(".")
        mod = __import__(pkg_name)
        for part in parts[1:]:
            mod = getattr(mod, part)
        
        pkg_path = Path(mod.__file__).parent
        for candidate in [pkg_path / "lib", pkg_path]:
            if candidate.exists() and list(candidate.glob("*.so*")):
                lib_dirs.append(str(candidate))
                break
    except (ImportError, AttributeError):
        continue

if lib_dirs:
    current = os.environ.get("LD_LIBRARY_PATH", "")
    new_paths = ":".join(lib_dirs)
    os.environ["LD_LIBRARY_PATH"] = f"{new_paths}:{current}" if current else new_paths

print("=== NVIDIA Driver vs CUDA Runtime Test ===\n")

# check nvidia-smi driver version
print("1. NVIDIA Driver Version:")
import subprocess
result = subprocess.run(
    ["nvidia-smi", "--query-gpu=driver_version,cuda_version", "--format=csv,noheader"],
    capture_output=True,
    text=True,
)
if result.returncode == 0:
    driver_ver, driver_cuda = result.stdout.strip().split(",")
    print(f"   Driver: {driver_ver}")
    print(f"   Driver CUDA: {driver_cuda.strip()}")
else:
    print(f"   Error: {result.stderr}")

print("\n2. PyTorch CUDA Runtime Version:")
import torch
print(f"   Compiled with: CUDA {torch.version.cuda}")

print("\n3. Testing CUDA Runtime directly:")
try:
    import ctypes
    # try to load cudart and get version
    for lib_dir in lib_dirs:
        cudart_files = list(Path(lib_dir).glob("libcudart.so.12*"))
        if cudart_files:
            lib = ctypes.CDLL(str(cudart_files[0]))
            
            # get CUDA runtime version
            version = ctypes.c_int()
            result = lib.cudaRuntimeGetVersion(ctypes.byref(version))
            if result == 0:
                cuda_version = version.value
                major = cuda_version // 1000
                minor = (cuda_version % 1000) // 10
                print(f"   ✓ CUDA Runtime {major}.{minor} loaded successfully")
            else:
                print(f"   ✗ cudaRuntimeGetVersion failed with code {result}")
            
            # try to get device count
            device_count = ctypes.c_int()
            result = lib.cudaGetDeviceCount(ctypes.byref(device_count))
            print(f"   cudaGetDeviceCount result: {result}")
            if result == 0:
                print(f"   ✓ Device count: {device_count.value}")
            elif result == 35:  # cudaErrorInsufficientDriver
                print(f"   ✗ Error 35: Insufficient driver (driver too old for CUDA 12.4)")
                print(f"   Your driver supports CUDA {driver_cuda.strip()}, but PyTorch needs CUDA 12.4")
            elif result == 100:  # cudaErrorNoDevice
                print(f"   ✗ Error 100: No CUDA devices found")
            else:
                print(f"   ✗ Error code: {result}")
            break
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n4. PyTorch CUDA availability:")
print(f"   torch.cuda.is_available(): {torch.cuda.is_available()}")

print("\n=== End Test ===")
