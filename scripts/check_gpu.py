#!/usr/bin/env python3
"""diagnostic script to check gpu detection."""

import sys

print("=== GPU Detection Diagnostic ===\n")

# check torch.
print("1. checking pytorch...")
try:
    import torch
    print(f"   ✓ torch version: {torch.__version__}")
    print(f"   • cuda available: {torch.cuda.is_available()}")
    print(f"   • cuda version: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A'}")
    
    if torch.cuda.is_available():
        print(f"   • device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"   • gpu {i}: {props.name} ({props.total_memory / (1024**3):.1f} GB)")
    else:
        print("   ✗ no cuda devices detected by pytorch")
except ImportError as e:
    print(f"   ✗ torch not installed: {e}")
except Exception as e:
    print(f"   ✗ error: {e}")

print()

# check nvidia-smi.
print("2. checking nvidia-smi...")
import subprocess
try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode == 0:
        print("   ✓ nvidia-smi output:")
        for line in result.stdout.strip().split("\n"):
            print(f"     {line}")
    else:
        print(f"   ✗ nvidia-smi failed: {result.stderr}")
except FileNotFoundError:
    print("   ✗ nvidia-smi not found in PATH")
except Exception as e:
    print(f"   ✗ error: {e}")

print()

# check pynvml.
print("3. checking pynvml...")
try:
    import pynvml
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    print(f"   ✓ pynvml initialized")
    print(f"   • device count: {device_count}")
    
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"   • gpu {i}: {name} ({mem_info.total / (1024**3):.1f} GB)")
    
    pynvml.nvmlShutdown()
except ImportError as e:
    print(f"   ✗ pynvml not installed: {e}")
except Exception as e:
    print(f"   ✗ error: {e}")

print()

# check cuda environment variables.
print("4. checking cuda environment variables...")
import os
cuda_vars = {k: v for k, v in os.environ.items() if "CUDA" in k or "NVIDIA" in k}
if cuda_vars:
    for k, v in cuda_vars.items():
        print(f"   • {k}={v}")
else:
    print("   • no cuda-related environment variables found")

print()

# check ld_library_path.
print("5. checking LD_LIBRARY_PATH...")
ld_path = os.environ.get("LD_LIBRARY_PATH", "")
if ld_path:
    for path in ld_path.split(":"):
        if "cuda" in path.lower() or "nvidia" in path.lower():
            print(f"   • {path}")
    if not any("cuda" in p.lower() or "nvidia" in p.lower() for p in ld_path.split(":")):
        print("   • no cuda/nvidia paths found in LD_LIBRARY_PATH")
else:
    print("   • LD_LIBRARY_PATH not set")

print("\n=== End Diagnostic ===")
