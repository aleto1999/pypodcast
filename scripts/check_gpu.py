#!/usr/bin/env python3
"""diagnostic script to check gpu detection."""

import sys
import os

# CRITICAL: setup cuda library paths BEFORE importing torch.
try:
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
        print(f"✓ setup: added {len(lib_dirs)} nvidia library paths\n", file=sys.stderr)
except Exception as e:
    print(f"✗ setup failed: {e}\n", file=sys.stderr)

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

# check pynvml/nvidia-ml-py.
print("3. checking nvidia management library...")
try:
    import pynvml
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    print(f"   ✓ nvidia-ml initialized")
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
    print(f"   ✗ nvidia-ml not installed: {e}")
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

print()

# check for cuda libraries in python packages.
print("6. checking bundled cuda libraries...")
try:
    import torch
    import pathlib
    torch_path = pathlib.Path(torch.__file__).parent
    
    # check for cuda libraries in torch package
    cuda_libs = list(torch_path.rglob("*cudart*.so*"))
    if cuda_libs:
        print(f"   ✓ found {len(cuda_libs)} cudart libraries in torch:")
        for lib in cuda_libs[:3]:
            print(f"     {lib}")
    else:
        print("   ✗ no cudart libraries found in torch package")
    
    # check nvidia packages
    nvidia_packages = [
        "nvidia.cuda_runtime",
        "nvidia.cublas", 
        "nvidia.cudnn",
    ]
    
    for pkg in nvidia_packages:
        try:
            mod = __import__(pkg)
            pkg_path = pathlib.Path(mod.__file__).parent
            libs = list(pkg_path.rglob("*.so*"))
            if libs:
                print(f"   ✓ {pkg}: found {len(libs)} libraries")
            else:
                print(f"   • {pkg}: installed but no .so files found")
        except ImportError:
            print(f"   ✗ {pkg}: not installed")
        except Exception as e:
            print(f"   • {pkg}: error - {e}")
            
except Exception as e:
    print(f"   ✗ error checking packages: {e}")

print()

# check detailed torch cuda initialization.
print("7. torch cuda initialization details...")
try:
    import torch
    print(f"   • torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"   • torch.version.cuda: {torch.version.cuda}")
    print(f"   • torch.backends.cudnn.enabled: {torch.backends.cudnn.enabled}")
    
    # try to get more error details
    try:
        print(f"   • attempting torch.cuda.init()...")
        torch.cuda.init()
        print(f"   ✓ torch.cuda.init() succeeded")
    except Exception as e:
        print(f"   ✗ torch.cuda.init() failed: {e}")
        
except Exception as e:
    print(f"   ✗ error: {e}")

print("\n=== End Diagnostic ===")
