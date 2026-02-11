#!/usr/bin/env python3
"""
setup cuda library paths from nvidia python packages.

run this before importing torch to ensure cuda libraries are found.
"""

import os
import sys
from pathlib import Path


def setup_cuda_path():
    """add nvidia cuda libraries to LD_LIBRARY_PATH."""
    nvidia_packages = [
        "nvidia.cuda_runtime",
        "nvidia.cublas",
        "nvidia.cudnn",
    ]
    
    lib_dirs = []
    
    for pkg_name in nvidia_packages:
        try:
            # import the package to find its location
            parts = pkg_name.split(".")
            mod = __import__(pkg_name)
            for part in parts[1:]:
                mod = getattr(mod, part)
            
            # find library directory
            pkg_path = Path(mod.__file__).parent
            
            # look for lib directory or .so files
            lib_candidates = [
                pkg_path / "lib",
                pkg_path,
            ]
            
            for candidate in lib_candidates:
                if candidate.exists() and list(candidate.glob("*.so*")):
                    lib_dirs.append(str(candidate))
                    break
                    
        except (ImportError, AttributeError):
            continue
    
    if lib_dirs:
        current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        new_paths = ":".join(lib_dirs)
        
        if current_ld_path:
            os.environ["LD_LIBRARY_PATH"] = f"{new_paths}:{current_ld_path}"
        else:
            os.environ["LD_LIBRARY_PATH"] = new_paths
        
        print(f"✓ added {len(lib_dirs)} nvidia library paths to LD_LIBRARY_PATH", file=sys.stderr)
        return True
    else:
        print("✗ no nvidia cuda libraries found", file=sys.stderr)
        return False


if __name__ == "__main__":
    setup_cuda_path()
    
    # test if it worked
    import torch
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
