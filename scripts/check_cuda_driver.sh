#!/bin/bash
# comprehensive cuda driver check

echo "=== CUDA Driver Compatibility Check ==="
echo ""

echo "1. Driver version:"
nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1
echo ""

echo "2. Driver CUDA version:"
nvidia-smi | grep "CUDA Version" || nvidia-smi --query-gpu=cuda_version --format=csv,noheader | head -1
echo ""

echo "3. GPU compute mode (should be 'Default' or '0'):"
nvidia-smi --query-gpu=compute_mode --format=csv,noheader
echo ""

echo "4. GPU persistence mode:"
nvidia-smi --query-gpu=persistence_mode --format=csv,noheader
echo ""

echo "5. Detailed GPU info:"
nvidia-smi -q | grep -A 5 "Product Name\|Driver Version\|CUDA Version\|Compute Mode"
echo ""

echo "6. Test basic CUDA sample (if available):"
if command -v /usr/local/cuda*/samples/bin/*/deviceQuery &> /dev/null; then
    /usr/local/cuda*/samples/bin/*/deviceQuery | grep -A 3 "CUDA Driver"
else
    echo "   CUDA samples not found, trying manual driver check..."
    
    # check if driver library exists
    if [ -f /usr/lib64/libcuda.so.1 ] || [ -f /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then
        echo "   ✓ NVIDIA driver library found"
        
        # try loading it
        export LD_LIBRARY_PATH=$(find .venv/lib/python*/site-packages/nvidia -name "lib" -type d | tr '\n' ':')$LD_LIBRARY_PATH
        
        python3 << 'EOF'
import ctypes
import os

# try to load driver library
driver_paths = ['/usr/lib64/libcuda.so.1', '/usr/lib/x86_64-linux-gnu/libcuda.so.1']
for path in driver_paths:
    if os.path.exists(path):
        try:
            lib = ctypes.CDLL(path)
            print(f"   ✓ Loaded NVIDIA driver library: {path}")
            
            # try cuInit
            result = lib.cuInit(0)
            if result == 0:
                print("   ✓ cuInit succeeded")
            else:
                print(f"   ✗ cuInit failed with error: {result}")
            break
        except Exception as e:
            print(f"   ✗ Failed to load {path}: {e}")
EOF
    else
        echo "   ✗ NVIDIA driver library not found"
    fi
fi
echo ""

echo "7. Check if running in container/restricted environment:"
if [ -f /.dockerenv ]; then
    echo "   Running in Docker container"
elif grep -q docker /proc/1/cgroup 2>/dev/null; then
    echo "   Running in containerized environment"
elif command -v systemd-detect-virt &> /dev/null; then
    VIRT=$(systemd-detect-virt)
    if [ "$VIRT" != "none" ]; then
        echo "   Virtualization detected: $VIRT"
    else
        echo "   No virtualization detected"
    fi
else
    echo "   No container/virtualization detected"
fi

echo ""
echo "=== End Check ==="
