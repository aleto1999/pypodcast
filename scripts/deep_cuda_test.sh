#!/bin/bash
# deep dive into cuda runtime and driver linkage

echo "=== CUDA Runtime-Driver Linkage Test ==="
echo ""

# setup environment
export LD_LIBRARY_PATH=/usr/lib64:$(find .venv/lib/python*/site-packages/nvidia -name "lib" -type d | tr '\n' ':')$LD_LIBRARY_PATH

echo "1. Check what libcudart.so.12 links to:"
CUDART=$(find .venv -name "libcudart.so.12" | head -1)
if [ -n "$CUDART" ]; then
    echo "   Found: $CUDART"
    echo "   Dependencies:"
    ldd "$CUDART" | grep -E "libcuda|not found"
fi
echo ""

echo "2. Check PyTorch's CUDA C++ library dependencies:"
TORCH_CUDA=$(find .venv -path "*/torch/lib/libtorch_cuda.so" | head -1)
if [ -n "$TORCH_CUDA" ]; then
    echo "   Found: $TORCH_CUDA"
    echo "   Checking for libcuda.so linkage:"
    ldd "$TORCH_CUDA" | grep -E "libcuda.so|not found" | head -5
fi
echo ""

echo "3. Try loading libcuda.so.1 directly with Python:"
uv run python << 'EOF'
import ctypes
import ctypes.util

# find libcuda
libcuda_path = ctypes.util.find_library("cuda")
print(f"   find_library('cuda'): {libcuda_path}")

# try to load it
try:
    cuda = ctypes.CDLL("libcuda.so.1")
    print("   ✓ libcuda.so.1 loaded")
    
    # initialize
    result = cuda.cuInit(0)
    print(f"   cuInit(0) returned: {result}")
    
    if result == 0:
        # get version
        version = ctypes.c_int()
        cuda.cuDriverGetVersion(ctypes.byref(version))
        print(f"   ✓ Driver version: {version.value}")
        
        # get device count
        count = ctypes.c_int()
        result = cuda.cuDeviceGetCount(ctypes.byref(count))
        print(f"   cuDeviceGetCount returned: {result}, count: {count.value}")
    else:
        print(f"   ✗ cuInit failed - error code meanings:")
        print(f"      100 = CUDA_ERROR_NO_DEVICE")
        print(f"      999 = CUDA_ERROR_UNKNOWN")
        
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
EOF
echo ""

echo "4. Check for CUDA_ERROR_NO_DEVICE causes:"
echo "   Possible reasons for error 100:"
echo "   a) GPU is in exclusive process mode (checked: Default)"
echo "   b) User lacks permissions (checked: /dev/nvidia* are rw)"
echo "   c) Driver module not loaded properly"
echo ""

echo "5. Check loaded kernel modules:"
lsmod | grep nvidia | head -10
echo ""

echo "6. Check nvidia device info from kernel:"
cat /proc/driver/nvidia/gpus/*/information 2>/dev/null | grep -E "Model|IRQ|VBIOS" | head -10
echo ""

echo "7. Try strace to see system calls:"
echo "   Running minimal CUDA test with strace..."
uv run strace -e openat,access python -c "import torch; torch.cuda.is_available()" 2>&1 | grep -E "libcuda|nvidia" | head -20

echo ""
echo "=== End Test ==="
