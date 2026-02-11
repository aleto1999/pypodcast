#!/bin/bash
# check if pytorch can find the nvidia driver library

echo "=== NVIDIA Driver Library Path Check ==="
echo ""

echo "1. System driver library location:"
find /usr/lib* -name "libcuda.so*" 2>/dev/null
echo ""

echo "2. Current LD_LIBRARY_PATH contains /usr/lib64:"
echo "$LD_LIBRARY_PATH" | grep -q "/usr/lib64" && echo "   ✓ Yes" || echo "   ✗ No"
echo ""

echo "3. Testing with system driver library in path:"
export LD_LIBRARY_PATH=/usr/lib64:$(find .venv/lib/python*/site-packages/nvidia -name "lib" -type d | tr '\n' ':')$LD_LIBRARY_PATH
echo "   LD_LIBRARY_PATH now includes: /usr/lib64"
echo ""

echo "4. Test CUDA with driver library accessible:"
uv run python << 'EOF'
import torch
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA compiled: {torch.version.cuda}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   Device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    # try direct CUDA init to get error code
    try:
        import ctypes
        lib = ctypes.CDLL("libcuda.so.1")
        result = lib.cuInit(0)
        print(f"   cuInit result: {result}")
    except Exception as e:
        print(f"   cuInit error: {e}")
EOF

echo ""
echo "=== End Check ==="
