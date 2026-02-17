#!/bin/bash
# detailed cuda diagnostic for pytorch linking issues.

echo "=== CUDA Linking Diagnostic ==="
echo ""

echo "1. Environment check..."
if [[ "$LD_LIBRARY_PATH" == *"nvidia"* ]]; then
    echo "   ✓ LD_LIBRARY_PATH contains nvidia paths"
else
    echo "   ✗ LD_LIBRARY_PATH does NOT contain nvidia paths"
    echo "   Run: export LD_LIBRARY_PATH=\$(find .venv/lib/python*/site-packages/nvidia -name 'lib' -type d | tr '\n' ':')\$LD_LIBRARY_PATH"
    exit 1
fi

echo ""
echo "2. PyTorch CUDA error details..."
uv run python -c "
import torch
import os

print('   • LD_LIBRARY_PATH set:', 'nvidia' in os.environ.get('LD_LIBRARY_PATH', ''))
print('   • Torch version:', torch.__version__)
print('   • CUDA compiled version:', torch.version.cuda)
print('   • CUDA available:', torch.cuda.is_available())

if not torch.cuda.is_available():
    print('')
    print('   Attempting CUDA initialization...')
    try:
        _ = torch.zeros(1).cuda()
    except Exception as e:
        print(f'   ✗ Error: {e}')
        import traceback
        print('')
        print('   Full traceback:')
        traceback.print_exc()
"

echo ""
echo "3. Checking PyTorch CUDA library dependencies..."
TORCH_CUDA_LIB=$(find .venv/lib/python*/site-packages/torch/lib -name "libtorch_cuda.so" 2>/dev/null | head -1)

if [ -n "$TORCH_CUDA_LIB" ]; then
    echo "   Found: $TORCH_CUDA_LIB"
    echo "   Checking library dependencies (first 30 lines)..."
    ldd "$TORCH_CUDA_LIB" 2>&1 | head -30
else
    echo "   ✗ libtorch_cuda.so not found"
fi

echo ""
echo "4. Checking for missing CUDA libraries..."
ldd "$TORCH_CUDA_LIB" 2>&1 | grep "not found" || echo "   ✓ All libraries found"

echo ""
echo "=== End Diagnostic ==="
