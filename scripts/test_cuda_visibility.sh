#!/bin/bash
# test cuda device visibility

echo "=== CUDA Device Visibility Test ==="
echo ""

echo "1. Current CUDA_VISIBLE_DEVICES:"
echo "   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
echo ""

echo "2. nvidia-smi device list:"
nvidia-smi --query-gpu=index,name,uuid --format=csv
echo ""

echo "3. Testing with CUDA_VISIBLE_DEVICES unset:"
unset CUDA_VISIBLE_DEVICES
export LD_LIBRARY_PATH=$(find .venv/lib/python*/site-packages/nvidia -name "lib" -type d | tr '\n' ':')$LD_LIBRARY_PATH
uv run python -c "import torch; print('   CUDA available:', torch.cuda.is_available()); print('   Device count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"
echo ""

echo "4. Testing with CUDA_VISIBLE_DEVICES=0:"
export CUDA_VISIBLE_DEVICES=0
uv run python -c "import torch; print('   CUDA available:', torch.cuda.is_available()); print('   Device count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"
echo ""

echo "5. Testing with CUDA_VISIBLE_DEVICES=0,1,2:"
export CUDA_VISIBLE_DEVICES=0,1,2
uv run python -c "import torch; print('   CUDA available:', torch.cuda.is_available()); print('   Device count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"
echo ""

echo "6. Checking GPU permissions:"
ls -la /dev/nvidia* 2>&1 | head -10
echo ""

echo "=== End Test ==="
