#!/bin/bash
# check if system cuda module has cudnn 8

echo "=== System CUDA cuDNN Check ==="
echo ""

echo "1. Load CUDA module:"
module load cuda/12.1.1
module list 2>&1 | grep cuda
echo ""

echo "2. Check for system cuDNN libraries:"
find /usr/local/cuda* -name "libcudnn*.so*" 2>/dev/null | head -10
find /usr/lib* -name "libcudnn*.so*" 2>/dev/null | head -10
ls -la /usr/local/cuda/lib64/libcudnn* 2>/dev/null || echo "Not in /usr/local/cuda/lib64"
echo ""

echo "3. Check module environment:"
echo "CUDA_HOME: ${CUDA_HOME:-not set}"
echo "CUDNN_PATH: ${CUDNN_PATH:-not set}"
echo ""

echo "4. Available cudnn modules:"
module avail cudnn 2>&1 | grep -i cudnn
echo ""

echo "=== End Check ==="
