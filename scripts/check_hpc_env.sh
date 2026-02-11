#!/bin/bash
# check hpc environment and gpu access

echo "=== HPC Environment Check ==="
echo ""

echo "1. Current hostname and node type:"
hostname
echo ""

echo "2. Check if on compute node with GPU access:"
if command -v squeue &> /dev/null; then
    echo "   SLURM detected"
    echo "   SLURM_JOB_ID: ${SLURM_JOB_ID:-<not set>}"
    echo "   SLURM_GPUS: ${SLURM_GPUS:-<not set>}"
    echo "   SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-<not set>}"
elif command -v qstat &> /dev/null; then
    echo "   PBS/Torque detected"
else
    echo "   No job scheduler detected"
fi
echo ""

echo "3. Check cgroup GPU restrictions:"
if [ -f /sys/fs/cgroup/devices/devices.list ]; then
    echo "   Checking cgroup device allowlist:"
    grep "195:" /sys/fs/cgroup/devices/devices.list 2>/dev/null || echo "   No GPU devices (195:*) in cgroup allowlist"
elif [ -f /sys/fs/cgroup/user.slice/*/devices.list ]; then
    echo "   Checking cgroup v2:"
    cat /sys/fs/cgroup/user.slice/*/devices.list 2>/dev/null | grep "195:" || echo "   No GPU devices in cgroup"
else
    echo "   Could not check cgroup restrictions"
fi
echo ""

echo "4. Environment modules:"
if command -v module &> /dev/null; then
    echo "   Module system available"
    echo "   Loaded modules:"
    module list 2>&1 | head -10
    echo ""
    echo "   Available CUDA modules:"
    module avail cuda 2>&1 | grep -i cuda | head -5
else
    echo "   No module system found"
fi
echo ""

echo "5. Check if GPUs are accessible to this process:"
echo "   Trying to open /dev/nvidia0:"
python3 << 'EOF'
import os
try:
    fd = os.open("/dev/nvidia0", os.O_RDWR)
    print("   ✓ Successfully opened /dev/nvidia0")
    os.close(fd)
except PermissionError:
    print("   ✗ Permission denied - cgroup or security restriction")
except Exception as e:
    print(f"   ✗ Error: {e}")
EOF
echo ""

echo "6. Check for MIG (Multi-Instance GPU) mode:"
nvidia-smi -L 2>&1 | head -5
echo ""

echo "=== Recommendations ==="
echo "If on login node:"
echo "  - Request GPU compute node with: salloc --gpus=1 or srun --gpus=1"
echo "  - Or submit batch job with GPU allocation"
echo ""
echo "If modules available:"
echo "  - Load CUDA module: module load cuda/12.1 or similar"
echo ""

echo "=== End Check ==="
