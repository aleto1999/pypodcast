#!/bin/bash
# wrapper script to run transcription with CUDA libraries properly configured.
# usage: ./scripts/transcribe_cuda.sh [args...]

set -e

# find the script directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# activate virtual environment if not already active.
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d "$PROJECT_ROOT/.venv" ]; then
        source "$PROJECT_ROOT/.venv/bin/activate"
    else
        echo "Error: virtual environment not found at $PROJECT_ROOT/.venv"
        echo "Run: uv sync --extra gpu"
        exit 1
    fi
fi

# find nvidia cuda library paths.
VENV_LIB="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia"

if [ ! -d "$VENV_LIB" ]; then
    echo "Error: nvidia packages not found"
    echo "Run: uv sync --extra gpu"
    exit 1
fi

# build LD_LIBRARY_PATH with all nvidia libraries.
CUDA_LIB_PATHS=""

for pkg in cuda_runtime cublas cudnn cuda_nvrtc nvtx cusparse cusolver cufft curand; do
    LIB_DIR="$VENV_LIB/$pkg/lib"
    if [ -d "$LIB_DIR" ]; then
        CUDA_LIB_PATHS="$LIB_DIR:$CUDA_LIB_PATHS"
    fi
done

# export with existing LD_LIBRARY_PATH.
export LD_LIBRARY_PATH="$CUDA_LIB_PATHS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "✓ CUDA libraries configured"
echo "  Using GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

# run transcription script with all arguments.
exec python "$SCRIPT_DIR/transcribe_batch.py" "$@"
