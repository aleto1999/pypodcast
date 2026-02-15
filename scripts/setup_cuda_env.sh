#!/bin/bash
# setup CUDA library paths from nvidia python packages.
# source this script before running python: source scripts/setup_cuda_env.sh

# find nvidia cuda packages in the uv virtual environment.
VENV_PATH="${VIRTUAL_ENV:-.venv}"

if [ ! -d "$VENV_PATH" ]; then
    echo "Error: virtual environment not found at $VENV_PATH"
    return 1 2>/dev/null || exit 1
fi

# find nvidia package directories.
NVIDIA_DIRS=$(find "$VENV_PATH/lib" -type d -name "nvidia" 2>/dev/null | head -1)

if [ -z "$NVIDIA_DIRS" ]; then
    echo "Error: nvidia packages not found in $VENV_PATH"
    return 1 2>/dev/null || exit 1
fi

# collect library paths.
LIB_PATHS=""

for pkg in cuda_runtime cublas cudnn; do
    PKG_DIR="$NVIDIA_DIRS/$pkg"
    if [ -d "$PKG_DIR/lib" ]; then
        LIB_PATHS="$PKG_DIR/lib:$LIB_PATHS"
    elif [ -d "$PKG_DIR" ]; then
        # check if .so files exist directly in package dir
        if ls "$PKG_DIR"/*.so* >/dev/null 2>&1; then
            LIB_PATHS="$PKG_DIR:$LIB_PATHS"
        fi
    fi
done

# update LD_LIBRARY_PATH.
if [ -n "$LIB_PATHS" ]; then
    export LD_LIBRARY_PATH="$LIB_PATHS:$LD_LIBRARY_PATH"
    echo "✓ CUDA libraries added to LD_LIBRARY_PATH"
    echo "  Paths: $LIB_PATHS"
else
    echo "✗ No CUDA libraries found"
    return 1 2>/dev/null || exit 1
fi
