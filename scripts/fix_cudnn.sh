#!/bin/bash
# Quick fix for cuDNN missing library on NVIDIA GPU servers

set -e

echo "🔧 Installing cuDNN for CUDA 12.x..."

# Check if we're in a uv environment
if command -v uv &> /dev/null; then
    echo "Using uv to install nvidia-cudnn..."
    uv pip install nvidia-cudnn-cu12
elif command -v pip &> /dev/null; then
    echo "Using pip to install nvidia-cudnn..."
    pip install nvidia-cudnn-cu12
else
    echo "❌ Error: Neither uv nor pip found"
    exit 1
fi

echo ""
echo "✅ cuDNN installed successfully"
echo ""
echo "Verifying CUDA/cuDNN setup..."

python3 << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("⚠️  Warning: CUDA not available")
EOF

echo ""
echo "🎉 Setup complete! You can now run transcription with GPU acceleration."
