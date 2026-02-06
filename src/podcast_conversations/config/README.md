# Configuration Package

Central configuration constants for podcast conversations analysis.

## Overview

This package provides default configuration values used across the codebase, primarily for diarization settings. It ensures consistent defaults while allowing runtime overrides via command-line arguments.

## Configuration Constants

### Diarization Model

```python
DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
```

The default HuggingFace model for speaker diarization. This is a state-of-the-art model from pyannote.audio that:
- Identifies speaker segments in audio
- Assigns speaker labels (SPEAKER_00, SPEAKER_01, etc.)
- Handles overlapping speech
- Works with any number of speakers

**Override at runtime:**
```bash
./scripts/diarize_optimized.sh --model "pyannote/speaker-diarization-2.1"
```

### Device Configuration

```python
DEFAULT_DEVICE = "cuda:0"
```

Default compute device constant for programmatic use. Options:
- `cuda:0` - First CUDA GPU (default constant)
- `cuda:1` - Second CUDA GPU
- `mps` - Apple Silicon GPU (Metal Performance Shaders)
- `cpu` - CPU processing

**Note:** CLI scripts use `--device auto` as their default, which auto-detects the best available device (CUDA → MPS → CPU). The constant `DEFAULT_DEVICE` is used when explicitly referenced in code.

**Override at runtime:**
```bash
# Use auto-detection (recommended)
uv run python scripts/diarize_batch.py --device auto ...

# Or specify explicitly
uv run python scripts/diarize_batch.py --device cuda:1 ...
uv run python scripts/diarize_batch.py --device mps ...  # Apple Silicon
uv run python scripts/diarize_batch.py --device cpu ...
```

### Optimization Defaults

```python
DEFAULT_USE_BF16 = True
DEFAULT_USE_COMPILE = False
DEFAULT_COMPILE_MODE = "reduce-overhead"
```

**BFloat16 Mixed Precision:**
- `DEFAULT_USE_BF16 = True` - Enable BFloat16 for faster inference
- Requires H100, A100, or newer GPUs
- Automatically falls back to FP32 if unsupported

**torch.compile:**
- `DEFAULT_USE_COMPILE = False` - Disabled by default for multiprocessing
- Can be enabled with `--torch-compile` flag
- Provides 20-40% speedup after initial compilation

**Compilation Mode:**
- `DEFAULT_COMPILE_MODE = "reduce-overhead"` - Balanced optimization
- Options: `default`, `reduce-overhead`, `max-autotune`
- Only applies when torch.compile is enabled

## Usage

### Importing Constants

```python
from podcast_conversations.config import (
    DEFAULT_DIARIZATION_MODEL,
    DEFAULT_DEVICE,
    DEFAULT_USE_BF16,
    DEFAULT_USE_COMPILE,
    DEFAULT_COMPILE_MODE,
)

# Use in your code
model_name = DEFAULT_DIARIZATION_MODEL
device = DEFAULT_DEVICE
```

### Runtime Overrides

Command-line scripts allow overriding defaults:

```bash
# Override model
./scripts/diarize_optimized.sh \
  --model "custom/model-name" \
  --transcripts-dir outputs/transcripts \
  --audio-base-dir /path/to/audio \
  --output-dir outputs/diarizations

# Override device
./scripts/diarize_optimized.sh \
  --device cuda:1 \
  --transcripts-dir outputs/transcripts \
  --audio-base-dir /path/to/audio \
  --output-dir outputs/diarizations

# Override optimizations
./scripts/diarize_optimized.sh \
  --no-bf16 \
  --torch-compile \
  --compile-mode max-autotune \
  --transcripts-dir outputs/transcripts \
  --audio-base-dir /path/to/audio \
  --output-dir outputs/diarizations
```

### In Python Scripts

```python
from podcast_conversations.config import DEFAULT_DEVICE, DEFAULT_USE_BF16
from podcast_conversations.diarization import DiarizationPipeline

# Use defaults
pipeline = DiarizationPipeline(
    device=DEFAULT_DEVICE,
    use_bf16=DEFAULT_USE_BF16,
)

# Or override
pipeline = DiarizationPipeline(
    device="cuda:1",
    use_bf16=False,
)
```

## Configuration Files

While this package provides Python constants, external configuration files are located in the `config/` directory at the project root:

- `config/taxonomy.yaml` - Dehumanization taxonomy for embeddings method
- `config/keyword_analysis_config.yaml` - Keywords and settings for keyword analysis
- `config/classifiers.yaml` - Transformer models for utterance classification

See respective package documentation for details on these configuration files.

## Modifying Defaults

To change default values permanently:

1. Edit `src/podcast_conversations/config/__init__.py`
2. Update the constant values
3. Changes apply to all scripts using defaults

**Example:**
```python
# Change default device to CPU
DEFAULT_DEVICE = "cpu"

# Disable BFloat16 by default
DEFAULT_USE_BF16 = False

# Enable torch.compile by default
DEFAULT_USE_COMPILE = True
```

**Note:** It's generally better to use runtime overrides than modifying defaults.

## Best Practices

### Use Defaults

Use constants for consistency:
```python
# Good
from podcast_conversations.config import DEFAULT_DEVICE
pipeline = DiarizationPipeline(device=DEFAULT_DEVICE)

# Less ideal
pipeline = DiarizationPipeline(device="cuda:0")
```

### Override When Needed

Override via arguments, not hardcoded values:
```python
# Good
def process(device: str = DEFAULT_DEVICE):
    pipeline = DiarizationPipeline(device=device)

# Less ideal
def process():
    pipeline = DiarizationPipeline(device="cuda:0")
```

### Document Overrides

When overriding defaults, document why:
```python
# Use CPU for compatibility with systems without GPU
pipeline = DiarizationPipeline(device="cpu", use_bf16=False)
```

## Architecture

### Centralized Configuration

Benefits of centralized constants:
- **Consistency**: Same defaults across all scripts
- **Maintainability**: Single place to update defaults
- **Discoverability**: Easy to find configuration options
- **Documentation**: Clear documentation of defaults

### Configuration Hierarchy

```
1. Package defaults (this module)
   ↓
2. Environment variables (e.g., HF_TOKEN)
   ↓
3. Configuration files (YAML)
   ↓
4. Command-line arguments (highest priority)
```

## Examples

### Basic Usage

```python
from podcast_conversations.config import (
    DEFAULT_DIARIZATION_MODEL,
    DEFAULT_DEVICE,
)

print(f"Using model: {DEFAULT_DIARIZATION_MODEL}")
print(f"Using device: {DEFAULT_DEVICE}")
```

### Conditional Configuration

```python
import torch
from podcast_conversations.config import DEFAULT_DEVICE, DEFAULT_USE_BF16

# Auto-detect GPU availability
if torch.cuda.is_available():
    device = DEFAULT_DEVICE
    use_bf16 = DEFAULT_USE_BF16
else:
    device = "cpu"
    use_bf16 = False

print(f"Device: {device}, BF16: {use_bf16}")
```

### Configuration for Different Environments

```python
import os
from podcast_conversations.config import DEFAULT_DEVICE

# Development: use CPU
if os.getenv("ENVIRONMENT") == "development":
    device = "cpu"
# Production: use default GPU
else:
    device = DEFAULT_DEVICE
```

## Related Documentation

- [Diarization Package](../diarization/README.md) - Uses these configuration constants
- [Main README.md](../../../README.md) - Overall project configuration

## Future Enhancements

Potential future additions to this package:
- Configuration validation
- Environment-specific configs (dev/prod)
- Configuration schema definitions
- Runtime configuration reloading

For now, the simple constant-based approach provides clarity and ease of use.
