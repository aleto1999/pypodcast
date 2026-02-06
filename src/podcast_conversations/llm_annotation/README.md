# LLM Annotation Package

Large language model-based annotation for podcast utterances using local GPU inference with multi-platform support for NVIDIA GPUs (CUDA) and Apple Silicon (MPS).

## Overview

This package provides high-performance LLM-based annotation of podcast transcripts for detecting hate speech, identifying target groups, classifying hate speech types, detecting advertisements, and extracting topics. It's optimized for multiple platforms including HPC environments with NVIDIA GPUs and Apple Silicon Macs.

## Platform Support

| Platform | Quantization | Flash Attention | Recommended Models | Memory |
|----------|--------------|-----------------|-------------------|--------|
| **NVIDIA CUDA** | 4-bit/8-bit (bitsandbytes) | Yes | Llama 3.3 70B | H100 80GB, A100 40GB |
| **Apple Silicon MPS** | float16 (or MLX 4-bit) | No | Llama 3.2 7B-13B | M2/M3 Pro/Max 32GB+ |
| **CPU** | float32 | No | Small models | 32GB+ RAM |

## Features

- **Hate Speech Detection**: Binary classification for harmful content
- **Target Group Identification**: Identifies 6 target population groups
- **Hate Speech Type Classification**: Classifies into 4 dehumanization strategies
- **Advertisement Detection**: Identifies promotional content
- **Topic Extraction**: Extracts main topic per utterance
- **Multi-Platform Support**: Auto-detects NVIDIA CUDA, Apple Silicon MPS, or CPU
- **Local GPU Inference**: Direct model inference without API calls
- **4-bit/8-bit Quantization**: Memory-efficient inference with bitsandbytes (NVIDIA)
- **Apple Silicon Optimization**: Unified memory detection and MPS backend support
- **MLX Integration**: Native Apple Silicon quantization support via mlx-lm
- **Flash Attention 2**: Optimized attention for faster processing (NVIDIA)
- **torch.compile()**: 10-30% inference speedup on CUDA devices (auto-enabled)
- **Dynamic Batching**: Auto-detects optimal batch size based on device and memory
- **Parallel I/O Pipeline**: Concurrent file loading/saving overlaps with GPU compute
- **Cross-File Batching**: Batches segments across multiple files for optimal GPU utilization
- **Prefetching**: Loads files ahead of processing to minimize GPU idle time
- **Throughput Tracking**: Real-time metrics for segments/second and files/minute
- **Smart Caching**: Skips already-annotated files
- **Error Handling**: Graceful degradation with error annotations

## Installation

The package is automatically installed with the main project:

```bash
uv sync
```

### NVIDIA GPU Setup

**Additional requirements for local inference:**
```bash
uv pip install torch transformers bitsandbytes accelerate
```

**Optional for Flash Attention 2 (2-4x faster):**
```bash
uv pip install flash-attn --no-build-isolation
```

### Apple Silicon Setup

For Apple Silicon Macs (M1/M2/M3/M4), the pipeline automatically uses MPS:

```bash
# Core requirements (automatically installed)
uv pip install torch transformers accelerate
```

**Optional for native quantization support:**
```bash
# Install MLX for Apple Silicon optimized inference
pip install mlx mlx-lm

# Pre-quantize a model to 4-bit
mlx_lm.convert --hf-path meta-llama/Llama-3.2-8B-Instruct -q --q-bits 4 -o ./llama-3.2-8b-4bit

# Run inference with quantized model
mlx_lm.generate --model ./llama-3.2-8b-4bit --prompt "Hello"
```

**Note:** bitsandbytes and Flash Attention 2 are not available on Apple Silicon. The pipeline automatically uses float16 precision. For memory-efficient inference with quantization, use MLX.

## Configuration

### Annotation Configuration

Default configuration in `AnnotationConfig`:

```python
# Target Groups
target_groups = [
    "racial and ethnic minorities",
    "religious minorities",
    "women",
    "lgbtq+ population",
    "disabled (physical and/or mental) population",
    "immigrant population",
]

# Hate Speech Types
hate_speech_types = [
    "Threat to culture or identity",
    "Threat to survival or physical security",
    "Vilification or villainization",
    "Explicit dehumanization",
]

# Model Configuration
model_name = "meta-llama/Llama-3.3-70B-Instruct"
temperature = 0.1  # Low temperature for consistent annotations
max_tokens = 256
timeout = 60  # seconds

# Retry Configuration
max_retries = 3
retry_delay = 1.0  # seconds
retry_backoff = 2.0  # exponential backoff multiplier
```

## Usage

### Python API

```python
from pathlib import Path
from podcast_conversations.llm_annotation import (
    # Configuration
    AnnotationConfig,
    load_questions_config,
    # Annotator
    LLMAnnotator,
    detect_optimal_batch_size,
    # File processing
    process_transcript_file,
    check_annotations_exist,
    # Output
    save_annotation_results,
)

# Initialize configuration
config = AnnotationConfig(
    model_name="meta-llama/Llama-3.3-70B-Instruct",
    temperature=0.1,
    max_tokens=256,
)

# Or load from YAML config file
questions_config = load_questions_config("config/llm_questions.yaml")

# Get optimal batch size for your GPU
batch_size = detect_optimal_batch_size()

# Create annotator with local GPU inference
annotator = LLMAnnotator(
    config=config,
    device="cuda",
    batch_size=batch_size,
    use_flash_attention=True,
    load_in_4bit=True,  # 4-bit quantization for memory efficiency
)

# Check if output already exists (useful for resuming)
input_path = Path("outputs/transcripts/episode.json")
output_path = Path("outputs/analysis/llm_annotation_method/episode.json")

if not check_annotations_exist(output_path):
    # Process single file
    result = process_transcript_file(
        input_path=input_path,
        output_path=output_path,
        annotator=annotator,
        skip_existing=True,
    )

    print(f"Annotated {result['segments_annotated']} segments")

    # Save results summary
    save_annotation_results(
        output_dir=Path("outputs/analysis/llm_annotation_method"),
        results=[result],
    )

# Unload model when done
annotator.unload_model()
```

### Batch Processing

```python
from pathlib import Path
from podcast_conversations.llm_annotation import (
    AnnotationConfig,
    LLMAnnotator,
)
import json

config = AnnotationConfig()
annotator = LLMAnnotator(
    config=config,
    load_in_4bit=True,
    use_flash_attention=True,
)

# Load transcript
with open("outputs/transcripts/episode.json") as f:
    transcript_data = json.load(f)

segments = transcript_data.get("segments", [])
texts = [seg.get("text", "") for seg in segments]

# Annotate all segments in batches
annotations = annotator.annotate_batch(texts)

# Add annotations to segments
for segment, annotation in zip(segments, annotations):
    segment["llm_annotation"] = annotation

# Save annotated transcript
with open("outputs/analysis/llm_annotation_method/episode.json", "w") as f:
    json.dump(transcript_data, f, indent=2, ensure_ascii=False)

# Cleanup
annotator.unload_model()
```

### High-Throughput Parallel Processing

For maximum throughput when processing many files, use the parallel processing pipeline:

```python
from pathlib import Path
from podcast_conversations.llm_annotation import (
    AnnotationConfig,
    LLMAnnotator,
    ParallelAnnotationPipeline,
    FileTask,
)

# Initialize annotator
config = AnnotationConfig()
annotator = LLMAnnotator(
    config=config,
    load_in_4bit=True,
    use_flash_attention=True,
)

# Create parallel pipeline
pipeline = ParallelAnnotationPipeline(
    annotator=annotator,
    prefetch_count=4,          # Files to load ahead
    io_workers=2,              # File loading threads
    save_workers=2,            # File saving threads
    cross_file_batching=True,  # Batch segments across files
    max_files_per_batch=8,     # Max files per GPU batch
)

# Create file tasks
input_dir = Path("outputs/transcripts")
output_dir = Path("outputs/analysis/llm_annotation_method")
file_tasks = [
    FileTask(input_path=f, output_path=output_dir / f.relative_to(input_dir))
    for f in input_dir.rglob("*.json")
]

# Process with progress callback
def on_progress(processed: int, total: int):
    print(f"Progress: {processed}/{total}")

files_processed, files_failed, total_segments = pipeline.process_files(
    file_tasks,
    progress_callback=on_progress,
)

# Get throughput metrics
metrics = pipeline.get_metrics_summary()
print(f"Segments/second: {metrics['segments_per_second']:.2f}")
print(f"Files/minute: {metrics['files_per_minute']:.2f}")
print(f"Avg batch time: {metrics['avg_batch_time_ms']:.1f}ms")

# Cleanup
annotator.unload_model()
```

**Parallel Pipeline Features:**

| Feature | Description |
|---------|-------------|
| **Prefetching** | Loads files in background while GPU processes |
| **Cross-File Batching** | Combines segments from multiple files for optimal batch size |
| **Async Saving** | Saves files in background while processing continues |
| **Throughput Metrics** | Tracks segments/second, files/minute, batch times |

## Annotation Schema

Each utterance is annotated with the following fields:

```json
{
  "has_hate_speech": true,
  "has_advertisement": false,
  "target_group": "immigrant population",
  "hate_speech_type": "Threat to culture or identity",
  "main_topic": "immigration policy debate"
}
```

**Fields:**
- `has_hate_speech` (boolean): Whether utterance contains hate speech
- `has_advertisement` (boolean): Whether utterance is promotional content
- `target_group` (string|null): Target population if hate speech detected
- `hate_speech_type` (string|null): Type of dehumanization strategy used
- `main_topic` (string): Brief description of utterance's main topic

**Validation Rules:**
- `target_group` and `hate_speech_type` are `null` when `has_hate_speech` is `false`
- `main_topic` is always present (1-10 words)
- JSON-only response from LLM (no markdown or additional text)

## Input/Output Formats

### Input Format

Transcript JSON files with segments:

```json
{
  "metadata": {
    "title": "Episode Title",
    "show_name": "Show Name"
  },
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 5.2,
      "text": "They refuse to assimilate into our culture.",
      "speaker": "SPEAKER_00"
    }
  ]
}
```

### Output Format

Enriched transcript with `llm_annotation` field added to each segment:

```json
{
  "metadata": {
    "title": "Episode Title",
    "show_name": "Show Name"
  },
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 5.2,
      "text": "They refuse to assimilate into our culture.",
      "speaker": "SPEAKER_00",
      "llm_annotation": {
        "has_hate_speech": true,
        "has_advertisement": false,
        "target_group": "immigrant population",
        "hate_speech_type": "Threat to culture or identity",
        "main_topic": "immigration and cultural assimilation"
      }
    }
  ]
}
```

## Batch Size Optimization

### Automatic Detection

The package automatically detects optimal batch size based on device type and memory:

```python
from podcast_conversations.llm_annotation import (
    detect_optimal_batch_size,
    detect_device,
    DeviceType,
)

# Auto-detect device and get optimal batch size
device_info = detect_device()
batch_size = detect_optimal_batch_size(
    gpu_memory_gb=device_info.total_memory_gb,
    device_type=device_info.device_type,
)
print(f"Device: {device_info.name}")
print(f"Recommended batch size: {batch_size}")
```

**NVIDIA GPU Batch Sizes:**

| GPU Memory | Recommended Batch Size |
|------------|------------------------|
| >= 80 GB (H100) | 8 |
| >= 40 GB (A100) | 4 |
| >= 24 GB (RTX 3090/4090) | 2 |
| < 24 GB | 1 |

**Apple Silicon Batch Sizes:**

| Unified Memory | Chip | Recommended Batch Size |
|----------------|------|------------------------|
| >= 128 GB | M2/M3 Ultra | 8 |
| >= 64 GB | M2/M3 Max | 4 |
| >= 32 GB | M2/M3 Pro | 2 |
| < 32 GB | M1/M2/M3 | 1 |

**Note:** Apple Silicon uses unified memory shared between CPU and GPU, so batch sizes are more conservative to leave headroom for the OS and other applications.

### Manual Configuration

```python
# Conservative (lower memory usage)
annotator = LLMAnnotator(config, batch_size=1)

# For H100 80GB (NVIDIA)
annotator = LLMAnnotator(config, batch_size=8)

# For Apple Silicon (no quantization available via bitsandbytes)
annotator = LLMAnnotator(config, device="mps", batch_size=2)

# Disable quantization (requires more memory, NVIDIA only)
annotator = LLMAnnotator(config, load_in_4bit=False, load_in_8bit=False)

# Auto-detect device (recommended)
annotator = LLMAnnotator(config, device=None)  # auto-detects best device
```

## Performance

### Memory Requirements

**NVIDIA GPUs - Llama 3.3 70B with 4-bit quantization:**
- Model: ~35 GB
- KV cache + activations: ~40 GB
- Total: ~75 GB (fits H100 80GB)

**8-bit quantization:**
- Model: ~70 GB
- Requires tensor parallelism or A100 80GB

**Apple Silicon - Llama 3.2 8B with float16:**
- Model: ~16 GB
- KV cache + activations: ~8 GB
- Total: ~24 GB (fits M2/M3 Pro 32GB+)

**Apple Silicon - With MLX 4-bit quantization:**
- Model: ~4 GB (8B model)
- KV cache: ~4 GB
- Total: ~8 GB (fits most Apple Silicon Macs)

### torch.compile() Optimization

On NVIDIA GPUs with CUDA, `torch.compile()` is automatically applied to the model for 10-30% faster inference:

```python
# Auto-enabled by default on CUDA devices
annotator = LLMAnnotator(config, use_torch_compile=True)

# Disable if experiencing issues
annotator = LLMAnnotator(config, use_torch_compile=False)
```

**CLI usage:**
```bash
# Enable torch.compile (default on CUDA)
uv run python scripts/annotate_with_llm.py --torch-compile ...

# Disable torch.compile
uv run python scripts/annotate_with_llm.py --no-torch-compile ...
```

**Requirements:**
- PyTorch 2.0+ (auto-detected)
- CUDA device only (skipped on MPS/CPU)
- Uses `reduce-overhead` compilation mode for optimal balance

### Quantization Options

**NVIDIA (bitsandbytes):**

| Option | Memory | Speed | Quality |
|--------|--------|-------|---------|
| 4-bit (NF4) | ~35 GB | Fast | Good |
| 8-bit | ~70 GB | Medium | Better |
| FP16 | ~140 GB | Fastest | Best |

**Apple Silicon (MLX):**

| Option | Memory | Speed | Quality |
|--------|--------|-------|---------|
| 4-bit (MLX) | ~4 GB (8B) | Fast | Good |
| 8-bit (MLX) | ~8 GB (8B) | Medium | Better |
| float16 | ~16 GB (8B) | Fast | Best |

## Apple Silicon Guide

### Recommended Models

For Apple Silicon, we recommend smaller models due to unified memory constraints:

| Model | Size | Unified Memory Required |
|-------|------|------------------------|
| Llama 3.2 1B | 1B | 4 GB |
| Llama 3.2 3B | 3B | 8 GB |
| Llama 3.2 8B | 8B | 24 GB |
| Llama 3.2 13B | 13B | 48 GB |
| Llama 3.3 70B | 70B | 192 GB (M2 Ultra max) |

### Using MLX for Quantized Inference

MLX provides native Apple Silicon support with quantization:

```bash
# Install MLX
pip install mlx mlx-lm

# Download and quantize a model
mlx_lm.convert \
    --hf-path meta-llama/Llama-3.2-8B-Instruct \
    -q \
    --q-bits 4 \
    -o ./llama-3.2-8b-4bit

# Test the quantized model
mlx_lm.generate \
    --model ./llama-3.2-8b-4bit \
    --prompt "Analyze this text for hate speech:"
```

### Memory Management

Apple Silicon uses unified memory, which is shared between CPU and GPU:

```python
# The pipeline automatically handles MPS memory management
annotator = LLMAnnotator(config, device="mps")

# After processing, memory is cleaned up
annotator.unload_model()  # Calls torch.mps.synchronize() + gc.collect()
```

### Limitations on Apple Silicon

1. **No bitsandbytes**: 4-bit/8-bit quantization via bitsandbytes is not supported
2. **No Flash Attention 2**: Flash Attention requires CUDA
3. **Unified Memory**: Must leave headroom for OS and other apps
4. **Smaller Models**: Recommended to use 7B-13B models instead of 70B

## Architecture

### Components

- **annotator.py**: Local GPU inference with batching and quantization
- **config.py**: Configuration for prompts, models, and target categories
- **output.py**: Results aggregation and summary generation
- **parallel.py**: High-throughput parallel processing pipeline

### Data Flow

```
Transcript JSON
      ↓
LLMAnnotator.annotate_batch()
      ↓
┌─────────────────────────────────────────┐
│  Local GPU Inference                    │
│  - 4-bit quantization (bitsandbytes)    │
│  - Flash Attention 2                    │
│  - Dynamic batching                     │
│  - KV cache optimization                │
└──────────────────┬──────────────────────┘
                   ↓
         Segment Annotations
                   ↓
    ┌──────────────┴──────────────┐
    ↓                             ↓
Enriched Transcript          Summary Statistics
    ↓                             ↓
Save to JSON                 Save to JSON
```

### Error Handling

**Graceful Degradation:**
- Failed requests return error annotations
- Processing continues for remaining segments
- Error details logged for debugging

**Error Annotation Format:**
```json
{
  "has_hate_speech": null,
  "has_advertisement": null,
  "target_group": null,
  "hate_speech_type": null,
  "main_topic": null,
  "error": "JSON parse error: ..."
}
```

## Troubleshooting

### CUDA Out of Memory

**Error:** `CUDA out of memory`

**Fix:**
1. Reduce batch size:
   ```python
   annotator = LLMAnnotator(config, batch_size=1)
   ```
2. Enable 4-bit quantization:
   ```python
   annotator = LLMAnnotator(config, load_in_4bit=True)
   ```
3. Clear cache between files:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### Apple Silicon Memory Pressure

**Error:** System becomes slow or unresponsive during processing

**Fix:**
1. Use smaller models (8B instead of 70B):
   ```python
   config = AnnotationConfig(model_name="meta-llama/Llama-3.2-8B-Instruct")
   ```
2. Reduce batch size:
   ```python
   annotator = LLMAnnotator(config, device="mps", batch_size=1)
   ```
3. Close other applications to free unified memory
4. Use MLX with 4-bit quantization for smaller memory footprint

### MPS Not Available

**Error:** `MPS (Apple Silicon) not available`

**Fix:**
1. Ensure you're running macOS 12.3 or later
2. Verify you have an Apple Silicon Mac (M1/M2/M3/M4)
3. Update PyTorch:
   ```bash
   pip install --upgrade torch
   ```

### Model Loading Errors

**Error:** `ImportError: transformers and bitsandbytes are required`

**Fix (NVIDIA):**
```bash
uv pip install transformers bitsandbytes accelerate
```

**Fix (Apple Silicon):**
```bash
# bitsandbytes not needed on Apple Silicon
uv pip install transformers accelerate
```

### Quantization Not Supported on MPS

**Warning:** `Quantization (4-bit/8-bit) not supported on Apple Silicon`

This is expected. The pipeline automatically uses float16 on Apple Silicon:
```python
# Pipeline automatically handles this - no action needed
annotator = LLMAnnotator(config, device="mps")  # Uses float16
```

For quantization on Apple Silicon, use MLX:
```bash
pip install mlx mlx-lm
mlx_lm.convert --hf-path <model> -q --q-bits 4
```

### Flash Attention Errors

**Error:** `Flash Attention 2 not available`

**Fix (NVIDIA):**
1. Install flash-attn:
   ```bash
   uv pip install flash-attn --no-build-isolation
   ```
2. Or disable Flash Attention:
   ```python
   annotator = LLMAnnotator(config, use_flash_attention=False)
   ```

**Note:** Flash Attention is not available on Apple Silicon. The pipeline automatically disables it for MPS devices.

### JSON Parsing Errors

**Error:** `Failed to parse LLM response as JSON`

**Fix:**
1. Increase temperature slightly if responses too rigid
2. Check model supports JSON output mode
3. Try different model checkpoint

## Credits

Built with:
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Transformers](https://huggingface.co/docs/transformers) - Model loading and inference
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) - Quantization (NVIDIA)
- [MLX](https://github.com/ml-explore/mlx) - Apple Silicon ML framework
- [mlx-lm](https://pypi.org/project/mlx-lm/) - LLM inference for MLX
- [Llama 3.3 70B Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) - LLM model (NVIDIA)
- [Llama 3.2 8B Instruct](https://huggingface.co/models) - LLM model (Apple Silicon)
