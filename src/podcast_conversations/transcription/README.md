# Transcription Pipeline

GPU-accelerated speech-to-text transcription using WhisperX with word-level timestamps and automatic language detection.

## Overview

This package provides batch transcription of audio files using WhisperX, generating JSON transcripts with word-level timestamps. It's optimized for GPU processing with automatic batch size calculation based on available VRAM.

## Features

- **WhisperX Integration**: State-of-the-art transcription with faster-whisper backend
- **Word-Level Timestamps**: Precise timing for each word via alignment models
- **Auto Language Detection**: Detects language or uses specified language code
- **GPU Acceleration**: Optimized for CUDA GPUs with automatic batch sizing
- **Multiple Model Sizes**: tiny, base, small, medium, large-v2, large-v3
- **Compute Type Options**: float16, int8, float32 for memory/speed tradeoffs
- **Smart Caching**: Skips already-transcribed files
- **Resource Monitoring**: Real-time CPU, memory, and GPU display during processing

## Installation

The package is automatically installed with the main project:

```bash
uv sync
```

**WhisperX is installed automatically** via the project dependencies.

### Apple Silicon (MLX Backend)

For optimized transcription on Apple Silicon (M1/M2/M3/M4), install the optional MLX dependencies:

```bash
uv sync --extra macos
```

This installs `mlx-whisper` for native Apple Silicon acceleration, which is significantly faster than CPU and comparable to CUDA performance.

## Usage

### Command Line

Process audio files using the batch transcription script:

```bash
# Basic usage
uv run python scripts/transcribe_batch.py \
    --audio-dir downloads/the_daily \
    --output-dir outputs/transcripts \
    --model large-v3 \
    --device cuda

# With language specification
uv run python scripts/transcribe_batch.py \
    --audio-dir downloads/spanish_podcast \
    --output-dir outputs/transcripts \
    --model large-v3 \
    --language es

# CPU processing with int8 for lower memory
uv run python scripts/transcribe_batch.py \
    --audio-dir downloads/the_daily \
    --output-dir outputs/transcripts \
    --model small \
    --device cpu \
    --compute-type int8

# Preview what would be processed
uv run python scripts/transcribe_batch.py \
    --audio-dir downloads/ \
    --output-dir outputs/transcripts \
    --dry-run

# Force reprocessing of existing files
uv run python scripts/transcribe_batch.py \
    --audio-dir downloads/ \
    --output-dir outputs/transcripts \
    --force
```

### Python API

```python
from pathlib import Path
from podcast_conversations.transcription import TranscriptionConfig, TranscriptionPipeline

# Create configuration
config = TranscriptionConfig(
    model_size="large-v3",
    language=None,  # Auto-detect
    compute_type="float16",
    batch_size=None,  # Auto-calculate
    device="cuda",
)

# Initialize pipeline
pipeline = TranscriptionPipeline(config)

# Transcribe single file
result = pipeline.transcribe(Path("audio/episode.mp3"))
print(f"Language: {result.language}")
print(f"Duration: {result.duration_seconds:.1f}s")
print(f"Words: {result.word_count}")

# Or transcribe and save to JSON
result = pipeline.transcribe_to_json(
    audio_path=Path("audio/episode.mp3"),
    output_path=Path("outputs/transcripts/episode.json"),
)

# Clear GPU cache between files
pipeline.clear_cache()

# Unload models when done
pipeline.unload()
```

## Command Line Options

```
--audio-dir PATH       Root directory containing audio files [required]
--output-dir PATH      Output directory for transcript JSON files [required]
--model TEXT           Whisper model: tiny, base, small, medium, large-v2, large-v3 (default: large-v3)
--language TEXT        Language code (en, es, etc.) or omit for auto-detect
--compute-type TEXT    Precision: float16, int8, float32 (default: float16)
--batch-size INTEGER   Batch size (auto-calculated if not specified)
--device TEXT          Device: auto, cuda, mlx, cpu (default: auto)
--force                Force reprocessing of files with existing transcripts
--dry-run              Show what would be processed without running
--log-level TEXT       Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)
```

## Configuration

### Model Sizes

| Model | Parameters | VRAM | Speed | Accuracy |
|-------|------------|------|-------|----------|
| tiny | 39M | ~1GB | Fastest | Low |
| base | 74M | ~1GB | Fast | Medium |
| small | 244M | ~2GB | Medium | Good |
| medium | 769M | ~5GB | Slow | Better |
| large-v2 | 1.5B | ~10GB | Slower | Best |
| large-v3 | 1.5B | ~10GB | Slower | Best (improved) |

**Recommended**: `large-v3` for production, `small` for testing.

### MLX-Specific Models (Apple Silicon)

When using `--device mlx`, additional optimized models are available:

| Model | Description | Memory | Speed |
|-------|-------------|--------|-------|
| large-v3-turbo | Faster variant of large-v3 | ~8GB | 2x faster |
| large-v3-8bit | 8-bit quantized large-v3 | ~5GB | Faster |
| large-v3-4bit | 4-bit quantized large-v3 | ~3GB | Fastest |
| distil-large-v3 | Distilled variant | ~6GB | 1.5x faster |

**Usage:**
```bash
# Use turbo model for faster transcription
uv run python scripts/transcribe_batch.py \
    --audio-dir downloads/ \
    --output-dir outputs/transcripts \
    --model large-v3-turbo \
    --device mlx

# Use quantized model for lower memory usage
uv run python scripts/transcribe_batch.py \
    --audio-dir downloads/ \
    --output-dir outputs/transcripts \
    --model large-v3-4bit \
    --device mlx
```

### Device Selection

| Device | Platform | Backend | Notes |
|--------|----------|---------|-------|
| auto | Any | Best available | Recommended default |
| cuda | Linux/Windows | WhisperX + faster-whisper | Requires NVIDIA GPU |
| mlx | macOS | mlx-whisper | Apple Silicon only |
| cpu | Any | WhisperX + faster-whisper | Slowest, always available |

### Compute Types

| Type | Memory | Speed | Quality |
|------|--------|-------|---------|
| float16 | Low | Fast | Standard |
| int8 | Lower | Medium | Slightly lower |
| float32 | High | Slow | Best |

### Auto Batch Sizing

Batch size is automatically calculated based on GPU memory:

```python
# Formula: (gpu_gb / 10) * (40 / (model_factor * compute_factor))

# Example for H100 80GB with large-v3 float16:
# (80 / 10) * (40 / (5.0 * 1.0)) = 8 * 8 = 64
```

**Manual Override:**
```bash
uv run python scripts/transcribe_batch.py \
    --audio-dir audio/ \
    --output-dir transcripts/ \
    --batch-size 32
```

## Output Format

### Transcript JSON Structure

```json
{
  "audio_file": "episode_001.mp3",
  "language": "en",
  "text": "Hello and welcome to the podcast. Today we're discussing...",
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "Hello and welcome to the podcast.",
      "words": [
        {"word": "Hello", "start": 0.0, "end": 0.4},
        {"word": "and", "start": 0.5, "end": 0.6},
        {"word": "welcome", "start": 0.7, "end": 1.1},
        {"word": "to", "start": 1.2, "end": 1.3},
        {"word": "the", "start": 1.4, "end": 1.5},
        {"word": "podcast", "start": 1.6, "end": 2.5}
      ]
    },
    {
      "start": 2.8,
      "end": 5.2,
      "text": "Today we're discussing...",
      "words": [...]
    }
  ]
}
```

### Directory Structure

Output mirrors input directory structure:

```
downloads/
  ├── the_daily/
  │   ├── episode_001.mp3
  │   ├── episode_002.mp3
  │   └── ...
  └── lex_fridman/
      ├── episode_001.mp3
      └── ...

outputs/transcripts/
  ├── the_daily/
  │   ├── episode_001.json
  │   ├── episode_002.json
  │   └── ...
  └── lex_fridman/
      ├── episode_001.json
      └── ...
```

## Supported Audio Formats

- `.wav` - Waveform Audio
- `.mp3` - MPEG Audio Layer 3
- `.flac` - Free Lossless Audio Codec
- `.ogg` - Ogg Vorbis
- `.m4a` - MPEG-4 Audio
- `.wma` - Windows Media Audio
- `.aac` - Advanced Audio Coding
- `.mp4` - MPEG-4 (audio extracted)

## Performance

### GPU Recommendations

| GPU | Model | Batch Size | Speed |
|-----|-------|------------|-------|
| H100 80GB | large-v3 | 64 | ~50x realtime |
| A100 40GB | large-v3 | 32 | ~30x realtime |
| RTX 4090 24GB | large-v3 | 16 | ~20x realtime |
| RTX 3090 24GB | large-v3 | 12 | ~15x realtime |
| CPU | small | 4 | ~1x realtime |

### Memory Usage

With `large-v3` model:
- **Model**: ~10GB VRAM
- **Per-batch**: ~1-2GB additional
- **Total**: ~12-15GB minimum recommended

## Troubleshooting

### Out of Memory (OOM)

**Fix:**
1. Reduce batch size: `--batch-size 8`
2. Use smaller model: `--model small`
3. Use int8 compute: `--compute-type int8`
4. Clear cache between files (automatic)

### Slow Transcription

**Fix:**
1. Ensure GPU is being used: check `using gpu:` in logs
2. Increase batch size if VRAM allows
3. Use float16 compute type
4. Use larger model for better batch utilization

### Language Detection Issues

**Fix:**
1. Specify language explicitly: `--language en`
2. Ensure audio quality is sufficient
3. Use larger model for better detection

### WhisperX Import Errors

**Error:** `ModuleNotFoundError: No module named 'whisperx'`

**Fix:**
```bash
uv sync
# or
uv pip install whisperx
```

## Architecture

### Components

- **config.py**: Configuration models with auto batch sizing
- **pipeline.py**: WhisperX pipeline with CUDA/CPU support
- **mlx_backend.py**: MLX-whisper pipeline for Apple Silicon

### Data Flow

```
Audio File
    ↓
whisperx.load_audio()
    ↓
whisperx model.transcribe() (batched)
    ↓
whisperx.align() (word-level timestamps)
    ↓
TranscriptionResult
    ↓
JSON Output
```

### Lazy Loading

Models are loaded on first use:
1. Transcription model loads on first `transcribe()` call
2. Alignment model loads when alignment is needed
3. Both remain loaded for subsequent files
4. Call `pipeline.unload()` to free memory

## Integration with Pipeline

The transcription step typically comes before diarization:

```bash
# Step 0: Download podcasts (optional)
uv run python -m podcast_downloader download "The Daily" -o downloads/

# Step 1: Transcribe audio files
uv run python scripts/transcribe_batch.py \
    --audio-dir downloads/ \
    --output-dir outputs/transcripts

# Step 2: Diarize (identify speakers)
uv run python scripts/diarize_batch.py \
    --transcripts-dir outputs/transcripts \
    --audio-base-dir downloads/ \
    --output-dir outputs/diarizations

# Step 3: Combine transcripts with speakers
uv run python scripts/combine_transcripts_with_speakers.py ...
```

## Requirements

- Python 3.11+
- CUDA GPU (recommended) or CPU
- WhisperX (auto-installed)
- FFmpeg (for audio processing)
- ~10GB disk space for large-v3 model (first run)

## Credits

Built with:
- [WhisperX](https://github.com/m-bain/whisperX) - Fast Whisper with word alignment
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - CTranslate2 backend
- [mlx-whisper](https://pypi.org/project/mlx-whisper/) - MLX-native Whisper for Apple Silicon
- [MLX](https://github.com/ml-explore/mlx) - Apple Silicon ML framework
- [PyTorch](https://pytorch.org/) - Deep learning framework
