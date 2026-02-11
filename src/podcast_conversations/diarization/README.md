# Speaker Diarization

GPU-accelerated speaker diarization for podcast conversations using pyannote.audio with multi-processing optimizations.

## Overview

This package identifies "who spoke when" in podcast audio files using state-of-the-art speaker diarization models. It's optimized for high-performance batch processing with GPU acceleration, achieving 4-5x speedup through parallel workers and torch.compile optimizations.

## Features

- **GPU Acceleration**: Optimized for CUDA GPUs (H100, A100, etc.) and Apple Silicon (MPS)
- **Auto Device Detection**: Automatically selects the best available device (CUDA → MPS → CPU)
- **Multi-Processing**: Process multiple files in parallel with independent workers
- **torch.compile**: 20-40% faster inference after initial compilation (CUDA only)
- **BFloat16 Precision**: Automatic mixed precision for supported GPUs (CUDA only)
- **Dynamic Batching**: Group files by duration for optimal GPU utilization
- **RTTM Output**: Standard RTTM format for diarization results
- **Automatic Retry**: OOM error handling with cache clearing
- **Fallback Audio Loading**: torchaudio with ffmpeg fallback



## Installation

The package is automatically installed with the main project:

```bash
uv sync
```

## Configuration

### HuggingFace Token

Speaker diarization requires a HuggingFace token:

```bash
export HF_TOKEN="your_token_here"
```

Or pass it via `--hf-token` flag.

### Configuration Constants

Default settings in `src/podcast_conversations/config/__init__.py`:

```python
DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
DEFAULT_DEVICE = "cuda:0"  # CLI default is "auto" for auto-detection
DEFAULT_USE_BF16 = True   # auto-disabled on non-CUDA devices
DEFAULT_USE_COMPILE = False  # auto-disabled on non-CUDA devices
```

**Note:** While the config constant defaults to `cuda:0`, CLI scripts use `--device auto` as their default, which auto-detects the best available device (CUDA → MPS → CPU).

## Usage

### Command Line

#### Process Multiple Podcast Directories (Recommended)

```bash
./scripts/diarize_all_podcasts.sh \
  --transcripts-root outputs/transcripts \
  --audio-base-dir /path/to/audio \
  --output-dir outputs/diarizations \
  --hf-token YOUR_HF_TOKEN \
  --workers 5 \
  --compile-mode max-autotune
```

This automatically discovers all podcast subdirectories:
```
transcripts/
  ├── the_ezra_klein_show/
  ├── the_joe_rogan_experience/
  ├── lex_fridman_podcast/
  └── ... (auto-discovered)
```

**Preview before processing:**
```bash
./scripts/diarize_all_podcasts.sh \
  --transcripts-root outputs/transcripts \
  --audio-base-dir /path/to/audio \
  --output-dir outputs/diarizations \
  --dry-run
```

#### Process Single Podcast Directory

```bash
./scripts/diarize_optimized.sh \
  --transcripts-dir outputs/transcripts/the_daily \
  --audio-base-dir /path/to/audio/the_daily \
  --output-dir outputs/diarizations/the_daily \
  --hf-token YOUR_HF_TOKEN
```

### Python API

```python
from pathlib import Path
from podcast_conversations.diarization import (
    DiarizationPipeline,
    FileDiscovery,
    RTTMWriter,
    detect_diarization_device,
    get_diarization_device_info,
)

# Check available devices
print(get_diarization_device_info())
# {'is_apple_silicon': True, 'cuda_available': False, 'mps_available': True, 'recommended_device': 'mps'}

# Initialize pipeline with auto device detection
pipeline = DiarizationPipeline(
    model_name="pyannote/speaker-diarization-3.1",
    hf_token="your_token",
    device="auto",  # auto-detects: CUDA → MPS → CPU
    use_bf16=True,  # auto-disabled on non-CUDA devices
    use_compile=True,  # auto-disabled on non-CUDA devices
    compile_mode="reduce-overhead",
)

# Discover transcript-audio pairs
discovery = FileDiscovery(
    transcripts_dir=Path("outputs/transcripts/the_daily"),
    audio_base_dir=Path("/path/to/audio/the_daily"),
)
file_mappings = discovery.discover_files()

# Process audio files
for mapping in file_mappings:
    try:
        diarization = pipeline.process(mapping.audio_path)

        # Write RTTM file
        output_path = Path(f"outputs/diarizations/{mapping.base_name}.rttm")
        RTTMWriter.write(diarization, output_path, mapping.base_name)

        print(f"✓ {mapping.base_name}: {len(diarization.segments)} segments")
    except Exception as e:
        print(f"✗ {mapping.base_name}: {e}")
```

## Command Line Options

### diarize_all_podcasts.sh

```
--transcripts-root PATH       Root directory with podcast subdirectories [required]
--audio-base-dir PATH        Root audio directory [required]
--output-dir PATH            Output directory for RTTM files [required]
--hf-token TEXT              HuggingFace API token (or set HF_TOKEN env var)
--model TEXT                 Model identifier (default: pyannote/speaker-diarization-3.1)
--device TEXT                Device (default: auto, can use cuda:0, cuda:1, mps, cpu)
--use-bf16/--no-bf16         Enable BFloat16 (default: enabled, auto-disabled on non-CUDA)
--torch-compile/--no-compile Enable torch.compile (default: enabled, CUDA only)
--compile-mode TEXT          Compilation mode: default, reduce-overhead, max-autotune (CUDA only)
--workers INTEGER            Parallel workers (default: 4)
--batch-size INTEGER         Files per batch (default: 4)
--dry-run                    Preview without processing
--log-level TEXT             Logging level: DEBUG, INFO, WARNING, ERROR
```

### diarize_optimized.sh

Same options as above, but for single podcast directory. Use `--transcripts-dir` instead of `--transcripts-root`.

## Performance Optimization

### Worker Count Recommendations

| Device | Memory | Audio Length | Recommended Workers |
|--------|--------|--------------|---------------------|
| H100 80GB  | 80 GB | < 30 min     | 5-6                 |
| H100 80GB  | 80 GB | 30-60 min    | 4-5                 |
| H100 80GB  | 80 GB | > 60 min     | 2-3                 |
| A100 40GB  | 40 GB | < 30 min     | 3-4                 |
| A100 40GB  | 40 GB | 30-60 min    | 2-3                 |
| Apple Silicon (MPS) | 16-32 GB unified | any | 1-2 |
| CPU | varies | any | 1 (slow) |

**Note**: Apple Silicon uses unified memory shared with system RAM. The CLI auto-reduces workers to max 2 when using MPS.

### Compilation Modes

- `default`: Balanced performance and compilation time
- `reduce-overhead` (default): Good balance, faster subsequent runs
- `max-autotune`: Maximum performance, slowest first compilation (~5 min)

### Performance Configurations

**Maximum Performance (H100/A100 80GB):**
```bash
./scripts/diarize_optimized.sh \
  --transcripts-dir outputs/transcripts/show_name \
  --audio-base-dir /path/to/audio \
  --output-dir outputs/diarizations \
  --hf-token YOUR_TOKEN \
  --workers 5 \
  --use-bf16 \
  --compile-mode max-autotune \
  --batch-size 5
```

**Conservative Settings (OOM issues):**
```bash
./scripts/diarize_optimized.sh \
  --transcripts-dir outputs/transcripts/show_name \
  --audio-base-dir /path/to/audio \
  --output-dir outputs/diarizations \
  --hf-token YOUR_TOKEN \
  --workers 2 \
  --batch-size 2
```

**Apple Silicon (macOS with M1/M2/M3/M4):**
```bash
uv run python scripts/diarize_batch.py \
  --transcripts-dir outputs/transcripts/show_name \
  --audio-base-dir /path/to/audio \
  --output-dir outputs/diarizations \
  --hf-token YOUR_TOKEN \
  --device auto \
  --workers 2
```

Note: torch.compile and BFloat16 are automatically disabled on MPS. The CLI auto-detects Apple Silicon and uses MPS acceleration.

**CPU-Only Processing:**
```bash
./scripts/diarize_optimized.sh \
  --transcripts-dir outputs/transcripts/show_name \
  --audio-base-dir /path/to/audio \
  --output-dir outputs/diarizations \
  --hf-token YOUR_TOKEN \
  --device cpu \
  --no-bf16 \
  --workers 10
```

### Expected Speedup

With H100 80GB and 4 workers processing 1-hour podcasts:

| Optimization       | Speedup vs Sequential |
|--------------------|-----------------------|
| torch.compile only | 1.3x                  |
| 4 workers only     | 3.5x                  |
| All optimizations  | 4.5-5x                |

**Example**: 100 one-hour podcasts
- Before: ~50 hours total
- After: ~10-11 hours total

### Monitoring Performance

**GPU utilization:**
```bash
watch -n 1 nvidia-smi
```

You should see:
- Multiple Python processes using GPU
- GPU memory: 60-80% (with 4 workers)
- GPU utilization: 90-100%

**CPU usage:**
```bash
htop
```

You should see:
- 4+ Python processes running
- CPU distributed across cores

## Output Format

### RTTM Format

Standard RTTM (Rich Transcription Time Marked) format:

```
SPEAKER episode_name 1 1.077 6.463 <NA> <NA> SPEAKER_08 <NA> <NA>
SPEAKER episode_name 1 7.540 3.561 <NA> <NA> SPEAKER_00 <NA> <NA>
SPEAKER episode_name 1 11.101 2.427 <NA> <NA> SPEAKER_01 <NA> <NA>
```

**Fields:**
```
TYPE FILE CHANNEL START DURATION <NA> <NA> SPEAKER <NA> <NA>
```

- `TYPE`: Always "SPEAKER"
- `FILE`: Episode/file identifier
- `CHANNEL`: Always 1
- `START`: Start time in seconds
- `DURATION`: Duration in seconds
- `SPEAKER`: Speaker label (SPEAKER_00, SPEAKER_01, etc.)

### Output Directory Structure

```
outputs/diarizations/
├── the_ezra_klein_show/
│   ├── episode_001.rttm
│   ├── episode_002.rttm
│   └── ...
├── the_joe_rogan_experience/
│   ├── episode_001.rttm
│   └── ...
└── ... (one folder per show)
```

## Architecture

### Components

- **DiarizationPipeline**: pyannote.audio wrapper with GPU optimizations
- **FileDiscovery**: Transcript-audio file mapping and discovery
- **RTTMWriter**: RTTM format output writer
- **FileMapping**: Data class for file relationships

### Multi-Processing Architecture

- **Main process**: File discovery, batching, progress tracking
- **Worker processes**: Each initializes its own pipeline independently
- **Process spawn method**: Avoids CUDA context sharing issues
- **Result aggregation**: Uses `concurrent.futures.as_completed()` for efficiency

### Memory Management

- Each worker loads model independently (~2-3GB GPU memory per worker)
- Automatic cache clearing on OOM errors
- Retry logic: 2 attempts per file on OOM
- Dynamic batching groups similar-duration files

### File Processing Order

1. Discover all transcript-audio pairs
2. Skip files with existing RTTM outputs
3. Compute audio durations (fast metadata read)
4. Sort files by duration
5. Group into batches of similar durations
6. Submit all files to worker pool
7. Process results as they complete

## Troubleshooting

### BFloat16 Errors

**Error:** `Got unsupported ScalarType BFloat16`

**Fix:** The script automatically retries with FP32. If it persists:
1. Ensure you're not explicitly using `--use-bf16`
2. Update PyTorch to latest version
3. BFloat16 requires H100, A100, or newer GPUs (V100 not supported)

### CUDA Device Errors

**Error:** `device >= 0 && device < num_gpus`

**Fix:** The script now properly manages CUDA device visibility. If it persists:
1. Explicitly set device: `--device cuda:0`
2. Reduce workers if issue continues

### Out of Memory (OOM)

**Fix:**
1. Reduce `--workers` (try 2 or 3)
2. Reduce `--batch-size` (try 2)
3. Process longer files separately
4. Disable BFloat16: `--no-bf16`

### PyTorch Multiprocessing Errors

**Error:** `PythonDispatcherTLS was not set`

**Fix:** Already resolved in latest version. The script:
1. Sets `TORCH_COMPILE_DISABLE=1` environment variable
2. Initializes PyTorch in worker processes
3. Disables torch.compile by default in multiprocessing mode

### Slower Than Expected

**Fix:**
1. Check GPU utilization with `nvidia-smi`
2. Increase `--workers` if GPU usage < 80%
3. Try `--compile-mode max-autotune`
4. Ensure torch.compile is enabled: `--torch-compile`

## Technical Details

### Optimizations Implemented

1. **Multi-Processing (3-5x speedup)**
   - Parallel file processing with ProcessPoolExecutor
   - Independent worker processes with isolated pipelines
   - Spawn method for CUDA context isolation

2. **torch.compile (20-40% speedup)**
   - First file slower (~1-2 min compilation)
   - Subsequent files 20-40% faster
   - Configurable compilation modes

3. **Dynamic Batching**
   - Groups files by duration
   - Improves GPU utilization
   - Reduces memory fragmentation

4. **GPU Optimizations**
   - BFloat16 mixed precision (H100/A100)
   - TF32 matrix operations
   - cuDNN auto-tuning
   - Automatic OOM handling

### Model Information

**Default Model:** `pyannote/speaker-diarization-3.1`

This is a state-of-the-art speaker diarization model that:
- Segments audio by speaker
- Assigns speaker labels (SPEAKER_00, SPEAKER_01, etc.)
- Handles overlapping speech
- Works with any number of speakers

## Examples

### Process All Podcasts in Directory

```bash
# Discover and process all shows
./scripts/diarize_all_podcasts.sh \
  --transcripts-root outputs/transcripts \
  --audio-base-dir /media/audio \
  --output-dir outputs/diarizations \
  --hf-token $HF_TOKEN
```

### Process Single Show with Maximum Performance

```bash
# H100 GPU with maximum optimization
./scripts/diarize_optimized.sh \
  --transcripts-dir outputs/transcripts/the_daily \
  --audio-base-dir /media/audio/the_daily \
  --output-dir outputs/diarizations/the_daily \
  --hf-token $HF_TOKEN \
  --workers 5 \
  --compile-mode max-autotune
```

### Preview What Would Be Processed

```bash
# Dry run to see file mappings
./scripts/diarize_all_podcasts.sh \
  --transcripts-root outputs/transcripts \
  --audio-base-dir /media/audio \
  --output-dir outputs/diarizations \
  --dry-run
```

## Requirements

- Python 3.11+
- GPU acceleration (optional, recommended):
  - **NVIDIA**: CUDA-capable GPU with PyTorch CUDA support
  - **Apple Silicon**: M1/M2/M3/M4 Mac with MPS support (PyTorch 2.0+)
  - **CPU**: Fallback option (significantly slower)
- FFmpeg for audio processing
- HuggingFace account and token
- pyannote.audio 3.1+

## Credits

Built with:
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) - Speaker diarization models
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [torchaudio](https://pytorch.org/audio/) - Audio processing
