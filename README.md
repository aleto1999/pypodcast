# Podcast Conversations Analysis

A high-performance Python toolkit for analyzing podcast conversations through GPU-accelerated transcription, speaker diarization, utterance classification, semantic embeddings analysis, LLM-based content annotation, and keyword extraction. Includes a podcast downloader for fetching episodes from RSS feeds and YouTube.

## Features

- **Podcast Downloader**: Download episodes from RSS feeds and YouTube with automatic feed discovery, including the creation of placeholder transcript files to streamline downstream processing
- **GPU-Accelerated Transcription**: Generate word-level transcripts using WhisperX with batch processing
- **GPU-Accelerated Speaker Diarization**: Identify who spoke when using pyannote.audio with multi-processing and GPU optimization
- **Speaker Labeling**: Replace generic speaker IDs with actual names using a 5-stage pipeline involving metadata, NER, and optional LLM-based role classification.
- **Utterance Classification**: Apply transformer-based models to classify utterances for content analysis
- **Semantic Embeddings**: Detect language patterns using sentence embeddings and cosine similarity
- **LLM Annotation**: Analyze utterances with Llama 3.3 70B for hate speech detection, target group identification, and topic extraction using local GPU inference with 4-bit quantization (NVIDIA) or MLX (Apple Silicon)
- **Feature Extraction**: Extract conversation-level features including question patterns, turn-taking dynamics, vocabulary metrics, and politeness markers
- **Keyword Analysis**: Search transcripts for specific keywords with context extraction
- **Terminal Data Visualizer**: Interactive TUI for exploring all pipeline outputs with rich statistics and filtering
- **Resource Monitoring**: Real-time CPU, memory, and GPU monitoring with Rich terminal displays
- **Smart Caching**: Skip already-processed files to save time on incremental runs
- **Dynamic Batch Sizing**: Automatic batch size optimization based on GPU memory, RAM, and CPU cores
- **Rich CLI**: Interactive command-line interface with progress tracking
- **Multiple Export Formats**: JSON, CSV, and TXT output formats
- **High Performance**: 4-5x speedup with multi-processing and torch.compile optimizations

## Pipeline Overview

The toolkit consists of modular pipelines that can be combined in different ways:

| # | Pipeline | Script/Command | Description |
|---|----------|----------------|-------------|
| 🚀 | **Full Corpus Pipeline** | `run_full_corpus_pipeline.py` | Run 7 core stages: transcription, diarization, combining, embeddings, keywords, and classification. |
| ⚡ | **Analysis Pipeline** | `run_analysis_pipeline.py` | Run analysis stages: embeddings, keywords, and classification. |
| 🎧 | **Podcast Download** | `podcast-dl` | Download episodes from RSS/YouTube. |
| 🗣️ | **Speaker Labeling** | `run_speaker_labeling_pipeline.py` | Replace generic speaker IDs with actual names. |
| 📊 | **Terminal Visualizer** | `terminal-visualizer` | Interactive TUI for exploring all outputs. |
| | **Transcription** | `transcribe_batch.py` | Generate word-level transcripts with WhisperX. |
| | **Diarization** | `diarize_batch.py` | Identify who spoke when in audio files. |
| | **Combine Transcripts** | `combine_transcripts_with_speakers.py` | Merge speaker labels with transcripts. |
| | **Combine Consecutive** | `combine_consecutive_speakers.py` | Merge consecutive same-speaker segments. |
| | **Embeddings** | `analyze_embeddings.py` | Semantic embeddings and similarity analysis. |
| | **Keywords** | (via `run_full_corpus_pipeline.py`) | Keyword matching with context extraction. |
| | **Classification** | `classify_utterances.py` | Multi-label utterance classification. |
| | **Document Labels** | `generate_document_labels.py` | Aggregate classifications to per-episode/show labels. |
| | **LLM Annotation** | `annotate_with_llm.py` | LLM-based content analysis (standalone). |
| | **Feature Extraction** | `extract_features.py` | Conversation features (standalone). |

### Common CLI Flags

All pipelines support these standard flags:

| Flag | Description |
|------|-------------|
| `--transcripts-dir` | Input directory with transcript JSON files |
| `--output-dir` | Output directory for results |
| `--force` | Force reprocessing of files with existing outputs |
| `--dry-run` | Preview what would be processed without running |
| `--log-level` | Logging level: DEBUG, INFO, WARNING, ERROR |

### Data Flow

```
Podcast Sources (RSS Feeds / YouTube)
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  0. Download (podcast-dl)                                   │
│     RSS/YouTube → Audio files & Transcript Placeholders     │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  1. Transcription (transcribe_batch.py)                     │
│     Audio → JSON transcripts with word-level timestamps     │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Diarization (diarize_batch.py)                         │
│     Audio → RTTM speaker timing files                       │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Combine Transcripts (combine_transcripts_with_speakers) │
│     Transcripts + RTTM → Speaker-labeled transcripts        │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Combine Consecutive (combine_consecutive_speakers)      │
│     Merge fragmented same-speaker segments                  │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  5. Embeddings (analyze_embeddings.py)                      │
│     Semantic similarity analysis (MLX/CUDA/MPS/CPU)         │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  6. Keywords (keyword_matcher module)                       │
│     Keyword matching with context extraction                │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  7. Classification (classify_utterances.py)                 │
│     Multi-label utterance classification                    │
│     (can filter by keyword matches)                         │
└─────────────────────────────────────────────────────────────┘

        ┌─────────────────────────────────────────────────────┐
        │  Standalone Pipelines (run independently)           │
        │                                                     │
        │  🗣️ Speaker Labeling (run_speaker_labeling_pipeline.py)      │
        │     Replace generic speaker IDs with actual names           │
        │                                                     │
        │  ⚡ LLM Annotation (annotate_with_llm.py)           │
        │     Llama 3.3 70B content analysis                  │
        │                                                     │
        │  ⚡ Feature Extraction (extract_features.py)        │
        │     Questions, turn-taking, vocabulary metrics      │
        │                                                     │
        │  📝 Document Labels (generate_document_labels.py)    │
        │     Aggregate classifications to per-episode/show labels    │
        └─────────────────────────────────────────────────────┘
```

**Pipeline Integration:** The classification pipeline can optionally filter input files based on keyword analysis results, processing only transcripts that contain keyword matches for target population groups.

## Quick Start

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install dependencies

```bash
uv sync
```

### Install Flash Attention 2 (Optional, NVIDIA GPUs only)

Flash Attention 2 provides 2-4x faster inference for the LLM annotation pipeline. It requires CUDA and special build flags:

```bash
# Install flash-attn (requires CUDA toolkit)
uv pip install flash-attn --no-build-isolation
```

**Note:** Flash Attention 2 is optional and only available on NVIDIA GPUs. The LLM annotation pipeline will work without it, just slower.

### Apple Silicon Support (Optional)

For Apple Silicon Macs (M1/M2/M3/M4), install the optional MLX dependencies for optimized embeddings:

```bash
# Install MLX dependencies for embeddings
uv sync --extra macos
```

This enables:
- **MLX Transcription**: Optimized speech-to-text using `mlx-whisper`
- **MLX Embeddings**: Optimized embedding generation using `mlx-embedding-models`
- **Unified Memory**: Efficient batch sizes for Apple Silicon's unified memory architecture

For LLM annotation, the pipeline automatically uses MPS (Metal Performance Shaders). For better performance with quantization, install MLX-LM:

```bash
# Install MLX for Apple Silicon optimized LLM inference
uv pip install mlx mlx-lm

# Pre-quantize a model to 4-bit for faster inference
mlx_lm.convert --hf-path meta-llama/Llama-3.2-8B-Instruct -q --q-bits 4
```

**Note:** For Apple Silicon, we recommend using smaller models (7B-13B) due to unified memory constraints. See the [LLM Annotation Package](src/podcast_conversations/llm_annotation/README.md) for detailed Apple Silicon setup.

### Configure HuggingFace Token

For speaker diarization and gated classification models, you need a HuggingFace token:

```bash
export HF_TOKEN="your_token_here"
```

Or pass it via `--hf-token` flag when running scripts.

**Get your token at:** https://huggingface.co/settings/tokens

**Required for:**
- Speaker diarization (all runs)
- Gated classification models (if configured in `config/classifiers.yaml`)

## Usage

### Podcast Download (Optional First Step)

Download podcast episodes from RSS feeds or YouTube:

```bash
# Discover podcasts by name
uv run python -m podcast_downloader discover "The Daily"

# Download episodes from a podcast
uv run python -m podcast_downloader download "The Daily" --max-episodes 10

# Download from a direct RSS feed
uv run python -m podcast_downloader download --feed-url https://example.com/feed.xml -o downloads/

# Download with date filtering
uv run python -m podcast_downloader download "The Ezra Klein Show" \
    --start-date 2025-01-01 \
    --end-date 2025-12-31 \
    --output-dir downloads/
```

See [Podcast Downloader Documentation](src/podcast_downloader/README.md) for all options.

### Transcription (Audio to Text)

Generate word-level transcripts from audio files using WhisperX:

```bash
uv run python scripts/transcribe_batch.py \
    --audio-dir downloads/the_daily \
    --output-dir outputs/transcripts \
    --model large-v3 \
    --device cuda
```

See [Transcription Documentation](src/podcast_conversations/transcription/README.md) for all options.

### Complete Pipeline (One Command)

Run all 7 stages of the main corpus processing pipeline with a single command:

```bash
uv run python scripts/run_full_corpus_pipeline.py \
    --audio-dir /path/to/audio \
    --hf-token YOUR_HF_TOKEN \
    --torch-compile
```

This runs: Transcription → Diarization → Combine Speakers → Combine Consecutive → Embeddings → Keywords → Classification.

**Common Options:**
- `--skip-transcription`: Use existing transcripts
- `--skip-diarization`: Use existing RTTM files
- `--skip-embeddings`: Skip embeddings analysis
- `--skip-classification`: Skip GPU-intensive classification
- `--dry-run`: Preview without executing
- `--force`: Reprocess all files

See the [Scripts Documentation](scripts/README.md) for detailed usage of all available pipelines and their options.

## Smart Caching & Dynamic Batch Sizing

The classification and embeddings pipelines now feature intelligent resource management that automatically optimizes performance based on your hardware.

### Dynamic Batch Sizing

Both **utterance classification** and **embeddings analysis** automatically determine optimal batch sizes based on:

- **Device Type**: CUDA GPU, Apple Silicon (MPS), or CPU
- **Available Memory**: GPU VRAM for CUDA, unified memory for MPS, system RAM for CPU
- **CPU Cores**: For CPU processing

**Batch Size Ranges:**

| Pipeline | CUDA GPU | Apple Silicon (MPS) | CPU |
|----------|----------|---------------------|-----|
| **Classification** | 4-32 | 8-24 | 2-8 |
| **Embeddings** | 16-128 | 32-96 | 8-32 |

Classification uses smaller batches because transformer models require more memory per item.

**How to Use:**
```bash
# Auto batch sizing (recommended) - no --batch-size flag
uv run python scripts/classify_utterances.py \
  --transcripts-dir outputs/transcripts_with_speakers \
  --output-dir outputs/transcripts_with_speakers_and_labels

# Manual override if needed
uv run python scripts/classify_utterances.py \
  --transcripts-dir outputs/transcripts_with_speakers \
  --output-dir outputs/transcripts_with_speakers_and_labels \
  --batch-size 16
```

### Smart Caching

Both pipelines automatically skip files that have already been processed:

**Utterance Classification:**
- Checks if output files exist with valid classifications
- Skips files where all segments have classification data
- Use `--force` flag to reprocess everything

```bash
# Normal run - skips existing classifications
uv run python scripts/classify_utterances.py \
  --transcripts-dir outputs/transcripts_with_speakers \
  --output-dir outputs/transcripts_with_speakers_and_labels

# Force reprocessing of all files
uv run python scripts/classify_utterances.py \
  --transcripts-dir outputs/transcripts_with_speakers \
  --output-dir outputs/transcripts_with_speakers_and_labels \
  --force
```

**Embeddings Analysis:**
- Checks if embedding files exist with valid embeddings
- Skips files where all segments have non-empty embedding vectors
- Use `--force` flag to reprocess everything

```bash
# Normal run - uses cached embeddings
uv run python scripts/analyze_embeddings.py

# Force regeneration
uv run python scripts/analyze_embeddings.py --force
```

**Benefits:**
- **Faster re-runs**: Incremental processing saves hours on large datasets
- **Automatic optimization**: No manual tuning required
- **Memory safety**: Conservative batch sizes prevent OOM errors
- **Clear feedback**: See exactly what's being processed vs. skipped

**Example Output:**
```
found existing classifications for 150/200 files
generating embeddings for 50 files (batch size: 64)...
✓ processed 50 files, skipped 150 existing
```

## Performance Optimizations

### Implemented Optimizations

1. **Multi-Processing (3-5x speedup)**
   - Processes multiple audio files in parallel using `ProcessPoolExecutor`
   - Each worker runs its own diarization pipeline instance
   - Default: 4 parallel workers (configurable with `--workers`)
   - Uses `spawn` method to avoid CUDA context issues

2. **torch.compile (20-40% speedup)**
   - Enabled by default (previously disabled)
   - First file will be slower (~1-2 minutes compilation)
   - Subsequent files process 20-40% faster
   - Can disable with `--no-compile` if needed

3. **Dynamic Batching**
   - Groups files by similar duration before processing
   - Improves GPU utilization and reduces memory fragmentation
   - Default batch size: 4 files (configurable with `--batch-size`)
   - Processes shorter files together, longer files together

4. **I/O Parallelization**
   - Implicitly handled by parallel workers
   - Each worker independently: loads audio → processes → writes RTTM
   - No blocking I/O operations

5. **GPU Optimizations (already in place)**
   - BFloat16 mixed precision (enabled by default on H100)
   - TF32 for matrix operations
   - cuDNN auto-tuning
   - Automatic OOM handling with cache clearing

### Performance Tuning

#### Worker Count Recommendations

| GPU Memory | Audio Length | Recommended Workers |
|------------|--------------|---------------------|
| H100 80GB  | < 30 min     | 5-6                 |
| H100 80GB  | 30-60 min    | 4-5                 |
| H100 80GB  | > 60 min     | 2-3                 |
| A100 40GB  | < 30 min     | 3-4                 |
| A100 40GB  | 30-60 min    | 2-3                 |

#### Compilation Modes

- `default`: Balanced performance and compilation time
- `reduce-overhead` (default): Good balance, faster subsequent runs
- `max-autotune`: Maximum performance, slowest first compilation (~5 min)

#### Expected Speedup

With H100 80GB and 4 workers processing 1-hour podcasts:

| Optimization       | Speedup vs Sequential |
|--------------------|-----------------------|
| torch.compile only | 1.3x                  |
| 4 workers only     | 3.5x                  |
| All optimizations  | 4.5-5x                |

Example: 100 one-hour podcasts
- Before: ~50 hours total processing time
- After: ~10-11 hours total processing time

### Monitoring Performance

**Watch GPU utilization:**
```bash
watch -n 1 nvidia-smi
```

You should see:
- Multiple Python processes using the GPU
- GPU memory usage: 60-80% (with 4 workers)
- GPU utilization: 90-100%

**Check CPU usage:**
```bash
htop
```

You should see:
- 4+ Python processes running
- CPU usage distributed across cores

## Configuration

### Diarization Configuration

Configuration is managed via constants in `src/podcast_conversations/config/__init__.py`:

```python
DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
DEFAULT_DEVICE = "cuda:0"
DEFAULT_USE_BF16 = True
DEFAULT_USE_COMPILE = False
```

### Keyword Analysis Configuration

Keywords and settings are defined in `config/keyword_analysis_config.yaml`:

```yaml
keyword_analysis:
  keywords:
    - "immigrant"
    - "minority"
    - "ethnic"
    # ... 730+ more keywords

  settings:
    case_sensitive: false
    partial_matches: true
    context_window: 100
    include_speaker_info: true
    include_timestamps: true
```

## Output Formats

All analysis stages produce structured JSON outputs with preserved data from previous stages:

- **Diarization**: RTTM files with speaker segments
- **Transcripts with Speakers**: JSON files with speaker labels added to segments
- **Classified Transcripts**: JSON files with classification labels from 5 models per utterance
- **Embeddings**: JSON files with 384-dim semantic embeddings per utterance
- **Keyword Analysis**: JSON/CSV/TXT files with keyword matches and context
- **LLM Annotations**: JSON files with hate speech detection, target groups, hate speech types, advertisement detection, and topic extraction per utterance
- **Feature Extraction**: JSON files with conversation-level metrics (questions, turn-taking, vocabulary, politeness)
- **Document Labels**: Markdown files with per-episode and per-show labels, plus comprehensive label distribution statistics for all classification models

## Terminal Data Visualizer

An interactive terminal user interface (TUI) for exploring and analyzing all pipeline outputs. Navigate through 21+ podcast shows, view classification results, LLM annotations, and generate statistics with an intuitive menu-driven interface.

### Features

- **📊 Classification Viewer**: Browse results from all 8 ML classification models
  - View per-model results with label distributions
  - Cross-model agreement analysis
  - Show-level and episode-level summaries
  - Export classification matrices to CSV
  
- **🤖 LLM Annotation Viewer**: Explore LLM-generated annotations
  - Filter by annotation type (hate speech/advertisements)
  - Filter by target group (women, LGBTQ+, minorities, etc.)
  - Filter by hate speech type (threats, vilification, dehumanization)
  - Topic frequency analysis
  - Show and episode-level summaries
  - Export to JSON/CSV

- **📈 Statistics Dashboard**: Comprehensive corpus statistics
  - Global statistics across all shows
  - Per-show metrics and trends
  - Per-episode detailed breakdowns
  
- **🔍 Search & Browse**: Navigate your data efficiently
  - Browse analysis files and folder structures
  - Search keywords across all transcripts
  - View episode transcripts with speaker labels
  - Bookmark favorite views

- **💾 Data Export**: Extract data in multiple formats
  - JSON for structured data
  - CSV for spreadsheet analysis
  - Markdown reports

### Quick Start

Launch the terminal visualizer:

```bash
uv run terminal-visualizer
```

### Navigation Guide

Main menu structure:
```
Terminal Data Visualizer v3.0
├── 1. Dashboard (outputs overview)
├── 2. Episode View (detailed episode analysis)
├── 3. Global Search (cross-corpus search)
├── 4. Summary Generator (show summaries)
├── 5. Report Generator (research reports)
├── 6. Visualizations (timelines, heatmaps)
├── 7. Bookmarks (saved views)
└── 8. Browse & Search (legacy menus)
    └── 7. Output Files Viewer
        ├── 4. Classification Viewer ⭐ NEW
        │   ├── View per-model results (8 models)
        │   ├── Label statistics
        │   ├── Cross-model analysis
        │   ├── Show classification summary
        │   └── Episode classification summary
        └── 5. LLM Annotation Viewer ⭐ NEW
            ├── Filter by annotation type
            ├── Filter by target group
            ├── Filter by hate speech type
            ├── Topic analysis
            ├── Show annotation summary
            └── Episode annotation summary
```

### Usage Examples

**View classification results for a specific model:**
1. Main Menu → 8 (Legacy Menu)
2. → 7 (Output Files)
3. → 4 (Classification Viewer)
4. → 1 (View Per-Model Results)
5. Select model (e.g., "hate_speech_detection")
6. Select podcast and episode

**Browse LLM annotations by target group:**
1. Main Menu → 8 (Legacy Menu)
2. → 7 (Output Files)
3. → 5 (LLM Annotation Viewer)
4. → 2 (View by Target Group)
5. Select target group (e.g., "women")
6. Select podcast and episode

**Generate show-level statistics:**
- Classification: Menu → 8 → 7 → 4 → 4 (Show Classification Summary)
- LLM Annotation: Menu → 8 → 7 → 5 → 5 (Show Annotation Summary)

### Keyboard Shortcuts

- `b` - Go back to previous menu
- `q` - Quit application
- `s` - Save current view (when available)
- `1-9` - Select menu options

### Data Requirements

The terminal visualizer works with outputs from:
- ✅ Classification pipeline (`classify_utterances.py`)
- ✅ LLM annotation pipeline (`annotate_with_llm.py`)
- ✅ Keyword analysis
- ✅ Feature extraction
- ✅ Transcripts with speaker labels

### Supported Analysis

**Classification Models** (8 total):
- hate_speech_detection
- fine_grained_hate_speech_detection  
- hate_against_minorities
- hostile_content
- ad_content_detection
- multilingual_hate_speech
- hate_speech_multilabel_bert
- beto_contextualized_hate_speech

**LLM Annotation Fields**:
- Hate speech detection (yes/no)
- Advertisement detection (yes/no)
- Target groups (6 categories)
- Hate speech types (4 categories)
- Main topics (extracted text)

### Performance

- **Fast loading**: LRU caching for file operations
- **Parallel processing**: Multi-threaded directory scanning
- **Memory efficient**: Lazy loading of large datasets
- **Responsive**: Instant navigation between views

### Export Capabilities

All views support exporting to:
- **CSV**: Classification matrices, annotation datasets
- **JSON**: Structured data exports
- **Markdown**: Summary reports
- **Screen capture**: Save terminal views

For more details, see [Terminal Visualizer Documentation](src/terminal_data_visualizer/README.md).

## Troubleshooting

### BFloat16 Errors (Got unsupported ScalarType BFloat16)

Your GPU doesn't support BFloat16 mixed precision.

**Fixed in latest version** - BFloat16 is now disabled by default and auto-falls back to FP32 if it fails.

If you still see this error:
1. The script should automatically retry with FP32
2. If it persists, ensure you're not explicitly using `--use-bf16`

**Supported GPUs for BFloat16**: H100, A100, and some newer architectures. V100 and older GPUs do NOT support it.

### CUDA Device Errors (device >= 0 && device < num_gpus)

This happens when you have a single GPU but workers try to access multiple devices.

**Fixed in latest version** - The script now properly manages CUDA device visibility for multiprocessing.

If you still see this error:
1. Explicitly set device: `--device cuda:0`
2. Reduce workers if issue persists

### Out of Memory (OOM) Errors

1. Reduce `--workers` (try 2 or 3)
2. Reduce `--batch-size` (try 2)
3. Process longer files separately

### Slower Than Expected

1. Check GPU utilization with `nvidia-smi`
2. Increase `--workers` if GPU usage < 80%
3. Try `--compile-mode max-autotune`

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'podcast_conversations'`

**Fix:**
1. Ensure you're running from the project directory:
   ```bash
   cd /path/to/podcast-conversations
   uv run python scripts/classify_utterances.py ...
   ```

2. For diarization, use the wrapper script:
   ```bash
   ./scripts/diarize_optimized.sh ...
   ```

3. If issues persist, install the package:
   ```bash
uv pip install -e .
   ```

### PyTorch/torchaudio Loading Issues

```bash
# The pipeline automatically falls back to ffmpeg if torchaudio fails
# Ensure ffmpeg is installed:
ffmpeg -version
```

## Development

### Install development dependencies

```bash
uv sync --all-extras
```

### Run tests

```bash
uv run pytest
```

### Static Analysis with Ruff

Ruff is used for linting and formatting. It enforces code quality rules defined in `pyproject.toml`.

**Check for issues:**
```bash
# Check all files
uv run ruff check .

# Check specific files or directories
uv run ruff check scripts/
uv run ruff check src/podcast_conversations/

# Show detailed error explanations
uv run ruff check . --show-source
```

**Fix auto-fixable issues:**
```bash
# Fix safe issues automatically
uv run ruff check --fix .

# Fix including unsafe fixes (review changes carefully)
uv run ruff check --fix --unsafe-fixes .
```

**Format code:**
```bash
# Format all files
uv run ruff format .

# Check formatting without making changes
uv run ruff format --check .

# Format specific files
uv run ruff format scripts/classify_utterances.py
```

**Common rules enforced:**
| Rule | Description |
|------|-------------|
| E | PEP 8 style errors |
| F | Pyflakes (unused imports, undefined names) |
| I | Import sorting |
| N | Naming conventions |
| W | PEP 8 warnings |

**Configuration:** See `pyproject.toml` for the full ruff configuration (line length: 100, target: Python 3.11).

## Project Structure

```
.
├── src/
│   ├── podcast_downloader/              # Podcast download from RSS/YouTube
│   │   ├── cli.py                       # CLI commands (download, discover, list)
│   │   ├── config.py                    # Download configuration
│   │   ├── discovery.py                 # Feed discovery via iTunes API
│   │   ├── downloader.py                # Async download with progress
│   │   ├── feed_parser.py               # RSS feed parsing
│   │   ├── metadata.py                  # Metadata storage
│   │   └── providers/                   # Download providers (RSS, YouTube)
│   │
│   ├── podcast_conversations/           # Core analysis package
│   │   ├── config/                      # Configuration constants
│   │   ├── monitoring/                  # Resource monitoring (CPU, memory, GPU)
│   │   ├── transcription/               # WhisperX transcription with MLX support
│   │   ├── diarization/                 # Speaker diarization with pyannote.audio
│   │   ├── embeddings_method/           # Semantic similarity analysis with MLX
│   │   ├── analysis/                    # Keyword matching and context extraction
│   │   ├── utterance_classification/    # Transformer-based classification
│   │   ├── feature_extraction/          # Questions, turn-taking, vocabulary, politeness
│   │   ├── llm_annotation/              # Llama 3.3 70B content annotation
│   │   ├── speaker_labeling/            # 5-stage speaker identification pipeline
│   │   └── types/                       # Pydantic data models and validation
│   │
│   └── terminal_data_visualizer/        # Interactive TUI for data exploration
│       ├── main.py                      # Entry point and menu system
│       ├── dashboard.py                 # Outputs health dashboard
│       ├── episode_view.py              # Episode-centric viewer
│       ├── global_search.py             # Cross-output search
│       ├── summary_generator.py         # Show-level summaries
│       ├── report_generator.py          # Research report generation
│       ├── features_explorer.py         # CSV data exploration with plots
│       ├── classification_viewer.py     # Classification results browser
│       ├── llm_annotation_viewer.py     # LLM annotation browser
│       ├── keyword_browser.py           # Keyword analysis exploration
│       ├── statistics.py                # Visualization and statistics
│       ├── visualizations/              # Advanced visualizations (timeline, heatmap)
│       └── screen_capture.py            # SVG/JPG export
│
├── scripts/                             # Pipeline CLI scripts (see scripts/README.md)
│   ├── run_full_corpus_pipeline.py      # Run all 7 stages with one command
│   ├── run_analysis_pipeline.py         # Run analysis stages only
│   ├── run_speaker_labeling_pipeline.py # Speaker identification pipeline
│   ├── transcribe_batch.py              # WhisperX batch transcription
│   ├── diarize_batch.py                 # Speaker diarization
│   ├── combine_transcripts_with_speakers.py  # Merge transcripts + diarization
│   ├── combine_consecutive_speakers.py  # Merge same-speaker segments
│   ├── analyze_embeddings.py            # Semantic similarity analysis
│   ├── download_and_process.py          # Download and process in one step
│   ├── classify_utterances.py           # Multi-label classification
│   ├── extract_features.py              # Conversation feature extraction
│   ├── annotate_with_llm.py             # LLM-based content annotation
│   ├── generate_document_labels.py      # Document-level label aggregation
│   └── generate_speaker_statistics.py   # Speaker statistics generation
│
├── config/                              # Configuration files (see config/README.md)
│   ├── taxonomy.yml                     # Dehumanization taxonomy for embeddings
│   ├── classifiers.yaml                 # Transformer models for classification
│   ├── keyword_analysis_config.yaml     # Keywords and matching settings
│   └── llm_annotation_config.yaml       # LLM annotation questions and prompts
│
├── docs/                                # Documentation
│   ├── pipelines/                       # Per-pipeline documentation
│   ├── features.md                      # Feature inventory
│   └── MLX_SUPPORT.md                   # Apple Silicon MLX support guide
│
├── tests/                               # Pytest test suite
├── outputs/                             # Processing outputs (not in git)
├── downloads/                           # Downloaded audio files (not in git)
├── pyproject.toml                       # Project configuration
├── CLAUDE.md                            # Development instructions for Claude Code
└── README.md                            # This file
```

## Requirements

- Python 3.11+
- GPU (optional, recommended for transcription, diarization, classification, embeddings, and LLM annotation):
  - **NVIDIA GPUs (CUDA)**: Full feature support with 4-bit/8-bit quantization and Flash Attention 2
  - **Apple Silicon (MPS)**: M1/M2/M3/M4 Macs with unified memory, float16 inference
  - **LLM Annotation (NVIDIA)**: H100 80GB (recommended), A100 40GB, or RTX 4090 24GB
  - **LLM Annotation (Apple Silicon)**: M2/M3 Pro/Max/Ultra with 32GB+ unified memory (smaller models recommended)
- FFmpeg (for audio processing)
- HuggingFace account (for diarization, gated models, and Llama access)
- ~2GB disk space for sentence embedding models (first run only)
- ~3GB disk space for WhisperX large-v3 model (first run only)
- ~35GB disk space for Llama 3.3 70B model weights (LLM annotation, first run only)
- Dependencies (auto-installed via `uv sync`):
  - whisperx (GPU-accelerated transcription)
  - pyannote.audio (speaker diarization)
  - transformers (classification and LLM inference)
  - sentence-transformers (embeddings)
  - bitsandbytes (4-bit/8-bit quantization, NVIDIA only)
  - mlx, mlx-whisper, mlx-embedding-models (Apple Silicon, optional via `uv sync --extra macos`)
  - mlx-lm (Apple Silicon LLM inference, optional)
  - psutil (resource detection for dynamic batch sizing)
  - torch, torchaudio (deep learning)
  - httpx, feedparser (podcast download)
  - See `pyproject.toml` for complete list

## Technical Details

### Architecture

- **Main process**: File discovery, batching, progress tracking
- **Worker processes**: Each initializes its own pipeline and processes files independently
- **Process spawn method**: Avoids CUDA context sharing issues
- **Result aggregation**: Uses `concurrent.futures.as_completed()` for efficient result collection

### Memory Management

- Each worker loads model independently (~2-3GB GPU memory per worker)
- Automatic cache clearing on OOM errors
- Retry logic: 2 attempts per file on OOM

### File Processing Order

1. Discover all transcript-audio pairs
2. Skip files with existing RTTM outputs
3. Compute audio durations (fast metadata read)
4. Sort files by duration
5. Group into batches of similar durations
6. Submit all files to worker pool
7. Process results as they complete



## Credits

Built with:
- [WhisperX](https://github.com/m-bain/whisperX) - GPU-accelerated transcription with word-level timestamps
- [mlx-whisper](https://pypi.org/project/mlx-whisper/) - Apple Silicon transcription
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) - Speaker diarization
- [sentence-transformers](https://www.sbert.net/) - Semantic embeddings
- [mlx-embedding-models](https://github.com/taylorai/mlx_embedding_models) - Apple Silicon embeddings
- [Llama 3.3 70B](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) - LLM annotation
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) - 4-bit/8-bit quantization (NVIDIA)
- [MLX](https://github.com/ml-explore/mlx) - Apple Silicon ML framework
- [torch](https://pytorch.org/) - Deep learning framework
- [polars](https://pola.rs/) - High-performance dataframes
- [click](https://click.palletsprojects.com/) - CLI framework
- [rich](https://rich.readthedocs.io/) - Terminal UI
- [pydantic](https://docs.pydantic.dev/) - Data validation
- [httpx](https://www.python-httpx.org/) - Async HTTP client
- [feedparser](https://feedparser.readthedocs.io/) - RSS feed parsing
- [uv](https://docs.astral.sh/uv/) - Package management

## License

This project is licensed under the terms of the license file.

## Contributing

1. Follow the code style defined in `pyproject.toml` (ruff configuration)
2. Add tests for new features
3. Update documentation as needed
4. Run linting and formatting before committing
