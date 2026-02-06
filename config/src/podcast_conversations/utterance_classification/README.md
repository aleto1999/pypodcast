# Utterance Classification

Classify utterances in podcast transcripts using multiple transformer-based models.

## Overview

This package applies text classification to each utterance (segment) in podcast transcripts. It supports multiple classification models configured via YAML and outputs enriched transcripts with classification labels, model names, and confidence scores.

## Features

- **Multiple Models**: Apply multiple classification models to each utterance
- **Configurable**: Models defined in YAML configuration file
- **Mirrored Structure**: Maintains input directory structure in output
- **GPU Acceleration**: Automatic device detection (CUDA → MPS → CPU)
- **Apple Silicon Optimized**: Native MPS acceleration on M1/M2/M3/M4 Macs
- **Dynamic Batch Sizing**: Auto-adjusts based on available memory
- **Rich Progress**: Beautiful terminal UI with progress tracking

## Installation

The package is automatically installed with the main project. To ensure all dependencies are available:

```bash
uv sync
```

## Configuration

Models are defined in `config/classifiers.yaml`:

```yaml
models:
  - name: hate_speech_detection
    description: "A model for binary hate speech detection."
    model_repo: "cardiffnlp/twitter-roberta-base-hate-latest"
    labels: ["NOT-HATE", "HATE"]

  - name: fine_grained_hate_speech_detection
    description: "A model for fine-grained hate speech detection."
    model_repo: "GroNLP/hateBERT"
    labels: ["acceptable", "inappropriate", "offensive", "violent"]
```

## Usage

### Command Line

Process all transcripts in a directory:

```bash
uv run python scripts/classify_utterances.py \
    --transcripts-dir outputs/transcripts_with_speakers \
    --output-dir outputs/transcripts_with_speakers_and_labels \
    --config config/classifiers.yaml
```

#### Options

- `--transcripts-dir`: Input directory with transcript JSON files (required)
- `--output-dir`: Output directory for classified transcripts (required)
- `--config`: Path to classifiers YAML file (default: `config/classifiers.yaml`)
- `--device`: Device to use - `auto`, `cpu`, `cuda`, `mps` (default: `auto`)
- `--batch-size`: Batch size for processing (default: 8)
- `--log-level`: Logging level - `DEBUG`, `INFO`, `WARNING`, `ERROR` (default: `INFO`)

### Python API

```python
from pathlib import Path
from podcast_conversations.utterance_classification import (
    ClassifierConfig,
    UtteranceClassifier,
    TranscriptProcessor,
    detect_classification_device,
    get_classification_device_info,
    get_optimal_batch_size,
    check_classifications_exist,
)

# Check available devices
print(get_classification_device_info())
# {'is_apple_silicon': True, 'cuda_available': False, 'mps_available': True,
#  'recommended_device': 'mps', 'system_memory_gb': 64.0}

# Load configuration
config = ClassifierConfig("config/classifiers.yaml")
models = config.get_models()

# Get optimal batch size for detected device
device = detect_classification_device()  # Returns "mps" on Apple Silicon
batch_size = get_optimal_batch_size(device)

# Initialize classifier with auto device detection
classifier = UtteranceClassifier(models=models, device="auto", batch_size=batch_size)

# Check if output already exists (useful for resuming)
if not check_classifications_exist(
    input_dir=Path("outputs/transcripts_with_speakers"),
    output_dir=Path("outputs/transcripts_with_speakers_and_labels"),
):
    # Process files
    processor = TranscriptProcessor(classifier)
    stats = processor.process_directory(
        input_dir=Path("outputs/transcripts_with_speakers"),
        output_dir=Path("outputs/transcripts_with_speakers_and_labels"),
    )

    print(f"Processed {stats['files_processed']} files")
    print(f"Classified {stats['total_utterances']} utterances")
```

## Input Format

Input JSON files should contain segments with text:

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
      "text": "Hello and welcome to the show.",
      "speaker": "SPEAKER_00"
    }
  ]
}
```

## Output Format

Output JSON files include the original data plus classifications:

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
      "text": "Hello and welcome to the show.",
      "speaker": "SPEAKER_00",
      "classifications": [
        {
          "model_name": "hate_speech_detection",
          "label": "NOT-HATE",
          "confidence": 0.9876
        },
        {
          "model_name": "fine_grained_hate_speech_detection",
          "label": "acceptable",
          "confidence": 0.9543
        }
      ]
    }
  ]
}
```

## Directory Structure

The output maintains the input directory structure:

```
Input:
  outputs/transcripts_with_speakers/
    the_megyn_kelly_show/
      episode1.json
      episode2.json
    the_ben_shapiro_show/
      episode1.json

Output:
  outputs/transcripts_with_speakers_and_labels/
    the_megyn_kelly_show/
      episode1.json  # with classifications
      episode2.json  # with classifications
    the_ben_shapiro_show/
      episode1.json  # with classifications
```

## Available Models

The default configuration includes:

1. **hate_speech_detection** - Binary hate speech detection
2. **fine_grained_hate_speech_detection** - Multi-class hate speech classification
3. **hate_against_minorities** - Toxic content detection with multiple categories
4. **hostile_content** - Offensive speech detection
5. **ad_content_detection** - Advertisement content detection

## Performance

### Device Support

| Device | Auto-Detected | Batch Size | Notes |
|--------|---------------|------------|-------|
| CUDA (NVIDIA) | ✅ | 16-32 | Fastest for large batches |
| MPS (Apple Silicon) | ✅ | 16-24 | Native M1/M2/M3/M4 acceleration |
| CPU | Fallback | 4-8 | Slowest but always available |

### Apple Silicon (macOS)

On Apple Silicon Macs, the pipeline automatically uses MPS acceleration:

```bash
uv run python scripts/classify_utterances.py \
    --transcripts-dir outputs/transcripts_with_speakers \
    --output-dir outputs/classified \
    --device auto
```

The `--device auto` setting (default) automatically detects Apple Silicon and uses MPS. Batch size is auto-adjusted based on unified memory (typically 16-24 for 16GB+ systems).

### CUDA (NVIDIA GPUs)

```bash
uv run python scripts/classify_utterances.py \
    --transcripts-dir outputs/transcripts_with_speakers \
    --output-dir outputs/classified \
    --device cuda \
    --batch-size 32
```

### CPU Only

```bash
uv run python scripts/classify_utterances.py \
    --transcripts-dir outputs/transcripts_with_speakers \
    --output-dir outputs/classified \
    --device cpu \
    --batch-size 4
```

## Error Handling

- Continues processing even if individual models fail to load
- Logs errors for failed files but continues with remaining files
- Provides comprehensive statistics at completion

## Requirements

- Python 3.11+
- PyTorch with CUDA, MPS, or CPU support
- transformers library
- GPU recommended for efficient batch processing

## Credits

Built with:
- [transformers](https://huggingface.co/docs/transformers) - Model loading and inference
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Rich](https://rich.readthedocs.io/) - Terminal UI
