# Embeddings Method

Semantic analysis of podcast conversations using sentence embeddings and cosine similarity to detect dehumanizing language patterns.

## Overview

This package uses sentence transformers to generate semantic embeddings for both podcast utterances and a taxonomy of dehumanizing language patterns. It finds matches using cosine similarity, providing contextual detection of harmful language beyond simple keyword matching.

## Features

- **Semantic Similarity**: Detects conceptually similar language, not just exact keywords
- **Taxonomy-Based**: Uses structured taxonomy of dehumanization patterns
- **GPU Acceleration**: 50-100× faster on CUDA/MPS vs CPU
- **Embeddings Caching**: Generated embeddings are cached for reuse
- **Contextual Results**: Includes surrounding utterances for each match
- **Interactive CLI**: User-friendly interface with progress tracking
- **Auto-Detection**: Automatically uses classified transcripts when available
- **Comprehensive Output**: Full results and summary statistics

## Installation

The package is automatically installed with the main project:

```bash
uv sync
```

### Apple Silicon (MLX Backend)

For optimized embedding generation on Apple Silicon (M1/M2/M3/M4), install the optional MLX dependencies:

```bash
uv sync --extra macos
```

This installs `mlx` and `mlx-embedding-models` for native Apple Silicon acceleration.

## Configuration

### Taxonomy Configuration

The dehumanization taxonomy is defined in `config/taxonomy.yaml`:

```yaml
target_groups:
  - id: immigrants
    name: Immigrants
  - id: political_opponents
    name: Political Opponents
  # ... more groups

strategies:
  - id: threat_to_culture
    name: Threat to Culture
    description: "Framing group as threats to cultural identity"
  # ... more strategies

examples:
  - target_group_id: immigrants
    strategy_id: threat_to_culture
    items:
      - "they don't share our values"
      - "they refuse to assimilate"
      # ... more examples
```

**Default Taxonomy:**
- 6 target groups (immigrants, political opponents, religious minorities, etc.)
- 4 othering strategies per group (threat framing, vilification, etc.)
- 24 total taxonomy items for comprehensive analysis

## Usage

### Command Line

#### Interactive Analysis

```bash
# Run interactive embeddings analysis
uv run python scripts/analyze_embeddings.py
```

This will:
1. Auto-detect classified transcripts (or fall back to speaker-only transcripts)
2. Show available podcast shows
3. Prompt for show selection
4. Load taxonomy
5. Generate embeddings (cached for reuse)
6. Prompt for similarity threshold
7. Analyze and save results

#### Force Regenerate Embeddings

```bash
# Regenerate embeddings even if cached
uv run python scripts/analyze_embeddings.py --force-regenerate
```

#### Specify Input Directory

```bash
# Use specific transcript directory
uv run python scripts/analyze_embeddings.py \
  --transcripts-dir outputs/transcripts_with_speakers_and_labels
```

### Python API

```python
from pathlib import Path
from podcast_conversations.embeddings_method import (
    # Device and model utilities
    detect_device,
    load_embedding_model,
    get_optimal_batch_size,
    # Taxonomy loading
    load_taxonomy,
    generate_taxonomy_embeddings,
    # Transcript processing
    process_transcript_file,
    check_embeddings_exist,
    # Analysis functions
    analyze_transcript,
    analyze_multiple_transcripts,
    cosine_similarity,
    # Discovery utilities
    discover_shows,
    display_available_shows,
    # Output functions
    save_analysis_results,
    generate_summary_statistics,
)

# 1. Initialize model
device = detect_device()  # Auto-detects CUDA/MPS/CPU
model = load_embedding_model(device)
batch_size = get_optimal_batch_size()

# 2. Load taxonomy
taxonomy = load_taxonomy()  # Loads from config/taxonomy.yaml
taxonomy_embeddings = generate_taxonomy_embeddings(taxonomy, model)

# 3. Discover available shows
shows = discover_shows()
display_available_shows(shows)  # Pretty prints available shows

# 4. Process transcript (generates embeddings) - skip if already exists
transcript_path = Path("outputs/transcripts_with_speakers/show/episode.json")
output_path = Path("outputs/analysis/embeddings_method/show/embeddings/episode.json")

if not check_embeddings_exist(output_path):
    process_transcript_file(transcript_path, output_path, model, batch_size=batch_size)

# 5. Analyze for matches
threshold = 0.35  # Similarity threshold
results = analyze_transcript(
    embedding_file=output_path,
    taxonomy_embeddings=taxonomy_embeddings,
    threshold=threshold,
)

# 6. Save results and generate statistics
save_analysis_results(
    show_name="show_name",
    results=[results],
    threshold=threshold,
    model_name="all-MiniLM-L6-v2",
)

stats = generate_summary_statistics([results])
print(f"Found {results['matches_found']} matches")
```

## How It Works

### 1. Embedding Generation

**Transcript Embeddings:**
```python
# Each utterance is converted to a 384-dimensional vector
model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode("They don't share our values")
# Result: array([0.123, -0.456, 0.789, ...])  # 384 dimensions
```

**Taxonomy Embeddings:**
```python
# Each (target_group, strategy) pair gets two embeddings:
# 1. Description only
# 2. Description + examples (up to 10)
combined_text = f"{description} {' '.join(examples[:10])}"
embedding = model.encode(combined_text)
```

### 2. Similarity Calculation

```python
# Cosine similarity between utterance and taxonomy embeddings
similarity = cosine_similarity(utterance_embedding, taxonomy_embedding)
# Result: 0.0 to 1.0 (higher = more similar)

# Match if similarity >= threshold
if similarity >= threshold:
    # Record as match
```

### 3. Context Extraction

For each match, includes:
- 3 previous utterances (with speaker and timestamp)
- The matched utterance
- 3 following utterances (with speaker and timestamp)

This provides conversational context for understanding the match.

## Similarity Thresholds

Embeddings-based similarity typically yields lower scores than exact text matching:

| Threshold | Interpretation | Use Case |
|-----------|----------------|----------|
| 0.5-0.6 | Very strong similarity | Strict matching, high precision |
| 0.35-0.45 | Meaningful similarity | **Recommended for general use** |
| 0.2-0.3 | Weak similarity | Lenient matching, high recall |

**Default: 0.35** - Balances precision and recall for semantic similarity.

## Input/Output

### Input

**Auto-Detection:**
1. Checks `outputs/transcripts_with_speakers_and_labels/` first (preferred)
2. Falls back to `outputs/transcripts_with_speakers/` if not found
3. Displays which directory is being used

**Input Format:**
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
      "text": "Example utterance",
      "speaker": "SPEAKER_00",
      "classifications": [...]  // Optional, from classification stage
    }
  ]
}
```

### Output

**Directory Structure:**
```
outputs/analysis/embeddings_method/
└── SHOW_NAME/
    ├── embeddings/                           # Cached embeddings
    │   ├── episode_001.json
    │   ├── episode_002.json
    │   └── ...
    ├── full_results_2026-01-05.json         # Detailed matches
    └── summary_2026-01-05.json              # Statistics
```

**Full Results Format:**
```json
{
  "show_name": "the_daily",
  "analysis_timestamp": "2026-01-05T10:30:00",
  "threshold": 0.35,
  "embedding_model": "all-MiniLM-L6-v2",
  "total_files": 5,
  "files_analyzed": [
    {
      "file": "outputs/.../episode_001.json",
      "matches_found": 23,
      "matches": [
        {
          "target_group": "immigrants",
          "othering_strategy": "threat_to_culture",
          "similarity_score": 0.487,
          "matched_text": "they don't share our values",
          "speaker": "SPEAKER_01",
          "timestamp": 932.5,
          "description": "framing immigrants as threats to cultural identity",
          "context": {
            "previous_utterances": [
              {"text": "...", "speaker": "SPEAKER_01", "timestamp": 920.1},
              {"text": "...", "speaker": "SPEAKER_01", "timestamp": 925.3},
              {"text": "...", "speaker": "SPEAKER_01", "timestamp": 929.8}
            ],
            "following_utterances": [
              {"text": "...", "speaker": "SPEAKER_01", "timestamp": 935.2},
              {"text": "...", "speaker": "SPEAKER_01", "timestamp": 938.5},
              {"text": "...", "speaker": "SPEAKER_02", "timestamp": 942.1}
            ]
          }
        }
      ]
    }
  ]
}
```

**Summary Statistics Format:**
```json
{
  "analysis_timestamp": "2026-01-05T10:30:00",
  "show_name": "the_daily",
  "threshold_used": 0.35,
  "embedding_model": "all-MiniLM-L6-v2",
  "summary_statistics": {
    "total_files_analyzed": 5,
    "total_matches_found": 87,
    "matches_by_target_group": {
      "immigrants": 34,
      "political_opponents": 28,
      "religious_minorities": 15
    },
    "matches_by_strategy": {
      "threat_to_culture": 40,
      "vilification": 30,
      "explicit_dehumanization": 17
    },
    "average_similarity_score": 0.412
  }
}
```

## Performance

### GPU Acceleration

| Device | Speed | Example |
|--------|-------|---------|
| CUDA (H100/A100) | 50-100× faster than CPU | 1000 utterances in ~2 seconds |
| MLX (Apple Silicon) | 30-50× faster than CPU | 1000 utterances in ~3 seconds |
| MPS (Apple Silicon) | 30-50× faster than CPU | 1000 utterances in ~3 seconds |
| CPU | Baseline | 1000 utterances in ~60 seconds |

**Device Selection:**
- `--device auto` (default): Automatically selects the best available device
- `--device mlx`: Use MLX backend (Apple Silicon only, requires `uv sync --extra macos`)
- `--device mps`: Use Metal Performance Shaders (Apple Silicon)
- `--device cuda`: Use NVIDIA GPU
- `--device cpu`: Force CPU processing

### Embeddings Caching

- Generated embeddings are saved to disk
- Reused across multiple analyses with different thresholds
- Use `--force-regenerate` to regenerate cached embeddings
- Significantly speeds up repeated analyses

### Batch Processing

- Processes utterances in batches of 32 (configurable)
- Efficient GPU utilization
- Progress bar shows real-time status

## Architecture

### Components

- **embeddings.py**: Embedding model loading and generation (supports CUDA, MPS, MLX, CPU)
- **mlx_backend.py**: MLX-optimized embedding backend for Apple Silicon
- **taxonomy.py**: Taxonomy loading and validation
- **analysis.py**: Similarity calculation and matching (MLX-accelerated on Apple Silicon)
- **discovery.py**: Show discovery and file mapping
- **output.py**: Results formatting and statistics

### Data Flow

```
1. Load taxonomy (config/taxonomy.yaml)
   ↓
2. Generate taxonomy embeddings (cached)
   ↓
3. Discover shows (auto-detect directory)
   ↓
4. User selects show
   ↓
5. Generate transcript embeddings (cached)
   ↓
6. User sets similarity threshold
   ↓
7. Calculate cosine similarities
   ↓
8. Filter matches above threshold
   ↓
9. Add context (previous/following utterances)
   ↓
10. Save results + statistics
```

### Embedding Model

**Model**: `all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Speed**: Fast (optimized for CPU/GPU)
- **Quality**: Good balance of speed and accuracy
- **Size**: ~80MB download (first run only)

## Advantages Over Keyword Matching

### Semantic Understanding

**Keyword Matching:**
```python
# Only finds exact text: "they refuse to assimilate"
```

**Embeddings Method:**
```python
# Also finds semantically similar:
# - "they won't adapt to our culture"
# - "they resist integration"
# - "they maintain their own customs"
```

### Context Preservation

- Includes 3 previous + 3 following utterances
- Provides conversational context
- Helps understand intent and tone

### Flexible Thresholds

- Adjust precision/recall balance
- No need to manually list all variations
- Captures conceptual similarity

## Troubleshooting

### No Shows Found

**Error:** `No shows with transcripts found`

**Fix:**
1. Ensure transcripts exist in one of:
   - `outputs/transcripts_with_speakers_and_labels/`
   - `outputs/transcripts_with_speakers/`
2. Check directory structure: `outputs/transcripts_with_speakers/SHOW_NAME/*.json`
3. Run classification step if needed

### Taxonomy File Not Found

**Error:** `Taxonomy configuration not found`

**Fix:**
1. Ensure `config/taxonomy.yaml` exists
2. Check YAML syntax is valid
3. Verify all required fields are present

### Low Similarity Scores

**Issue:** All similarity scores very low (< 0.2)

**Fix:**
1. Lower threshold (try 0.25)
2. Check taxonomy examples are relevant
3. Verify transcript quality
4. Try different shows/episodes

### High Similarity Scores (False Positives)

**Issue:** Too many matches, low precision

**Fix:**
1. Raise threshold (try 0.45)
2. Review taxonomy examples for specificity
3. Check for overly broad descriptions

### Slow Performance

**Fix:**
1. Ensure GPU acceleration is working:
   ```python
   from podcast_conversations.embeddings_method import detect_device
   print(detect_device())  # Should show 'cuda' or 'mps'
   ```
2. Install PyTorch with CUDA support
3. Use cached embeddings (don't use `--force-regenerate`)

## Examples

### Analyze Specific Show

```bash
# Interactive mode - select "The Daily" from menu
uv run python scripts/analyze_embeddings.py
```

### Batch Analysis Multiple Shows

```python
from pathlib import Path
from podcast_conversations.embeddings_method import (
    discover_shows,
    load_embedding_model,
    load_taxonomy,
    generate_taxonomy_embeddings,
    analyze_multiple_transcripts,
    save_analysis_results,
)

# Setup
device = detect_device()
model = load_embedding_model(device)
taxonomy = load_taxonomy()
taxonomy_embeddings = generate_taxonomy_embeddings(taxonomy, model)

# Analyze all shows
shows = discover_shows()
threshold = 0.35

for show_name, transcript_paths in shows.items():
    # Generate embeddings
    embedding_paths = []
    for path in transcript_paths:
        output_path = Path(f"outputs/analysis/embeddings_method/{show_name}/embeddings/{path.name}")
        process_transcript_file(path, output_path, model)
        embedding_paths.append(output_path)

    # Analyze
    results = analyze_multiple_transcripts(embedding_paths, taxonomy_embeddings, threshold)

    # Save
    save_analysis_results(show_name, results, threshold, "all-MiniLM-L6-v2")

    print(f"{show_name}: {sum(r['matches_found'] for r in results)} matches")
```

### Custom Threshold Analysis

```python
# Analyze with multiple thresholds
thresholds = [0.25, 0.30, 0.35, 0.40, 0.45]

for threshold in thresholds:
    results = analyze_multiple_transcripts(embedding_paths, taxonomy_embeddings, threshold)
    total_matches = sum(r['matches_found'] for r in results)
    print(f"Threshold {threshold}: {total_matches} matches")
```

## Integration with Pipeline

### Stage 5: Embeddings

The embeddings method is stage 5 in the full pipeline:

```bash
# Stage 1: Transcription
uv run python scripts/transcribe_batch.py ...

# Stage 2: Diarization
./scripts/diarize_all_podcasts.sh ...

# Stage 3: Combine Transcripts with Speakers
uv run python scripts/combine_transcripts_with_speakers.py ...

# Stage 4: Combine Consecutive Speakers
uv run python scripts/combine_consecutive_speakers.py ...

# Stage 5: Embeddings (THIS PACKAGE)
uv run python scripts/analyze_embeddings.py

# Stage 6: Keyword Analysis (via analysis pipeline)
uv run python scripts/run_analysis_pipeline.py --skip-features --skip-classification ...

# Stage 7: Classification
uv run python scripts/classify_utterances.py ...
```

### Data Preservation

The embeddings stage preserves ALL data from previous stages:
- Original metadata
- Speaker labels (from diarization)
- Classification labels (from classification)
- Timestamps
- Any other fields

Output files include both embeddings AND all previous enrichments.

## Advanced Usage

### Custom Embedding Model

```python
from sentence_transformers import SentenceTransformer

# Use different model
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
model = model.to(device)

# Rest of pipeline unchanged
```

### Custom Taxonomy

Edit `config/taxonomy.yaml`:
```yaml
target_groups:
  - id: custom_group
    name: Custom Target Group

strategies:
  - id: custom_strategy
    name: Custom Strategy
    description: "Description of the strategy"

examples:
  - target_group_id: custom_group
    strategy_id: custom_strategy
    items:
      - "example phrase 1"
      - "example phrase 2"
```

## Credits

Built with:
- [sentence-transformers](https://www.sbert.net/) - Semantic embeddings
- [mlx-embedding-models](https://github.com/taylorai/mlx_embedding_models) - Apple Silicon embeddings
- [MLX](https://github.com/ml-explore/mlx) - Apple Silicon ML framework
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) - Embedding model

## Further Reading

- [Embeddings Method](../../../README.md#embeddings-method) - Detailed methodology and pipeline documentation
