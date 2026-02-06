# Pipeline Scripts

This directory contains the CLI scripts for running the various stages of the podcast conversations analysis pipeline.

## Overview

The scripts are designed to be modular, allowing you to run individual stages or complete pipelines. All scripts are intended to be run from the project's root directory using `uv run`.

**Common CLI Flags:**

Most scripts support these standard flags:

| Flag | Description |
|------|-------------|
| `--output-dir` | Output directory for results. |
| `--force` | Force reprocessing of files with existing outputs. |
| `--dry-run` | Preview what would be processed without running. |
| `--log-level` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`. |

---

## Main Pipeline Scripts

### `run_full_corpus_pipeline.py`

Runs the main 7-stage corpus processing pipeline in sequence. This is the most comprehensive script for processing raw audio into fully analyzed transcripts.

**Stages:**
1.  Transcription (WhisperX)
2.  Diarization (pyannote.audio)
3.  Combine Transcripts with Speakers
4.  Combine Consecutive Speakers
5.  Embeddings Analysis
6.  Keyword Analysis
7.  Utterance Classification

**Usage:**
```bash
uv run python scripts/run_full_corpus_pipeline.py \
    --audio-dir /path/to/audio \
    --hf-token YOUR_HF_TOKEN \
    --torch-compile
```

**Common Options:**
- `--skip-transcription`: Use existing transcripts.
- `--skip-diarization`: Use existing RTTM files.
- `--skip-embeddings`: Skip embeddings analysis.
- `--skip-classification`: Skip GPU-intensive classification.

### `run_analysis_pipeline.py`

Runs only the analysis stages of the pipeline on existing post-processed transcripts.

**Stages:**
1.  Embeddings Analysis
2.  Keyword Analysis
3.  Utterance Classification

**Usage:**
```bash
uv run python scripts/run_analysis_pipeline.py \
    --transcripts-dir outputs/transcripts_postprocessed
```

---

## Individual Stage Scripts

### `transcribe_batch.py`

Generates word-level transcripts from audio files using WhisperX.

**Usage:**
```bash
uv run python scripts/transcribe_batch.py \
    --audio-dir downloads/the_daily \
    --output-dir outputs/transcripts \
    --model large-v3 \
    --device cuda
```
See the [Transcription Package README](src/podcast_conversations/transcription/README.md) for more details.

### `diarize_batch.py` & `diarize_all_podcasts.sh`

Identifies speakers in audio files. `diarize_batch.py` is the core Python script, while `diarize_all_podcasts.sh` is a convenience wrapper for processing multiple podcast directories.

**Usage:**
```bash
# Process all podcasts with optimal settings using the wrapper
./scripts/diarize_all_podcasts.sh \
  --transcripts-root outputs/transcripts \
  --audio-base-dir /path/to/audio \
  --output-dir outputs/diarizations \
  --hf-token YOUR_HF_TOKEN
```
See the [Diarization Package README](src/podcast_conversations/diarization/README.md) for detailed options.

### `combine_transcripts_with_speakers.py`

Merges the output of the transcription and diarization stages.

**Usage:**
```bash
uv run python scripts/combine_transcripts_with_speakers.py \
  --transcripts-dir outputs/transcripts \
  --diarizations-dir outputs/diarizations \
  --output-dir outputs/transcripts_with_speakers
```

### `combine_consecutive_speakers.py`

Post-processes transcripts to merge consecutive segments from the same speaker into a single segment.

**Usage:**
```bash
uv run python scripts/combine_consecutive_speakers.py \
  --transcripts-dir outputs/transcripts_with_speakers \
  --output-dir outputs/transcripts_postprocessed
```

### `analyze_embeddings.py`

Performs semantic similarity analysis using sentence embeddings.

**Usage:**
```bash
# Interactive mode (select one show)
uv run python scripts/analyze_embeddings.py

# Batch process all shows
uv run python scripts/analyze_embeddings.py --process-all --threshold 0.35
```
See the [Embeddings Method Package README](src/podcast_conversations/embeddings_method/README.md) for details.

### `analyze_keywords.py`

Searches transcripts for a predefined list of keywords.

**Usage:**
```bash
uv run python scripts/analyze_keywords.py \
  --transcripts-dir outputs/transcripts_postprocessed \
  --config config/keyword_analysis_config.yaml \
  --format all
```
See the [Analysis Package README](src/podcast_conversations/analysis/README.md) for configuration options.

### `classify_utterances.py`

Classifies each utterance in a transcript using multiple transformer-based models.

**Usage:**
```bash
# Classify all utterances
uv run python scripts/classify_utterances.py \
  --transcripts-dir outputs/transcripts_postprocessed \
  --output-dir outputs/classified \
  --config config/classifiers.yaml \
  --device auto

# Classify only transcripts with keyword matches
uv run python scripts/classify_utterances.py \
  --transcripts-dir outputs/transcripts_postprocessed \
  --output-dir outputs/classified \
  --keywords-dir outputs/analysis/keyword_analysis \
  --min-matches 5
```
See the [Utterance Classification Package README](src/podcast_conversations/utterance_classification/README.md) for model configuration details.

---

## Standalone Pipeline Scripts

These scripts run complex pipelines that are not part of the main corpus processing flow.

### `annotate_with_llm.py`

Analyzes each utterance using a Large Language Model (Llama 3.3 70B Instruct) with local GPU inference.

**Features:**
- Hate speech detection
- Target group identification
- Hate speech type classification
- Advertisement detection
- Topic extraction

**Usage:**
```bash
uv run python scripts/annotate_with_llm.py \
  --transcripts-dir outputs/transcripts_with_speakers \
  --output-dir outputs/analysis/llm_annotation_method \
  --hf-token $HF_TOKEN
```
See the [LLM Annotation Package README](src/podcast_conversations/llm_annotation/README.md) for hardware requirements and advanced options.

### `run_speaker_labeling_pipeline.py`

Replaces generic speaker IDs (`SPEAKER_00`) with actual speaker names using a 5-stage pipeline.

**Usage:**
```bash
# Basic usage with heuristic classification
uv run python scripts/run_speaker_labeling_pipeline.py --podcast "the_joe_rogan_experience"

# With LLM-based role classification (requires OpenAI API key)
uv run python scripts/run_speaker_labeling_pipeline.py --all --use-llm
```
See the [Speaker Labeling Package README](src/podcast_conversations/speaker_labeling/README.md) for detailed configuration.

### `extract_features.py`

Extracts conversation-level metrics from transcripts.

**Features:**
- Question patterns
- Turn-taking dynamics
- Vocabulary diversity
- Politeness markers

**Usage:**
```bash
uv run python scripts/extract_features.py \
  --transcripts-dir outputs/transcripts_with_speakers \
  --output-dir outputs/features
```
See the [Feature Extraction Package README](src/podcast_conversations/feature_extraction/README.md) for more info.

### `generate_document_labels.py`

Aggregates utterance-level classifications to generate per-episode and per-show labels, producing summary reports.

**Usage:**
```bash
uv run python scripts/generate_document_labels.py
```

This script reads from classification outputs and generates Markdown reports in `outputs/document_labels/`.

---

## Utility Scripts

These scripts provide additional functionality for data management, diagnostics, and maintenance.

### `download_rss_metadata.py`

Downloads and caches RSS feed metadata for podcasts. This metadata is used by the speaker labeling pipeline to extract host/guest names.

**Usage:**
```bash
uv run python scripts/download_rss_metadata.py \
  --output-dir outputs/rss_metadata
```

### `download_missing_episodes.py`

Downloads audio files for episodes that already have transcripts but are missing the original audio. Useful for dataset reconstruction.

**Usage:**
```bash
uv run python scripts/download_missing_episodes.py \
  --transcripts-dir outputs/transcripts \
  --output-dir downloads
```

### `generate_speaker_statistics.py`

Generates speaker-level statistics across transcripts, including speaking time distribution and turn-taking patterns.

**Usage:**
```bash
uv run python scripts/generate_speaker_statistics.py \
  --transcripts-dir outputs/transcripts_with_speakers
```

### `regenerate_classification_summary.py`

Regenerates the classification summary report from existing classification outputs without re-running classification.

**Usage:**
```bash
uv run python scripts/regenerate_classification_summary.py
```

### `check_classification_coverage.py`

Checks the coverage of classification outputs, identifying episodes that have been processed vs. those still pending.

**Usage:**
```bash
uv run python scripts/check_classification_coverage.py \
  --input-dir outputs/transcripts_with_speakers \
  --output-dir outputs/classified
```

### `download_and_process.py`

Combined workflow script that downloads podcast episodes and immediately runs them through the processing pipeline.

**Usage:**
```bash
uv run python scripts/download_and_process.py "The Daily" --max-episodes 5
```

### `run_full_pipeline.py`

Alternative full pipeline runner (see `run_full_corpus_pipeline.py` for the main implementation).

---

## Script Organization

| Category | Scripts |
|----------|---------|
| **Main Pipelines** | `run_full_corpus_pipeline.py`, `run_analysis_pipeline.py` |
| **Individual Stages** | `transcribe_batch.py`, `diarize_batch.py`, `combine_*.py`, `analyze_*.py`, `classify_utterances.py` |
| **Standalone Pipelines** | `annotate_with_llm.py`, `run_speaker_labeling_pipeline.py`, `extract_features.py`, `generate_document_labels.py` |
| **Utilities** | `download_rss_metadata.py`, `download_missing_episodes.py`, `generate_speaker_statistics.py`, `check_classification_coverage.py`, `regenerate_classification_summary.py` |
