# Analysis Package

Keyword matching and context extraction for podcast transcripts with multiple export formats.

## Overview

This package searches podcast transcripts for specific keywords and extracts surrounding context. Unlike the embeddings method which uses semantic similarity, this performs exact (or partial) text matching with configurable context windows.

## Features

- **Keyword Matching**: Search for specific words/phrases in transcripts
- **Context Extraction**: Configurable window of surrounding text
- **Speaker Information**: Track which speaker said each match
- **Timestamps**: Precise timing for each match
- **Multiple Formats**: Export to JSON, CSV, or TXT
- **Summary Reports**: Aggregate statistics across shows and episodes
- **Case Sensitivity**: Configurable case-sensitive/insensitive matching
- **Partial Matching**: Find variations (e.g., "immigrant" matches "immigrants")
- **Directory Mirroring**: Maintains show structure in output

## Installation

The package is automatically installed with the main project:

```bash
uv sync
```

## Configuration

### Keyword Configuration

Keywords and settings are defined in `config/keyword_analysis_config.yaml`:

```yaml
keyword_analysis:
  keywords:
    - "immigrant"
    - "immigration"
    - "minority"
    - "minorities"
    - "ethnic"
    - "refugee"
    # ... 730+ keywords

  settings:
    case_sensitive: false          # Case-insensitive matching
    partial_matches: true          # "immigrant" matches "immigrants"
    context_window: 100            # Characters before/after match
    include_speaker_info: true     # Include speaker labels
    include_timestamps: true       # Include start/end times
```

### Configuration Structure

- **keywords**: List of keywords to search for
- **case_sensitive**: Whether matching is case-sensitive
- **partial_matches**: Match keywords as substrings
- **context_window**: Characters of context before/after match
- **include_speaker_info**: Add speaker labels to results
- **include_timestamps**: Add timing information to results

## Usage

### Via Analysis Pipeline (Recommended)

Keyword analysis is integrated into the analysis pipeline. Use `run_analysis_pipeline.py`:

```bash
# Run keyword analysis with default settings
uv run python scripts/run_analysis_pipeline.py \
  --transcripts-dir outputs/transcripts_with_speakers \
  --keywords-config config/keyword_analysis_config.yaml

# Export in different formats (json, csv, txt, or all)
uv run python scripts/run_analysis_pipeline.py \
  --transcripts-dir outputs/transcripts \
  --keywords-format all

# Run only keyword analysis (skip other stages)
uv run python scripts/run_analysis_pipeline.py \
  --transcripts-dir outputs/transcripts \
  --skip-features --skip-classification
```

### Python API

```python
from pathlib import Path
from podcast_conversations.analysis import (
    # Configuration classes
    AnalysisConfig,
    KeywordAnalysisConfig,
    KeywordAnalysisSettings,
    KeywordCategory,
    # Matching classes
    KeywordMatcher,
    KeywordMatch,
    # Transcript loading
    TranscriptReader,
)

# 1. Load configuration
config = AnalysisConfig("config/keyword_analysis_config.yaml")
keyword_config: KeywordAnalysisConfig = config.get_keyword_analysis()
settings: KeywordAnalysisSettings = keyword_config.settings

# 2. Initialize keyword matcher with settings
matcher = KeywordMatcher(
    keywords=keyword_config.get_all_keywords(),
    case_sensitive=settings.case_sensitive,
    partial_matches=settings.partial_matches,
    context_window=settings.context_window,
)

# 3. Load transcript
reader = TranscriptReader()
transcript = reader.load_transcript(
    Path("outputs/transcripts/the_daily/episode_001.json")
)

# 4. Find matches
matches: list[KeywordMatch] = []
for segment in transcript["segments"]:
    segment_matches = matcher.find_matches(
        text=segment["text"],
        speaker=segment.get("speaker"),
        start_time=segment.get("start"),
        end_time=segment.get("end"),
    )
    matches.extend(segment_matches)

# 5. Process results
print(f"Found {len(matches)} keyword matches")

for match in matches:
    print(f"Keyword: {match.keyword}")
    print(f"Matched: {match.matched_text}")
    print(f"Speaker: {match.speaker}")
    print(f"Time: {match.start_time}s - {match.end_time}s")
    print(f"Context: {match.context_before}[{match.matched_text}]{match.context_after}")
    print()

# 6. Access keyword categories (if using categorized keywords)
for category in keyword_config.categories:
    print(f"Category: {category.name}")
    print(f"Keywords: {category.keywords}")
```

## Command Line Options

```
--transcripts-dir PATH    Directory containing transcript JSON files
--config PATH             Path to keyword config YAML file (default: config/keyword_analysis_config.yaml)
--format TEXT             Output format: json, csv, txt, all (default: json)
--show-stats              Display summary statistics
--log-level TEXT          Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)
```

## Input Format

Transcript JSON files should contain segments with text:

```json
{
  "metadata": {
    "title": "Episode Title",
    "show_name": "The Ezra Klein Show"
  },
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 5.2,
      "text": "Welcome to the show. Today we discuss immigration policy.",
      "speaker": "SPEAKER_00"
    },
    {
      "id": 1,
      "start": 5.5,
      "end": 10.3,
      "text": "Thanks for having me. Immigrants contribute significantly to our economy.",
      "speaker": "SPEAKER_01"
    }
  ]
}
```

## Output Formats

### Directory Structure

```
outputs/analysis/
├── keyword_analysis_summary_05_01_2026.md              # Main summary (overview + statistics)
├── the_daily_05_01_2026_143022.md                      # Per-show details
├── the_ezra_klein_show_05_01_2026_143022.md           # Per-show details
├── the_joe_rogan_experience_05_01_2026_143022.md      # Per-show details
└── keyword_analysis/
    ├── the_ezra_klein_show/
    │   ├── episode_001_keywords.json                   # Per-episode matches
    │   ├── episode_001_keywords.csv                    # If CSV format
    │   ├── episode_001_keywords.txt                    # If TXT format
    │   ├── episode_002_keywords.json
    │   └── ...
    └── the_joe_rogan_experience/
        ├── episode_001_keywords.json
        └── ...
```

**File Purposes:**
- **Main summary** - High-level overview, top keywords, show comparison table
- **Per-show detail files** - Episode-by-episode keyword breakdown for each show
- **Per-episode JSON/CSV/TXT** - Raw match data for programmatic analysis

### JSON Format

Per-episode file (`episode_keywords.json`):

```json
{
  "transcript_file": "episode_001.json",
  "metadata": {
    "title": "Episode Title",
    "show_name": "The Ezra Klein Show"
  },
  "matches": [
    {
      "keyword": "immigrant",
      "matched_text": "immigrants",
      "context_before": "Thanks for having me. ",
      "context_after": " contribute significantly to our economy.",
      "speaker": "SPEAKER_01",
      "start_time": 5.5,
      "end_time": 10.3
    },
    {
      "keyword": "immigration",
      "matched_text": "immigration",
      "context_before": "Today we discuss ",
      "context_after": " policy.",
      "speaker": "SPEAKER_00",
      "start_time": 0.0,
      "end_time": 5.2
    }
  ],
  "match_count": 2,
  "keyword_counts": {
    "immigrant": 1,
    "immigration": 1
  }
}
```

### CSV Format

Per-episode file (`episode_keywords.csv`):

```csv
keyword,matched_text,context_before,context_after,speaker,start_time,end_time
immigrant,immigrants,Thanks for having me. , contribute significantly to our economy.,SPEAKER_01,5.5,10.3
immigration,immigration,Today we discuss , policy.,SPEAKER_00,0.0,5.2
```

### TXT Format

Per-episode file (`episode_keywords.txt`):

```
Transcript: episode_001.json
Title: Episode Title
Show: The Ezra Klein Show
Total Matches: 2
================================================================================

Keyword: immigrant
Match: immigrants
Speaker: SPEAKER_01
Time: 5.50s - 10.30s
Context: Thanks for having me. [immigrants] contribute significantly to our economy.

--------------------------------------------------------------------------------

Keyword: immigration
Match: immigration
Speaker: SPEAKER_00
Time: 0.00s - 5.20s
Context: Today we discuss [immigration] policy.

================================================================================
```

### Summary Report Format

**Main Summary** (`keyword_analysis_summary_DD_MM_YYYY.md`):

```markdown
# Keyword Analysis Summary

**Analysis Date:** January 05, 2026

## Overall Statistics

- **Total Files Analyzed:** 150
- **Files with Matches:** 142
- **Total Keyword Matches:** 3,450
- **Unique Keywords Found:** 287
- **Shows Analyzed:** 18

## Keyword Match Counts (All Files)

| Keyword | Count |
|---------|-------|
| immigrant | 234 |
| immigration | 198 |
| minority | 187 |
| refugees | 145 |
| ... | ... |

## Statistics by Show

| Show | Episodes | Total Matches |
|------|----------|---------------|
| the_joe_rogan_experience | 25 | 856 |
| the_ezra_klein_show | 18 | 623 |
| the_daily | 32 | 512 |
| ... | ... | ... |

## Detailed Results

Detailed keyword analysis for each show has been written to separate files:

- `the_daily_05_01_2026_143022.md`
- `the_ezra_klein_show_05_01_2026_143022.md`
- `the_joe_rogan_experience_05_01_2026_143022.md`
```

**Per-Show Detail Files** (`{show_name}_DD_MM_YYYY_HHMMSS.md`):

```markdown
# Keyword Analysis Details - the_ezra_klein_show

**Analysis Date:** January 05, 2026

## Show Statistics

- **Total Episodes:** 18
- **Episodes with Matches:** 17
- **Total Matches:** 623

## Episodes

### Episode 001: Immigration Policy Discussion

- **File:** `episode_001.json`
- **Total Matches:** 42
- **Unique Keywords:** 15

**Keywords Found:**

| Keyword | Count |
|---------|-------|
| immigrant | 12 |
| immigration | 8 |
| minority | 7 |
| ... | ... |

### Episode 002: Economic Impacts

- **File:** `episode_002.json`
- **Total Matches:** 35
- **Unique Keywords:** 12

...
```

**Key Improvements (v2.0):**
- ✅ **Show names** automatically extracted from directory structure if not in transcript JSON
- ✅ **Per-show detail files** keep main summary concise and manageable
- ✅ **Timestamped filenames** prevent overwrites (`{show_name}_DD_MM_YYYY_HHMMSS.md`)
- ✅ **Scalable** handles thousands of episodes efficiently
- ✅ **Better organization** one focused file per show

## How It Works

### 1. Keyword Matching

**Case-Insensitive (default):**
```python
# Keyword: "immigrant"
# Matches: "immigrant", "Immigrant", "IMMIGRANT"
```

**Case-Sensitive:**
```python
# Keyword: "immigrant"
# Only matches: "immigrant"
```

**Partial Matching (default):**
```python
# Keyword: "immigrant"
# Matches: "immigrant", "immigrants", "immigration"
```

**Exact Matching:**
```python
# Keyword: "immigrant"
# Only matches: "immigrant" (as whole word)
```

### 2. Context Extraction

```python
# Text: "The policy affects immigrants and their families significantly"
# Keyword: "immigrant"
# Context window: 20 characters

# Result:
context_before = "policy affects "
matched_text = "immigrants"
context_after = " and their familie"
```

### 3. Speaker & Timestamp

Each match includes:
- **speaker**: Which speaker said it (SPEAKER_00, SPEAKER_01, etc.)
- **start_time**: When the utterance started (seconds)
- **end_time**: When the utterance ended (seconds)

## Architecture

### Components

- **config_loader.py**: YAML configuration loading
- **keyword_matcher.py**: Keyword matching logic with context extraction
- **transcript_reader.py**: JSON transcript parsing

### Matching Algorithm

```python
# For each segment in transcript:
1. Normalize text (case-insensitive if configured)
2. For each keyword:
   a. Find all occurrences in segment
   b. Extract context before/after
   c. Record speaker and timestamps
   d. Store match with metadata
3. Aggregate statistics
```

## Use Cases

### 1. Identify Specific Topics

Track mentions of specific topics across episodes:
- Immigration, refugees, minorities
- Political terms, policy discussions
- Social issues, current events

### 2. Speaker Analysis

See which speakers use which keywords:
```python
# Group matches by speaker
speaker_usage = {}
for match in matches:
    speaker = match["speaker"]
    keyword = match["keyword"]
    speaker_usage[speaker] = speaker_usage.get(speaker, {})
    speaker_usage[speaker][keyword] = speaker_usage[speaker].get(keyword, 0) + 1
```

### 3. Temporal Analysis

Track keyword usage over time:
```python
# Group by timestamp
for match in matches:
    time_bucket = int(match["start_time"] // 300)  # 5-minute buckets
    # Aggregate by time bucket
```

### 4. Cross-Show Comparison

Compare keyword usage across different shows using summary report.

## Differences from Embeddings Method

| Feature | Keyword Analysis | Embeddings Method |
|---------|------------------|-------------------|
| **Matching** | Exact text matching | Semantic similarity |
| **Flexibility** | Only finds specified keywords | Finds conceptually similar language |
| **Precision** | High (exact matches) | Variable (threshold-based) |
| **Configuration** | List of keywords | Taxonomy with examples |
| **Speed** | Very fast | Slower (GPU-accelerated) |
| **Use Case** | Track specific terms | Detect dehumanization patterns |

**When to use Keyword Analysis:**
- Need exact keyword counts
- Tracking specific terminology
- Fast, simple searches
- Known keywords to search for

**When to use Embeddings Method:**
- Detect conceptually similar language
- Don't know all keyword variations
- Semantic understanding needed
- Finding dehumanization patterns

## Performance

### Speed

- Very fast (CPU-based)
- Processes 1000 segments in ~1 second
- No GPU required
- Scales linearly with transcript size

### Memory

- Low memory usage
- Processes files one at a time
- No model loading required

## Troubleshooting

### No Matches Found

**Issue:** Analysis completes but finds 0 matches

**Fix:**
1. Check keyword spelling in config file
2. Verify transcripts contain expected text
3. Try case-insensitive matching
4. Enable partial matches
5. Review transcript content manually

### Too Many Matches

**Issue:** Overwhelming number of matches

**Fix:**
1. Narrow keyword list
2. Use exact matching (disable partial matches)
3. Enable case-sensitive matching
4. Filter results by speaker or time

### Context Too Short/Long

**Issue:** Context window not optimal

**Fix:**
1. Adjust `context_window` in config
2. Typical values: 50-200 characters
3. Depends on average utterance length

### Configuration File Not Found

**Error:** `Config file not found`

**Fix:**
1. Ensure `config/keyword_analysis_config.yaml` exists
2. Check YAML syntax is valid
3. Verify path in `--config` argument

## Examples

### Analyze All Transcripts

```bash
# Process all shows with default settings
uv run python scripts/run_analysis_pipeline.py \
  --transcripts-dir outputs/transcripts_with_speakers \
  --keywords-config config/keyword_analysis_config.yaml \
  --skip-features --skip-classification
```

### Export in All Formats

```bash
# Generate JSON, CSV, and TXT for each episode
uv run python scripts/run_analysis_pipeline.py \
  --transcripts-dir outputs/transcripts_with_speakers \
  --keywords-format all
```

### Custom Keyword List

Edit `config/keyword_analysis_config.yaml`:

```yaml
keyword_analysis:
  keywords:
    - "climate"
    - "environment"
    - "sustainability"
    - "renewable"
    - "carbon"

  settings:
    case_sensitive: false
    partial_matches: true
    context_window: 150
    include_speaker_info: true
    include_timestamps: true
```

Then run:
```bash
uv run python scripts/run_analysis_pipeline.py \
  --transcripts-dir outputs/transcripts \
  --keywords-config config/keyword_analysis_config.yaml
```

### Programmatic Analysis

```python
from pathlib import Path
from podcast_conversations.analysis import AnalysisConfig, KeywordMatcher, TranscriptReader

# Custom configuration
keywords = ["democracy", "freedom", "liberty", "rights"]
matcher = KeywordMatcher(
    keywords=keywords,
    case_sensitive=False,
    partial_matches=True,
    context_window=100,
)

# Process single file
reader = TranscriptReader()
transcript = reader.load_transcript(Path("outputs/transcripts/show/episode.json"))

matches = []
for segment in transcript["segments"]:
    segment_matches = matcher.find_matches(
        text=segment["text"],
        speaker=segment.get("speaker"),
        start_time=segment.get("start"),
        end_time=segment.get("end"),
    )
    matches.extend(segment_matches)

# Analyze results
keyword_counts = {}
for match in matches:
    keyword = match["keyword"]
    keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1

print(f"Total matches: {len(matches)}")
for keyword, count in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{keyword}: {count}")
```

## Integration with Pipeline

### Via Analysis Pipeline

Keyword analysis is part of the analysis pipeline (`run_analysis_pipeline.py`):

```bash
# Run full analysis pipeline (features + keywords + classification)
uv run python scripts/run_analysis_pipeline.py \
  --transcripts-dir outputs/transcripts_postprocessed

# Run only keyword analysis
uv run python scripts/run_analysis_pipeline.py \
  --transcripts-dir outputs/transcripts_postprocessed \
  --skip-features --skip-classification
```

### Via Full Corpus Pipeline

Keyword analysis is also integrated into the full corpus pipeline:

```bash
# Run full corpus pipeline (transcription through classification)
uv run python scripts/run_full_corpus_pipeline.py \
  --audio-dir outputs/downloads \
  --keywords-config config/keyword_analysis_config.yaml
```

## Credits

Built with:
- Python standard library (re for regex matching)
- [PyYAML](https://pyyaml.org/) - YAML configuration
- [Rich](https://rich.readthedocs.io/) - Terminal UI
