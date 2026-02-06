# Feature Extraction Package

Extract conversational features from podcast transcripts for quantitative analysis of dialogue patterns, engagement, and communication styles.

## Overview

This package provides comprehensive conversation analysis through multiple feature extractors that analyze linguistic patterns, turn-taking dynamics, vocabulary diversity, and politeness strategies in podcast conversations. It produces both hierarchical JSON and flat CSV outputs suitable for statistical analysis and machine learning.

## Features

- **Question Analysis**: Detect questions, direct address patterns, and interrogative frequency
- **Turn-Taking Analysis**: Measure conversation flow, dominance, and speaking time distribution
- **Vocabulary Analysis**: Calculate lexical diversity using TTR and RTTR metrics
- **Politeness Analysis**: Detect politeness strategies (convokit-based or rule-based)
- **Dual Output Formats**: Nested JSON for detailed inspection, flat CSV for statistical analysis
- **Speaker-Level Metrics**: Per-speaker breakdowns of all features
- **Flexible Architecture**: Modular analyzers can be used independently or combined

## Installation

The package is automatically installed with the main project:

```bash
uv sync
```

## Usage

### Command Line

Extract features from transcripts using the provided script:

```bash
# Process all transcripts in a directory
uv run python scripts/extract_features.py \
  --transcripts-dir outputs/transcripts_with_speakers \
  --output-dir outputs/features

# Export to CSV for statistical analysis
uv run python scripts/extract_features.py \
  --transcripts-dir outputs/transcripts_with_speakers \
  --output-dir outputs/features \
  --format csv

# Use convokit for advanced politeness analysis
uv run python scripts/extract_features.py \
  --transcripts-dir outputs/transcripts_with_speakers \
  --output-dir outputs/features \
  --use-convokit
```

### Python API

```python
from pathlib import Path
from podcast_conversations.feature_extraction import (
    # Main extractor
    FeatureExtractor,
    extract_features_from_transcript,
    # Individual analyzers (for custom pipelines)
    QuestionAnalyzer,
    TurnTakingAnalyzer,
    VocabularyAnalyzer,
    PolitenessAnalyzer,
)
import json

# Extract features from a single file
features = extract_features_from_transcript(
    file_path="outputs/transcripts_with_speakers/show/episode.json",
    use_convokit=False,
)

print(f"Questions: {features.questions.total_questions}")
print(f"Turn-taking: {features.turn_taking.total_turns}")
print(f"Vocabulary: {features.vocabulary.unique_words} unique words")

# Or use the full extractor with custom settings
extractor = FeatureExtractor(use_convokit=True)

with open("outputs/transcripts/episode.json") as f:
    transcript_data = json.load(f)

features = extractor.extract(transcript_data, file_path="episode.json")

# Convert to dictionary
features_dict = extractor.to_dict(features)

# Or convert to flat CSV format
flat_dict = extractor.to_flat_dict(features)

# Use individual analyzers for custom analysis
question_analyzer = QuestionAnalyzer()
turn_analyzer = TurnTakingAnalyzer()
vocab_analyzer = VocabularyAnalyzer()
politeness_analyzer = PolitenessAnalyzer(use_convokit=False)

# Analyze specific aspects
question_results = question_analyzer.analyze(transcript_data["segments"])
turn_results = turn_analyzer.analyze(transcript_data["segments"])
vocab_results = vocab_analyzer.analyze(transcript_data["segments"])
politeness_results = politeness_analyzer.analyze(transcript_data["segments"])
```

## Feature Categories

### 1. Question Analysis

Detects and classifies questions in conversation:

**Metrics:**
- `total_sentences`: Total number of sentences detected
- `total_questions`: Number of questions found
- `question_ratio`: Proportion of sentences that are questions
- `questions_per_minute`: Question frequency normalized by duration
- `direct_address_count`: Questions with direct address (you, your, imperatives)
- `direct_address_ratio`: Proportion of questions with direct address
- `questions_by_speaker`: Per-speaker question counts

**Detection Methods:**
- Question marks (`?`)
- Interrogative sentence starters (who, what, when, where, why, how)
- Auxiliary verb questions (is, are, do, does, can, could, will, would)
- Direct address patterns (second-person pronouns, imperatives)

**Example:**
```python
features.questions.total_questions  # 45
features.questions.question_ratio   # 0.23 (23% of sentences are questions)
features.questions.questions_per_minute  # 0.75 (0.75 questions per minute)
features.questions.questions_by_speaker  # {"SPEAKER_00": 30, "SPEAKER_01": 15}
```

### 2. Turn-Taking Analysis

Analyzes conversation flow and speaker dynamics:

**Metrics:**
- `total_turns`: Total number of speaker changes
- `turns_by_speaker`: Turn count per speaker
- `speaking_time_by_speaker`: Total seconds each speaker talked
- `average_turn_time_by_speaker`: Average turn duration per speaker
- `average_switch_time`: Average time between turn changes
- `dominance_index`: Measure of speaking time inequality (0-1)
- `total_duration_seconds`: Total conversation duration

**Calculations:**
- Dominance Index: Standard deviation of speaking time proportions
- Turn Changes: Count of speaker transitions
- Speaking Time: Sum of segment durations per speaker

**Example:**
```python
features.turn_taking.total_turns  # 120
features.turn_taking.dominance_index  # 0.35 (moderate dominance)
features.turn_taking.speaking_time_by_speaker  # {"SPEAKER_00": 1800, "SPEAKER_01": 1200}
features.turn_taking.average_turn_time_by_speaker  # {"SPEAKER_00": 15.0, "SPEAKER_01": 10.0}
```

### 3. Vocabulary Analysis

Measures lexical diversity and vocabulary richness:

**Metrics:**
- `total_words`: Total word count
- `total_words_no_stopwords`: Content words only
- `unique_words`: Unique word count
- `unique_words_no_stopwords`: Unique content words
- `type_token_ratio`: TTR = unique_words / total_words
- `root_type_token_ratio`: RTTR = unique_words / sqrt(total_words)
- `words_by_speaker`: Word count per speaker
- `unique_words_by_speaker`: Unique word count per speaker
- `ttr_by_speaker`: TTR per speaker

**Metrics Explained:**
- **TTR (Type-Token Ratio)**: Classic lexical diversity measure (0-1)
  - Higher = more diverse vocabulary
  - Sensitive to text length
- **RTTR (Root Type-Token Ratio)**: Length-normalized diversity
  - More stable across different text lengths
  - Better for comparing conversations of different durations

**Example:**
```python
features.vocabulary.total_words  # 5000
features.vocabulary.unique_words  # 1200
features.vocabulary.type_token_ratio  # 0.24 (24% of words are unique)
features.vocabulary.root_type_token_ratio  # 16.97
features.vocabulary.ttr_by_speaker  # {"SPEAKER_00": 0.28, "SPEAKER_01": 0.22}
```

### 4. Politeness Analysis

Detects politeness strategies and markers:

**Two Modes:**

#### Rule-Based Mode (Default)
Fast, simple politeness marker detection:
- Please/thank you counts
- Modal verbs (could, would, might)
- Hedge words (maybe, perhaps, possibly)
- Formal language markers

**Metrics:**
- `please_count`: "Please" occurrences
- `thank_count`: "Thank you" variations
- `politeness_score`: Simple aggregate score

#### Convokit Mode (Advanced)
Uses [ConvoKit](https://convokit.cornell.edu/) for sophisticated analysis:
- Request strategies
- Deference markers
- Gratitude expressions
- Hedge detection
- Positive/negative politeness

**Metrics:**
- `feature_politeness_*`: Multiple politeness strategy scores
- `strategy_counts`: Detected strategy frequencies
- `overall_politeness_score`: Aggregate politeness measure

**Example:**
```python
# Rule-based mode
features.politeness.please_count  # 12
features.politeness.thank_count  # 8
features.politeness.politeness_score  # 0.42

# Convokit mode
features.politeness.strategy_counts  # {"Gratitude": 8, "Deference": 15}
features.politeness.overall_politeness_score  # 0.68
```

## Input Format

Transcript JSON files should contain segments with text and optional speaker labels:

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
      "text": "Welcome to the show. What brings you here today?",
      "speaker": "SPEAKER_00"
    },
    {
      "id": 1,
      "start": 5.5,
      "end": 10.3,
      "text": "Thank you for having me. I wanted to discuss...",
      "speaker": "SPEAKER_01"
    }
  ]
}
```

## Output Formats

### JSON Format (Hierarchical)

Detailed nested structure preserving all feature categories:

```json
{
  "file_path": "outputs/transcripts/episode.json",
  "file_name": "episode",
  "metadata": {
    "title": "Episode Title",
    "show_name": "Show Name"
  },
  "questions": {
    "total_sentences": 195,
    "total_questions": 45,
    "question_ratio": 0.231,
    "questions_per_minute": 0.75,
    "direct_address_count": 28,
    "direct_address_ratio": 0.622,
    "questions_by_speaker": {
      "SPEAKER_00": 30,
      "SPEAKER_01": 15
    }
  },
  "turn_taking": {
    "total_turns": 120,
    "turns_by_speaker": {
      "SPEAKER_00": 65,
      "SPEAKER_01": 55
    },
    "speaking_time_by_speaker": {
      "SPEAKER_00": 1800.5,
      "SPEAKER_01": 1199.5
    },
    "average_turn_time_by_speaker": {
      "SPEAKER_00": 27.7,
      "SPEAKER_01": 21.8
    },
    "average_switch_time": 25.0,
    "dominance_index": 0.35,
    "total_duration_seconds": 3000.0
  },
  "vocabulary": {
    "total_words": 5000,
    "total_words_no_stopwords": 2800,
    "unique_words": 1200,
    "unique_words_no_stopwords": 950,
    "type_token_ratio": 0.24,
    "root_type_token_ratio": 16.97,
    "words_by_speaker": {
      "SPEAKER_00": 2800,
      "SPEAKER_01": 2200
    },
    "unique_words_by_speaker": {
      "SPEAKER_00": 780,
      "SPEAKER_01": 620
    },
    "ttr_by_speaker": {
      "SPEAKER_00": 0.279,
      "SPEAKER_01": 0.282
    }
  },
  "politeness": {
    "please_count": 12,
    "thank_count": 8,
    "politeness_score": 0.42
  }
}
```

### CSV Format (Flat)

Single-row format suitable for statistical analysis and machine learning:

```csv
file_path,file_name,total_sentences,total_questions,question_ratio,questions_per_minute,direct_address_count,direct_address_ratio,questions_SPEAKER_00,questions_SPEAKER_01,total_turns,turns_SPEAKER_00,turns_SPEAKER_01,speaking_time_SPEAKER_00,speaking_time_SPEAKER_01,avg_turn_time_SPEAKER_00,avg_turn_time_SPEAKER_01,average_switch_time,dominance_index,total_duration_seconds,total_words,total_words_no_stopwords,unique_words,unique_words_no_stopwords,type_token_ratio,root_type_token_ratio,words_SPEAKER_00,words_SPEAKER_01,unique_words_SPEAKER_00,unique_words_SPEAKER_01,ttr_SPEAKER_00,ttr_SPEAKER_01,please_count,thank_count,politeness_score
outputs/transcripts/episode.json,episode,195,45,0.231,0.75,28,0.622,30,15,120,65,55,1800.5,1199.5,27.7,21.8,25.0,0.35,3000.0,5000,2800,1200,950,0.24,16.97,2800,2200,780,620,0.279,0.282,12,8,0.42
```

**Benefits of CSV Format:**
- Easy import into pandas, R, SPSS, Excel
- Ready for statistical tests and correlation analysis
- Compatible with machine learning libraries
- Compact for large-scale analysis

## Architecture

### Components

- **extractor.py**: Main orchestrator coordinating all feature analyzers
- **questions.py**: Question detection and direct address analysis
- **turn_taking.py**: Conversation flow and dominance metrics
- **vocabulary.py**: Lexical diversity and vocabulary richness
- **politeness.py**: Politeness marker and strategy detection

### Data Flow

```
Transcript JSON
      ↓
FeatureExtractor.extract()
      ↓
┌─────────────────────────────────────────┐
│  Parallel Feature Extraction:           │
│  - QuestionAnalyzer.analyze()           │
│  - TurnTakingAnalyzer.analyze()         │
│  - VocabularyAnalyzer.analyze()         │
│  - PolitenessAnalyzer.analyze()         │
└──────────────────┬──────────────────────┘
                   ↓
         TranscriptFeatures
                   ↓
         ┌────────┴─────────┐
         ↓                  ↓
    to_dict()         to_flat_dict()
         ↓                  ↓
    JSON Output      CSV Output
```

## Use Cases

### 1. Comparative Show Analysis

Compare conversational dynamics across different podcast shows:

```python
import pandas as pd
from pathlib import Path
from podcast_conversations.feature_extraction import extract_features_from_transcript, features_to_csv_row

# Extract features from multiple shows
rows = []
for transcript_path in Path("outputs/transcripts").rglob("*.json"):
    features = extract_features_from_transcript(transcript_path)
    rows.append(features_to_csv_row(features))

# Create dataframe for analysis
df = pd.DataFrame(rows)

# Compare shows
show_comparison = df.groupby('show_name').agg({
    'question_ratio': 'mean',
    'dominance_index': 'mean',
    'type_token_ratio': 'mean',
    'politeness_score': 'mean',
})

print(show_comparison)
```

### 2. Host vs Guest Analysis

Analyze differences between hosts and guests:

```python
# Assuming SPEAKER_00 is host, SPEAKER_01 is guest
df['host_questions'] = df['questions_SPEAKER_00']
df['guest_questions'] = df['questions_SPEAKER_01']
df['host_speaking_time'] = df['speaking_time_SPEAKER_00']
df['guest_speaking_time'] = df['speaking_time_SPEAKER_01']

# Compare host vs guest patterns
print(f"Average host questions: {df['host_questions'].mean()}")
print(f"Average guest questions: {df['guest_questions'].mean()}")
print(f"Host speaking time proportion: {(df['host_speaking_time'] / df['total_duration_seconds']).mean()}")
```

### 3. Engagement Metrics

Identify highly engaging episodes:

```python
# High engagement indicators
df['engagement_score'] = (
    df['question_ratio'] * 0.3 +
    (1 - df['dominance_index']) * 0.3 +  # Lower dominance = more balanced
    df['type_token_ratio'] * 0.4
)

top_episodes = df.nlargest(10, 'engagement_score')[['file_name', 'engagement_score']]
print(top_episodes)
```

### 4. Communication Style Clustering

Group episodes by conversational style:

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Select features for clustering
features_for_clustering = [
    'question_ratio', 'dominance_index', 'type_token_ratio',
    'questions_per_minute', 'average_switch_time', 'politeness_score'
]

X = df[features_for_clustering].fillna(0)
X_scaled = StandardScaler().fit_transform(X)

# Cluster into 4 communication styles
kmeans = KMeans(n_clusters=4, random_state=42)
df['style_cluster'] = kmeans.fit_predict(X_scaled)

# Analyze clusters
for cluster in range(4):
    cluster_df = df[df['style_cluster'] == cluster]
    print(f"\nCluster {cluster} ({len(cluster_df)} episodes):")
    print(cluster_df[features_for_clustering].mean())
```

## Performance

- **Speed**: Fast CPU-based processing
- **Memory**: Low memory footprint (~50MB per transcript)
- **Throughput**: ~100-200 transcripts per minute on standard hardware
- **Dependencies**: Minimal (no GPU required)

Optional spaCy integration for better sentence splitting (recommended):
```bash
uv pip install spacy
python -m spacy download en_core_web_sm
```

## Troubleshooting

### No Sentences Detected

**Issue:** `total_sentences = 0`

**Fix:**
1. Verify transcript has "segments" with "text" fields
2. Check text contains valid sentences
3. Install spaCy for better sentence detection:
   ```bash
   uv pip install spacy
   python -m spacy download en_core_web_sm
   ```

### Low Question Detection

**Issue:** Questions not being detected

**Fix:**
1. Verify questions have question marks or interrogative patterns
2. Check for non-standard punctuation
3. Review `questions.py` regex patterns for your content

### Inaccurate Turn-Taking Metrics

**Issue:** Turn counts seem wrong

**Fix:**
1. Ensure segments have "speaker" field
2. Verify speaker labels are consistent
3. Check timestamp accuracy in "start" and "end" fields

### Convokit Installation Issues

**Issue:** Convokit politeness analysis fails

**Fix:**
1. Install convokit separately:
   ```bash
   uv pip install convokit
   ```
2. Or use rule-based mode (default, no convokit required)

## Integration with Pipeline

Feature extraction can be run independently on any transcripts with speaker labels:

```bash
# After diarization and combining transcripts with speakers:
uv run python scripts/extract_features.py \
  --transcripts-dir outputs/transcripts_with_speakers \
  --output-dir outputs/features

# Or on postprocessed transcripts (consecutive speakers combined):
uv run python scripts/extract_features.py \
  --transcripts-dir outputs/transcripts_postprocessed \
  --output-dir outputs/features
```

**Note:** Feature extraction is not part of the main `run_full_pipeline.py` workflow but can be run independently at any time on transcripts that have speaker labels.

## Credits

Built with:
- Python standard library (re, dataclasses)
- Optional: [spaCy](https://spacy.io/) - Sentence segmentation
- Optional: [ConvoKit](https://convokit.cornell.edu/) - Politeness strategies

## Further Reading

- [Feature Extraction Pipeline](../../../README.md#feature-extraction) - Main README documentation
- Main README.md - Overall project structure
