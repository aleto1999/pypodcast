# Types Package

Centralized type definitions for the podcast-conversations project using Pydantic for runtime validation.

## Overview

This package provides validated data types for all major components:

| Module | Description | Key Types |
|--------|-------------|-----------|
| `transcript.py` | Transcript data structures | `TranscriptData`, `TranscriptSegment`, `Word` |
| `annotations.py` | LLM annotation results | `AnnotationResult`, `AnnotatedTranscript` |
| `features.py` | Feature extraction stats | `TranscriptFeatures`, `QuestionStats`, `VocabularyStats` |
| `pipeline.py` | Processing task types | `FileTask`, `BatchResult`, `ProcessingResult` |
| `analysis.py` | Keyword analysis types | `KeywordMatch`, `KeywordAnalysisResult` |

## Usage

### Basic Import

```python
from podcast_conversations.types import (
    TranscriptData,
    TranscriptSegment,
    parse_transcript_file,
)
```

### Loading Transcript Files with Validation

```python
from podcast_conversations.types import parse_transcript_file

# load and validate transcript JSON.
transcript = parse_transcript_file("outputs/transcripts/show/episode.json")

# access validated data.
print(f"Segments: {transcript.segment_count}")
print(f"Speakers: {transcript.speaker_count}")
print(f"Duration: {transcript.total_duration:.1f}s")

for segment in transcript.segments:
    print(f"[{segment.speaker}]: {segment.text}")
```

### Validating JSON Data

```python
import json
from podcast_conversations.types import TranscriptData

with open("transcript.json") as f:
    raw_data = json.load(f)

# validate with Pydantic.
transcript = TranscriptData.model_validate(raw_data)
```

### Working with Annotations

```python
from podcast_conversations.types import (
    AnnotationResult,
    HateSpeechType,
    TargetGroup,
)

# create annotation result.
result = AnnotationResult(
    has_hate_speech=True,
    hate_speech_type=HateSpeechType.DEROGATORY,
    target_group=TargetGroup.RACIAL,
    main_topic="discussion about immigration",
)

# serialize to dict.
data = result.model_dump()
```

### Feature Extraction Types

```python
from podcast_conversations.types import (
    TranscriptFeatures,
    QuestionStats,
    TurnTakingStats,
)

# create features with validation.
features = TranscriptFeatures(
    file_name="episode.json",
    file_path="/path/to/episode.json",
    questions=QuestionStats(
        total_questions=25,
        wh_questions=15,
        yes_no_questions=10,
    ),
    turn_taking=TurnTakingStats(
        total_turns=150,
        unique_speakers=2,
        turn_switch_count=148,
    ),
)

# export to flat dict for CSV.
flat_data = features.to_flat_dict()
```

### Pipeline Task Management

```python
from pathlib import Path
from podcast_conversations.types import FileTask, BatchResult, TaskStatus

# create task from paths.
task = FileTask.from_paths(
    input_path=Path("inputs/audio.mp3"),
    output_path=Path("outputs/transcript.json"),
    base_dir=Path("inputs"),
)

# update status.
task.mark_completed()
# or
task.mark_failed("Transcription failed: out of memory")

# aggregate results.
batch = BatchResult(total_items=100)
batch.add_result(result)
print(f"Success rate: {batch.success_rate:.1%}")
```

## Design Principles

### 1. Runtime Validation at System Boundaries

All types use Pydantic for validation when loading external data:

```python
# validates structure, types, and constraints.
transcript = parse_transcript_file("file.json")

# raises ValidationError if data is invalid.
result = AnnotationResult.model_validate(json_data)
```

### 2. Computed Properties for Derived Values

Types include computed properties for common calculations:

```python
segment.duration        # end - start
segment.word_count      # len(text.split())
turn_taking.balance_score  # conversation balance metric
vocabulary.hapax_ratio  # proportion of unique words
```

### 3. Type Safety with Enums

Enumerated values prevent invalid states:

```python
class HateSpeechType(str, Enum):
    NONE = "none"
    DEROGATORY = "derogatory"
    DEHUMANIZING = "dehumanizing"
    # ...

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    # ...
```

### 4. Forward Compatibility

Models use `extra = "allow"` to accept unknown fields:

```python
class TranscriptMetadata(BaseModel):
    # ...
    class Config:
        extra = "allow"  # new fields won't break parsing.
```

### 5. Serialization Support

All types serialize cleanly to JSON:

```python
# to dict (excluding None values).
data = transcript.model_dump(exclude_none=True)

# to JSON string.
json_str = transcript.model_dump_json(indent=2)
```

## Validation Constraints

Types include constraints for data integrity:

| Field | Constraint | Example |
|-------|------------|---------|
| `start`, `end` | `>= 0` | Timestamps must be non-negative |
| `score`, `confidence` | `0.0 - 1.0` | Scores must be in valid range |
| `type_token_ratio` | `0.0 - 1.0` | Ratio must be valid proportion |
| `end` vs `start` | `end >= start` | End time must follow start |

## Error Handling

### Strict Mode

```python
# raises ValidationError on any issue.
transcript = parse_transcript_file("file.json", strict=True)
```

### Lenient Mode (Default)

```python
# attempts to parse valid portions.
transcript = parse_transcript_file("file.json", strict=False)
```

### Catching Validation Errors

```python
from pydantic import ValidationError

try:
    result = AnnotationResult.model_validate(data)
except ValidationError as e:
    print(f"Validation failed: {e}")
    for error in e.errors():
        print(f"  - {error['loc']}: {error['msg']}")
```

## Requirements

- Python 3.11+
- pydantic

## Module Reference

### transcript.py

- `Word` - Single word with timing
- `TranscriptSegment` - Spoken segment with speaker
- `TranscriptMetadata` - Source file metadata
- `TranscriptData` - Complete transcript
- `parse_transcript_file()` - Load and validate JSON file

### annotations.py

- `HateSpeechType` - Hate speech classification enum
- `TargetGroup` - Target group enum
- `AnnotationResult` - Single segment annotation
- `AnnotatedSegment` - Segment with annotation
- `AnnotationStats` - Aggregate statistics
- `AnnotatedTranscript` - Complete annotated transcript

### features.py

- `QuestionStats` - Question detection stats
- `TurnTakingStats` - Speaker turn patterns
- `VocabularyStats` - Lexical diversity metrics
- `PolitenessStats` - Politeness markers
- `TranscriptFeatures` - Combined features

### pipeline.py

- `TaskStatus` - Processing status enum
- `FileTask` - Single file task
- `SegmentBatch` - Batch for parallel processing
- `ProcessingResult` - Single item result
- `BatchResult` - Aggregated batch results

### analysis.py

- `ConfidenceTier` - Keyword confidence enum
- `KeywordMatch` - Single keyword occurrence
- `KeywordCategory` - Category definition
- `CategoryStats` - Category statistics
- `KeywordAnalysisResult` - Complete analysis
- `SearchResult` - Multi-file search result
