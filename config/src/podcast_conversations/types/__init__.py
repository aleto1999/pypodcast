"""Centralized type definitions for the podcast-conversations project.

This package provides Pydantic models and type definitions for:
- Transcript data structures (segments, words, metadata)
- LLM annotation results
- Feature extraction statistics
- Pipeline task management
- Analysis results

All types support runtime validation at system boundaries (JSON loading,
API responses, configuration parsing) to catch data issues early.

Usage:
    from podcast_conversations.types import (
        TranscriptSegment,
        TranscriptData,
        AnnotationResult,
        QuestionStats,
    )
"""

from podcast_conversations.types.transcript import (
    Word,
    TranscriptSegment,
    TranscriptMetadata,
    TranscriptData,
    parse_transcript_file,
)
from podcast_conversations.types.annotations import (
    HateSpeechType,
    TargetGroup,
    AnnotationResult,
    AnnotatedSegment,
    AnnotatedTranscript,
)
from podcast_conversations.types.features import (
    QuestionStats,
    TurnTakingStats,
    VocabularyStats,
    PolitenessStats,
    TranscriptFeatures,
)
from podcast_conversations.types.pipeline import (
    FileTask,
    SegmentBatch,
    ProcessingResult,
    BatchResult,
)
from podcast_conversations.types.analysis import (
    KeywordMatch,
    KeywordCategory,
    KeywordAnalysisResult,
    CategoryStats,
)

__all__ = [
    # transcript types.
    "Word",
    "TranscriptSegment",
    "TranscriptMetadata",
    "TranscriptData",
    "parse_transcript_file",
    # annotation types.
    "HateSpeechType",
    "TargetGroup",
    "AnnotationResult",
    "AnnotatedSegment",
    "AnnotatedTranscript",
    # feature types.
    "QuestionStats",
    "TurnTakingStats",
    "VocabularyStats",
    "PolitenessStats",
    "TranscriptFeatures",
    # pipeline types.
    "FileTask",
    "SegmentBatch",
    "ProcessingResult",
    "BatchResult",
    # analysis types.
    "KeywordMatch",
    "KeywordCategory",
    "KeywordAnalysisResult",
    "CategoryStats",
]
