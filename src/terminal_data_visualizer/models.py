"""Type-safe data models for podcast analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Segment:
    """Transcript segment with speaker and timing information."""

    text: str
    start: float
    end: float
    speaker: str
    confidence: float | None = None
    words: list[dict[str, Any]] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Segment:
        """Create segment from dictionary."""
        return cls(
            text=data.get("text", ""),
            start=data.get("start", 0.0),
            end=data.get("end", 0.0),
            speaker=data.get("speaker", "UNKNOWN"),
            confidence=data.get("confidence"),
            words=data.get("words"),
        )

    @property
    def duration(self) -> float:
        """Segment duration in seconds."""
        return self.end - self.start

    @property
    def word_count(self) -> int:
        """Number of words in segment."""
        return len(self.text.split())


@dataclass
class Match:
    """Keyword match with context and metadata."""

    keyword: str
    category: str
    speaker: str | None
    start_time: float | None
    end_time: float | None
    context: str | None = None
    confidence_tier: str | None = None
    episode_id: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Match:
        """Create match from dictionary."""
        return cls(
            keyword=data.get("keyword", ""),
            category=data.get("category", ""),
            speaker=data.get("speaker"),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            context=data.get("context"),
            confidence_tier=data.get("confidence_tier"),
            episode_id=data.get("episode_id"),
        )


@dataclass
class Classification:
    """ML model classification result."""

    model_name: str
    label: str
    confidence: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Classification:
        """Create classification from dictionary."""
        return cls(
            model_name=data.get("model_name", ""),
            label=data.get("label", ""),
            confidence=data.get("confidence"),
        )


@dataclass
class LLMAnnotation:
    """LLM annotation with hate speech analysis."""

    has_hate_speech: bool
    has_advertisement: bool
    target_group: str | None
    hate_speech_type: str | None
    main_topic: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LLMAnnotation:
        """Create LLM annotation from dictionary."""
        return cls(
            has_hate_speech=data.get("has_hate_speech", False),
            has_advertisement=data.get("has_advertisement", False),
            target_group=data.get("target_group"),
            hate_speech_type=data.get("hate_speech_type"),
            main_topic=data.get("main_topic", ""),
        )


@dataclass
class Annotation:
    """LLM annotation with classifications (legacy)."""

    segment_id: int | str
    hate_speech_detected: bool
    ad_content_detected: bool
    target_groups: list[str]
    classifications: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Annotation:
        """Create annotation from dictionary."""
        return cls(
            segment_id=data.get("segment_id", 0),
            hate_speech_detected=data.get("hate_speech_detected", False),
            ad_content_detected=data.get("ad_content_detected", False),
            target_groups=data.get("target_groups", []),
            classifications=data.get("classifications", {}),
        )


@dataclass
class SpeakerStats:
    """Statistics for a speaker."""

    speaker: str
    segment_count: int
    total_words: int
    total_duration: float
    avg_segment_length: float


@dataclass
class EpisodeStats:
    """Statistics for an episode."""

    filename: str
    total_segments: int
    total_duration: float
    speaker_count: int
    total_words: int
    speakers: dict[str, SpeakerStats]


@dataclass
class CategoryStats:
    """Statistics for a keyword category."""

    category: str
    total_matches: int
    files_with_matches: int
    unique_keywords: int
