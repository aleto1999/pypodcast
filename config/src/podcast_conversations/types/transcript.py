"""Transcript data types with runtime validation.

This module defines the core data structures for podcast transcripts:
- Word: Individual word with timing information
- TranscriptSegment: A spoken segment with text, timing, and speaker
- TranscriptMetadata: Metadata about the transcript source
- TranscriptData: Complete transcript with segments and metadata

All types use Pydantic for runtime validation when loading JSON files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Word(BaseModel):
    """A single word with timing information.

    Attributes:
        word: The transcribed word text.
        start: Start time in seconds from audio beginning.
        end: End time in seconds from audio beginning.
        score: Confidence score from transcription model (0.0 to 1.0).
    """

    word: str = Field(description="The transcribed word text")
    start: float = Field(ge=0, description="Start time in seconds")
    end: float = Field(ge=0, description="End time in seconds")
    score: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Confidence score from transcription (0.0 to 1.0)",
    )

    @field_validator("end")
    @classmethod
    def end_after_start(cls, v: float, info) -> float:
        """Validate that end time is not before start time."""
        start = info.data.get("start", 0)
        if v < start:
            raise ValueError(f"end time ({v}) cannot be before start time ({start})")
        return v


class TranscriptSegment(BaseModel):
    """A segment of transcribed speech from a single speaker.

    Represents a continuous utterance from one speaker with timing
    information and optional word-level detail.

    Attributes:
        text: The transcribed text for this segment.
        start: Start time in seconds from audio beginning.
        end: End time in seconds from audio beginning.
        speaker: Speaker identifier (e.g., "SPEAKER_00", "SPEAKER_01").
        words: Optional list of individual words with timing.
    """

    text: str = Field(description="The transcribed text content")
    start: float = Field(ge=0, description="Start time in seconds")
    end: float = Field(ge=0, description="End time in seconds")
    speaker: str = Field(
        default="UNKNOWN",
        description="Speaker identifier (e.g., SPEAKER_00)",
    )
    words: list[Word] = Field(
        default_factory=list,
        description="Word-level timing information",
    )

    @field_validator("end")
    @classmethod
    def end_after_start(cls, v: float, info) -> float:
        """Validate that end time is not before start time."""
        start = info.data.get("start", 0)
        if v < start:
            raise ValueError(f"end time ({v}) cannot be before start time ({start})")
        return v

    @property
    def duration(self) -> float:
        """Duration of this segment in seconds."""
        return self.end - self.start

    @property
    def word_count(self) -> int:
        """Number of words in this segment."""
        return len(self.text.split())


class TranscriptMetadata(BaseModel):
    """Metadata about a transcript and its source.

    Attributes:
        audio_file: Path to the source audio file.
        language: Detected or specified language code.
        duration: Total audio duration in seconds.
        model: Transcription model used (e.g., "large-v3").
        diarization_file: Path to RTTM diarization file if applied.
        diarization_applied: Whether speaker diarization was applied.
    """

    audio_file: str | None = Field(
        default=None,
        description="Path to source audio file",
    )
    language: str = Field(
        default="en",
        description="Language code (e.g., 'en', 'es')",
    )
    duration: float | None = Field(
        default=None,
        ge=0,
        description="Total audio duration in seconds",
    )
    model: str | None = Field(
        default=None,
        description="Transcription model used",
    )
    diarization_file: str | None = Field(
        default=None,
        description="Path to RTTM diarization file",
    )
    diarization_applied: bool = Field(
        default=False,
        description="Whether speaker diarization was applied",
    )

    class Config:
        """Allow extra fields for forward compatibility."""

        extra = "allow"


class TranscriptData(BaseModel):
    """Complete transcript data structure.

    This is the primary type for loading and validating transcript JSON files.
    Contains all segments plus optional metadata and formatted text.

    Attributes:
        segments: List of transcript segments with speaker labels.
        text: Full transcript text without speaker labels.
        text_with_speakers: Formatted text with speaker labels.
        metadata: Additional metadata about the transcript.

    Example:
        >>> data = TranscriptData.model_validate(json.load(f))
        >>> for segment in data.segments:
        ...     print(f"[{segment.speaker}]: {segment.text}")
    """

    segments: list[TranscriptSegment] = Field(
        default_factory=list,
        description="List of transcript segments",
    )
    text: str | None = Field(
        default=None,
        description="Full transcript text without speaker labels",
    )
    text_with_speakers: str | None = Field(
        default=None,
        description="Formatted text with speaker labels",
    )
    metadata: TranscriptMetadata = Field(
        default_factory=TranscriptMetadata,
        description="Transcript metadata",
    )

    class Config:
        """Allow extra fields for forward compatibility."""

        extra = "allow"

    @property
    def total_duration(self) -> float:
        """Total duration of all segments in seconds."""
        if not self.segments:
            return 0.0
        return max(s.end for s in self.segments)

    @property
    def speaker_count(self) -> int:
        """Number of unique speakers in the transcript."""
        speakers = {s.speaker for s in self.segments}
        speakers.discard("UNKNOWN")
        return len(speakers)

    @property
    def segment_count(self) -> int:
        """Total number of segments."""
        return len(self.segments)

    def get_speaker_segments(self, speaker: str) -> list[TranscriptSegment]:
        """Get all segments for a specific speaker.

        Args:
            speaker: Speaker identifier to filter by.

        Returns:
            List of segments from the specified speaker.
        """
        return [s for s in self.segments if s.speaker == speaker]

    def get_speakers(self) -> list[str]:
        """Get list of unique speakers in order of appearance.

        Returns:
            List of speaker identifiers in order of first appearance.
        """
        seen = set()
        speakers = []
        for segment in self.segments:
            if segment.speaker not in seen:
                seen.add(segment.speaker)
                speakers.append(segment.speaker)
        return speakers


def parse_transcript_file(
    file_path: Path | str,
    strict: bool = False,
) -> TranscriptData:
    """Load and validate a transcript JSON file.

    This is the primary entry point for loading transcript files with
    runtime validation. Use this instead of raw json.load() to catch
    data issues early.

    Args:
        file_path: Path to the transcript JSON file.
        strict: If True, raise on validation errors. If False, attempt
                to load with partial data.

    Returns:
        Validated TranscriptData instance.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        json.JSONDecodeError: If the file isn't valid JSON.
        pydantic.ValidationError: If strict=True and validation fails.

    Example:
        >>> transcript = parse_transcript_file("outputs/transcripts/show/ep1.json")
        >>> print(f"Found {transcript.segment_count} segments")
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Transcript file not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw_data = json.load(f)

    if strict:
        return TranscriptData.model_validate(raw_data)

    # lenient mode: try to parse what we can.
    try:
        return TranscriptData.model_validate(raw_data)
    except Exception:
        # fallback: create with raw segments if validation fails.
        segments = []
        for seg in raw_data.get("segments", []):
            try:
                segments.append(TranscriptSegment.model_validate(seg))
            except Exception:
                # skip invalid segments.
                pass

        return TranscriptData(
            segments=segments,
            text=raw_data.get("text"),
            text_with_speakers=raw_data.get("text_with_speakers"),
        )


def transcript_to_dict(transcript: TranscriptData) -> dict[str, Any]:
    """Convert a TranscriptData instance to a dictionary for JSON serialization.

    Args:
        transcript: The transcript data to convert.

    Returns:
        Dictionary representation suitable for json.dump().
    """
    return transcript.model_dump(exclude_none=True)
