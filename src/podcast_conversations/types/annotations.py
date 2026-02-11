"""LLM annotation types with runtime validation.

This module defines types for LLM-based content annotation:
- HateSpeechType: Classification of hate speech categories
- TargetGroup: Groups that may be targeted by harmful content
- AnnotationResult: Result of annotating a single segment
- AnnotatedSegment: Segment with its annotation results
- AnnotatedTranscript: Complete annotated transcript

Used by the llm_annotation pipeline to validate annotation outputs.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class HateSpeechType(str, Enum):
    """Types of hate speech that can be detected.

    Values:
        NONE: No hate speech detected.
        DEROGATORY: Derogatory language or slurs.
        DEHUMANIZING: Language that dehumanizes a group.
        THREATENING: Threatening or inciting violence.
        STEREOTYPING: Harmful stereotypes or generalizations.
        OTHER: Other forms of harmful speech.
    """

    NONE = "none"
    DEROGATORY = "derogatory"
    DEHUMANIZING = "dehumanizing"
    THREATENING = "threatening"
    STEREOTYPING = "stereotyping"
    OTHER = "other"


class TargetGroup(str, Enum):
    """Groups that may be targeted by harmful content.

    Values:
        NONE: No specific group targeted.
        RACIAL: Racial or ethnic groups.
        RELIGIOUS: Religious groups.
        GENDER: Gender-based targeting.
        LGBTQ: LGBTQ+ community.
        DISABILITY: People with disabilities.
        POLITICAL: Political groups or affiliations.
        NATIONALITY: National origin or immigration status.
        OTHER: Other identifiable groups.
    """

    NONE = "none"
    RACIAL = "racial"
    RELIGIOUS = "religious"
    GENDER = "gender"
    LGBTQ = "lgbtq"
    DISABILITY = "disability"
    POLITICAL = "political"
    NATIONALITY = "nationality"
    OTHER = "other"


class AnnotationResult(BaseModel):
    """Result of LLM annotation for a single segment.

    Contains flags for content classification and optional details
    about detected issues.

    Attributes:
        has_hate_speech: Whether hate speech was detected.
        has_advertisement: Whether segment contains advertising.
        hate_speech_type: Type of hate speech if detected.
        target_group: Targeted group if hate speech detected.
        main_topic: Primary topic or subject of the segment.
        confidence: Confidence score for the annotation (0.0 to 1.0).
        raw_response: Original LLM response for debugging.
    """

    has_hate_speech: bool = Field(
        default=False,
        description="Whether hate speech was detected",
    )
    has_advertisement: bool = Field(
        default=False,
        description="Whether segment contains advertising",
    )
    hate_speech_type: HateSpeechType | None = Field(
        default=None,
        description="Type of hate speech if detected",
    )
    target_group: TargetGroup | None = Field(
        default=None,
        description="Targeted group if hate speech present",
    )
    main_topic: str | None = Field(
        default=None,
        max_length=500,
        description="Primary topic or subject of the segment",
    )
    confidence: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Annotation confidence score (0.0 to 1.0)",
    )
    raw_response: str | None = Field(
        default=None,
        description="Original LLM response for debugging",
    )

    @field_validator("hate_speech_type", mode="before")
    @classmethod
    def parse_hate_speech_type(cls, v: Any) -> HateSpeechType | None:
        """Parse hate speech type from string or enum."""
        if v is None:
            return None
        if isinstance(v, HateSpeechType):
            return v
        if isinstance(v, str):
            try:
                return HateSpeechType(v.lower())
            except ValueError:
                return HateSpeechType.OTHER
        return None

    @field_validator("target_group", mode="before")
    @classmethod
    def parse_target_group(cls, v: Any) -> TargetGroup | None:
        """Parse target group from string or enum."""
        if v is None:
            return None
        if isinstance(v, TargetGroup):
            return v
        if isinstance(v, str):
            try:
                return TargetGroup(v.lower())
            except ValueError:
                return TargetGroup.OTHER
        return None

    class Config:
        """Allow extra fields for forward compatibility."""

        extra = "allow"


class AnnotatedSegment(BaseModel):
    """A transcript segment with its annotation results.

    Combines the original segment data with annotation results
    for convenient access.

    Attributes:
        text: The segment text that was annotated.
        start: Start time in seconds.
        end: End time in seconds.
        speaker: Speaker identifier.
        segment_index: Index of this segment in the original transcript.
        annotation: The annotation result for this segment.
    """

    text: str = Field(description="The segment text")
    start: float = Field(ge=0, description="Start time in seconds")
    end: float = Field(ge=0, description="End time in seconds")
    speaker: str = Field(default="UNKNOWN", description="Speaker identifier")
    segment_index: int = Field(ge=0, description="Index in original transcript")
    annotation: AnnotationResult = Field(
        default_factory=AnnotationResult,
        description="Annotation results",
    )


class AnnotationStats(BaseModel):
    """Statistics about annotations in a transcript.

    Attributes:
        total_segments: Total number of segments annotated.
        hate_speech_count: Number of segments with hate speech.
        advertisement_count: Number of segments with advertising.
        hate_speech_by_type: Count of each hate speech type.
        target_groups: Count of each targeted group.
    """

    total_segments: int = Field(ge=0, description="Total segments annotated")
    hate_speech_count: int = Field(ge=0, default=0, description="Segments with hate speech")
    advertisement_count: int = Field(ge=0, default=0, description="Segments with ads")
    hate_speech_by_type: dict[str, int] = Field(
        default_factory=dict,
        description="Count by hate speech type",
    )
    target_groups: dict[str, int] = Field(
        default_factory=dict,
        description="Count by target group",
    )


class AnnotatedTranscript(BaseModel):
    """Complete transcript with annotation results.

    This is the primary output type for the annotation pipeline.
    Contains all annotated segments plus aggregate statistics.

    Attributes:
        file_name: Name of the source transcript file.
        file_path: Full path to the source file.
        segments: List of annotated segments.
        stats: Aggregate statistics for the transcript.
        metadata: Additional metadata about the annotation process.
    """

    file_name: str = Field(description="Source transcript filename")
    file_path: str = Field(description="Full path to source file")
    segments: list[AnnotatedSegment] = Field(
        default_factory=list,
        description="Annotated segments",
    )
    stats: AnnotationStats = Field(
        default_factory=AnnotationStats,
        description="Annotation statistics",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    def compute_stats(self) -> AnnotationStats:
        """Compute statistics from annotated segments.

        Returns:
            AnnotationStats with counts computed from segments.
        """
        hate_speech_count = 0
        advertisement_count = 0
        hate_speech_by_type: dict[str, int] = {}
        target_groups: dict[str, int] = {}

        for segment in self.segments:
            ann = segment.annotation

            if ann.has_hate_speech:
                hate_speech_count += 1
                if ann.hate_speech_type:
                    key = ann.hate_speech_type.value
                    hate_speech_by_type[key] = hate_speech_by_type.get(key, 0) + 1
                if ann.target_group:
                    key = ann.target_group.value
                    target_groups[key] = target_groups.get(key, 0) + 1

            if ann.has_advertisement:
                advertisement_count += 1

        return AnnotationStats(
            total_segments=len(self.segments),
            hate_speech_count=hate_speech_count,
            advertisement_count=advertisement_count,
            hate_speech_by_type=hate_speech_by_type,
            target_groups=target_groups,
        )
