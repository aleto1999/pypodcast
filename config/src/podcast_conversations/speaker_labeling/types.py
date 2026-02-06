"""Type definitions for the speaker labeling pipeline.

Provides Pydantic models for:
- Extracted names and speaker roles
- Episode and podcast metadata
- Speaker statistics and host candidates
- Labeled transcripts
- Pipeline configuration and results
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, computed_field


class SpeakerRole(str, Enum):
    """Role classification for a speaker."""

    HOST = "HOST"
    GUEST = "GUEST"
    NEITHER = "NEITHER"
    UNKNOWN = "UNKNOWN"


class ExtractionMethod(str, Enum):
    """Method used to extract a name."""

    NER = "ner"
    PATTERN = "pattern"
    RSS_METADATA = "rss_metadata"
    FILENAME = "filename"
    FOLDER_NAME = "folder_name"
    LLM = "llm"
    COMBINED = "combined"


class AssignmentMethod(str, Enum):
    """Method used to assign a name to a speaker ID."""

    SELF_INTRODUCTION = "self_introduction"
    DIRECT_MENTION = "direct_mention"
    INTRODUCTION_CONTEXT = "introduction_context"
    CROSS_EPISODE_PATTERN = "cross_episode_pattern"
    SPEAKING_TIME_RANK = "speaking_time_rank"
    SPEAKING_TIME_MATCH = "speaking_time_match"
    FIRST_SPEAKER = "first_speaker"
    RSS_HINT = "rss_hint"
    FALLBACK = "fallback"
    UNASSIGNED = "unassigned"


class ExtractedName(BaseModel):
    """A person name extracted from a transcript or metadata.

    Attributes:
        name: The extracted name.
        context: Surrounding text where the name was found.
        position: Character position in the source text.
        extraction_method: How the name was extracted.
        confidence: Confidence score (0.0-1.0).
        source_speaker: Speaker ID who mentioned this name (if applicable).
    """

    name: str = Field(description="The extracted person name")
    context: str = Field(default="", description="Surrounding context text")
    position: int = Field(default=0, ge=0, description="Position in source text")
    extraction_method: ExtractionMethod = Field(
        default=ExtractionMethod.NER,
        description="Method used for extraction",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Extraction confidence score",
    )
    source_speaker: str | None = Field(
        default=None,
        description="Speaker ID who mentioned this name",
    )

    def __hash__(self) -> int:
        return hash(self.name.lower())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ExtractedName):
            return self.name.lower() == other.name.lower()
        return False


class RoleClassification(BaseModel):
    """Classification of a name as HOST, GUEST, or NEITHER.

    Attributes:
        name: The classified name.
        role: Assigned role.
        confidence: Classification confidence.
        reasoning: Explanation for the classification.
    """

    name: str = Field(description="The classified name")
    role: SpeakerRole = Field(description="Assigned role")
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Classification confidence",
    )
    reasoning: str = Field(default="", description="Explanation for classification")


class SpeakerMapping(BaseModel):
    """Mapping from a speaker ID to a named person.

    Attributes:
        speaker_id: Original speaker ID (e.g., SPEAKER_00).
        name: Name assigned to this speaker.
        role: Role of this speaker.
        confidence: Mapping confidence.
        assignment_method: How the assignment was determined.
        evidence: Supporting evidence for the assignment.
    """

    speaker_id: str = Field(description="Original speaker ID")
    name: str = Field(default="", description="Assigned speaker name")
    role: SpeakerRole = Field(default=SpeakerRole.UNKNOWN, description="Speaker role")
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Assignment confidence",
    )
    assignment_method: AssignmentMethod = Field(
        default=AssignmentMethod.UNASSIGNED,
        description="How the mapping was determined",
    )
    evidence: dict[str, Any] = Field(default_factory=dict, description="Supporting evidence")


class SpeakerStats(BaseModel):
    """Statistics for a speaker across episodes.

    Attributes:
        total_speaking_time: Total speaking time in seconds.
        total_segments: Total number of segments.
        episodes_appeared: Number of episodes this speaker appeared in.
        episode_speaking_times: Speaking time per episode.
    """

    total_speaking_time: float = Field(default=0.0, ge=0, description="Total speaking time")
    total_segments: int = Field(default=0, ge=0, description="Total segments")
    episodes_appeared: int = Field(default=0, ge=0, description="Episodes appeared in")
    episode_speaking_times: list[float] = Field(
        default_factory=list,
        description="Speaking time per episode",
    )

    @computed_field
    @property
    def avg_speaking_time_per_episode(self) -> float:
        """Average speaking time per episode."""
        if not self.episode_speaking_times:
            return 0.0
        return sum(self.episode_speaking_times) / len(self.episode_speaking_times)

    @computed_field
    @property
    def speaking_time_consistency(self) -> float:
        """Consistency score (lower variance = more consistent = likely host)."""
        if len(self.episode_speaking_times) < 2:
            return 0.0
        import statistics

        try:
            mean = statistics.mean(self.episode_speaking_times)
            stdev = statistics.stdev(self.episode_speaking_times)
            return 1 - (stdev / mean) if mean > 0 else 0.0
        except statistics.StatisticsError:
            return 0.0


class HostCandidate(BaseModel):
    """A candidate host identified through cross-episode analysis.

    Attributes:
        speaker_pattern: Pattern describing the speaker (e.g., "rank_0_by_speaking_time").
        confidence: Confidence that this is the host.
        evidence: Supporting evidence for the determination.
    """

    speaker_pattern: str = Field(description="Pattern identifying the speaker")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    evidence: dict[str, Any] = Field(default_factory=dict, description="Supporting evidence")


class EpisodeMetadata(BaseModel):
    """Metadata for a single episode.

    Attributes:
        podcast_name: Name of the podcast series.
        episode_filename: Filename of the episode.
        episode_title: Title from RSS feed.
        published: Publication date.
        description: Episode description.
        inferred_host: Host name inferred from metadata.
        inferred_guests: Guest names inferred from metadata.
        confidence: Confidence in the metadata extraction.
    """

    podcast_name: str = Field(description="Podcast series name")
    episode_filename: str = Field(description="Episode filename")
    episode_title: str | None = Field(default=None, description="Episode title from RSS")
    published: datetime | None = Field(default=None, description="Publication date")
    description: str | None = Field(default=None, description="Episode description")
    inferred_host: str | None = Field(default=None, description="Inferred host name")
    inferred_guests: list[str] = Field(default_factory=list, description="Inferred guest names")
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Metadata confidence",
    )


class PodcastMetadata(BaseModel):
    """Metadata for a podcast series.

    Attributes:
        title: Podcast title.
        feed_url: RSS feed URL.
        author: Podcast author (often the host).
        description: Podcast description.
        inferred_host: Inferred host name.
        frequent_guests: List of frequent guests.
        episode_count: Total number of episodes.
    """

    title: str = Field(description="Podcast title")
    feed_url: str | None = Field(default=None, description="RSS feed URL")
    author: str | None = Field(default=None, description="Podcast author")
    description: str | None = Field(default=None, description="Podcast description")
    inferred_host: str | None = Field(default=None, description="Inferred host name")
    frequent_guests: list[str] = Field(default_factory=list, description="Frequent guests")
    episode_count: int = Field(default=0, ge=0, description="Episode count")


class LabeledSegment(BaseModel):
    """A transcript segment with speaker name labels.

    Extends the standard segment with speaker identification.

    Attributes:
        text: Segment text content.
        start: Start time in seconds.
        end: End time in seconds.
        speaker_id: Original speaker ID (e.g., SPEAKER_00).
        speaker_name: Assigned speaker name.
        speaker_role: Speaker role (HOST/GUEST).
        words: Word-level timing (optional).
    """

    text: str = Field(description="Segment text content")
    start: float = Field(ge=0, description="Start time in seconds")
    end: float = Field(ge=0, description="End time in seconds")
    speaker_id: str = Field(description="Original speaker ID")
    speaker_name: str | None = Field(default=None, description="Assigned speaker name")
    speaker_role: SpeakerRole | None = Field(default=None, description="Speaker role")
    words: list[dict[str, Any]] = Field(default_factory=list, description="Word timing")

    @computed_field
    @property
    def duration(self) -> float:
        """Segment duration in seconds."""
        return self.end - self.start


class LabeledTranscript(BaseModel):
    """A complete transcript with speaker name labels.

    Attributes:
        original_file: Path to the original transcript file.
        audio_file: Path to the source audio file.
        language: Language code.
        speaker_mappings: Mapping from speaker IDs to names.
        segments: Labeled transcript segments.
        metadata: Additional metadata.
    """

    original_file: str = Field(description="Original transcript file path")
    audio_file: str | None = Field(default=None, description="Source audio file")
    language: str = Field(default="en", description="Language code")
    speaker_mappings: dict[str, SpeakerMapping] = Field(
        default_factory=dict,
        description="Speaker ID to name mappings",
    )
    segments: list[LabeledSegment] = Field(
        default_factory=list,
        description="Labeled segments",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @computed_field
    @property
    def total_duration(self) -> float:
        """Total duration in seconds."""
        if not self.segments:
            return 0.0
        return max(s.end for s in self.segments)

    @computed_field
    @property
    def speakers_identified(self) -> int:
        """Number of speakers with assigned names."""
        return sum(1 for m in self.speaker_mappings.values() if m.assigned_name)

    @computed_field
    @property
    def total_speakers(self) -> int:
        """Total number of unique speakers."""
        return len(self.speaker_mappings)


class PipelineConfig(BaseModel):
    """Configuration for the speaker labeling pipeline.

    Attributes:
        input_dir: Input transcripts directory.
        output_dir: Output directory for labeled transcripts.
        rss_metadata_dir: Directory containing RSS metadata.
        use_llm: Whether to use LLM API for role classification.
        llm_model: OpenAI LLM model to use.
        use_local_llm: Whether to use local transformers model with CUDA.
        local_model: HuggingFace model name for local inference.
        load_in_4bit: Use 4-bit quantization for local model.
        load_in_8bit: Use 8-bit quantization for local model.
        max_workers: Maximum parallel workers.
        batch_size: Batch size for processing.
        skip_existing: Skip files that already exist.
        min_confidence: Minimum confidence threshold for assignments.
    """

    model_config = {"arbitrary_types_allowed": True}

    input_dir: Path = Field(description="Input transcripts directory")
    output_dir: Path = Field(description="Output directory")
    rss_metadata_dir: Path | None = Field(default=None, description="RSS metadata directory")
    use_llm: bool = Field(default=False, description="Use OpenAI API for role classification")
    llm_model: str = Field(default="gpt-4o-mini", description="OpenAI model to use")
    use_local_llm: bool = Field(default=False, description="Use local CUDA model")
    local_model: str | None = Field(default=None, description="HuggingFace model for local inference")
    load_in_4bit: bool = Field(default=False, description="Use 4-bit quantization")
    load_in_8bit: bool = Field(default=False, description="Use 8-bit quantization")
    max_workers: int = Field(default=4, ge=1, description="Max parallel workers")
    batch_size: int = Field(default=8, ge=1, description="Batch size")
    skip_existing: bool = Field(default=True, description="Skip existing files")
    min_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold",
    )


class StageResult(BaseModel):
    """Result from a single pipeline stage.

    Attributes:
        stage_name: Name of the stage.
        success: Whether the stage succeeded.
        items_processed: Number of items processed.
        items_failed: Number of items that failed.
        duration_seconds: Time taken.
        error_message: Error message if failed.
        metrics: Stage-specific metrics.
    """

    stage_name: str = Field(description="Stage name")
    success: bool = Field(description="Whether stage succeeded")
    items_processed: int = Field(default=0, ge=0, description="Items processed")
    items_failed: int = Field(default=0, ge=0, description="Items failed")
    duration_seconds: float = Field(default=0.0, ge=0, description="Duration")
    error_message: str | None = Field(default=None, description="Error message")
    metrics: dict[str, Any] = Field(default_factory=dict, description="Stage metrics")


class EpisodeSummary(BaseModel):
    """Summary statistics for a single episode.

    Attributes:
        filename: Episode filename.
        total_segments: Total segments.
        total_speakers: Total unique speakers.
        speakers_identified: Speakers with assigned names.
        host_name: Identified host name.
        guest_names: Identified guest names.
        total_duration: Total duration in seconds.
        labeling_confidence: Average confidence of assignments.
    """

    filename: str = Field(description="Episode filename")
    total_segments: int = Field(default=0, ge=0, description="Total segments")
    total_speakers: int = Field(default=0, ge=0, description="Total speakers")
    speakers_identified: int = Field(default=0, ge=0, description="Speakers identified")
    host_name: str | None = Field(default=None, description="Identified host")
    guest_names: list[str] = Field(default_factory=list, description="Identified guests")
    total_duration: float = Field(default=0.0, ge=0, description="Duration in seconds")
    labeling_confidence: float = Field(default=0.0, ge=0, le=1, description="Average confidence")


class PodcastSummary(BaseModel):
    """Summary statistics for a podcast series.

    Attributes:
        podcast_name: Podcast name.
        total_episodes: Total episodes processed.
        episodes_with_labels: Episodes with speaker labels.
        primary_host: Most common host name.
        unique_guests: Set of unique guests.
        avg_confidence: Average labeling confidence.
        episode_summaries: Per-episode summaries.
    """

    podcast_name: str = Field(description="Podcast name")
    total_episodes: int = Field(default=0, ge=0, description="Total episodes")
    episodes_with_labels: int = Field(default=0, ge=0, description="Episodes with labels")
    primary_host: str | None = Field(default=None, description="Primary host")
    unique_guests: list[str] = Field(default_factory=list, description="Unique guests")
    avg_confidence: float = Field(default=0.0, ge=0, le=1, description="Average confidence")
    episode_summaries: list[EpisodeSummary] = Field(
        default_factory=list,
        description="Episode summaries",
    )


class EpisodeResult(BaseModel):
    """Result of processing a single episode.

    Attributes:
        episode_file: Path to the episode file.
        success: Whether processing succeeded.
        error: Error message if failed.
        speakers_labeled: Number of speakers labeled.
        speakers_total: Total speakers in episode.
        mappings: Speaker mappings generated.
    """

    episode_file: str = Field(description="Episode file path")
    success: bool = Field(default=True, description="Whether processing succeeded")
    error: str | None = Field(default=None, description="Error message if failed")
    speakers_labeled: int = Field(default=0, ge=0, description="Speakers labeled")
    speakers_total: int = Field(default=0, ge=0, description="Total speakers")
    mappings: list[SpeakerMapping] = Field(default_factory=list, description="Speaker mappings")


class PodcastResult(BaseModel):
    """Result of processing a podcast.

    Attributes:
        podcast_name: Name of the podcast.
        episodes_processed: Number of episodes processed.
        episodes_failed: Number of episodes that failed.
        episode_results: Results for each episode.
    """

    podcast_name: str = Field(description="Podcast name")
    episodes_processed: int = Field(default=0, ge=0, description="Episodes processed")
    episodes_failed: int = Field(default=0, ge=0, description="Episodes failed")
    episode_results: list[EpisodeResult] = Field(
        default_factory=list,
        description="Episode results",
    )


class PipelineSummary(BaseModel):
    """Summary of the entire pipeline run.

    Attributes:
        started_at: Pipeline start time.
        completed_at: Pipeline completion time.
        total_podcasts: Total podcasts processed.
        total_episodes_processed: Total episodes processed.
        total_episodes_failed: Total episodes that failed.
        total_speakers_labeled: Total speakers labeled.
        total_speakers_found: Total speakers found.
        average_confidence: Average confidence score.
        method_distribution: Assignment method distribution.
        role_distribution: Role distribution.
        podcast_results: Results for each podcast.
    """

    started_at: datetime = Field(default_factory=datetime.now, description="Start time")
    completed_at: datetime | None = Field(default=None, description="Completion time")
    total_podcasts: int = Field(default=0, ge=0, description="Total podcasts")
    total_episodes_processed: int = Field(default=0, ge=0, description="Total episodes processed")
    total_episodes_failed: int = Field(default=0, ge=0, description="Total episodes failed")
    total_speakers_labeled: int = Field(default=0, ge=0, description="Total speakers labeled")
    total_speakers_found: int = Field(default=0, ge=0, description="Total speakers found")
    average_confidence: float = Field(default=0.0, ge=0, le=1, description="Average confidence")
    method_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="Assignment method distribution",
    )
    role_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="Role distribution",
    )
    podcast_results: list[PodcastResult] = Field(
        default_factory=list,
        description="Podcast results",
    )

    @computed_field
    @property
    def duration_seconds(self) -> float:
        """Total pipeline duration in seconds."""
        if not self.completed_at:
            return 0.0
        return (self.completed_at - self.started_at).total_seconds()

    @computed_field
    @property
    def success_rate(self) -> float:
        """Proportion of episodes successfully processed."""
        total = self.total_episodes_processed + self.total_episodes_failed
        if total == 0:
            return 0.0
        return self.total_episodes_processed / total
