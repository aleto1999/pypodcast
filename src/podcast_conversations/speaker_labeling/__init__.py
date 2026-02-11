"""Speaker labeling module for identifying and naming speakers in podcast transcripts.

This module provides a 5-stage pipeline for replacing generic speaker labels
(SPEAKER_00, SPEAKER_01, etc.) with actual speaker names:

1. Metadata Extraction - Extract names from RSS feeds, filenames, folder names
2. Transcript NER - Extract person names from transcript text using NLP
3. Cross-Episode Analysis - Identify hosts by analyzing patterns across episodes
4. LLM Role Classification - Classify names as HOST/GUEST/NEITHER
5. Voice-Name Assignment - Map names to speaker IDs using heuristics

Usage:
    from podcast_conversations.speaker_labeling import SpeakerLabelingPipeline

    pipeline = SpeakerLabelingPipeline(
        input_dir=Path("outputs/transcripts_with_diarization_labels_postprocessed"),
        output_dir=Path("outputs/transcripts_with_speaker_labels_postprocessed"),
    )
    results = await pipeline.run()
"""

from podcast_conversations.speaker_labeling.types import (
    ExtractedName,
    SpeakerRole,
    RoleClassification,
    SpeakerMapping,
    EpisodeMetadata,
    PodcastMetadata,
    SpeakerStats,
    HostCandidate,
    LabeledSegment,
    LabeledTranscript,
    PipelineConfig,
    StageResult,
    PipelineSummary,
    EpisodeResult,
    PodcastResult,
    AssignmentMethod,
    ExtractionMethod,
)

from podcast_conversations.speaker_labeling.pipeline import (
    SpeakerLabelingPipeline,
    create_pipeline,
)

from podcast_conversations.speaker_labeling.metadata_extractor import MetadataExtractor
from podcast_conversations.speaker_labeling.transcript_ner import TranscriptNERExtractor
from podcast_conversations.speaker_labeling.cross_episode_analyzer import CrossEpisodeAnalyzer
from podcast_conversations.speaker_labeling.role_classifier import (
    HeuristicRoleClassifier,
    LLMRoleClassifier,
    LocalLLMRoleClassifier,
    create_role_classifier,
    create_local_classifier,
    detect_cuda_available,
)
from podcast_conversations.speaker_labeling.voice_name_mapper import (
    VoiceNameMapper,
    TranscriptLabeler,
)
from podcast_conversations.speaker_labeling.summary_generator import (
    SummaryGenerator,
    create_summary_generator,
)

__all__ = [
    # types.
    "ExtractedName",
    "SpeakerRole",
    "RoleClassification",
    "SpeakerMapping",
    "EpisodeMetadata",
    "PodcastMetadata",
    "SpeakerStats",
    "HostCandidate",
    "LabeledSegment",
    "LabeledTranscript",
    "PipelineConfig",
    "StageResult",
    "PipelineSummary",
    "EpisodeResult",
    "PodcastResult",
    "AssignmentMethod",
    "ExtractionMethod",
    # pipeline.
    "SpeakerLabelingPipeline",
    "create_pipeline",
    # extractors.
    "MetadataExtractor",
    "TranscriptNERExtractor",
    "CrossEpisodeAnalyzer",
    # classifiers.
    "HeuristicRoleClassifier",
    "LLMRoleClassifier",
    "LocalLLMRoleClassifier",
    "create_role_classifier",
    "create_local_classifier",
    "detect_cuda_available",
    # mappers.
    "VoiceNameMapper",
    "TranscriptLabeler",
    # summary.
    "SummaryGenerator",
    "create_summary_generator",
]
