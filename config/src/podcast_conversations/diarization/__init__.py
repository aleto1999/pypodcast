"""speaker diarization pipeline for podcast conversations."""

from podcast_conversations.diarization.file_discovery import FileDiscovery, FileMapping
from podcast_conversations.diarization.pipeline import (
    DiarizationPipeline,
    detect_diarization_device,
    get_diarization_device_info,
    is_apple_silicon,
)
from podcast_conversations.diarization.rttm_writer import RTTMWriter

__all__ = [
    "FileDiscovery",
    "FileMapping",
    "DiarizationPipeline",
    "RTTMWriter",
    # device detection.
    "detect_diarization_device",
    "get_diarization_device_info",
    "is_apple_silicon",
]
