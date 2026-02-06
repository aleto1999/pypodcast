"""transcription pipeline for podcast audio using whisperx and mlx-whisper."""

from podcast_conversations.transcription.config import TranscriptionConfig, TranscriptionResult
from podcast_conversations.transcription.mlx_backend import (
    MLX_WHISPER_AVAILABLE,
    MLXTranscriptionPipeline,
    get_mlx_transcription_status,
    is_apple_silicon,
    is_mlx_whisper_available,
)
from podcast_conversations.transcription.pipeline import (
    TranscriptionPipeline,
    create_transcription_pipeline,
    detect_transcription_device,
)

__all__ = [
    # config.
    "TranscriptionConfig",
    "TranscriptionResult",
    # pipelines.
    "TranscriptionPipeline",
    "MLXTranscriptionPipeline",
    "create_transcription_pipeline",
    # device detection.
    "detect_transcription_device",
    "is_apple_silicon",
    "is_mlx_whisper_available",
    "get_mlx_transcription_status",
    "MLX_WHISPER_AVAILABLE",
]
