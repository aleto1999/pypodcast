"""configuration models for transcription pipeline."""

import logging
from typing import Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# model size to VRAM multiplier for batch size calculation.
MODEL_VRAM_FACTORS = {
    "tiny": 1.0,
    "base": 1.5,
    "small": 2.0,
    "medium": 3.0,
    "large-v2": 5.0,
    "large-v3": 5.0,
    "turbo": 4.0,
    "large-v3-turbo": 4.0,
    "large-v3-8bit": 2.5,
    "large-v3-4bit": 1.5,
    "distil-large-v3": 3.0,
}

# compute type to speed multiplier.
COMPUTE_TYPE_FACTORS = {
    "float16": 1.0,
    "int8": 0.8,
    "float32": 2.0,
}


class TranscriptionConfig(BaseModel):
    """configuration for whisperx transcription."""

    model_size: Literal[
        "tiny", "base", "small", "medium", "large-v2", "large-v3",
        "turbo", "large-v3-turbo", "large-v3-8bit", "large-v3-4bit", "distil-large-v3"
    ] = Field(
        default="large-v3",
        description="whisper model size",
    )
    language: str | None = Field(
        default=None,
        description="language code (en, es, etc.) or None for auto-detect",
    )
    compute_type: Literal["float16", "int8", "float32"] = Field(
        default="float16",
        description="compute precision type (ignored for mlx backend)",
    )
    batch_size: int | None = Field(
        default=None,
        description="batch size for inference (auto-calculated if None)",
    )
    device: str = Field(
        default="cuda",
        description="device to use (cuda, mlx, or cpu)",
    )

    def calculate_optimal_batch_size(self, gpu_memory_gb: float = 80.0) -> int:
        """
        calculate optimal batch size based on gpu memory and model.

        uses free gpu memory when available for dynamic allocation,
        falls back to total memory estimate.

        args:
            gpu_memory_gb: total gpu memory in gigabytes (used as fallback).

        returns:
            optimal batch size for the configuration.
        """
        if self.batch_size is not None:
            return self.batch_size

        if self.device == "cpu":
            return 4  # cpu default.

        available_gb = self._get_free_gpu_memory_gb(gpu_memory_gb)

        model_factor = MODEL_VRAM_FACTORS.get(self.model_size, 5.0)
        compute_factor = COMPUTE_TYPE_FACTORS.get(self.compute_type, 1.0)

        optimal = int((available_gb / 10) * (40 / (model_factor * compute_factor)))

        # clamp to reasonable range.
        result = max(4, min(optimal, 128))
        logger.debug(
            f"batch size: {result} "
            f"(available VRAM: {available_gb:.1f}GB, model: {self.model_size})"
        )
        return result

    @staticmethod
    def _get_free_gpu_memory_gb(fallback_gb: float) -> float:
        """get free gpu memory in GB, falling back to provided value."""
        try:
            import torch

            if torch.cuda.is_available():
                free_bytes, total_bytes = torch.cuda.mem_get_info(0)
                free_gb = free_bytes / (1024**3)
                total_gb = total_bytes / (1024**3)
                logger.debug(f"GPU memory: {free_gb:.1f}GB free / {total_gb:.1f}GB total")
                return free_gb
        except Exception:
            pass
        return fallback_gb


class TranscriptionResult(BaseModel):
    """result of a transcription operation."""

    audio_file: str
    language: str | None = None
    text: str = ""
    segments: list[dict] = Field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        """get total duration from segments."""
        if not self.segments:
            return 0.0
        return max(s.get("end", 0) for s in self.segments)

    @property
    def word_count(self) -> int:
        """get total word count."""
        return len(self.text.split())
