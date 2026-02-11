"""configuration models for transcription pipeline."""

from typing import Literal

from pydantic import BaseModel, Field

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

        formula: (gpu_gb / 10) * (40 / (model_factor * compute_factor))

        args:
            gpu_memory_gb: available gpu memory in gigabytes.

        returns:
            optimal batch size for the configuration.
        """
        if self.batch_size is not None:
            return self.batch_size

        if self.device == "cpu":
            return 4  # cpu default.

        model_factor = MODEL_VRAM_FACTORS.get(self.model_size, 5.0)
        compute_factor = COMPUTE_TYPE_FACTORS.get(self.compute_type, 1.0)

        optimal = int((gpu_memory_gb / 10) * (40 / (model_factor * compute_factor)))

        # clamp to reasonable range.
        return max(4, min(optimal, 128))


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
