"""mlx-optimized transcription backend for Apple Silicon."""

import logging
import platform
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# mlx-whisper availability flag.
MLX_WHISPER_AVAILABLE = False

try:
    import mlx_whisper

    MLX_WHISPER_AVAILABLE = True
except ImportError:
    mlx_whisper = None


def is_mlx_whisper_available() -> bool:
    """check if mlx-whisper is available for use."""
    return MLX_WHISPER_AVAILABLE


def is_apple_silicon() -> bool:
    """check if running on Apple Silicon."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def get_mlx_transcription_status() -> dict[str, Any]:
    """get mlx-whisper availability status and recommendations."""
    use_mlx = is_apple_silicon() and MLX_WHISPER_AVAILABLE
    status = {
        "is_apple_silicon": is_apple_silicon(),
        "mlx_whisper_available": MLX_WHISPER_AVAILABLE,
        "recommended_backend": "mlx" if use_mlx else "whisperx",
        "install_command": None,
    }

    if is_apple_silicon() and not MLX_WHISPER_AVAILABLE:
        status["install_command"] = "uv sync --extra macos"

    return status


# mlx model registry mapping standard names to mlx-community models.
MLX_MODEL_REGISTRY = {
    # standard whisper model sizes.
    "tiny": "mlx-community/whisper-tiny",
    "base": "mlx-community/whisper-base",
    "small": "mlx-community/whisper-small",
    "medium": "mlx-community/whisper-medium",
    "large": "mlx-community/whisper-large-v3",
    "large-v2": "mlx-community/whisper-large-v2",
    "large-v3": "mlx-community/whisper-large-v3",
    # turbo variants (faster, slightly less accurate).
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    # quantized variants for lower memory.
    "large-v3-8bit": "mlx-community/whisper-large-v3-mlx-8bit",
    "large-v3-4bit": "mlx-community/whisper-large-v3-mlx-4bit",
    # distil variants (faster, good accuracy).
    "distil-large-v3": "mlx-community/distil-whisper-large-v3",
}


def get_mlx_model_name(model_size: str) -> str:
    """map standard model size to mlx-community model name."""
    return MLX_MODEL_REGISTRY.get(model_size, f"mlx-community/whisper-{model_size}")


def get_available_mlx_models() -> list[str]:
    """get list of available mlx model keys."""
    return list(MLX_MODEL_REGISTRY.keys())


class MLXTranscriptionResult:
    """result container for mlx transcription with whisperx-compatible interface."""

    def __init__(
        self,
        audio_file: str,
        language: str | None,
        text: str,
        segments: list[dict],
    ):
        self.audio_file = audio_file
        self.language = language
        self.text = text
        self.segments = segments

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

    def model_dump(self) -> dict:
        """serialize to dictionary (pydantic-compatible interface)."""
        return {
            "audio_file": self.audio_file,
            "language": self.language,
            "text": self.text,
            "segments": self.segments,
        }


class MLXTranscriptionPipeline:
    """mlx-whisper based transcription pipeline for Apple Silicon."""

    def __init__(
        self,
        model_size: str = "large-v3",
        language: str | None = None,
    ):
        """
        initialize mlx transcription pipeline.

        args:
            model_size: model size (tiny, base, small, medium, large-v2, large-v3,
                       turbo, large-v3-turbo, large-v3-8bit, large-v3-4bit, distil-large-v3)
            language: language code (en, es, etc.) or None for auto-detect
        """
        if not MLX_WHISPER_AVAILABLE:
            raise ImportError(
                "mlx-whisper not installed. Install with: uv sync --extra macos"
            )

        self.model_name = get_mlx_model_name(model_size)
        self.language = language
        self._model_size = model_size

        logger.info(f"mlx transcription pipeline initialized with model: {self.model_name}")
        print(f"loading mlx whisper model: {self.model_name}")
        print("✓ mlx-whisper ready on Apple Silicon")

    def transcribe(self, audio_path: Path) -> MLXTranscriptionResult:
        """
        transcribe an audio file using mlx-whisper.

        args:
            audio_path: path to audio file.

        returns:
            MLXTranscriptionResult with segments and text.
        """
        start_time = time.time()
        logger.info(f"transcribing with mlx: {audio_path.name}")

        # prepare transcription options.
        transcribe_kwargs = {
            "path_or_hf_repo": self.model_name,
            "word_timestamps": True,  # always get word-level timestamps.
            "verbose": False,
        }

        # add language if specified.
        if self.language:
            transcribe_kwargs["language"] = self.language

        # run transcription.
        result = mlx_whisper.transcribe(
            str(audio_path),
            **transcribe_kwargs,
        )

        # extract results.
        text = result.get("text", "").strip()
        language = result.get("language", self.language or "en")
        raw_segments = result.get("segments", [])

        # convert segments to standard format.
        segments = self._convert_segments(raw_segments)

        elapsed = time.time() - start_time
        logger.info(f"transcribed {audio_path.name} in {elapsed:.1f}s")

        return MLXTranscriptionResult(
            audio_file=audio_path.name,
            language=language,
            text=text,
            segments=segments,
        )

    def _convert_segments(self, raw_segments: list[dict]) -> list[dict]:
        """convert mlx-whisper segments to standard format."""
        segments = []

        for seg in raw_segments:
            segment = {
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
                "text": seg.get("text", "").strip(),
            }

            # add word-level timestamps if available.
            if "words" in seg:
                segment["words"] = [
                    {
                        "word": w.get("word", ""),
                        "start": w.get("start", 0.0),
                        "end": w.get("end", 0.0),
                    }
                    for w in seg["words"]
                ]

            segments.append(segment)

        return segments

    def transcribe_to_json(
        self, audio_path: Path, output_path: Path
    ) -> MLXTranscriptionResult:
        """
        transcribe audio and save result to json file.

        args:
            audio_path: path to audio file.
            output_path: path for output json file.

        returns:
            MLXTranscriptionResult.
        """
        import json

        result = self.transcribe(audio_path)

        # ensure output directory exists.
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # write json.
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)

        logger.info(f"saved transcript to {output_path}")
        return result

    def clear_cache(self) -> None:
        """clear memory cache (no-op for mlx, included for interface compatibility)."""
        # mlx manages memory automatically via unified memory.
        pass

    def unload(self) -> None:
        """unload models (no-op for mlx, included for interface compatibility)."""
        # mlx-whisper loads models on-demand per transcription.
        logger.info("mlx pipeline unloaded")


def get_optimal_batch_size_mlx(model_size: str = "large-v3") -> int:
    """
    get optimal batch size for mlx on Apple Silicon.

    note: mlx-whisper handles batching internally, so this returns
    a recommendation for any external batching needs.
    """
    import psutil

    try:
        ram_gb = psutil.virtual_memory().total / (1024**3)

        # mlx efficiently uses unified memory.
        # larger memory allows processing longer audio files.
        if ram_gb >= 64:
            batch_size = 32
        elif ram_gb >= 32:
            batch_size = 16
        elif ram_gb >= 16:
            batch_size = 8
        else:
            batch_size = 4

        logger.info(f"unified memory: {ram_gb:.1f}GB -> mlx batch size: {batch_size}")
        return batch_size

    except Exception:
        return 8


def load_mlx_transcription_pipeline(
    model_size: str = "large-v3",
    language: str | None = None,
) -> MLXTranscriptionPipeline:
    """
    load mlx transcription pipeline.

    args:
        model_size: model size (tiny, base, small, medium, large-v2, large-v3,
                   turbo, distil-large-v3, or quantized variants)
        language: language code or None for auto-detect

    returns:
        MLXTranscriptionPipeline ready for transcription
    """
    return MLXTranscriptionPipeline(model_size=model_size, language=language)
