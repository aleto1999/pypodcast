"""whisperx transcription pipeline with gpu optimizations."""

import gc
import logging
import time
import warnings
from pathlib import Path

import torch

from podcast_conversations.transcription.config import TranscriptionConfig, TranscriptionResult
from podcast_conversations.transcription.mlx_backend import (
    MLX_WHISPER_AVAILABLE,
    MLXTranscriptionPipeline,
    is_apple_silicon,
)

# suppress whisperx/faster-whisper warnings.
warnings.filterwarnings("ignore", category=UserWarning, module="whisperx")
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


def detect_transcription_device() -> str:
    """detect the best available device for transcription."""
    # check for CUDA first.
    if torch.cuda.is_available():
        return "cuda"

    # check for Apple Silicon with MLX.
    if is_apple_silicon() and MLX_WHISPER_AVAILABLE:
        return "mlx"

    # check for MPS (Metal Performance Shaders) - note: whisperx doesn't support MPS.
    # fallback to cpu.
    return "cpu"


class TranscriptionPipeline:
    """whisperx-based transcription pipeline with gpu optimizations."""

    def __init__(self, config: TranscriptionConfig | None = None):
        """
        initialize transcription pipeline.

        args:
            config: transcription configuration. uses defaults if None.
        """
        self.config = config or TranscriptionConfig()
        self._model = None
        self._align_model = None
        self._align_metadata = None

        # detect device and gpu memory.
        self._setup_device()

    def _setup_device(self) -> None:
        """setup device and detect gpu memory."""
        requested_device = self.config.device

        # handle 'auto' device detection.
        if requested_device == "auto":
            requested_device = detect_transcription_device()
            logger.info(f"auto-detected device: {requested_device}")

        if requested_device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
            gpu_props = torch.cuda.get_device_properties(0)
            self.gpu_memory_gb = gpu_props.total_memory / (1024**3)
            logger.info(f"using gpu: {gpu_props.name} ({self.gpu_memory_gb:.1f}GB)")
        elif requested_device == "mlx":
            # mlx is handled separately via MLXTranscriptionPipeline.
            # this shouldn't be reached if using create_transcription_pipeline().
            self.device = "mlx"
            self.gpu_memory_gb = 0
            logger.info("mlx device requested - use create_transcription_pipeline() instead")
        else:
            self.device = "cpu"
            self.gpu_memory_gb = 0
            logger.info("using cpu for transcription")

        # calculate optimal batch size.
        self.batch_size = self.config.calculate_optimal_batch_size(self.gpu_memory_gb)
        logger.info(f"batch size: {self.batch_size}")

    def _load_model(self) -> None:
        """lazy load whisperx model."""
        if self._model is not None:
            return

        import whisperx

        logger.info(f"loading whisperx model: {self.config.model_size}")
        start = time.time()

        self._model = whisperx.load_model(
            self.config.model_size,
            device=self.device,
            compute_type=self.config.compute_type,
            language=self.config.language,
        )

        elapsed = time.time() - start
        logger.info(f"model loaded in {elapsed:.1f}s")

    def _load_align_model(self, language: str) -> None:
        """load alignment model for word-level timestamps."""
        import whisperx

        # only reload if language changed.
        if self._align_model is not None and self._current_align_lang == language:
            return

        logger.debug(f"loading alignment model for language: {language}")
        self._align_model, self._align_metadata = whisperx.load_align_model(
            language_code=language,
            device=self.device,
        )
        self._current_align_lang = language

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """
        transcribe an audio file.

        args:
            audio_path: path to audio file.

        returns:
            TranscriptionResult with segments and text.
        """
        import whisperx

        self._load_model()

        start_time = time.time()
        logger.info(f"transcribing: {audio_path.name}")

        # load audio.
        audio = whisperx.load_audio(str(audio_path))

        # transcribe with batched inference.
        result = self._model.transcribe(
            audio,
            batch_size=self.batch_size,
            language=self.config.language,
        )

        # get detected language.
        language = result.get("language", self.config.language or "en")

        # align for word-level timestamps.
        try:
            self._load_align_model(language)
            result = whisperx.align(
                result["segments"],
                self._align_model,
                self._align_metadata,
                audio,
                self.device,
                return_char_alignments=False,
            )
        except Exception as e:
            logger.warning(f"alignment failed, using segment-level timestamps: {e}")

        # build result.
        segments = result.get("segments", [])
        full_text = " ".join(s.get("text", "").strip() for s in segments)

        elapsed = time.time() - start_time
        logger.info(f"transcribed {audio_path.name} in {elapsed:.1f}s")

        return TranscriptionResult(
            audio_file=audio_path.name,
            language=language,
            text=full_text,
            segments=segments,
        )

    def transcribe_to_json(self, audio_path: Path, output_path: Path) -> TranscriptionResult:
        """
        transcribe audio and save result to json file.

        args:
            audio_path: path to audio file.
            output_path: path for output json file.

        returns:
            TranscriptionResult.
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
        """clear gpu cache and run garbage collection."""
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    def unload(self) -> None:
        """unload models to free memory."""
        self._model = None
        self._align_model = None
        self._align_metadata = None
        self.clear_cache()
        logger.info("models unloaded")


def create_transcription_pipeline(
    config: TranscriptionConfig | None = None,
) -> TranscriptionPipeline | MLXTranscriptionPipeline:
    """
    factory function to create the appropriate transcription pipeline.

    automatically selects MLX backend on Apple Silicon when available,
    otherwise uses WhisperX with CUDA/CPU.

    args:
        config: transcription configuration. uses defaults if None.

    returns:
        appropriate transcription pipeline for the platform.
    """
    config = config or TranscriptionConfig()

    # determine device.
    device = config.device
    if device == "auto":
        device = detect_transcription_device()

    # use mlx backend on Apple Silicon.
    if device == "mlx":
        if not MLX_WHISPER_AVAILABLE:
            logger.warning("mlx-whisper not available, falling back to cpu")
            config.device = "cpu"
            return TranscriptionPipeline(config)

        logger.info("using mlx backend for Apple Silicon")
        return MLXTranscriptionPipeline(
            model_size=config.model_size,
            language=config.language,
        )

    # use whisperx for cuda/cpu.
    return TranscriptionPipeline(config)
