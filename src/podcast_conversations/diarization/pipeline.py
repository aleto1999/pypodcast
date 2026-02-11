"""GPU-optimized pyannote.audio diarization pipeline wrapper.

Supports CUDA (NVIDIA), MPS (Apple Silicon), and CPU backends.
"""

import logging
import platform
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any

# CRITICAL: apply PyTorch 2.6+ compatibility fix BEFORE importing torch/pyannote.
# this registers omegaconf classes as safe globals for torch.load.
from podcast_conversations.transcription import torchaudio_compat  # noqa: F401

import numpy as np
import torch
import torchaudio

from podcast_conversations.config import DEFAULT_DIARIZATION_MODEL

# suppress torchaudio 2.9 deprecation warnings (UserWarning category).
warnings.filterwarnings(
    "ignore", category=UserWarning, module="torchaudio._backend.utils"
)

# suppress pyannote TF32 reproducibility warning (we intentionally enable TF32).
warnings.filterwarnings(
    "ignore", category=UserWarning, module="pyannote.audio.utils.reproducibility"
)

# suppress pyannote torchcodec warning (we use torchaudio + ffmpeg fallback instead).
warnings.filterwarnings(
    "ignore", category=UserWarning, module="pyannote.audio.core.io"
)

logger = logging.getLogger(__name__)


def is_apple_silicon() -> bool:
    """check if running on Apple Silicon."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def detect_diarization_device() -> str:
    """detect the best available device for diarization."""
    # check for CUDA first.
    if torch.cuda.is_available():
        return "cuda:0"

    # check for MPS (Apple Silicon).
    if is_apple_silicon() and torch.backends.mps.is_available():
        return "mps"

    # fallback to cpu.
    return "cpu"


def get_diarization_device_info() -> dict[str, Any]:
    """get information about available diarization devices."""
    info = {
        "is_apple_silicon": is_apple_silicon(),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False,
        "recommended_device": detect_diarization_device(),
    }

    if info["cuda_available"]:
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    return info


class DiarizationPipeline:
    """wrapper for pyannote.audio diarization with GPU optimizations.

    Supports CUDA (NVIDIA), MPS (Apple Silicon), and CPU backends.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_DIARIZATION_MODEL,
        hf_token: str | None = None,
        device: str = "auto",
        use_bf16: bool = True,
        use_compile: bool = False,
        compile_mode: str = "reduce-overhead",
    ):
        """
        initialize diarization pipeline with optimizations.

        Args:
            model_name: HuggingFace model identifier.
            hf_token: HuggingFace API token for model access.
            device: device to use (auto, cuda:0, mps, cpu).
            use_bf16: enable BFloat16 mixed precision (CUDA only, H100/A100).
            use_compile: enable torch.compile optimization (CUDA only).
            compile_mode: compilation mode (default, reduce-overhead, max-autotune).
        """
        self.model_name = model_name

        # resolve device.
        if device == "auto":
            self.device = detect_diarization_device()
            logger.info(f"auto-detected device: {self.device}")
        else:
            self.device = device

        # determine device type for optimization decisions.
        self.device_type = self._get_device_type()

        # disable CUDA-only optimizations on non-CUDA devices.
        if self.device_type != "cuda":
            if use_bf16:
                logger.info("bf16 disabled (only available on CUDA)")
            if use_compile:
                logger.info("torch.compile disabled (only available on CUDA)")
            use_bf16 = False
            use_compile = False

        self.use_bf16 = use_bf16
        self.use_compile = use_compile
        self.compile_mode = compile_mode

        logger.info(f"initializing diarization pipeline: {model_name}")
        logger.info(f"device: {self.device}, bf16: {use_bf16}, compile: {use_compile}")

        self._load_pipeline(hf_token)
        self._apply_optimizations()

    def _get_device_type(self) -> str:
        """get the device type (cuda, mps, or cpu)."""
        if self.device.startswith("cuda"):
            return "cuda"
        elif self.device == "mps":
            return "mps"
        else:
            return "cpu"

    def _load_pipeline(self, hf_token: str | None) -> None:
        """load pyannote.audio pipeline from HuggingFace."""
        from pyannote.audio import Pipeline

        try:
            # pyannote.audio 4.x uses 'token', 3.x uses 'use_auth_token'.
            try:
                self.pipeline = Pipeline.from_pretrained(
                    self.model_name,
                    token=hf_token,
                )
            except TypeError:
                self.pipeline = Pipeline.from_pretrained(
                    self.model_name,
                    use_auth_token=hf_token,
                )
            self.pipeline.to(torch.device(self.device))
            logger.info("pipeline loaded successfully")
        except Exception as e:
            raise RuntimeError(f"failed to load pipeline {self.model_name}: {e}") from e

    def _apply_optimizations(self) -> None:
        """apply device-specific optimizations."""
        # check if BFloat16 is actually supported on this device.
        self.bf16_available = (
            self.use_bf16
            and self.device_type == "cuda"
            and torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        )

        if self.bf16_available:
            logger.info("enabling BFloat16 mixed precision")
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32
            if self.use_bf16 and self.device_type == "cuda":
                logger.warning("BFloat16 not supported on this device, using FP32")

        # apply CUDA-specific optimizations.
        if self.device_type == "cuda" and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            logger.info("enabled TF32 and cuDNN optimizations")

        # apply MPS-specific optimizations.
        if self.device_type == "mps":
            # MPS doesn't support BFloat16 well, ensure FP32.
            self.dtype = torch.float32
            logger.info("using MPS (Apple Silicon GPU) with FP32 precision")

        # torch.compile only works reliably on CUDA.
        if self.use_compile and self.device_type == "cuda":
            logger.info("compiling model (this may take 1-2 minutes on first run)...")
            try:
                self.pipeline = torch.compile(
                    self.pipeline,
                    mode=self.compile_mode,
                    backend="inductor",
                )
                logger.info("model compilation enabled")
            except Exception as e:
                logger.warning(f"torch.compile failed, continuing without: {e}")
                self.use_compile = False

    def _load_audio_with_ffmpeg(self, audio_path: Path) -> dict:
        """
        load audio using ffmpeg as fallback when torchaudio fails.

        Args:
            audio_path: path to audio file.

        Returns:
            dict with 'waveform' (torch.Tensor) and 'sample_rate' (int).
        """
        logger.info(f"using ffmpeg fallback to load {audio_path.name}")

        try:
            # use ffmpeg to decode audio to raw PCM at 16kHz mono.
            cmd = [
                "ffmpeg",
                "-i",
                str(audio_path),
                "-f",
                "f32le",  # 32-bit float PCM
                "-acodec",
                "pcm_f32le",
                "-ar",
                "16000",  # sample rate
                "-ac",
                "1",  # mono
                "-hide_banner",
                "-loglevel",
                "error",
                "pipe:1",  # output to stdout
            ]

            result = subprocess.run(cmd, capture_output=True, check=True)

            # convert raw PCM bytes to numpy array, then to torch tensor.
            audio_np = np.frombuffer(result.stdout, dtype=np.float32)
            # copy array to make it writable before converting to tensor.
            waveform = torch.from_numpy(audio_np.copy()).unsqueeze(0)  # add channel dim

            return {"waveform": waveform, "sample_rate": 16000}

        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"ffmpeg failed to load {audio_path}: {e.stderr.decode()}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"failed to process ffmpeg output: {e}") from e

    def _load_audio(self, audio_path: Path) -> dict:
        """
        load and preprocess audio file using torchaudio with ffmpeg fallback.

        tries torchaudio first for performance, falls back to ffmpeg if that fails
        (e.g., due to PyTorch version issues or corrupted files).

        Args:
            audio_path: path to audio file.

        Returns:
            dict with 'waveform' (torch.Tensor) and 'sample_rate' (int).

        Note:
            libmpg123 may print "dequantization failed" warnings for some MP3 files.
            these are usually non-fatal and can be ignored if audio loads successfully.
        """
        try:
            waveform, sample_rate = torchaudio.load(str(audio_path))

            # convert stereo to mono if needed.
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # resample to 16kHz if needed (pyannote expects 16kHz).
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=16000
                )
                waveform = resampler(waveform)
                sample_rate = 16000

            return {"waveform": waveform, "sample_rate": sample_rate}

        except Exception as e:
            # torchaudio failed, try ffmpeg fallback.
            logger.warning(f"torchaudio failed to load {audio_path.name}: {e}")
            try:
                return self._load_audio_with_ffmpeg(audio_path)
            except Exception as ffmpeg_error:
                raise RuntimeError(
                    f"failed to load audio {audio_path} with both torchaudio and ffmpeg. "
                    f"torchaudio error: {e}. ffmpeg error: {ffmpeg_error}"
                ) from ffmpeg_error

    def process_file(
        self,
        audio_path: Path,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        retry_count: int = 0,
    ) -> Any:
        """
        process single audio file and return diarization.

        Args:
            audio_path: path to audio file.
            min_speakers: minimum number of speakers (optional hint).
            max_speakers: maximum number of speakers (optional hint).
            retry_count: current retry attempt (for OOM handling).

        Returns:
            pyannote.core.Annotation object with speaker diarization.

        Raises:
            MemoryError: if GPU OOM after retries.
            RuntimeError: if processing fails.
        """
        try:
            start_time = time.time()

            audio_data = self._load_audio(audio_path)

            # use autocast for mixed precision if BFloat16 is available (CUDA only).
            # MPS doesn't support autocast well, so we skip it.
            if self.device_type == "cuda" and self.bf16_available:
                with torch.amp.autocast(
                    device_type="cuda", dtype=self.dtype, enabled=True
                ):
                    diarization = self.pipeline(
                        audio_data,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                    )
            else:
                # MPS and CPU: run without autocast.
                diarization = self.pipeline(
                    audio_data,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                )

            elapsed = time.time() - start_time
            logger.info(f"processed {audio_path.name} in {elapsed:.2f}s")

            return diarization

        except torch.cuda.OutOfMemoryError:
            if retry_count < 2:
                logger.warning(
                    f"GPU OOM on {audio_path.name}, clearing cache and retrying..."
                )
                torch.cuda.empty_cache()
                time.sleep(1)
                return self.process_file(audio_path, min_speakers, max_speakers, retry_count + 1)
            else:
                raise MemoryError(
                    f"GPU OOM after {retry_count} retries on {audio_path}"
                ) from None

        except RuntimeError as e:
            # handle BFloat16 not supported error.
            if "unsupported ScalarType BFloat16" in str(e) or "BFloat16" in str(e):
                if self.bf16_available and retry_count == 0:
                    logger.warning(
                        f"BFloat16 not supported on this GPU, falling back to FP32 for {audio_path.name}"
                    )
                    # disable BFloat16 for this instance.
                    self.bf16_available = False
                    self.dtype = torch.float32
                    # retry with FP32.
                    torch.cuda.empty_cache()
                    return self.process_file(audio_path, min_speakers, max_speakers, retry_count + 1)
            raise RuntimeError(f"failed to process {audio_path}: {e}") from e

        except Exception as e:
            raise RuntimeError(f"failed to process {audio_path}: {e}") from e
