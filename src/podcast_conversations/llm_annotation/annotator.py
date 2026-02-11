"""LLM annotator with local GPU inference and dynamic batching."""

import json
import logging
import os
import platform
from dataclasses import dataclass
from enum import Enum
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import torch

from .config import AnnotationConfig

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Supported device types for inference."""

    CUDA = "cuda"
    MPS = "mps"  # apple silicon.
    CPU = "cpu"


@dataclass
class DeviceInfo:
    """Information about the compute device."""

    device_type: DeviceType
    device_str: str
    name: str
    total_memory_gb: float
    available_memory_gb: float
    is_apple_silicon: bool = False
    compute_capability: tuple[int, int] | None = None  # for CUDA devices.


def detect_device() -> DeviceInfo:
    """Detect the best available compute device.

    Returns:
        DeviceInfo with device details and memory information.
    """
    # check for CUDA first (highest priority for ML workloads).
    if torch.cuda.is_available():
        device_idx = 0
        props = torch.cuda.get_device_properties(device_idx)
        total_mem = props.total_memory / (1024**3)
        # get available memory.
        free_mem, _ = torch.cuda.mem_get_info(device_idx)
        available_mem = free_mem / (1024**3)

        return DeviceInfo(
            device_type=DeviceType.CUDA,
            device_str=f"cuda:{device_idx}",
            name=props.name,
            total_memory_gb=total_mem,
            available_memory_gb=available_mem,
            compute_capability=(props.major, props.minor),
        )

    # check for Apple Silicon MPS.
    if torch.backends.mps.is_available():
        # get system memory info for unified memory architecture.
        total_mem, available_mem = _get_apple_silicon_memory()
        chip_name = _get_apple_chip_name()

        return DeviceInfo(
            device_type=DeviceType.MPS,
            device_str="mps",
            name=chip_name,
            total_memory_gb=total_mem,
            available_memory_gb=available_mem,
            is_apple_silicon=True,
        )

    # fallback to CPU.
    total_mem, available_mem = _get_system_memory()
    return DeviceInfo(
        device_type=DeviceType.CPU,
        device_str="cpu",
        name=platform.processor() or "CPU",
        total_memory_gb=total_mem,
        available_memory_gb=available_mem,
    )


def _get_apple_silicon_memory() -> tuple[float, float]:
    """Get memory info for Apple Silicon unified memory.

    Returns:
        Tuple of (total_gb, available_gb).
    """
    try:
        import psutil

        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        available_gb = mem.available / (1024**3)
        return total_gb, available_gb
    except ImportError:
        # fallback: try to get from sysctl on macOS.
        try:
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                check=True,
            )
            total_bytes = int(result.stdout.strip())
            total_gb = total_bytes / (1024**3)
            # estimate 70% available without psutil.
            return total_gb, total_gb * 0.7
        except Exception:
            return 16.0, 12.0  # conservative default.


def _get_system_memory() -> tuple[float, float]:
    """Get system memory info.

    Returns:
        Tuple of (total_gb, available_gb).
    """
    try:
        import psutil

        mem = psutil.virtual_memory()
        return mem.total / (1024**3), mem.available / (1024**3)
    except ImportError:
        return 16.0, 8.0  # conservative default.


def _get_apple_chip_name() -> str:
    """Get the Apple Silicon chip name."""
    try:
        import subprocess

        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "Apple Silicon"


def is_flash_attention_available() -> bool:
    """Check if Flash Attention 2 is installed and available."""
    return find_spec("flash_attn") is not None


def is_mlx_available() -> bool:
    """Check if MLX (Apple's ML framework) is available."""
    return find_spec("mlx") is not None and find_spec("mlx_lm") is not None


def get_mlx_quantization_info() -> dict[str, Any]:
    """Get information about MLX quantization support.

    MLX supports efficient quantization on Apple Silicon:
    - 4-bit quantization (similar to bitsandbytes NF4)
    - 8-bit quantization
    - Native unified memory optimization

    Returns:
        Dict with MLX quantization capabilities.
    """
    info = {
        "available": is_mlx_available(),
        "supports_4bit": True,
        "supports_8bit": True,
        "install_command": "pip install mlx mlx-lm",
        "quantize_command": "mlx_lm.convert --hf-path <model> -q --q-bits 4",
        "notes": [
            "MLX is Apple's ML framework optimized for Apple Silicon",
            "mlx-lm provides LLM inference with native quantization",
            "4-bit quantization significantly reduces memory usage",
            "Models can be pre-quantized or quantized on-the-fly",
            "Unified memory allows larger models than discrete GPUs with same RAM",
        ],
    }
    return info


def detect_optimal_batch_size(
    gpu_memory_gb: float = 80.0,
    model_memory_gb: float = 0.0,
    use_available: bool = True,
    device_type: DeviceType = DeviceType.CUDA,
) -> int:
    """
    Detect optimal batch size based on GPU memory.

    For Llama 3.3 70B with 4-bit quantization:
    - Model takes ~35-40GB
    - Each batch item needs ~2.5-3GB for KV cache (depends on sequence length)
    - We target 60-70% GPU utilization to leave headroom for fragmentation

    For Apple Silicon (MPS):
    - Uses unified memory (shared with system)
    - More conservative memory allocation needed
    - Smaller models recommended (7B-13B)

    Args:
        gpu_memory_gb: Total GPU memory in GB
        model_memory_gb: Memory already used by model (0 = estimate from total)
        use_available: If True, calculate based on available memory after model
        device_type: Type of compute device (CUDA, MPS, CPU)

    Returns:
        Recommended batch size
    """
    # apple silicon uses unified memory - be more conservative.
    if device_type == DeviceType.MPS:
        return _detect_mps_batch_size(gpu_memory_gb, model_memory_gb, use_available)

    if use_available and model_memory_gb > 0:
        # calculate available memory for batching.
        available_gb = gpu_memory_gb - model_memory_gb
        # leave 20% headroom for fragmentation and PyTorch reserved memory.
        usable_gb = available_gb * 0.6
        # estimate ~3GB per batch item for KV cache with 70B model.
        # this is conservative to avoid OOM during generation.
        batch_size = max(1, int(usable_gb / 3.0))
        return batch_size

    # fallback to total memory based estimation.
    # calculate dynamically: assume ~35GB for model, use 60% of remaining for batching.
    if gpu_memory_gb >= 40:
        available_for_batch = (gpu_memory_gb - 35) * 0.6
        batch_size = max(1, int(available_for_batch / 3.0))
        return batch_size
    elif gpu_memory_gb >= 24:
        # medium VRAM - estimate ~20GB for model.
        available_for_batch = (gpu_memory_gb - 20) * 0.6
        return max(1, int(available_for_batch / 3.0))
    elif gpu_memory_gb >= 16:
        # limited VRAM.
        return 1
    else:
        return 1


def _detect_mps_batch_size(
    total_memory_gb: float,
    model_memory_gb: float = 0.0,
    use_available: bool = True,
) -> int:
    """Detect optimal batch size for Apple Silicon MPS.

    Apple Silicon uses unified memory shared between CPU and GPU.
    We need to be more conservative to leave memory for the OS and other apps.

    Args:
        total_memory_gb: Total unified memory in GB.
        model_memory_gb: Estimated memory used by model.
        use_available: If True, calculate based on available memory.

    Returns:
        Recommended batch size for MPS.
    """
    if use_available and model_memory_gb > 0:
        available_gb = total_memory_gb - model_memory_gb
        # be very conservative on MPS - only use 40% of available.
        # unified memory is shared with system, so we need more headroom.
        usable_gb = available_gb * 0.40
        # MPS memory bandwidth is lower, use larger per-item estimate.
        gb_per_item = 4.0
        batch_size = max(1, int(usable_gb / gb_per_item))
        return batch_size

    # fallback based on total unified memory.
    # calculate dynamically: assume ~15GB for model on smaller models, use 40% of remaining.
    if total_memory_gb >= 64:
        available_for_batch = (total_memory_gb - 15) * 0.40
        return max(1, int(available_for_batch / 4.0))
    elif total_memory_gb >= 32:
        available_for_batch = (total_memory_gb - 12) * 0.40
        return max(1, int(available_for_batch / 4.5))
    elif total_memory_gb >= 16:
        return 1
    else:
        return 1


def detect_optimal_io_workers(cpu_count: int | None = None) -> int:
    """
    Detect optimal number of I/O workers based on CPU cores.

    I/O workers handle file loading/saving in parallel with GPU compute.
    Too few = I/O becomes bottleneck. Too many = context switching overhead.

    Args:
        cpu_count: Number of CPU cores (auto-detected if None)

    Returns:
        Recommended number of I/O workers
    """
    import os

    if cpu_count is None:
        cpu_count = os.cpu_count() or 4

    # scale I/O workers: use ~10-15% of cores, min 2, max 8.
    # I/O is not CPU-intensive, so we don't need many workers.
    workers = max(2, min(8, cpu_count // 8))
    return workers


def detect_optimal_prefetch(
    available_ram_gb: float | None = None,
    avg_file_size_mb: float = 1.0,
) -> int:
    """
    Detect optimal prefetch count based on available RAM.

    Prefetching loads files ahead of processing to minimize I/O wait.
    More prefetch = less GPU idle time, but uses more RAM.

    Args:
        available_ram_gb: Available RAM in GB (auto-detected if None)
        avg_file_size_mb: Estimated average file size in MB

    Returns:
        Recommended prefetch count
    """
    if available_ram_gb is None:
        try:
            import psutil
            available_ram_gb = psutil.virtual_memory().available / (1024**3)
        except ImportError:
            available_ram_gb = 8.0  # conservative default

    # allocate up to 5% of available RAM for prefetch buffer.
    prefetch_budget_gb = available_ram_gb * 0.05
    prefetch_budget_mb = prefetch_budget_gb * 1024

    # calculate prefetch count based on file size.
    prefetch_count = max(2, int(prefetch_budget_mb / avg_file_size_mb))

    # cap at reasonable limits.
    return min(prefetch_count, 32)


class BatchConfig:
    """Configuration for batch processing with resource-based recommendations.

    Attributes:
        batch_size: Number of segments per GPU batch.
        total_segments: Total segments to process.
        total_batches: Calculated total number of batches.
        gpu_memory_gb: Total GPU memory in GB.
        available_memory_gb: Available GPU memory after model loading.
        recommended_batch_size: System-recommended batch size.
        recommendation_reason: Explanation for the recommendation.
    """

    def __init__(
        self,
        batch_size: int,
        total_segments: int,
        gpu_memory_gb: float,
        available_memory_gb: float,
        recommended_batch_size: int,
        recommendation_reason: str,
    ):
        self.batch_size = batch_size
        self.total_segments = total_segments
        self.gpu_memory_gb = gpu_memory_gb
        self.available_memory_gb = available_memory_gb
        self.recommended_batch_size = recommended_batch_size
        self.recommendation_reason = recommendation_reason

    @property
    def total_batches(self) -> int:
        """Calculate total number of batches needed."""
        if self.batch_size <= 0:
            return self.total_segments
        return (self.total_segments + self.batch_size - 1) // self.batch_size

    @property
    def recommended_total_batches(self) -> int:
        """Calculate total batches with recommended batch size."""
        if self.recommended_batch_size <= 0:
            return self.total_segments
        return (self.total_segments + self.recommended_batch_size - 1) // self.recommended_batch_size

    def get_batch_options(self) -> list[dict]:
        """Get a list of batch size options with trade-offs.

        Returns:
            List of dicts with 'size', 'batches', 'label', and 'description'.
        """
        options = []

        # calculate max batch size dynamically based on available memory.
        # no hardcoded limits - scales with available resources.
        if self.available_memory_gb >= 40:
            # high memory: max aggressive batch = 2x recommended.
            absolute_max = self.recommended_batch_size * 2
        elif self.available_memory_gb >= 25:
            # medium-high memory: max aggressive batch = 1.75x recommended.
            absolute_max = int(self.recommended_batch_size * 1.75)
        elif self.available_memory_gb >= 15:
            # medium memory: max aggressive batch = 1.5x recommended.
            absolute_max = int(self.recommended_batch_size * 1.5)
        else:
            # limited memory: cap at 1.25x recommended for safety.
            absolute_max = int(self.recommended_batch_size * 1.25)

        # ensure minimum of 1.
        absolute_max = max(1, absolute_max)

        # conservative option (smaller batches, safer).
        conservative_size = max(1, self.recommended_batch_size // 2)
        conservative_batches = (self.total_segments + conservative_size - 1) // conservative_size
        options.append({
            "size": conservative_size,
            "batches": conservative_batches,
            "label": "Conservative",
            "description": "Smaller batches, lower memory usage, more stable",
        })

        # recommended option.
        options.append({
            "size": self.recommended_batch_size,
            "batches": self.recommended_total_batches,
            "label": "Recommended",
            "description": self.recommendation_reason,
        })

        # aggressive option (larger batches, faster but higher memory).
        aggressive_size = min(int(self.recommended_batch_size * 1.5), absolute_max)
        if aggressive_size > self.recommended_batch_size:
            aggressive_batches = (self.total_segments + aggressive_size - 1) // aggressive_size
            options.append({
                "size": aggressive_size,
                "batches": aggressive_batches,
                "label": "Aggressive",
                "description": "Larger batches, faster processing, higher memory usage",
            })

        # maximum option (highest throughput, OOM risk).
        max_size = absolute_max
        if max_size > aggressive_size:
            max_batches = (self.total_segments + max_size - 1) // max_size
            options.append({
                "size": max_size,
                "batches": max_batches,
                "label": "Maximum",
                "description": f"Maximum throughput ({max_size} batch), OOM risk on long sequences",
            })

        return options


def calculate_optimal_batch_config(
    total_segments: int,
    gpu_memory_gb: float | None = None,
    model_memory_gb: float | None = None,
    current_batch_size: int | None = None,
    device_info: DeviceInfo | None = None,
) -> BatchConfig:
    """Calculate optimal batch configuration based on available resources.

    Analyzes GPU memory, total workload, and provides batch size recommendations
    with explanations for the user to make an informed choice.

    Supports CUDA GPUs, Apple Silicon (MPS), and CPU.

    Args:
        total_segments: Total number of segments to process.
        gpu_memory_gb: Total GPU memory in GB (auto-detected if None).
        model_memory_gb: Memory used by loaded model (auto-detected if None).
        current_batch_size: Currently configured batch size (if any).
        device_info: Device information (auto-detected if None).

    Returns:
        BatchConfig with recommendations and options.
    """
    # auto-detect device if not provided.
    if device_info is None:
        device_info = detect_device()

    # use device-specific memory detection.
    detected_gpu_memory: float = 0.0
    detected_model_memory: float = 0.0

    if gpu_memory_gb is not None:
        detected_gpu_memory = gpu_memory_gb
    else:
        detected_gpu_memory = device_info.total_memory_gb

    if model_memory_gb is not None:
        detected_model_memory = model_memory_gb
    elif device_info.device_type == DeviceType.CUDA and torch.cuda.is_available():
        detected_model_memory = torch.cuda.memory_allocated() / (1024**3)
    elif device_info.device_type == DeviceType.MPS:
        # MPS doesn't have direct memory tracking - estimate based on model size.
        detected_model_memory = 0.0  # will use conservative defaults.

    available_memory_gb = max(0.0, detected_gpu_memory - detected_model_memory)

    # apple silicon specific batch configuration.
    if device_info.device_type == DeviceType.MPS:
        return _calculate_mps_batch_config(
            total_segments=total_segments,
            total_memory_gb=detected_gpu_memory,
            available_memory_gb=available_memory_gb,
            current_batch_size=current_batch_size,
        )

    # CUDA/CPU batch configuration.
    # calculate recommended batch size dynamically based on available memory.
    # no hardcoded limits - scales automatically with available resources.
    if available_memory_gb >= 40:
        # large memory (H100 with 70B 4-bit model): use 70% of available, ~2.5GB per item.
        usable_gb = available_memory_gb * 0.70
        gb_per_item = 2.5
        recommended_size = max(1, int(usable_gb / gb_per_item))
        reason = f"Aggressive for {available_memory_gb:.1f}GB available (high throughput)"
    elif available_memory_gb >= 25:
        # medium-large memory: use 65% of available, ~2.8GB per item.
        usable_gb = available_memory_gb * 0.65
        gb_per_item = 2.8
        recommended_size = max(1, int(usable_gb / gb_per_item))
        reason = f"Optimal for {available_memory_gb:.1f}GB available memory"
    elif available_memory_gb >= 15:
        # medium memory: use 60% of available, ~3GB per item.
        usable_gb = available_memory_gb * 0.60
        gb_per_item = 3.0
        recommended_size = max(1, int(usable_gb / gb_per_item))
        reason = f"Balanced for {available_memory_gb:.1f}GB available memory"
    elif available_memory_gb > 0:
        # limited memory: conservative settings.
        usable_gb = available_memory_gb * 0.55
        gb_per_item = 3.5
        recommended_size = max(1, int(usable_gb / gb_per_item))
        reason = f"Conservative for limited {available_memory_gb:.1f}GB memory"
    else:
        # fallback based on total GPU memory.
        recommended_size = detect_optimal_batch_size(detected_gpu_memory, use_available=False)
        reason = f"Estimated for {detected_gpu_memory:.1f}GB total GPU memory"

    # use current batch size if provided, otherwise use recommended.
    batch_size = current_batch_size if current_batch_size is not None else recommended_size

    return BatchConfig(
        batch_size=batch_size,
        total_segments=total_segments,
        gpu_memory_gb=detected_gpu_memory,
        available_memory_gb=available_memory_gb,
        recommended_batch_size=recommended_size,
        recommendation_reason=reason,
    )


def _calculate_mps_batch_config(
    total_segments: int,
    total_memory_gb: float,
    available_memory_gb: float,
    current_batch_size: int | None = None,
) -> BatchConfig:
    """Calculate batch configuration optimized for Apple Silicon MPS.

    Apple Silicon uses unified memory architecture where GPU and CPU share
    the same memory pool. This requires more conservative allocation.

    Args:
        total_segments: Total segments to process.
        total_memory_gb: Total unified memory.
        available_memory_gb: Available memory after model loading.
        current_batch_size: User-specified batch size (if any).

    Returns:
        BatchConfig optimized for MPS.
    """
    # MPS batch sizes need to be smaller due to:
    # 1. unified memory is shared with OS and apps.
    # 2. memory bandwidth is lower than dedicated GPUs.
    # 3. MPS backend has different memory management.

    if total_memory_gb >= 128:
        # M2/M3 Ultra with 128GB+.
        usable_gb = available_memory_gb * 0.50
        gb_per_item = 3.5
        recommended_size = max(1, int(usable_gb / gb_per_item))
        reason = f"MPS optimized for {total_memory_gb:.0f}GB unified memory (Ultra)"
    elif total_memory_gb >= 64:
        # M2/M3 Max with 64GB.
        usable_gb = available_memory_gb * 0.45
        gb_per_item = 4.0
        recommended_size = max(1, int(usable_gb / gb_per_item))
        reason = f"MPS optimized for {total_memory_gb:.0f}GB unified memory (Max)"
    elif total_memory_gb >= 32:
        # M2/M3 Pro with 32GB.
        usable_gb = available_memory_gb * 0.40
        gb_per_item = 4.5
        recommended_size = max(1, int(usable_gb / gb_per_item))
        reason = f"MPS optimized for {total_memory_gb:.0f}GB unified memory (Pro)"
    else:
        # Base M1/M2/M3 with 8-16GB.
        recommended_size = 1
        reason = f"MPS conservative for {total_memory_gb:.0f}GB unified memory"

    batch_size = current_batch_size if current_batch_size is not None else recommended_size

    return BatchConfig(
        batch_size=batch_size,
        total_segments=total_segments,
        gpu_memory_gb=total_memory_gb,
        available_memory_gb=available_memory_gb,
        recommended_batch_size=recommended_size,
        recommendation_reason=reason,
    )


def check_annotations_exist(output_path: Path) -> bool:
    """
    Check if annotations file exists and has valid LLM annotations.

    Returns True if file exists and contains llm_annotation for all segments.
    """
    if not output_path.exists():
        return False

    try:
        with open(output_path, encoding="utf-8") as f:
            data = json.load(f)

        segments = data.get("segments", [])
        if not segments:
            return False

        # check if all segments have llm_annotation.
        for segment in segments:
            if "llm_annotation" not in segment:
                return False
            annotation = segment["llm_annotation"]
            # verify annotation has required fields.
            if not isinstance(annotation, dict):
                return False
            if "has_hate_speech" not in annotation:
                return False

        return True

    except Exception:
        return False


class LLMAnnotator:
    """
    LLM-based annotator with local GPU inference.

    Optimized for multiple platforms:
    - NVIDIA GPUs (CUDA): 4-bit/8-bit quantization, Flash Attention 2
    - Apple Silicon (MPS): Unified memory optimization, float16/bfloat16
    - CPU: Fallback for systems without GPU

    For H100 80GB with Llama 3.3 70B using:
    - 4-bit quantization (bitsandbytes)
    - Flash Attention 2
    - Dynamic batching
    - KV cache optimization

    For Apple Silicon (M1/M2/M3):
    - Smaller models recommended (7B-13B)
    - No quantization (bitsandbytes not supported on MPS)
    - Optimized for unified memory architecture
    """

    def __init__(
        self,
        config: AnnotationConfig,
        device: str | None = None,
        batch_size: int | None = None,
        use_flash_attention: bool | None = None,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        hf_token: str | None = None,
        auto_optimize: bool = True,
        use_torch_compile: bool = True,
    ):
        """
        Initialize LLM annotator with local model.

        Args:
            config: Annotation configuration
            device: Device to run on ("cuda", "mps", "cpu", or None for auto-detect)
            batch_size: Batch size for inference (auto-detected if None)
            use_flash_attention: Use Flash Attention 2 for faster inference (auto-detected if None)
            load_in_4bit: Load model in 4-bit precision (CUDA only, recommended for 70B on 80GB)
            load_in_8bit: Load model in 8-bit precision (CUDA only)
            hf_token: HuggingFace token for gated models (Llama 3.3 requires access)
            auto_optimize: Auto-optimize batch size after model loading based on actual memory
            use_torch_compile: Use torch.compile() for faster inference (CUDA only, ~10-30% speedup)
        """
        self.config = config
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.auto_optimize = auto_optimize
        self.use_torch_compile = use_torch_compile
        self._user_batch_size = batch_size  # store user's explicit choice.

        # auto-detect device if not specified.
        if device is None:
            self.device_info = detect_device()
            self.device = self.device_info.device_str
        else:
            self.device = device
            # create device info based on specified device.
            if device.startswith("cuda"):
                self.device_info = detect_device()  # will detect CUDA.
            elif device == "mps":
                self.device_info = DeviceInfo(
                    device_type=DeviceType.MPS,
                    device_str="mps",
                    name=_get_apple_chip_name(),
                    total_memory_gb=_get_apple_silicon_memory()[0],
                    available_memory_gb=_get_apple_silicon_memory()[1],
                    is_apple_silicon=True,
                )
            else:
                mem_total, mem_avail = _get_system_memory()
                self.device_info = DeviceInfo(
                    device_type=DeviceType.CPU,
                    device_str="cpu",
                    name="CPU",
                    total_memory_gb=mem_total,
                    available_memory_gb=mem_avail,
                )

        # handle MPS-specific limitations.
        if self.device_info.device_type == DeviceType.MPS:
            # bitsandbytes quantization not supported on MPS.
            if load_in_4bit or load_in_8bit:
                logger.warning(
                    "Quantization (4-bit/8-bit) is not supported on Apple Silicon MPS. "
                    "Using float16 instead. Consider using smaller models (7B-13B)."
                )
            self.load_in_4bit = False
            self.load_in_8bit = False
            # flash attention not supported on MPS.
            self.use_flash_attention = False
            if use_flash_attention:
                logger.warning(
                    "Flash Attention 2 is not supported on Apple Silicon MPS. "
                    "Using default attention implementation."
                )
        else:
            self.load_in_4bit = load_in_4bit
            self.load_in_8bit = load_in_8bit

            # auto-detect flash attention availability for CUDA.
            if use_flash_attention is None:
                self.use_flash_attention = (
                    is_flash_attention_available()
                    and self.device_info.device_type == DeviceType.CUDA
                )
                if self.use_flash_attention:
                    logger.info("Flash Attention 2 detected and will be used")
                else:
                    logger.info("Flash Attention 2 not available, using default attention")
            else:
                self.use_flash_attention = (
                    use_flash_attention
                    and is_flash_attention_available()
                    and self.device_info.device_type == DeviceType.CUDA
                )
                if use_flash_attention and not self.use_flash_attention:
                    logger.warning(
                        "Flash Attention 2 requested but not available. "
                        "Install with: pip install flash-attn --no-build-isolation"
                    )

        # initial batch size estimate (will be refined after model loading).
        if batch_size is None:
            self.batch_size = detect_optimal_batch_size(
                gpu_memory_gb=self.device_info.total_memory_gb,
                use_available=False,
                device_type=self.device_info.device_type,
            )
            logger.info(
                f"Initial batch size estimate: {self.batch_size} for "
                f"{self.device_info.total_memory_gb:.1f}GB on {self.device_info.name}"
            )
        else:
            self.batch_size = batch_size

        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load_model(self) -> None:
        """Load the model and tokenizer."""
        if self._loaded:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required. "
                "Install with: pip install transformers accelerate"
            )

        # import bitsandbytes config only if needed (CUDA only).
        BitsAndBytesConfig = None
        if self.load_in_4bit or self.load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig as BnBConfig

                BitsAndBytesConfig = BnBConfig
            except ImportError:
                logger.warning(
                    "bitsandbytes not available. "
                    "Install with: pip install bitsandbytes"
                )
                self.load_in_4bit = False
                self.load_in_8bit = False

        logger.info(f"Loading model: {self.config.model_name}")
        logger.info(f"Device: {self.device} ({self.device_info.name})")

        # configure quantization (CUDA only).
        quantization_config = None
        if self.load_in_4bit and BitsAndBytesConfig:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            logger.info("Using 4-bit quantization (NF4)")
        elif self.load_in_8bit and BitsAndBytesConfig:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            logger.info("Using 8-bit quantization")

        # load tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            token=self.hf_token,
        )

        # set padding token if not set.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # use left-padding for decoder-only models (required for correct batched generation).
        self.tokenizer.padding_side = "left"

        # configure model loading based on device type.
        model_kwargs = self._get_model_loading_kwargs(quantization_config)

        # load model.
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs,
        )

        self.model.eval()
        self._loaded = True

        # apply torch.compile() for faster inference (CUDA only).
        self._apply_torch_compile()

        # log memory usage and optimize batch size.
        self._post_load_optimization()

    def _get_model_loading_kwargs(self, quantization_config: Any) -> dict[str, Any]:
        """Get model loading kwargs based on device type.

        Args:
            quantization_config: Quantization config (CUDA only).

        Returns:
            Dict of kwargs for from_pretrained().
        """
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "token": self.hf_token,
        }

        if self.device_info.device_type == DeviceType.CUDA:
            # CUDA: use device_map auto, quantization, flash attention.
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = torch.bfloat16

            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config

            if self.use_flash_attention:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Using Flash Attention 2")

        elif self.device_info.device_type == DeviceType.MPS:
            # MPS: Apple Silicon specific settings.
            # no device_map auto (not well supported), no quantization.
            # use float16 for better MPS compatibility.
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = {"": self.device}
            # enable low_cpu_mem_usage for faster loading on unified memory.
            model_kwargs["low_cpu_mem_usage"] = True

            logger.info("Using float16 for Apple Silicon MPS")
            logger.info(
                "Note: For large models, consider using MLX or llama.cpp for "
                "better Apple Silicon performance"
            )

        else:
            # CPU fallback.
            model_kwargs["device_map"] = "cpu"
            model_kwargs["torch_dtype"] = torch.float32
            model_kwargs["low_cpu_mem_usage"] = True

            logger.warning(
                "Running on CPU - inference will be slow. "
                "Consider using a GPU or Apple Silicon Mac."
            )

        return model_kwargs

    def _apply_torch_compile(self) -> None:
        """Apply torch.compile() for faster inference on CUDA."""
        if not self.use_torch_compile:
            return

        # torch.compile() is only beneficial on CUDA.
        if self.device_info.device_type != DeviceType.CUDA:
            if self.use_torch_compile:
                logger.info(
                    "torch.compile() skipped: only beneficial on CUDA devices"
                )
            return

        # check PyTorch version (torch.compile requires 2.0+).
        torch_version = tuple(int(x) for x in torch.__version__.split(".")[:2])
        if torch_version < (2, 0):
            logger.warning(
                f"torch.compile() requires PyTorch 2.0+, found {torch.__version__}"
            )
            return

        # torch.compile() may not work well with quantized models.
        if self.load_in_4bit or self.load_in_8bit:
            logger.info(
                "torch.compile() with quantization: using 'reduce-overhead' mode"
            )
            compile_mode = "reduce-overhead"
        else:
            compile_mode = "reduce-overhead"

        try:
            logger.info(f"Applying torch.compile() with mode='{compile_mode}'...")
            self.model = torch.compile(self.model, mode=compile_mode)
            logger.info("torch.compile() applied successfully (speedup on first batch)")
        except Exception as e:
            logger.warning(f"torch.compile() failed, continuing without: {e}")

    def _post_load_optimization(self) -> None:
        """Optimize settings after model is loaded."""
        if self.device_info.device_type == DeviceType.CUDA:
            # CUDA: log GPU memory and optimize batch size.
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

            # auto-optimize batch size based on actual available memory.
            if self.auto_optimize and self._user_batch_size is None:
                old_batch_size = self.batch_size
                self.batch_size = detect_optimal_batch_size(
                    gpu_memory_gb=total,
                    model_memory_gb=allocated,
                    use_available=True,
                    device_type=DeviceType.CUDA,
                )
                if self.batch_size != old_batch_size:
                    logger.info(
                        f"Optimized batch size: {old_batch_size} → {self.batch_size} "
                        f"(based on {total - allocated:.1f}GB available)"
                    )

        elif self.device_info.device_type == DeviceType.MPS:
            # MPS: log unified memory usage.
            total_mem, available_mem = _get_apple_silicon_memory()
            logger.info(
                f"Unified memory: {total_mem:.1f}GB total, "
                f"~{available_mem:.1f}GB available"
            )

            # auto-optimize batch size for MPS.
            if self.auto_optimize and self._user_batch_size is None:
                old_batch_size = self.batch_size
                self.batch_size = _detect_mps_batch_size(
                    total_memory_gb=total_mem,
                    model_memory_gb=total_mem - available_mem,
                    use_available=True,
                )
                if self.batch_size != old_batch_size:
                    logger.info(
                        f"Optimized batch size for MPS: {old_batch_size} → {self.batch_size}"
                    )

    def _build_prompt(self, utterance: str) -> str:
        """Build the full prompt for the model."""
        user_prompt = self.config.get_user_prompt(utterance)

        # format as chat template if available.
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # fallback format.
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{self.config.system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return prompt

    def _parse_response(self, content: str) -> dict[str, Any]:
        """Parse LLM response content into annotation dict."""
        try:
            # try to extract JSON from response.
            content = content.strip()

            # handle markdown code blocks.
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                if end > start:
                    content = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                if end > start:
                    content = content[start:end].strip()

            # extract only the FIRST complete JSON object.
            content = self._extract_first_json_object(content)

            # attempt to fix common JSON issues from LLMs.
            content = self._fix_json_syntax(content)

            annotation = json.loads(content)

            # validate required fields.
            required_fields = ["has_hate_speech", "has_advertisement", "main_topic"]
            for field in required_fields:
                if field not in annotation:
                    annotation[field] = None

            # ensure proper types.
            annotation["has_hate_speech"] = bool(annotation.get("has_hate_speech", False))
            annotation["has_advertisement"] = bool(annotation.get("has_advertisement", False))

            # clean up optional fields.
            if not annotation["has_hate_speech"]:
                annotation["target_group"] = None
                annotation["hate_speech_type"] = None

            return annotation

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Raw response: {content[:500]}")
            return self._get_error_annotation(f"JSON parse error: {e}")

    def _extract_first_json_object(self, content: str) -> str:
        """Extract only the first complete JSON object from content."""
        if "{" not in content:
            return content

        start = content.find("{")
        if start == -1:
            return content

        # count braces to find the matching closing brace.
        brace_count = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(content[start:], start):
            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    return content[start : i + 1]

        # if no complete object found, return from start to last }.
        end = content.rfind("}")
        if end > start:
            return content[start : end + 1]

        return content[start:]

    def _fix_json_syntax(self, content: str) -> str:
        """Fix common JSON syntax errors produced by LLMs."""
        import re

        # remove trailing commas before } or ].
        content = re.sub(r",\s*}", "}", content)
        content = re.sub(r",\s*]", "]", content)

        # replace single quotes with double quotes (but not within strings).
        # this is a simple heuristic - replace ' with " when it looks like JSON structure.
        # handle cases like {'key': 'value'} -> {"key": "value"}.
        content = re.sub(r"(?<=[{,\[])\s*'", ' "', content)
        content = re.sub(r"'\s*(?=[}\],:])", '"', content)
        content = re.sub(r"(?<=:)\s*'", ' "', content)
        content = re.sub(r"'\s*(?=[,}\]])", '"', content)

        # fix unquoted keys (e.g., {key: "value"} -> {"key": "value"}).
        content = re.sub(r"{\s*(\w+)\s*:", r'{"\1":', content)
        content = re.sub(r",\s*(\w+)\s*:", r', "\1":', content)

        # fix True/False to true/false.
        content = re.sub(r"\bTrue\b", "true", content)
        content = re.sub(r"\bFalse\b", "false", content)
        content = re.sub(r"\bNone\b", "null", content)

        # remove any control characters except newlines and tabs.
        content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", content)

        # fix newlines within string values (escape them).
        # this is tricky - try to find unescaped newlines within quoted strings.
        def escape_newlines_in_strings(match: re.Match) -> str:
            s = match.group(0)
            # escape unescaped newlines.
            s = s.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
            return s

        # match string values (this is a simplified pattern).
        content = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', escape_newlines_in_strings, content)

        return content

    def _get_error_annotation(self, error_msg: str) -> dict[str, Any]:
        """Return annotation dict for failed LLM calls."""
        return {
            "has_hate_speech": None,
            "has_advertisement": None,
            "target_group": None,
            "hate_speech_type": None,
            "main_topic": None,
            "error": error_msg,
        }

    @torch.inference_mode()
    def annotate_single(self, utterance: str) -> dict[str, Any]:
        """
        Annotate a single utterance.

        Args:
            utterance: Text to annotate

        Returns:
            Annotation dict
        """
        if not self._loaded:
            self.load_model()

        prompt = self._build_prompt(utterance)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            do_sample=self.config.temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # decode only new tokens.
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        return self._parse_response(response)

    @torch.inference_mode()
    def annotate_batch(self, utterances: list[str]) -> list[dict[str, Any]]:
        """
        Annotate a batch of utterances.

        Args:
            utterances: List of texts to annotate

        Returns:
            List of annotation dicts in same order as input
        """
        if not self._loaded:
            self.load_model()

        if not utterances:
            return []

        # process in sub-batches.
        all_annotations = []

        for i in range(0, len(utterances), self.batch_size):
            batch = utterances[i : i + self.batch_size]
            prompts = [self._build_prompt(u) for u in batch]

            # tokenize batch.
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            ).to(self.model.device)

            # generate.
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            # decode each response.
            for j, output in enumerate(outputs):
                input_len = inputs["input_ids"][j].ne(self.tokenizer.pad_token_id).sum()
                response = self.tokenizer.decode(
                    output[input_len:],
                    skip_special_tokens=True,
                )
                annotation = self._parse_response(response)
                all_annotations.append(annotation)

            # clear cache between batches.
            self._clear_memory_cache()

        return all_annotations

    def _clear_memory_cache(self) -> None:
        """Clear memory cache for the current device type."""
        if self.device_info.device_type == DeviceType.CUDA:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        elif self.device_info.device_type == DeviceType.MPS:
            if torch.backends.mps.is_available():
                # MPS doesn't have empty_cache, but we can synchronize.
                torch.mps.synchronize()
                # force garbage collection for unified memory.
                import gc

                gc.collect()

    def unload_model(self) -> None:
        """Unload model to free GPU/unified memory."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        self._loaded = False

        # clear caches based on device type.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if torch.backends.mps.is_available():
            torch.mps.synchronize()

        # force garbage collection (especially important for MPS unified memory).
        import gc

        gc.collect()

        logger.info("Model unloaded")


def process_transcript_file(
    input_path: Path,
    output_path: Path,
    annotator: LLMAnnotator,
    skip_existing: bool = True,
) -> dict[str, Any]:
    """
    Process a single transcript file and add LLM annotations.

    Preserves all existing data and adds 'llm_annotation' field to each segment.

    Args:
        input_path: Path to input transcript file
        output_path: Path to save annotated transcript
        annotator: LLM annotator instance
        skip_existing: If True, skip files that already have annotations

    Returns:
        Dict with processing statistics
    """
    # check if we should skip.
    if skip_existing and check_annotations_exist(output_path):
        return {"skipped": True, "segments_annotated": 0}

    # load transcript.
    with open(input_path, encoding="utf-8") as f:
        transcript_data = json.load(f)

    segments = transcript_data.get("segments", [])

    if not segments:
        # save empty file to preserve structure.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(transcript_data, f, indent=2, ensure_ascii=False)
        return {"skipped": False, "segments_annotated": 0}

    # extract texts for batch processing.
    texts = [segment.get("text", "") for segment in segments]

    # annotate all segments.
    annotations = annotator.annotate_batch(texts)

    # add annotations to segments.
    for segment, annotation in zip(segments, annotations):
        segment["llm_annotation"] = annotation

    # save annotated transcript.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, indent=2, ensure_ascii=False)

    return {"skipped": False, "segments_annotated": len(segments)}
