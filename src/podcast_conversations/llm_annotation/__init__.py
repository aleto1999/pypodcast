"""LLM-based annotation for podcast utterances.

This package provides utilities for annotating podcast utterances using
large language models (LLama 3.3 70B Instruct) for:
- Hate speech detection
- Target group identification
- Hate speech type classification
- Advertisement detection
- Topic extraction

Optimized for multiple platforms:
- NVIDIA GPUs (CUDA): 4-bit/8-bit quantization, Flash Attention 2
- Apple Silicon (MPS): Unified memory optimization, float16
- Apple Silicon (MLX): 4-bit quantized models via llm CLI
- CPU: Fallback for systems without GPU

Features:
- Dynamic batching based on available memory
- KV cache optimization
- Parallel I/O with prefetching
- Cross-file segment batching
- Auto-optimization based on available resources
- MLX support for efficient Apple Silicon inference
"""

from .annotator import (
    BatchConfig,
    DeviceInfo,
    DeviceType,
    LLMAnnotator,
    calculate_optimal_batch_config,
    check_annotations_exist,
    detect_device,
    detect_optimal_batch_size,
    detect_optimal_io_workers,
    detect_optimal_prefetch,
    get_mlx_quantization_info,
    is_mlx_available,
    process_transcript_file,
)
from .config import AnnotationConfig, load_questions_config
from .mlx_annotator import MLXAnnotator, MLXModelConfig
from .output import save_annotation_results
from .parallel import (
    CrossFileBatcher,
    FileTask,
    ParallelAnnotationPipeline,
    PrefetchingFileLoader,
    SegmentBatch,
    ThroughputMetrics,
)

__all__ = [
    "AnnotationConfig",
    "BatchConfig",
    "CrossFileBatcher",
    "DeviceInfo",
    "DeviceType",
    "FileTask",
    "LLMAnnotator",
    "MLXAnnotator",
    "MLXModelConfig",
    "ParallelAnnotationPipeline",
    "PrefetchingFileLoader",
    "SegmentBatch",
    "ThroughputMetrics",
    "calculate_optimal_batch_config",
    "check_annotations_exist",
    "detect_device",
    "detect_optimal_batch_size",
    "detect_optimal_io_workers",
    "detect_optimal_prefetch",
    "get_mlx_quantization_info",
    "is_mlx_available",
    "load_questions_config",
    "process_transcript_file",
    "save_annotation_results",
]
