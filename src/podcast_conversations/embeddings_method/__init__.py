"""embeddings-based semantic analysis for dehumanization detection."""

from .analysis import analyze_multiple_transcripts, analyze_transcript, cosine_similarity
from .discovery import discover_shows, display_available_shows
from .embeddings import (
    check_embeddings_exist,
    detect_device,
    generate_taxonomy_embeddings,
    get_optimal_batch_size,
    load_embedding_model,
    process_transcript_file,
)
from .mlx_backend import (
    get_mlx_status,
    is_apple_silicon,
    is_mlx_available,
    is_mlx_embeddings_available,
)
from .output import generate_summary_statistics, save_analysis_results
from .taxonomy import load_taxonomy

__all__ = [
    # analysis.
    "analyze_multiple_transcripts",
    "analyze_transcript",
    "cosine_similarity",
    # discovery.
    "discover_shows",
    "display_available_shows",
    # embeddings.
    "check_embeddings_exist",
    "detect_device",
    "generate_summary_statistics",
    "generate_taxonomy_embeddings",
    "get_optimal_batch_size",
    "load_embedding_model",
    "process_transcript_file",
    "save_analysis_results",
    # mlx backend.
    "get_mlx_status",
    "is_apple_silicon",
    "is_mlx_available",
    "is_mlx_embeddings_available",
    # taxonomy.
    "load_taxonomy",
]
