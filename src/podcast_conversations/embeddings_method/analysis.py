"""semantic analysis and similarity calculations."""

import json
from pathlib import Path
from typing import Any

import numpy as np
from rich.progress import track

from .mlx_backend import batch_cosine_similarity_mlx, cosine_similarity_mlx, is_mlx_available


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """calculate cosine similarity between two vectors.

    Uses MLX acceleration on Apple Silicon when available.
    """
    # use mlx-accelerated version if available.
    if is_mlx_available():
        return cosine_similarity_mlx(vec_a, vec_b)

    # fallback to numpy.
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))


def analyze_transcript(
    transcript_path: Path, taxonomy_embeddings: list[dict[str, Any]], threshold: float
) -> dict[str, Any]:
    """find semantic matches between transcript and taxonomy.

    Uses batch similarity computation with MLX acceleration on Apple Silicon.
    """
    # load transcript with embeddings.
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)

    segments = transcript_data.get("segments", [])
    matches: list[dict[str, Any]] = []

    # pre-extract taxonomy embeddings into a matrix for batch computation.
    taxonomy_matrix = np.array([t["combined_embedding"] for t in taxonomy_embeddings])

    for idx, segment in enumerate(segments):
        if "embedding" not in segment:
            continue

        utterance_embedding = np.array(segment["embedding"])

        # batch compute similarities against all taxonomy items.
        if is_mlx_available():
            similarities = batch_cosine_similarity_mlx(utterance_embedding, taxonomy_matrix)
        else:
            # vectorized numpy fallback.
            query_norm = np.linalg.norm(utterance_embedding)
            if query_norm == 0:
                continue
            query_normalized = utterance_embedding / query_norm
            matrix_norms = np.linalg.norm(taxonomy_matrix, axis=1, keepdims=True)
            matrix_norms = np.where(matrix_norms == 0, 1, matrix_norms)
            matrix_normalized = taxonomy_matrix / matrix_norms
            similarities = np.dot(matrix_normalized, query_normalized)

        # find matches above threshold.
        match_indices = np.where(similarities >= threshold)[0]

        if len(match_indices) == 0:
            continue

        # get surrounding context once (shared across all matches for this segment).
        previous_context = []
        for i in range(max(0, idx - 3), idx):
            prev_seg = segments[i]
            previous_context.append({
                "text": prev_seg.get("text", ""),
                "speaker": prev_seg.get("speaker", "UNKNOWN"),
                "timestamp": prev_seg.get("start", 0.0),
            })

        following_context = []
        for i in range(idx + 1, min(len(segments), idx + 4)):
            next_seg = segments[i]
            following_context.append({
                "text": next_seg.get("text", ""),
                "speaker": next_seg.get("speaker", "UNKNOWN"),
                "timestamp": next_seg.get("start", 0.0),
            })

        # create match entries for all taxonomy items above threshold.
        for match_idx in match_indices:
            tax_item = taxonomy_embeddings[match_idx]
            match = {
                "target_group": tax_item["target_group"],
                "othering_strategy": tax_item["othering_strategy"],
                "similarity_score": float(similarities[match_idx]),
                "matched_text": segment.get("text", ""),
                "speaker": segment.get("speaker", "UNKNOWN"),
                "timestamp": segment.get("start", 0.0),
                "description": tax_item["description"],
                "context": {
                    "previous_utterances": previous_context,
                    "following_utterances": following_context,
                },
            }
            matches.append(match)

    return {
        "file": str(transcript_path),
        "matches_found": len(matches),
        "matches": matches,
    }


def analyze_multiple_transcripts(
    transcript_paths: list[Path],
    taxonomy_embeddings: list[dict[str, Any]],
    threshold: float,
    batch_size: int = 50,
) -> list[dict[str, Any]]:
    """process multiple transcripts with progress tracking."""

    results: list[dict[str, Any]] = []
    all_max_scores = []  # Track max similarity scores for diagnostics

    for i in track(
        range(0, len(transcript_paths), batch_size), description="analyzing transcripts"
    ):
        batch = transcript_paths[i : i + batch_size]

        for path in batch:
            try:
                result = analyze_transcript(path, taxonomy_embeddings, threshold)
                results.append(result)

                # Track highest similarity score from this file for diagnostics
                if result["matches"]:
                    max_score = max(m["similarity_score"] for m in result["matches"])
                    all_max_scores.append(max_score)

            except Exception as e:
                print(f"⚠️  error analyzing {path.name}: {e}")
                continue

    # Print diagnostic info
    if all_max_scores:
        print(f"\n[dim]similarity score range: {min(all_max_scores):.3f} - {max(all_max_scores):.3f}")
        print(f"average max score: {np.mean(all_max_scores):.3f}[/dim]")
    else:
        print(f"\n[yellow]⚠️  no matches found at threshold {threshold}")
        print("consider lowering the threshold (try 0.25-0.35)[/yellow]")

    return results
