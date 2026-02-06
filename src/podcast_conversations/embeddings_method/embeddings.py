"""embedding model initialization and generation."""

import json
from pathlib import Path
from typing import Any

import psutil
import torch
from sentence_transformers import SentenceTransformer

from .mlx_backend import (
    MLXEmbeddingWrapper,
    get_mlx_status,
    get_optimal_batch_size_mlx,
    is_apple_silicon,
    is_mlx_embeddings_available,
    load_mlx_embedding_model,
)

# type alias for embedding models.
EmbeddingModel = SentenceTransformer | MLXEmbeddingWrapper


def detect_device(prefer_mlx: bool = True) -> str:
    """detect available compute device.

    Args:
        prefer_mlx: If True, prefer MLX on Apple Silicon when available.

    Returns:
        Device string: "mlx", "cuda", "mps", or "cpu"
    """
    # check for mlx availability first on Apple Silicon.
    if prefer_mlx and is_apple_silicon() and is_mlx_embeddings_available():
        print("🍎 apple silicon with mlx detected (optimized)")
        return "mlx"

    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🚀 cuda gpu detected: {gpu_name}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("🍎 apple silicon gpu (mps) detected")

        # suggest mlx if not available.
        if is_apple_silicon() and not is_mlx_embeddings_available():
            status = get_mlx_status()
            print(f"💡 tip: install mlx for better performance: {status['install_command']}")
    else:
        device = "cpu"
        print("💻 using cpu")

    return device


def get_optimal_batch_size(device: str, model: EmbeddingModel | None = None) -> int:
    """
    determine optimal batch size based on available resources.

    considers:
    - device type (mlx, cuda, mps, cpu)
    - available memory (gpu or ram)
    - model size

    returns conservative batch size to avoid oom errors.
    """
    # mlx has its own optimized batch size calculation.
    if device == "mlx":
        return get_optimal_batch_size_mlx()

    if device == "cuda":
        # get gpu memory.
        try:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

            # conservative batch sizing based on gpu memory.
            if gpu_memory_gb >= 16:
                batch_size = 128
            elif gpu_memory_gb >= 8:
                batch_size = 64
            elif gpu_memory_gb >= 4:
                batch_size = 32
            else:
                batch_size = 16

            print(f"📊 gpu memory: {gpu_memory_gb:.1f}gb → batch size: {batch_size}")

        except Exception:
            batch_size = 32
            print(f"⚠️  could not detect gpu memory, using default batch size: {batch_size}")

    elif device == "mps":
        # apple silicon - check system ram.
        try:
            ram_gb = psutil.virtual_memory().total / (1024**3)

            # mps uses unified memory, so base on total ram.
            if ram_gb >= 32:
                batch_size = 96
            elif ram_gb >= 16:
                batch_size = 64
            else:
                batch_size = 32

            print(f"📊 system memory: {ram_gb:.1f}gb → batch size: {batch_size}")

        except Exception:
            batch_size = 32
            print(f"⚠️  could not detect system memory, using default batch size: {batch_size}")

    else:  # cpu
        # cpu - base on available ram and cores.
        try:
            ram_gb = psutil.virtual_memory().available / (1024**3)
            cpu_count = psutil.cpu_count(logical=False) or 4

            # conservative sizing for cpu.
            if ram_gb >= 16 and cpu_count >= 8:
                batch_size = 32
            elif ram_gb >= 8 and cpu_count >= 4:
                batch_size = 16
            else:
                batch_size = 8

            print(f"📊 available memory: {ram_gb:.1f}gb, cores: {cpu_count} → batch size: {batch_size}")

        except Exception:
            batch_size = 16
            print(f"⚠️  could not detect system resources, using default batch size: {batch_size}")

    return batch_size


def load_embedding_model(
    device: str,
    model_name: str = "all-MiniLM-L6-v2",
    hf_token: str | None = None,
) -> EmbeddingModel:
    """load and initialize embedding model.

    Automatically uses MLX backend on Apple Silicon when available.

    Args:
        device: Device to load model on (mlx, cuda, mps, cpu)
        model_name: HuggingFace model name for sentence-transformers
        hf_token: HuggingFace token for gated models (ignored for MLX)

    Returns:
        Embedding model (MLXEmbeddingWrapper or SentenceTransformer)
    """
    # use mlx backend on Apple Silicon.
    if device == "mlx":
        return load_mlx_embedding_model(model_name)

    # fall back to sentence-transformers with torch.
    print(f"loading model: {model_name}")

    # pass token if provided (for gated models).
    model_kwargs = {}
    if hf_token:
        model_kwargs["token"] = hf_token

    model = SentenceTransformer(model_name, **model_kwargs)
    model = model.to(device)

    print(f"✓ model loaded on {device}")

    return model


def check_embeddings_exist(output_path: Path) -> bool:
    """
    check if embeddings file exists and has valid embeddings.

    returns True if file exists and contains embeddings for all segments.
    """
    if not output_path.exists():
        return False

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        segments = data.get("segments", [])

        if not segments:
            return False

        # check if all segments have embeddings.
        for segment in segments:
            if "embedding" not in segment:
                return False
            # verify embedding is non-empty list.
            if not isinstance(segment["embedding"], list) or len(segment["embedding"]) == 0:
                return False

        return True

    except Exception:
        return False


def process_transcript_file(
    input_path: Path,
    output_path: Path,
    model: SentenceTransformer,
    batch_size: int = 32,
    skip_existing: bool = True,
) -> bool:
    """
    generate embeddings for all utterances in transcript.

    preserves all existing data in the transcript including:
    - metadata
    - speaker information
    - classification labels (if present)
    - timestamps
    - any other fields

    adds 'embedding' field to each segment.

    args:
        input_path: path to input transcript file
        output_path: path to save enriched transcript with embeddings
        model: sentence transformer model
        batch_size: batch size for encoding
        skip_existing: if True, skip files that already have embeddings

    returns:
        True if embeddings were generated, False if skipped
    """

    # check if we should skip this file.
    if skip_existing and check_embeddings_exist(output_path):
        return False

    # load transcript (preserves all existing data).
    with open(input_path, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)

    segments = transcript_data.get("segments", [])

    if not segments:
        print(f"⚠️  no segments found in {input_path.name}")
        # still save the file to preserve metadata and structure.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(transcript_data, f, indent=2, ensure_ascii=False)
        return True

    # extract texts for batch processing.
    texts = [segment.get("text", "") for segment in segments]

    # generate embeddings in batches.
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True
    )

    # add embeddings to each segment (preserves existing fields including 'classifications').
    for segment, embedding in zip(segments, embeddings):
        segment["embedding"] = embedding.tolist()

    # save enriched transcript with all original data plus embeddings.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, indent=2, ensure_ascii=False)

    return True


def generate_taxonomy_embeddings(
    taxonomy: dict[str, Any], model: SentenceTransformer
) -> list[dict[str, Any]]:
    """create embeddings for all taxonomy items.

    combines target groups, strategies, and examples to create
    embeddings for each (target_group, strategy) pair.
    """

    taxonomy_items: list[dict[str, Any]] = []

    # extract components.
    target_groups = taxonomy.get("target_groups", [])
    strategies = taxonomy.get("strategies", [])
    examples = taxonomy.get("examples", [])

    # create lookup dicts.
    target_group_map = {tg["id"]: tg["name"] for tg in target_groups}
    strategy_map = {s["id"]: s for s in strategies}

    # create examples lookup: (target_group_id, strategy_id) -> [items].
    examples_map: dict[tuple, list[str]] = {}
    for ex in examples:
        key = (ex["target_group_id"], ex["strategy_id"])
        examples_map[key] = ex.get("items", [])

    # generate embeddings for each (target_group, strategy) combination.
    for target_group in target_groups:
        tg_id = target_group["id"]
        tg_name = target_group["name"]

        for strategy in strategies:
            strat_id = strategy["id"]
            strat_name = strategy["name"]
            strat_description = strategy.get("description", "")

            # get examples for this combination.
            key = (tg_id, strat_id)
            example_items = examples_map.get(key, [])

            # concatenate description and examples.
            examples_text = " ".join(example_items[:10])  # limit to 10 examples for embedding.
            combined_text = f"{strat_description} {examples_text}".strip()

            # generate embeddings.
            description_embedding = model.encode(strat_description, convert_to_numpy=True)
            combined_embedding = model.encode(combined_text, convert_to_numpy=True)

            # store with metadata.
            item = {
                "target_group": tg_name,
                "target_group_id": tg_id,
                "othering_strategy": strat_name,
                "strategy_id": strat_id,
                "description": strat_description,
                "example_count": len(example_items),
                "description_embedding": description_embedding.tolist(),
                "combined_embedding": combined_embedding.tolist(),
            }

            taxonomy_items.append(item)

    return taxonomy_items
