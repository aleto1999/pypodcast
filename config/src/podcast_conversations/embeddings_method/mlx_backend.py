"""mlx-optimized embedding backend for Apple Silicon."""

import platform
from typing import Any

import numpy as np

# mlx availability flags.
MLX_AVAILABLE = False
MLX_EMBEDDINGS_AVAILABLE = False

try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except ImportError:
    mx = None

try:
    from mlx_embedding_models.embedding import EmbeddingModel as MLXEmbeddingModel

    MLX_EMBEDDINGS_AVAILABLE = True
except ImportError:
    MLXEmbeddingModel = None


def is_mlx_available() -> bool:
    """check if mlx is available for use."""
    return MLX_AVAILABLE


def is_mlx_embeddings_available() -> bool:
    """check if mlx-embedding-models is available."""
    return MLX_EMBEDDINGS_AVAILABLE


def is_apple_silicon() -> bool:
    """check if running on Apple Silicon."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def get_mlx_status() -> dict[str, Any]:
    """get mlx availability status and recommendations."""
    use_mlx = is_apple_silicon() and MLX_EMBEDDINGS_AVAILABLE
    status = {
        "is_apple_silicon": is_apple_silicon(),
        "mlx_available": MLX_AVAILABLE,
        "mlx_embeddings_available": MLX_EMBEDDINGS_AVAILABLE,
        "recommended_backend": "mlx" if use_mlx else "torch",
        "install_command": None,
    }

    if is_apple_silicon() and not MLX_EMBEDDINGS_AVAILABLE:
        status["install_command"] = "uv sync --extra macos"

    return status


# mlx model registry mapping to standard model names.
MLX_MODEL_REGISTRY = {
    # bge family.
    "BAAI/bge-small-en-v1.5": "bge-small",
    "BAAI/bge-base-en-v1.5": "bge-base",
    "BAAI/bge-large-en-v1.5": "bge-large",
    "bge-small": "bge-small",
    "bge-base": "bge-base",
    "bge-large": "bge-large",
    # all-MiniLM family - map to bge-small as closest alternative.
    "all-MiniLM-L6-v2": "bge-small",
    "sentence-transformers/all-MiniLM-L6-v2": "bge-small",
    # paraphrase models - map to bge equivalents.
    "paraphrase-MiniLM-L6-v2": "bge-small",
    "sentence-transformers/paraphrase-MiniLM-L6-v2": "bge-small",
}


def get_mlx_model_name(model_name: str) -> str | None:
    """map standard model name to mlx registry name."""
    return MLX_MODEL_REGISTRY.get(model_name)


class MLXEmbeddingWrapper:
    """wrapper for mlx embedding model with sentence-transformers compatible interface."""

    def __init__(self, model_name: str):
        """initialize mlx embedding model.

        Args:
            model_name: Model name (either mlx registry name or standard HF name)
        """
        if not MLX_EMBEDDINGS_AVAILABLE:
            raise ImportError(
                "mlx-embedding-models not installed. "
                "Install with: uv sync --extra macos"
            )

        # resolve model name to mlx registry name.
        mlx_name = get_mlx_model_name(model_name)
        if mlx_name is None:
            # try using directly as registry name.
            mlx_name = model_name

        print(f"loading mlx model: {mlx_name}")
        self._model = MLXEmbeddingModel.from_registry(mlx_name)
        self._model_name = mlx_name
        print("✓ mlx model loaded on Apple Silicon GPU")

    def encode(
        self,
        sentences: list[str] | str,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """encode sentences to embeddings.

        Compatible with sentence-transformers interface.

        Args:
            sentences: Single sentence or list of sentences
            batch_size: Batch size for encoding (used for memory management)
            show_progress_bar: Ignored (for compatibility)
            convert_to_numpy: Always returns numpy array
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            Numpy array of embeddings with shape (num_sentences, embedding_dim)
        """
        if isinstance(sentences, str):
            sentences = [sentences]

        # process in batches to manage memory.
        all_embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            batch_embeddings = self._model.encode(batch)

            # convert mlx array to numpy if needed.
            if hasattr(batch_embeddings, "tolist"):
                # mlx array - convert to numpy.
                batch_embeddings = np.array(batch_embeddings)

            all_embeddings.append(batch_embeddings)

        # concatenate all batches.
        if len(all_embeddings) == 1:
            return all_embeddings[0]

        return np.vstack(all_embeddings)

    def to(self, device: str) -> "MLXEmbeddingWrapper":
        """no-op for compatibility with torch interface."""
        # mlx automatically uses Apple Silicon GPU.
        return self


def load_mlx_embedding_model(model_name: str = "bge-small") -> MLXEmbeddingWrapper:
    """load mlx embedding model.

    Args:
        model_name: Model name (mlx registry name or standard HF name)

    Returns:
        MLXEmbeddingWrapper with sentence-transformers compatible interface
    """
    return MLXEmbeddingWrapper(model_name)


def cosine_similarity_mlx(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """calculate cosine similarity using mlx for acceleration.

    Falls back to numpy if mlx not available.
    """
    if not MLX_AVAILABLE:
        # fallback to numpy.
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot_product / (norm_a * norm_b))

    # use mlx for acceleration.
    a = mx.array(vec_a)
    b = mx.array(vec_b)

    dot_product = mx.sum(a * b)
    norm_a = mx.sqrt(mx.sum(a * a))
    norm_b = mx.sqrt(mx.sum(b * b))

    if float(norm_a) == 0 or float(norm_b) == 0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))


def batch_cosine_similarity_mlx(
    query_embedding: np.ndarray, embeddings_matrix: np.ndarray
) -> np.ndarray:
    """compute cosine similarity between query and all embeddings in matrix.

    Optimized for batch operations using mlx.

    Args:
        query_embedding: Single embedding vector (embedding_dim,)
        embeddings_matrix: Matrix of embeddings (num_embeddings, embedding_dim)

    Returns:
        Array of similarity scores (num_embeddings,)
    """
    if not MLX_AVAILABLE:
        # fallback to numpy vectorized computation.
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return np.zeros(len(embeddings_matrix))

        query_normalized = query_embedding / query_norm
        matrix_norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
        matrix_norms = np.where(matrix_norms == 0, 1, matrix_norms)  # avoid div by zero.
        matrix_normalized = embeddings_matrix / matrix_norms

        return np.dot(matrix_normalized, query_normalized)

    # use mlx for acceleration.
    query = mx.array(query_embedding)
    matrix = mx.array(embeddings_matrix)

    # normalize query.
    query_norm = mx.sqrt(mx.sum(query * query))
    if float(query_norm) == 0:
        return np.zeros(len(embeddings_matrix))
    query_normalized = query / query_norm

    # normalize matrix rows.
    matrix_norms = mx.sqrt(mx.sum(matrix * matrix, axis=1, keepdims=True))
    matrix_norms = mx.where(matrix_norms == 0, 1, matrix_norms)
    matrix_normalized = matrix / matrix_norms

    # compute similarities.
    similarities = mx.matmul(matrix_normalized, query_normalized)

    return np.array(similarities)


def get_optimal_batch_size_mlx() -> int:
    """get optimal batch size for mlx on Apple Silicon.

    Apple Silicon unified memory allows for larger batches compared to
    discrete GPU memory management.
    """
    import psutil

    try:
        ram_gb = psutil.virtual_memory().total / (1024**3)

        # mlx efficiently uses unified memory.
        # larger batches are generally beneficial on Apple Silicon.
        if ram_gb >= 64:
            batch_size = 256
        elif ram_gb >= 32:
            batch_size = 192
        elif ram_gb >= 16:
            batch_size = 128
        else:
            batch_size = 64

        print(f"📊 unified memory: {ram_gb:.1f}gb → mlx batch size: {batch_size}")
        return batch_size

    except Exception:
        return 64
