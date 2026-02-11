"""utterance classifier using transformer models."""

import logging
import platform
import psutil
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from podcast_conversations.utterance_classification.config_loader import ModelConfig

logger = logging.getLogger(__name__)


def is_apple_silicon() -> bool:
    """check if running on Apple Silicon."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def detect_classification_device() -> str:
    """detect the best available device for classification.

    Returns:
        str: device identifier ("cuda", "mps", or "cpu").
    """
    # check for CUDA first.
    if torch.cuda.is_available():
        return "cuda"

    # check for MPS (Apple Silicon).
    if is_apple_silicon() and torch.backends.mps.is_available():
        return "mps"

    # fallback to cpu.
    return "cpu"


def get_classification_device_info() -> dict[str, Any]:
    """get information about available classification devices.

    Returns:
        dict with device availability and recommendations.
    """
    info = {
        "is_apple_silicon": is_apple_silicon(),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False,
        "recommended_device": detect_classification_device(),
    }

    if info["cuda_available"]:
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    if info["is_apple_silicon"]:
        info["system_memory_gb"] = psutil.virtual_memory().total / (1024**3)

    return info


@dataclass
class ClassificationResult:
    """result of classifying a single utterance with a single model."""

    model_name: str
    label: str
    confidence: float


def get_optimal_batch_size(device: str) -> int:
    """
    determine optimal batch size for classification based on available resources.

    considers:
    - device type (cuda, mps, cpu)
    - available memory (gpu or ram)

    returns conservative batch size to avoid oom errors.
    """

    if device == "cuda":
        # get gpu memory.
        try:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

            # conservative batch sizing for classification (typically needs more memory per item).
            if gpu_memory_gb >= 16:
                batch_size = 32
            elif gpu_memory_gb >= 8:
                batch_size = 16
            elif gpu_memory_gb >= 4:
                batch_size = 8
            else:
                batch_size = 4

            logger.info(f"gpu memory: {gpu_memory_gb:.1f}gb → batch size: {batch_size}")

        except Exception:
            batch_size = 8
            logger.warning(f"could not detect gpu memory, using default batch size: {batch_size}")

    elif device == "mps":
        # apple silicon - check system ram.
        try:
            ram_gb = psutil.virtual_memory().total / (1024**3)

            # mps uses unified memory.
            if ram_gb >= 32:
                batch_size = 24
            elif ram_gb >= 16:
                batch_size = 16
            else:
                batch_size = 8

            logger.info(f"system memory: {ram_gb:.1f}gb → batch size: {batch_size}")

        except Exception:
            batch_size = 8
            logger.warning(f"could not detect system memory, using default batch size: {batch_size}")

    else:  # cpu
        # cpu - base on available ram and cores.
        try:
            ram_gb = psutil.virtual_memory().available / (1024**3)
            cpu_count = psutil.cpu_count(logical=False) or 4

            # conservative sizing for cpu (classification is compute-intensive).
            if ram_gb >= 16 and cpu_count >= 8:
                batch_size = 8
            elif ram_gb >= 8 and cpu_count >= 4:
                batch_size = 4
            else:
                batch_size = 2

            logger.info(
                f"available memory: {ram_gb:.1f}gb, cores: {cpu_count} → batch size: {batch_size}"
            )

        except Exception:
            batch_size = 4
            logger.warning(f"could not detect system resources, using default batch size: {batch_size}")

    return batch_size


class UtteranceClassifier:
    """classifies utterances using multiple transformer models."""

    def __init__(
        self,
        models: list[ModelConfig],
        device: str = "auto",
        batch_size: int | None = None,
        hf_token: str | None = None,
    ):
        """
        initialize utterance classifier.

        Args:
            models: list of model configurations to load.
            device: device to use ("auto", "cpu", "cuda", "mps").
            batch_size: batch size for processing. if None, automatically determined.
            hf_token: hugging face token for accessing gated models.
        """
        self.models = models
        self.device = self._determine_device(device)
        self.hf_token = hf_token

        # determine batch size (auto or provided).
        if batch_size is None:
            self.batch_size = get_optimal_batch_size(self.device)
            logger.info(f"using dynamic batch size: {self.batch_size}")
        else:
            self.batch_size = batch_size
            logger.info(f"using provided batch size: {self.batch_size}")

        self.pipelines = {}

        logger.info(f"initializing {len(models)} classification models on {self.device}")
        self._load_models()

    def _determine_device(self, device: str) -> str:
        """determine the device to use for inference."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    def _load_models(self) -> None:
        """load all classifier models."""
        for model_config in self.models:
            try:
                logger.info(f"loading {model_config.name} from {model_config.model_repo}")

                # determine device parameter for pipeline.
                # transformers 4.30+ supports MPS directly.
                if self.device == "mps":
                    # use device_map for MPS to ensure proper tensor placement.
                    device_param = "mps"
                elif self.device == "cuda":
                    device_param = 0  # cuda:0
                else:
                    device_param = -1  # cpu

                # create classification pipeline.
                # note: token parameter is used for authentication with hugging face hub.
                clf_pipeline = pipeline(
                    "text-classification",
                    model=model_config.model_repo,
                    device=device_param,
                    tokenizer=model_config.model_repo,
                    return_all_scores=True,
                    token=self.hf_token,
                )

                self.pipelines[model_config.name] = {
                    "pipeline": clf_pipeline,
                    "labels": model_config.labels,
                }

                logger.info(f"✓ loaded {model_config.name}")

            except Exception as e:
                logger.error(f"failed to load {model_config.name}: {e}")
                # continue loading other models even if one fails.
                continue

    def classify_utterance(self, text: str) -> list[ClassificationResult]:
        """
        classify a single utterance with all loaded models.

        Args:
            text: utterance text to classify.

        Returns:
            list of ClassificationResult objects, one per model.
        """
        if not text or not text.strip():
            return []

        results = []

        for model_name, model_data in self.pipelines.items():
            try:
                # run classification.
                predictions = model_data["pipeline"](text, truncation=True, max_length=512)

                # predictions is a list of lists: [[{label, score}, {label, score}, ...]]
                # get the top prediction.
                if predictions and len(predictions) > 0:
                    # get all scores for this input.
                    scores = predictions[0]

                    # find the prediction with highest score.
                    top_prediction = max(scores, key=lambda x: x["score"])

                    result = ClassificationResult(
                        model_name=model_name,
                        label=top_prediction["label"],
                        confidence=float(top_prediction["score"]),
                    )
                    results.append(result)

            except Exception as e:
                logger.warning(f"classification failed for {model_name}: {e}")
                continue

        return results

    def classify_batch(self, texts: list[str]) -> list[list[ClassificationResult]]:
        """
        classify a batch of utterances.

        Args:
            texts: list of utterance texts to classify.

        Returns:
            list of lists of ClassificationResult objects.
        """
        results = []

        for text in texts:
            utterance_results = self.classify_utterance(text)
            results.append(utterance_results)

        return results

    def get_loaded_models(self) -> list[str]:
        """get list of successfully loaded model names."""
        return list(self.pipelines.keys())
