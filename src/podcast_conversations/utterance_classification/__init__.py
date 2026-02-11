"""utterance classification using transformer models."""

from podcast_conversations.utterance_classification.classifier import (
    UtteranceClassifier,
    detect_classification_device,
    get_classification_device_info,
    get_optimal_batch_size,
    is_apple_silicon,
)
from podcast_conversations.utterance_classification.config_loader import ClassifierConfig
from podcast_conversations.utterance_classification.processor import (
    TranscriptProcessor,
    check_classifications_exist,
)

__all__ = [
    "UtteranceClassifier",
    "ClassifierConfig",
    "TranscriptProcessor",
    # device detection.
    "detect_classification_device",
    "get_classification_device_info",
    "get_optimal_batch_size",
    "is_apple_silicon",
    # utilities.
    "check_classifications_exist",
]
