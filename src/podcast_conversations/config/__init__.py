"""configuration constants for podcast conversations."""

# diarization model configuration.
DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

# device configuration.
DEFAULT_DEVICE = "cuda:0"

# optimization defaults.
DEFAULT_USE_BF16 = True
DEFAULT_USE_COMPILE = False
DEFAULT_COMPILE_MODE = "reduce-overhead"

__all__ = [
    "DEFAULT_DIARIZATION_MODEL",
    "DEFAULT_DEVICE",
    "DEFAULT_USE_BF16",
    "DEFAULT_USE_COMPILE",
    "DEFAULT_COMPILE_MODE",
]
