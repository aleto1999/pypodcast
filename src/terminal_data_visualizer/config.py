"""Configuration settings for terminal data visualizer."""

from __future__ import annotations

from pathlib import Path

# visualization settings.
BAR_WIDTH = 60
TIMELINE_BINS = 10
MAX_FILES_DISPLAY = 20
MAX_CATEGORIES_DISPLAY = 50
MAX_KEYWORDS_DISPLAY = 100

# table settings.
TABLE_MAX_ROWS = 50
TABLE_PAGE_SIZE = 20

# performance settings.
CACHE_SIZE = 128
CHUNK_SIZE = 1000
MAX_SAMPLE_FILES = 10

# path settings.
OUTPUTS_PATH = Path("outputs")

# transcripts directory with fallback.
# try primary path first, then fallback if it doesn't exist.
_PRIMARY_TRANSCRIPTS_DIR = "transcripts_with_diarization_labels_postprocessed_with_utterance_and_document_labels"
_FALLBACK_TRANSCRIPTS_DIR = "transcripts_with_diarization_labels_postprocessed"

if (OUTPUTS_PATH / _PRIMARY_TRANSCRIPTS_DIR).exists():
    TRANSCRIPTS_DIR = _PRIMARY_TRANSCRIPTS_DIR
elif (OUTPUTS_PATH / _FALLBACK_TRANSCRIPTS_DIR).exists():
    TRANSCRIPTS_DIR = _FALLBACK_TRANSCRIPTS_DIR
else:
    # default to primary even if it doesn't exist (will show empty/error states).
    TRANSCRIPTS_DIR = _PRIMARY_TRANSCRIPTS_DIR

ANALYSIS_DIR = "analysis/keyword_analysis"
DOCUMENT_LABELS_DIR = "document_labels"
FEATURES_FILE = "features/features.csv"
LLM_ANNOTATIONS_DIR = "analysis/llm_annotation_method"

# screenshot settings.
SCREENSHOT_DIR = OUTPUTS_PATH / "terminal_viewer" / "screenshots"
SCREENSHOT_DPI = 300
SCREENSHOT_QUALITY = 95

# keyboard shortcuts.
KEY_QUIT = "q"
KEY_BACK = "b"
KEY_SAVE = "s"
KEY_HELP = "h"
KEY_SEARCH = "/"

# colors and styling.
COLOR_PRIMARY = "cyan"
COLOR_SECONDARY = "magenta"
COLOR_SUCCESS = "green"
COLOR_WARNING = "yellow"
COLOR_ERROR = "red"
COLOR_INFO = "blue"
COLOR_DIM = "dim"

# session state.
STATE_FILE = Path.home() / ".terminal_data_visualizer_state.json"
REMEMBER_LAST_SELECTION = True
