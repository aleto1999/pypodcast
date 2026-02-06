"""Caching layer for performance optimization."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from terminal_data_visualizer.config import CACHE_SIZE
from terminal_data_visualizer.models import EpisodeStats, Segment, SpeakerStats


def _get_file_key(file_path: Path) -> tuple[str, float]:
    """Generate cache key based on file path and modification time."""
    return (str(file_path.absolute()), file_path.stat().st_mtime)


@lru_cache(maxsize=CACHE_SIZE)
def load_json_cached(file_path: Path, mtime: float) -> dict[str, Any]:
    """Load JSON file with caching based on modification time."""
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


def load_transcript(file_path: Path) -> dict[str, Any] | None:
    """Load transcript with caching."""
    try:
        key = _get_file_key(file_path)
        return load_json_cached(file_path, key[1])
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def load_keyword_analysis(file_path: Path) -> dict[str, Any] | None:
    """Load keyword analysis with caching."""
    try:
        key = _get_file_key(file_path)
        return load_json_cached(file_path, key[1])
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


@lru_cache(maxsize=CACHE_SIZE)
def calculate_episode_stats_cached(file_path: str, mtime: float) -> EpisodeStats | None:
    """Calculate episode statistics with caching."""
    from collections import defaultdict

    path = Path(file_path)
    data = load_transcript(path)
    if not data:
        return None

    segments_data = data.get("segments", [])
    if not segments_data:
        return None

    # convert to Segment objects.
    segments = [Segment.from_dict(s) for s in segments_data]

    # calculate speaker statistics.
    speaker_data: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"segments": 0, "words": 0, "duration": 0.0}
    )

    for seg in segments:
        speaker_data[seg.speaker]["segments"] += 1
        speaker_data[seg.speaker]["words"] += seg.word_count
        speaker_data[seg.speaker]["duration"] += seg.duration

    # create SpeakerStats objects.
    speaker_stats = {}
    for speaker, data in speaker_data.items():
        avg_length = data["duration"] / data["segments"] if data["segments"] > 0 else 0.0
        speaker_stats[speaker] = SpeakerStats(
            speaker=speaker,
            segment_count=data["segments"],
            total_words=data["words"],
            total_duration=data["duration"],
            avg_segment_length=avg_length,
        )

    total_words = sum(s.total_words for s in speaker_stats.values())
    total_duration = sum(s.total_duration for s in speaker_stats.values())

    return EpisodeStats(
        filename=path.stem,
        total_segments=len(segments),
        total_duration=total_duration,
        speaker_count=len(speaker_stats),
        total_words=total_words,
        speakers=speaker_stats,
    )


def get_episode_stats(file_path: Path) -> EpisodeStats | None:
    """Get episode statistics with caching."""
    try:
        key = _get_file_key(file_path)
        return calculate_episode_stats_cached(key[0], key[1])
    except OSError:
        return None


def clear_cache() -> None:
    """Clear all caches."""
    load_json_cached.cache_clear()
    calculate_episode_stats_cached.cache_clear()


def get_cache_info() -> dict[str, Any]:
    """Get cache statistics."""
    return {
        "json_cache": load_json_cached.cache_info()._asdict(),
        "stats_cache": calculate_episode_stats_cached.cache_info()._asdict(),
    }
