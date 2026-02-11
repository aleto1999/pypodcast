"""configuration models for podcast downloader."""

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class EpisodeFilter(BaseModel):
    """filters for selecting which episodes to download."""

    max_episodes: int | None = Field(default=None, description="maximum episodes to download")
    start_date: datetime | None = Field(default=None, description="earliest episode date")
    end_date: datetime | None = Field(default=None, description="latest episode date")
    random_sample: int | None = Field(
        default=None, description="randomly sample n episodes instead of newest"
    )


class DownloadConfig(BaseModel):
    """configuration for download operations."""

    output_dir: Path = Field(default=Path("outputs/downloads"), description="base output directory for audio files")
    transcripts_dir: Path = Field(default=Path("outputs/transcripts"), description="base directory for transcript placeholders")
    metadata_dir: Path = Field(default=Path("outputs/metadata"), description="metadata storage directory")

    # concurrency settings.
    max_concurrent_downloads: int = Field(default=4, ge=1, le=20)
    rate_limit_delay: float = Field(default=1.0, ge=0, description="seconds between requests")

    # retry settings.
    max_retries: int = Field(default=3, ge=0)
    retry_backoff_base: float = Field(default=2.0, description="exponential backoff base")

    # timeout settings.
    connect_timeout: float = Field(default=15.0)
    read_timeout: float = Field(default=300.0)

    # feed fallback settings.
    enable_feed_fallback: bool = Field(default=True, description="try alternative feeds on fail")
    max_alternative_feeds: int = Field(default=3, ge=1)

    # file handling.
    skip_existing: bool = Field(default=True, description="skip already downloaded files")
    audio_format: Literal["mp3", "m4a", "wav", "ogg", "flac"] = Field(default="mp3")
    validate_audio: bool = Field(default=True, description="validate downloaded audio files")

    # enhanced validation settings.
    deep_validation: bool = Field(default=True, description="use ffprobe for deep file validation")
    min_file_size: int = Field(default=1024, description="minimum valid file size in bytes")
    max_file_size: int = Field(default=500 * 1024 * 1024, description="maximum valid file size in bytes")
    min_disk_space: int = Field(default=100 * 1024 * 1024, description="minimum disk space required in bytes")

    # youtube fallback settings.
    enable_youtube_fallback: bool = Field(default=True, description="fallback to youtube when rss fails")
    youtube_min_confidence: float = Field(default=0.5, ge=0, le=1, description="minimum confidence for youtube matches")
    youtube_max_retries: int = Field(default=2, ge=0, description="max retries per youtube video")

    # fuzzy matching settings.
    enable_fuzzy_matching: bool = Field(default=True, description="use fuzzy matching for discovery")
    fuzzy_min_score: float = Field(default=0.3, ge=0, le=1, description="minimum fuzzy match score")

    # statistics collection.
    collect_statistics: bool = Field(default=True, description="collect download statistics")

    # episode filtering (can be overridden per-download).
    default_filter: EpisodeFilter = Field(default_factory=EpisodeFilter)


class FeedSearchResult(BaseModel):
    """result from a feed discovery search."""

    title: str
    feed_url: str
    author: str | None = None
    description: str | None = None
    artwork_url: str | None = None
    episode_count: int | None = None
    source: str = Field(description="api source: itunes, podcastindex, etc")
    confidence: float = Field(default=1.0, ge=0, le=1, description="match confidence score")


class DownloadStats(BaseModel):
    """statistics for a download session."""

    podcasts_processed: int = 0
    episodes_found: int = 0
    episodes_downloaded: int = 0
    episodes_skipped: int = 0
    episodes_failed: int = 0
    total_bytes: int = 0
    errors: list[str] = Field(default_factory=list)

    def add_error(self, error: str) -> None:
        """add an error message."""
        self.errors.append(error)

    @property
    def success_rate(self) -> float:
        """calculate download success rate."""
        total = self.episodes_downloaded + self.episodes_failed
        if total == 0:
            return 0.0
        return self.episodes_downloaded / total
