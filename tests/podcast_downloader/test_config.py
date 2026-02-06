"""tests for config module."""

from datetime import datetime
from pathlib import Path

import pytest

from podcast_downloader.config import (
    DownloadConfig,
    DownloadStats,
    EpisodeFilter,
    FeedSearchResult,
)


class TestEpisodeFilter:
    """tests for EpisodeFilter model."""

    def test_default_values(self):
        """should have None defaults."""
        filter_ = EpisodeFilter()
        assert filter_.max_episodes is None
        assert filter_.start_date is None
        assert filter_.end_date is None
        assert filter_.random_sample is None

    def test_max_episodes(self):
        """should accept max_episodes."""
        filter_ = EpisodeFilter(max_episodes=10)
        assert filter_.max_episodes == 10

    def test_date_range(self):
        """should accept date range."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)
        filter_ = EpisodeFilter(start_date=start, end_date=end)
        assert filter_.start_date == start
        assert filter_.end_date == end

    def test_random_sample(self):
        """should accept random_sample."""
        filter_ = EpisodeFilter(random_sample=5)
        assert filter_.random_sample == 5


class TestDownloadConfig:
    """tests for DownloadConfig model."""

    def test_default_values(self):
        """should have sensible defaults."""
        config = DownloadConfig()
        assert config.output_dir == Path("outputs/downloads")
        assert config.metadata_dir == Path("outputs/metadata")
        assert config.max_concurrent_downloads == 4
        assert config.skip_existing is True
        assert config.validate_audio is True

    def test_custom_output_dir(self):
        """should accept custom output directory."""
        config = DownloadConfig(output_dir=Path("/custom/output"))
        assert config.output_dir == Path("/custom/output")

    def test_concurrency_limits(self):
        """should enforce concurrency limits."""
        config = DownloadConfig(max_concurrent_downloads=10)
        assert config.max_concurrent_downloads == 10

        # test bounds
        config = DownloadConfig(max_concurrent_downloads=1)
        assert config.max_concurrent_downloads == 1

    def test_retry_settings(self):
        """should accept retry settings."""
        config = DownloadConfig(
            max_retries=5,
            retry_backoff_base=3.0,
        )
        assert config.max_retries == 5
        assert config.retry_backoff_base == 3.0

    def test_timeout_settings(self):
        """should accept timeout settings."""
        config = DownloadConfig(
            connect_timeout=30.0,
            read_timeout=600.0,
        )
        assert config.connect_timeout == 30.0
        assert config.read_timeout == 600.0

    def test_feed_fallback_settings(self):
        """should accept feed fallback settings."""
        config = DownloadConfig(
            enable_feed_fallback=False,
            max_alternative_feeds=5,
        )
        assert config.enable_feed_fallback is False
        assert config.max_alternative_feeds == 5

    def test_audio_format(self):
        """should accept audio format."""
        config = DownloadConfig(audio_format="m4a")
        assert config.audio_format == "m4a"

    def test_default_filter(self):
        """should have default episode filter."""
        config = DownloadConfig()
        assert isinstance(config.default_filter, EpisodeFilter)

    def test_custom_default_filter(self):
        """should accept custom default filter."""
        filter_ = EpisodeFilter(max_episodes=20)
        config = DownloadConfig(default_filter=filter_)
        assert config.default_filter.max_episodes == 20

    def test_deep_validation_settings(self):
        """should accept deep validation settings."""
        config = DownloadConfig(
            deep_validation=True,
            min_file_size=2048,
            max_file_size=100 * 1024 * 1024,
            min_disk_space=50 * 1024 * 1024,
        )
        assert config.deep_validation is True
        assert config.min_file_size == 2048

    def test_youtube_fallback_settings(self):
        """should accept youtube fallback settings."""
        config = DownloadConfig(
            enable_youtube_fallback=True,
            youtube_min_confidence=0.6,
            youtube_max_retries=3,
        )
        assert config.enable_youtube_fallback is True
        assert config.youtube_min_confidence == 0.6
        assert config.youtube_max_retries == 3

    def test_fuzzy_matching_settings(self):
        """should accept fuzzy matching settings."""
        config = DownloadConfig(
            enable_fuzzy_matching=True,
            fuzzy_min_score=0.4,
        )
        assert config.enable_fuzzy_matching is True
        assert config.fuzzy_min_score == 0.4

    def test_statistics_settings(self):
        """should accept statistics settings."""
        config = DownloadConfig(collect_statistics=False)
        assert config.collect_statistics is False


class TestFeedSearchResult:
    """tests for FeedSearchResult model."""

    def test_required_fields(self):
        """should require title and feed_url."""
        result = FeedSearchResult(
            title="Test Podcast",
            feed_url="https://example.com/feed.xml",
            source="itunes",
        )
        assert result.title == "Test Podcast"
        assert result.feed_url == "https://example.com/feed.xml"

    def test_optional_fields(self):
        """should accept optional fields."""
        result = FeedSearchResult(
            title="Test Podcast",
            feed_url="https://example.com/feed.xml",
            author="Test Author",
            description="A test podcast",
            artwork_url="https://example.com/art.jpg",
            episode_count=100,
            source="itunes",
            confidence=0.95,
        )
        assert result.author == "Test Author"
        assert result.episode_count == 100
        assert result.confidence == 0.95

    def test_default_confidence(self):
        """should default confidence to 1.0."""
        result = FeedSearchResult(
            title="Test",
            feed_url="https://example.com/feed.xml",
            source="itunes",
        )
        assert result.confidence == 1.0

    def test_confidence_bounds(self):
        """should enforce confidence bounds."""
        result = FeedSearchResult(
            title="Test",
            feed_url="https://example.com/feed.xml",
            source="itunes",
            confidence=0.5,
        )
        assert 0 <= result.confidence <= 1


class TestDownloadStats:
    """tests for DownloadStats model."""

    def test_default_values(self):
        """should initialize with zeros."""
        stats = DownloadStats()
        assert stats.podcasts_processed == 0
        assert stats.episodes_found == 0
        assert stats.episodes_downloaded == 0
        assert stats.episodes_skipped == 0
        assert stats.episodes_failed == 0
        assert stats.total_bytes == 0
        assert stats.errors == []

    def test_add_error(self):
        """should add error message."""
        stats = DownloadStats()
        stats.add_error("test error 1")
        stats.add_error("test error 2")
        assert len(stats.errors) == 2
        assert "test error 1" in stats.errors

    def test_success_rate_zero(self):
        """should return 0 when no downloads."""
        stats = DownloadStats()
        assert stats.success_rate == 0.0

    def test_success_rate_all_success(self):
        """should return 1.0 when all succeed."""
        stats = DownloadStats(
            episodes_downloaded=10,
            episodes_failed=0,
        )
        assert stats.success_rate == 1.0

    def test_success_rate_partial(self):
        """should calculate partial success rate."""
        stats = DownloadStats(
            episodes_downloaded=8,
            episodes_failed=2,
        )
        assert stats.success_rate == 0.8

    def test_success_rate_all_failed(self):
        """should return 0 when all fail."""
        stats = DownloadStats(
            episodes_downloaded=0,
            episodes_failed=10,
        )
        assert stats.success_rate == 0.0

    def test_mutable_tracking(self):
        """should allow updating values."""
        stats = DownloadStats()
        stats.podcasts_processed = 5
        stats.episodes_found = 50
        stats.episodes_downloaded = 45
        stats.total_bytes = 1000000000

        assert stats.podcasts_processed == 5
        assert stats.episodes_found == 50


class TestConfigIntegration:
    """integration tests for config models."""

    def test_config_with_filter(self):
        """should work with nested filter."""
        config = DownloadConfig(
            output_dir=Path("/downloads"),
            max_concurrent_downloads=8,
            default_filter=EpisodeFilter(
                max_episodes=10,
                start_date=datetime(2024, 1, 1),
            ),
        )

        assert config.output_dir == Path("/downloads")
        assert config.default_filter.max_episodes == 10
        assert config.default_filter.start_date == datetime(2024, 1, 1)

    def test_config_serialization(self):
        """should be serializable to dict."""
        config = DownloadConfig(
            output_dir=Path("/downloads"),
            max_concurrent_downloads=8,
        )
        data = config.model_dump()
        assert "output_dir" in data
        assert "max_concurrent_downloads" in data

    def test_stats_aggregation(self):
        """should support aggregating multiple stats."""
        stats1 = DownloadStats(
            podcasts_processed=1,
            episodes_found=10,
            episodes_downloaded=8,
            episodes_failed=2,
            total_bytes=1000000,
        )
        stats2 = DownloadStats(
            podcasts_processed=1,
            episodes_found=20,
            episodes_downloaded=15,
            episodes_failed=5,
            total_bytes=2000000,
        )

        # aggregate
        total = DownloadStats(
            podcasts_processed=stats1.podcasts_processed + stats2.podcasts_processed,
            episodes_found=stats1.episodes_found + stats2.episodes_found,
            episodes_downloaded=stats1.episodes_downloaded + stats2.episodes_downloaded,
            episodes_failed=stats1.episodes_failed + stats2.episodes_failed,
            total_bytes=stats1.total_bytes + stats2.total_bytes,
        )

        assert total.podcasts_processed == 2
        assert total.episodes_found == 30
        assert total.episodes_downloaded == 23
        assert total.success_rate == pytest.approx(23 / 30, rel=0.01)
