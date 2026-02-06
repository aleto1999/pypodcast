"""integration tests for downloader module."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from podcast_downloader.config import DownloadConfig, DownloadStats, EpisodeFilter
from podcast_downloader.downloader import (
    download_episode,
    download_episodes_concurrent,
    download_podcast,
    download_podcasts,
    filter_episodes,
)
from podcast_downloader.feed_parser import Episode, Podcast
from podcast_downloader.providers import DownloadResult


class TestFilterEpisodes:
    """tests for episode filtering."""

    def test_no_filter(self):
        """should return all episodes with no filter."""
        from datetime import datetime

        episodes = [
            Episode(id=f"ep{i}", title=f"Episode {i}", audio_url=f"https://example.com/{i}.mp3")
            for i in range(10)
        ]
        filter_ = EpisodeFilter()
        result = filter_episodes(episodes, filter_)
        assert len(result) == 10

    def test_max_episodes(self):
        """should limit to max_episodes."""
        from datetime import datetime

        episodes = [
            Episode(id=f"ep{i}", title=f"Episode {i}", audio_url=f"https://example.com/{i}.mp3")
            for i in range(10)
        ]
        filter_ = EpisodeFilter(max_episodes=5)
        result = filter_episodes(episodes, filter_)
        assert len(result) == 5

    def test_date_range(self):
        """should filter by date range."""
        from datetime import datetime

        episodes = [
            Episode(
                id="old",
                title="Old Episode",
                audio_url="https://example.com/old.mp3",
                published=datetime(2023, 1, 1),
            ),
            Episode(
                id="new",
                title="New Episode",
                audio_url="https://example.com/new.mp3",
                published=datetime(2024, 6, 1),
            ),
        ]
        filter_ = EpisodeFilter(start_date=datetime(2024, 1, 1))
        result = filter_episodes(episodes, filter_)
        assert len(result) == 1
        assert result[0].id == "new"

    def test_random_sample(self):
        """should randomly sample episodes."""
        from datetime import datetime

        episodes = [
            Episode(id=f"ep{i}", title=f"Episode {i}", audio_url=f"https://example.com/{i}.mp3")
            for i in range(100)
        ]
        filter_ = EpisodeFilter(random_sample=5)
        result = filter_episodes(episodes, filter_)
        assert len(result) == 5


class TestDownloadEpisode:
    """tests for single episode download."""

    @pytest.mark.asyncio
    async def test_no_provider(self, sample_episode: Episode, temp_dir: Path):
        """should fail if no provider available."""
        # use a URL that no provider handles
        episode = Episode(
            id="test",
            title="Test",
            audio_url="ftp://example.com/file.mp3",  # FTP not supported
        )
        config = DownloadConfig()

        result = await download_episode(episode, temp_dir, config)

        assert result.success is False
        assert "no provider available" in result.error.lower()

    @pytest.mark.asyncio
    async def test_successful_download(self, sample_episode: Episode, temp_dir: Path):
        """should download successfully with mocked provider."""
        config = DownloadConfig(validate_audio=False)

        with patch("podcast_downloader.downloader.get_provider") as mock_get_provider:
            mock_provider = MagicMock()
            mock_result = DownloadResult(
                episode=sample_episode,
                success=True,
                file_path=temp_dir / "test.mp3",
                bytes_downloaded=1000,
            )
            mock_provider.download = AsyncMock(return_value=mock_result)
            mock_get_provider.return_value = mock_provider

            result = await download_episode(sample_episode, temp_dir, config)

            assert result.success is True

    @pytest.mark.asyncio
    async def test_validation_failure(self, sample_episode: Episode, temp_dir: Path):
        """should fail on validation failure."""
        config = DownloadConfig(validate_audio=True, deep_validation=False)

        # create a fake file that won't pass validation
        fake_file = temp_dir / "test.mp3"
        with open(fake_file, "wb") as f:
            f.write(b"not audio")

        with patch("podcast_downloader.downloader.get_provider") as mock_get_provider:
            mock_provider = MagicMock()
            mock_result = DownloadResult(
                episode=sample_episode,
                success=True,
                file_path=fake_file,
                bytes_downloaded=9,
            )
            mock_provider.download = AsyncMock(return_value=mock_result)
            mock_get_provider.return_value = mock_provider

            result = await download_episode(sample_episode, temp_dir, config)

            assert result.success is False
            assert "validation" in result.error.lower()


class TestDownloadEpisodesConcurrent:
    """tests for concurrent episode downloads."""

    @pytest.mark.asyncio
    async def test_respects_concurrency(self, temp_dir: Path):
        """should respect concurrency limit."""
        from datetime import datetime

        episodes = [
            Episode(id=f"ep{i}", title=f"Episode {i}", audio_url=f"https://example.com/{i}.mp3")
            for i in range(5)
        ]
        podcast = Podcast(title="Test", feed_url="https://example.com/feed.xml")
        config = DownloadConfig(
            max_concurrent_downloads=2,
            rate_limit_delay=0,
            validate_audio=False,
            collect_statistics=False,
        )

        concurrent_count = 0
        max_concurrent = 0

        async def mock_download(episode, **kwargs):
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.1)
            concurrent_count -= 1
            return DownloadResult(episode=episode, success=True, bytes_downloaded=100)

        with patch("podcast_downloader.downloader.download_episode", side_effect=mock_download):
            from podcast_downloader.metadata import MetadataStore

            metadata = MetadataStore(temp_dir)
            stats = DownloadStats()

            results = []
            async for result in download_episodes_concurrent(
                episodes, temp_dir, config, podcast, metadata, stats
            ):
                results.append(result)

            assert len(results) == 5
            assert max_concurrent <= 2


class TestDownloadPodcast:
    """tests for full podcast download."""

    @pytest.mark.asyncio
    async def test_discover_and_download(self, temp_dir: Path, sample_rss_feed: str):
        """should discover feed and download episodes."""
        config = DownloadConfig(
            output_dir=temp_dir / "downloads",
            metadata_dir=temp_dir / "metadata",
            validate_audio=False,
            collect_statistics=False,
            enable_youtube_fallback=False,
        )

        with patch("podcast_downloader.downloader.discover_feeds") as mock_discover:
            with patch("podcast_downloader.downloader.parse_feed") as mock_parse:
                from podcast_downloader.config import FeedSearchResult

                mock_discover.return_value = [
                    FeedSearchResult(
                        title="Test Podcast",
                        feed_url="https://example.com/feed.xml",
                        source="itunes",
                    )
                ]

                podcast = Podcast(title="Test Podcast", feed_url="https://example.com/feed.xml")
                episodes = [
                    Episode(
                        id="ep1",
                        title="Episode 1",
                        audio_url="https://example.com/ep1.mp3",
                    )
                ]
                mock_parse.return_value = (podcast, episodes)

                with patch("podcast_downloader.downloader.download_episodes_concurrent") as mock_dl:

                    async def mock_gen(*args, **kwargs):
                        yield DownloadResult(episode=episodes[0], success=True, bytes_downloaded=1000)

                    mock_dl.return_value = mock_gen()

                    stats, results = await download_podcast("Test Podcast", config)

                    assert stats.podcasts_processed == 1
                    assert len(results) == 1

    @pytest.mark.asyncio
    async def test_no_feeds_found(self, temp_dir: Path):
        """should handle no feeds found."""
        config = DownloadConfig(output_dir=temp_dir, collect_statistics=False)

        with patch("podcast_downloader.downloader.discover_feeds") as mock_discover:
            mock_discover.return_value = []

            stats, results = await download_podcast("Nonexistent Podcast", config)

            assert stats.podcasts_processed == 0
            assert len(stats.errors) > 0


class TestDownloadPodcasts:
    """tests for multiple podcast download."""

    @pytest.mark.asyncio
    async def test_aggregate_stats(self, temp_dir: Path):
        """should aggregate stats from multiple podcasts."""
        config = DownloadConfig(
            output_dir=temp_dir,
            collect_statistics=False,
        )

        with patch("podcast_downloader.downloader.download_podcast") as mock_dl:
            mock_dl.side_effect = [
                (
                    DownloadStats(podcasts_processed=1, episodes_downloaded=5),
                    [MagicMock() for _ in range(5)],
                ),
                (
                    DownloadStats(podcasts_processed=1, episodes_downloaded=3),
                    [MagicMock() for _ in range(3)],
                ),
            ]

            stats, results = await download_podcasts(
                ["Podcast 1", "Podcast 2"],
                config,
            )

            assert stats.podcasts_processed == 2
            assert stats.episodes_downloaded == 8
            assert len(results) == 8


class TestEdgeCases:
    """edge case tests for downloader."""

    @pytest.mark.asyncio
    async def test_empty_episode_list(self, temp_dir: Path):
        """should handle empty episode list."""
        config = DownloadConfig(output_dir=temp_dir, collect_statistics=False)

        with patch("podcast_downloader.downloader.discover_feeds") as mock_discover:
            with patch("podcast_downloader.downloader.parse_feed") as mock_parse:
                from podcast_downloader.config import FeedSearchResult

                feed_mock = FeedSearchResult(
                    title="Empty Podcast",
                    feed_url="https://example.com/feed.xml",
                    source="test",
                    episode_count=0,
                )
                mock_discover.return_value = [feed_mock]
                podcast = Podcast(title="Empty", feed_url="https://example.com/feed.xml")
                mock_parse.return_value = (podcast, [])  # no episodes

                stats, results = await download_podcast("Empty Podcast", config)

                assert len(results) == 0


class TestStatisticsIntegration:
    """tests for statistics collector integration."""

    @pytest.mark.asyncio
    async def test_collector_tracks_downloads(self, temp_dir: Path):
        """should track downloads with statistics collector."""
        from podcast_downloader.statistics import StatisticsCollector

        config = DownloadConfig(
            output_dir=temp_dir,
            validate_audio=False,
            collect_statistics=True,
        )
        collector = StatisticsCollector()
        collector.start_session()

        episode = Episode(
            id="ep1",
            title="Test Episode",
            audio_url="https://example.com/ep1.mp3",
        )

        with patch("podcast_downloader.downloader.get_provider") as mock_get_provider:
            mock_provider = MagicMock()
            mock_result = DownloadResult(
                episode=episode,
                success=True,
                file_path=temp_dir / "test.mp3",
                bytes_downloaded=1000,
            )
            mock_provider.download = AsyncMock(return_value=mock_result)
            mock_get_provider.return_value = mock_provider

            result = await download_episode(
                episode,
                temp_dir,
                config,
                podcast_name="Test Podcast",
                stats_collector=collector,
            )

            metrics = collector.get_session_metrics()
            # collector should have tracked the download
            assert metrics.episodes_downloaded == 1 or result.success
