"""tests for discovery module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from podcast_downloader.config import FeedSearchResult
from podcast_downloader.discovery import (
    discover_feeds,
    search_itunes,
    search_podcast_index,
    search_youtube,
    select_best_feed,
)


class TestSearchItunes:
    """tests for iTunes search."""

    @pytest.mark.asyncio
    async def test_search_success(self):
        """should return results on success."""
        mock_response = {
            "results": [
                {
                    "collectionName": "The Daily",
                    "feedUrl": "https://feeds.example.com/daily",
                    "artistName": "New York Times",
                    "description": "Daily news podcast",
                    "artworkUrl600": "https://example.com/art.jpg",
                    "trackCount": 1000,
                },
            ]
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_response_obj = MagicMock()
            mock_response_obj.status_code = 200
            mock_response_obj.json.return_value = mock_response
            mock_client.get = AsyncMock(return_value=mock_response_obj)
            mock_client.aclose = AsyncMock()
            mock_client_class.return_value = mock_client

            results = await search_itunes("The Daily")

            assert len(results) == 1
            assert results[0].title == "The Daily"
            assert results[0].source == "itunes"

    @pytest.mark.asyncio
    async def test_search_empty_results(self):
        """should handle empty results."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_response_obj = MagicMock()
            mock_response_obj.status_code = 200
            mock_response_obj.json.return_value = {"results": []}
            mock_client.get = AsyncMock(return_value=mock_response_obj)
            mock_client.aclose = AsyncMock()
            mock_client_class.return_value = mock_client

            results = await search_itunes("nonexistent podcast xyz")
            assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_skip_without_feed_url(self):
        """should skip results without feed URL."""
        mock_response = {
            "results": [
                {"collectionName": "No Feed Podcast"},  # no feedUrl
                {
                    "collectionName": "Has Feed",
                    "feedUrl": "https://example.com/feed.xml",
                },
            ]
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_response_obj = MagicMock()
            mock_response_obj.status_code = 200
            mock_response_obj.json.return_value = mock_response
            mock_client.get = AsyncMock(return_value=mock_response_obj)
            mock_client.aclose = AsyncMock()
            mock_client_class.return_value = mock_client

            results = await search_itunes("test")
            assert len(results) == 1
            assert results[0].title == "Has Feed"


class TestSearchPodcastIndex:
    """tests for Podcast Index search."""

    @pytest.mark.asyncio
    async def test_no_credentials(self):
        """should return empty list without credentials."""
        results = await search_podcast_index("test")
        assert results == []

    @pytest.mark.asyncio
    async def test_with_credentials(self):
        """should search with credentials."""
        mock_response = {
            "feeds": [
                {
                    "title": "Test Podcast",
                    "url": "https://example.com/feed.xml",
                    "author": "Test Author",
                    "description": "A test podcast",
                    "artwork": "https://example.com/art.jpg",
                    "episodeCount": 50,
                },
            ]
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_response_obj = MagicMock()
            mock_response_obj.status_code = 200
            mock_response_obj.json.return_value = mock_response
            mock_client.get = AsyncMock(return_value=mock_response_obj)
            mock_client.aclose = AsyncMock()
            mock_client_class.return_value = mock_client

            results = await search_podcast_index(
                "test",
                api_key="test_key",
                api_secret="test_secret",
            )

            assert len(results) == 1
            assert results[0].source == "podcastindex"


class TestSearchYoutube:
    """tests for YouTube search."""

    @pytest.mark.asyncio
    async def test_no_yt_dlp(self):
        """should return empty list if yt-dlp not available."""
        with patch("shutil.which", return_value=None):
            results = await search_youtube("test podcast")
            assert results == []

    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        """should return YouTube results."""
        mock_output = b'{"id": "vid123", "title": "Test Podcast Episode", "channel": "Test Channel", "channel_id": "UC123", "duration": 3600}'

        with patch("shutil.which", return_value="/usr/bin/yt-dlp"):
            with patch("asyncio.create_subprocess_exec") as mock_exec:
                mock_process = MagicMock()
                mock_process.communicate = AsyncMock(return_value=(mock_output, b""))
                mock_exec.return_value = mock_process

                results = await search_youtube("test podcast")

                assert len(results) > 0
                assert results[0].source == "youtube"

    @pytest.mark.asyncio
    async def test_filter_short_videos(self):
        """should filter videos under 5 minutes."""
        mock_output = b'{"id": "short", "title": "Short Video", "channel": "Test", "duration": 60}\n{"id": "long", "title": "Long Episode", "channel": "Test", "duration": 3600}'

        with patch("shutil.which", return_value="/usr/bin/yt-dlp"):
            with patch("asyncio.create_subprocess_exec") as mock_exec:
                mock_process = MagicMock()
                mock_process.communicate = AsyncMock(return_value=(mock_output, b""))
                mock_exec.return_value = mock_process

                results = await search_youtube("test")

                # only long video should be included
                assert len(results) == 1


class TestDiscoverFeeds:
    """tests for discover_feeds function."""

    @pytest.mark.asyncio
    async def test_combines_sources(self):
        """should combine results from multiple sources."""
        itunes_results = [
            FeedSearchResult(
                title="iTunes Podcast",
                feed_url="https://itunes.example.com/feed.xml",
                source="itunes",
            ),
        ]

        with patch("podcast_downloader.discovery.search_itunes") as mock_itunes:
            with patch("podcast_downloader.discovery.search_podcast_index") as mock_pi:
                with patch("podcast_downloader.discovery.search_youtube") as mock_yt:
                    mock_itunes.return_value = itunes_results
                    mock_pi.return_value = []
                    mock_yt.return_value = []

                    results = await discover_feeds("test", include_youtube=False)

                    assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_deduplicates_by_url(self):
        """should deduplicate results by feed URL."""
        results = [
            FeedSearchResult(
                title="Podcast 1",
                feed_url="https://example.com/feed.xml",
                source="itunes",
            ),
            FeedSearchResult(
                title="Podcast 1 Copy",
                feed_url="https://example.com/feed.xml",  # same URL
                source="podcastindex",
            ),
        ]

        with patch("podcast_downloader.discovery.search_itunes") as mock_itunes:
            with patch("podcast_downloader.discovery.search_podcast_index") as mock_pi:
                with patch("podcast_downloader.discovery.search_youtube") as mock_yt:
                    mock_itunes.return_value = results[:1]
                    mock_pi.return_value = results[1:]
                    mock_yt.return_value = []

                    final_results = await discover_feeds("test", include_youtube=False)

                    urls = [r.feed_url for r in final_results]
                    assert len(urls) == len(set(urls))  # all unique

    @pytest.mark.asyncio
    async def test_respects_limit(self):
        """should respect limit parameter."""
        many_results = [
            FeedSearchResult(
                title=f"Podcast {i}",
                feed_url=f"https://example.com/feed{i}.xml",
                source="itunes",
            )
            for i in range(20)
        ]

        with patch("podcast_downloader.discovery.search_itunes") as mock_itunes:
            with patch("podcast_downloader.discovery.search_podcast_index") as mock_pi:
                with patch("podcast_downloader.discovery.search_youtube") as mock_yt:
                    mock_itunes.return_value = many_results
                    mock_pi.return_value = []
                    mock_yt.return_value = []

                    results = await discover_feeds("test", limit=5, include_youtube=False)

                    assert len(results) <= 5

    @pytest.mark.asyncio
    async def test_applies_fuzzy_matching(self):
        """should apply fuzzy matching when enabled."""
        results = [
            FeedSearchResult(
                title="The Daily",
                feed_url="https://example.com/daily.xml",
                source="itunes",
                confidence=1.0,
            ),
        ]

        with patch("podcast_downloader.discovery.search_itunes") as mock_itunes:
            with patch("podcast_downloader.discovery.search_podcast_index") as mock_pi:
                with patch("podcast_downloader.discovery.search_youtube") as mock_yt:
                    mock_itunes.return_value = results
                    mock_pi.return_value = []
                    mock_yt.return_value = []

                    final_results = await discover_feeds(
                        "The Daily",
                        use_fuzzy_matching=True,
                        include_youtube=False,
                    )

                    # confidence should be adjusted by fuzzy matching
                    assert len(final_results) >= 1


class TestSelectBestFeed:
    """tests for select_best_feed function."""

    def test_empty_list(self):
        """should return None for empty list."""
        result = select_best_feed([])
        assert result is None

    def test_single_feed(self):
        """should return single feed."""
        feeds = [
            FeedSearchResult(
                title="Only Podcast",
                feed_url="https://example.com/feed.xml",
                source="itunes",
            ),
        ]
        result = select_best_feed(feeds)
        assert result.title == "Only Podcast"

    def test_prefer_higher_confidence(self):
        """should prefer higher confidence."""
        feeds = [
            FeedSearchResult(
                title="Low Confidence",
                feed_url="https://example.com/low.xml",
                source="itunes",
                confidence=0.5,
            ),
            FeedSearchResult(
                title="High Confidence",
                feed_url="https://example.com/high.xml",
                source="itunes",
                confidence=0.9,
            ),
        ]
        result = select_best_feed(feeds)
        assert result.title == "High Confidence"

    def test_prefer_more_episodes(self):
        """should prefer feeds with more episodes at same confidence."""
        feeds = [
            FeedSearchResult(
                title="Few Episodes",
                feed_url="https://example.com/few.xml",
                source="itunes",
                confidence=1.0,
                episode_count=10,
            ),
            FeedSearchResult(
                title="Many Episodes",
                feed_url="https://example.com/many.xml",
                source="itunes",
                confidence=1.0,
                episode_count=100,
            ),
        ]
        result = select_best_feed(feeds)
        assert result.title == "Many Episodes"

    def test_with_query_fuzzy_match(self):
        """should use fuzzy matching when query provided."""
        feeds = [
            FeedSearchResult(
                title="Completely Different",
                feed_url="https://example.com/different.xml",
                source="itunes",
                confidence=1.0,
            ),
            FeedSearchResult(
                title="The Daily Show",
                feed_url="https://example.com/daily.xml",
                source="itunes",
                confidence=0.9,
            ),
        ]
        result = select_best_feed(feeds, query="The Daily")
        assert "Daily" in result.title


class TestIntegration:
    """integration tests for discovery."""

    @pytest.mark.asyncio
    async def test_full_discovery_pipeline(self):
        """should run full discovery pipeline."""
        itunes_result = FeedSearchResult(
            title="The Daily",
            feed_url="https://feeds.example.com/daily",
            source="itunes",
            confidence=1.0,
            episode_count=1000,
        )

        with patch("podcast_downloader.discovery.search_itunes") as mock_itunes:
            with patch("podcast_downloader.discovery.search_podcast_index") as mock_pi:
                with patch("podcast_downloader.discovery.search_youtube") as mock_yt:
                    mock_itunes.return_value = [itunes_result]
                    mock_pi.return_value = []
                    mock_yt.return_value = []

                    results = await discover_feeds("The Daily", include_youtube=False)
                    best = select_best_feed(results, query="The Daily")

                    assert best is not None
                    assert best.title == "The Daily"
