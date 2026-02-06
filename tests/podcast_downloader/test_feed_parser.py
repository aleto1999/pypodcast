"""tests for feed_parser module."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from podcast_downloader.feed_parser import (
    Episode,
    Podcast,
    _extract_audio_url,
    _extract_file_size,
    _extract_mime_type,
    _generate_episode_id,
    _parse_duration,
    _parse_published_date,
    parse_feed,
    parse_feed_content,
)


class TestEpisode:
    """tests for Episode model."""

    def test_create_episode(self):
        """should create episode with required fields."""
        episode = Episode(
            id="ep1",
            title="Test Episode",
            audio_url="https://example.com/ep1.mp3",
        )
        assert episode.id == "ep1"
        assert episode.title == "Test Episode"

    def test_optional_fields(self):
        """should accept optional fields."""
        episode = Episode(
            id="ep1",
            title="Test Episode",
            audio_url="https://example.com/ep1.mp3",
            published=datetime(2024, 1, 15),
            duration=3600,
            description="A test episode",
            episode_number=1,
            season_number=1,
            file_size=50000000,
            mime_type="audio/mpeg",
        )
        assert episode.duration == 3600
        assert episode.episode_number == 1

    def test_safe_filename(self):
        """should generate safe filename."""
        episode = Episode(
            id="ep1",
            title="Episode: Test & Special!",
            audio_url="https://example.com/ep1.mp3",
        )
        filename = episode.safe_filename
        assert filename.endswith(".mp3")
        assert ":" not in filename
        assert "&" not in filename

    def test_safe_filename_lowercase(self):
        """should generate lowercase filename."""
        episode = Episode(
            id="ep1",
            title="UPPERCASE Episode",
            audio_url="https://example.com/ep1.mp3",
        )
        assert episode.safe_filename == episode.safe_filename.lower()


class TestPodcast:
    """tests for Podcast model."""

    def test_create_podcast(self):
        """should create podcast with required fields."""
        podcast = Podcast(
            title="Test Podcast",
            feed_url="https://example.com/feed.xml",
        )
        assert podcast.title == "Test Podcast"
        assert podcast.feed_url == "https://example.com/feed.xml"

    def test_optional_fields(self):
        """should accept optional fields."""
        podcast = Podcast(
            title="Test Podcast",
            feed_url="https://example.com/feed.xml",
            description="A test podcast",
            author="Test Author",
            artwork_url="https://example.com/art.jpg",
            website="https://example.com",
            language="en",
            episode_count=100,
        )
        assert podcast.author == "Test Author"
        assert podcast.episode_count == 100

    def test_safe_dirname(self):
        """should generate safe directory name."""
        podcast = Podcast(
            title="The Daily: News & Analysis!",
            feed_url="https://example.com/feed.xml",
        )
        dirname = podcast.safe_dirname
        assert ":" not in dirname
        assert "&" not in dirname
        assert len(dirname) <= 50


class TestParseDuration:
    """tests for duration parsing."""

    def test_hh_mm_ss_format(self):
        """should parse HH:MM:SS format."""
        assert _parse_duration("01:30:00") == 5400
        assert _parse_duration("00:45:30") == 2730

    def test_mm_ss_format(self):
        """should parse MM:SS format."""
        assert _parse_duration("45:30") == 2730
        assert _parse_duration("60:00") == 3600

    def test_seconds_only(self):
        """should parse seconds only."""
        assert _parse_duration("3600") == 3600
        assert _parse_duration("90") == 90

    def test_none_input(self):
        """should handle None input."""
        assert _parse_duration(None) is None

    def test_invalid_format(self):
        """should handle invalid format."""
        assert _parse_duration("invalid") is None
        assert _parse_duration("1:2:3:4") is None


class TestGenerateEpisodeId:
    """tests for episode ID generation."""

    def test_use_guid_if_available(self):
        """should use guid if available."""
        entry = {"id": "guid-123"}
        result = _generate_episode_id(entry, "https://example.com/feed.xml")
        assert result == "guid-123"

    def test_fallback_to_hash(self):
        """should generate hash if no guid."""
        entry = {"title": "Test Episode", "link": "https://example.com/ep1"}
        result = _generate_episode_id(entry, "https://example.com/feed.xml")
        assert len(result) == 16  # truncated sha256

    def test_deterministic_hash(self):
        """should generate same hash for same input."""
        entry = {"title": "Test Episode"}
        result1 = _generate_episode_id(entry, "https://example.com/feed.xml")
        result2 = _generate_episode_id(entry, "https://example.com/feed.xml")
        assert result1 == result2


class TestExtractAudioUrl:
    """tests for audio URL extraction."""

    def test_from_enclosure_href(self):
        """should extract from enclosure href."""
        entry = {
            "enclosures": [{"href": "https://example.com/ep1.mp3", "type": "audio/mpeg"}]
        }
        assert _extract_audio_url(entry) == "https://example.com/ep1.mp3"

    def test_from_enclosure_url(self):
        """should extract from enclosure url."""
        entry = {
            "enclosures": [{"url": "https://example.com/ep1.mp3", "type": "audio/mpeg"}]
        }
        assert _extract_audio_url(entry) == "https://example.com/ep1.mp3"

    def test_prefer_audio_type(self):
        """should prefer audio enclosure."""
        entry = {
            "enclosures": [
                {"href": "https://example.com/image.jpg", "type": "image/jpeg"},
                {"href": "https://example.com/ep1.mp3", "type": "audio/mpeg"},
            ]
        }
        assert _extract_audio_url(entry) == "https://example.com/ep1.mp3"

    def test_detect_by_extension(self):
        """should detect audio by file extension."""
        entry = {
            "enclosures": [{"href": "https://example.com/ep1.mp3", "type": ""}]
        }
        assert _extract_audio_url(entry) == "https://example.com/ep1.mp3"

    def test_no_audio_url(self):
        """should return None if no audio found."""
        entry = {
            "enclosures": [{"href": "https://example.com/image.jpg", "type": "image/jpeg"}]
        }
        assert _extract_audio_url(entry) is None

    def test_fallback_to_links(self):
        """should fallback to links if no enclosure."""
        entry = {
            "enclosures": [],
            "links": [{"href": "https://example.com/ep1.mp3", "type": "audio/mpeg"}],
        }
        assert _extract_audio_url(entry) == "https://example.com/ep1.mp3"


class TestExtractFileSize:
    """tests for file size extraction."""

    def test_extract_size(self):
        """should extract file size."""
        entry = {"enclosures": [{"length": "50000000"}]}
        assert _extract_file_size(entry) == 50000000

    def test_integer_length(self):
        """should handle integer length."""
        entry = {"enclosures": [{"length": 50000000}]}
        assert _extract_file_size(entry) == 50000000

    def test_no_length(self):
        """should return None if no length."""
        entry = {"enclosures": [{"href": "https://example.com/ep1.mp3"}]}
        assert _extract_file_size(entry) is None

    def test_invalid_length(self):
        """should handle invalid length."""
        entry = {"enclosures": [{"length": "invalid"}]}
        assert _extract_file_size(entry) is None


class TestExtractMimeType:
    """tests for MIME type extraction."""

    def test_extract_mime_type(self):
        """should extract MIME type."""
        entry = {"enclosures": [{"type": "audio/mpeg"}]}
        assert _extract_mime_type(entry) == "audio/mpeg"

    def test_no_mime_type(self):
        """should return None if no type."""
        entry = {"enclosures": [{"href": "https://example.com/ep1.mp3"}]}
        assert _extract_mime_type(entry) is None


class TestParseFeedContent:
    """tests for feed content parsing."""

    def test_parse_valid_feed(self, sample_rss_feed: str):
        """should parse valid RSS feed."""
        podcast, episodes = parse_feed_content(
            sample_rss_feed,
            "https://example.com/feed.xml",
        )

        assert podcast.title == "Test Podcast"
        assert podcast.author == "Test Author"
        assert len(episodes) == 2

    def test_parse_episodes(self, sample_rss_feed: str):
        """should parse episode details."""
        _, episodes = parse_feed_content(
            sample_rss_feed,
            "https://example.com/feed.xml",
        )

        ep1 = next(e for e in episodes if "Introduction" in e.title)
        assert ep1.audio_url == "https://example.com/episode1.mp3"
        assert ep1.duration == 3600

    def test_episodes_sorted_by_date(self, sample_rss_feed: str):
        """should sort episodes by date, newest first."""
        _, episodes = parse_feed_content(
            sample_rss_feed,
            "https://example.com/feed.xml",
        )

        # Episode 2 is newer
        assert "Episode 2" in episodes[0].title

    def test_skip_entries_without_audio(self):
        """should skip entries without audio URL."""
        feed_content = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test Podcast</title>
    <item>
      <title>No Audio Entry</title>
      <description>No enclosure</description>
    </item>
    <item>
      <title>Has Audio</title>
      <enclosure url="https://example.com/ep.mp3" type="audio/mpeg"/>
    </item>
  </channel>
</rss>"""

        _, episodes = parse_feed_content(feed_content, "https://example.com/feed.xml")
        assert len(episodes) == 1
        assert "Has Audio" in episodes[0].title


class TestParseFeed:
    """tests for async feed parsing."""

    @pytest.mark.asyncio
    async def test_parse_feed_success(self, sample_rss_feed: str):
        """should fetch and parse feed."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = sample_rss_feed
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.aclose = AsyncMock()
            mock_client_class.return_value = mock_client

            podcast, episodes = await parse_feed("https://example.com/feed.xml")

            assert podcast.title == "Test Podcast"
            assert len(episodes) == 2

    @pytest.mark.asyncio
    async def test_parse_feed_network_error(self):
        """should raise NetworkError on network failure."""
        import httpx

        from podcast_downloader.errors import NetworkError

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            mock_client.aclose = AsyncMock()
            mock_client_class.return_value = mock_client

            with pytest.raises(NetworkError):
                await parse_feed("https://example.com/feed.xml")


class TestEdgeCases:
    """edge case tests."""

    def test_empty_feed(self):
        """should handle empty feed."""
        feed_content = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Empty Podcast</title>
  </channel>
</rss>"""

        podcast, episodes = parse_feed_content(feed_content, "https://example.com/feed.xml")
        assert podcast.title == "Empty Podcast"
        assert len(episodes) == 0

    def test_missing_optional_fields(self):
        """should handle missing optional fields."""
        feed_content = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Minimal Podcast</title>
    <item>
      <title>Episode</title>
      <enclosure url="https://example.com/ep.mp3" type="audio/mpeg"/>
    </item>
  </channel>
</rss>"""

        podcast, episodes = parse_feed_content(feed_content, "https://example.com/feed.xml")
        assert podcast.author is None
        assert podcast.description is None
        assert episodes[0].duration is None

    def test_unicode_content(self):
        """should handle unicode content."""
        feed_content = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Podcast café émission</title>
    <item>
      <title>Épisode spécial: München</title>
      <enclosure url="https://example.com/ep.mp3" type="audio/mpeg"/>
    </item>
  </channel>
</rss>"""

        podcast, episodes = parse_feed_content(feed_content, "https://example.com/feed.xml")
        assert "café" in podcast.title
        assert "Épisode" in episodes[0].title
