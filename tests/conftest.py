"""pytest configuration and shared fixtures."""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import asyncio
import tempfile
from typing import Generator

import pytest


@pytest.fixture(scope="session")
def event_loop():
    """create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_mp3_file(temp_dir: Path) -> Path:
    """create a minimal valid mp3 file for testing."""
    mp3_path = temp_dir / "test_episode.mp3"

    # minimal mp3 file with id3v2 header and one frame
    # id3v2 header (10 bytes) + mp3 frame header (4 bytes) + padding
    id3_header = b"ID3\x04\x00\x00\x00\x00\x00\x00"

    # mp3 frame sync (0xff 0xfb) + bitrate/sample rate + padding
    # this is a valid mpeg audio layer 3 frame header
    mp3_frame = b"\xff\xfb\x90\x00" + b"\x00" * 417  # ~418 bytes for 128kbps frame

    # write multiple frames to make file large enough
    with open(mp3_path, "wb") as f:
        f.write(id3_header)
        for _ in range(10):
            f.write(mp3_frame)

    return mp3_path


@pytest.fixture
def sample_invalid_file(temp_dir: Path) -> Path:
    """create an invalid audio file for testing."""
    invalid_path = temp_dir / "invalid.mp3"
    with open(invalid_path, "wb") as f:
        f.write(b"this is not an audio file at all")
    return invalid_path


@pytest.fixture
def sample_small_file(temp_dir: Path) -> Path:
    """create a file that's too small to be valid."""
    small_path = temp_dir / "small.mp3"
    with open(small_path, "wb") as f:
        f.write(b"ID3" + b"\x00" * 10)
    return small_path


@pytest.fixture
def sample_rss_feed() -> str:
    """return sample rss feed xml content."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
  <channel>
    <title>Test Podcast</title>
    <link>https://example.com/podcast</link>
    <description>A test podcast for unit testing</description>
    <itunes:author>Test Author</itunes:author>
    <itunes:image href="https://example.com/artwork.jpg"/>
    <item>
      <title>Episode 1: Introduction</title>
      <description>The first episode</description>
      <pubDate>Mon, 01 Jan 2024 00:00:00 GMT</pubDate>
      <guid>episode-1</guid>
      <enclosure url="https://example.com/episode1.mp3" type="audio/mpeg" length="1000000"/>
      <itunes:duration>3600</itunes:duration>
    </item>
    <item>
      <title>Episode 2: Deep Dive</title>
      <description>The second episode</description>
      <pubDate>Mon, 08 Jan 2024 00:00:00 GMT</pubDate>
      <guid>episode-2</guid>
      <enclosure url="https://example.com/episode2.mp3" type="audio/mpeg" length="2000000"/>
      <itunes:duration>5400</itunes:duration>
    </item>
  </channel>
</rss>"""


@pytest.fixture
def sample_episode():
    """create a sample episode for testing."""
    from podcast_downloader.feed_parser import Episode
    from datetime import datetime

    return Episode(
        id="test-episode-1",
        title="Test Episode: Special Characters & More!",
        description="A test episode with special characters",
        published=datetime(2024, 1, 15, 12, 0, 0),
        duration=3600,
        audio_url="https://example.com/test-episode.mp3",
        episode_number=1,
    )


@pytest.fixture
def sample_podcast():
    """create a sample podcast for testing."""
    from podcast_downloader.feed_parser import Podcast

    return Podcast(
        title="The Test Podcast Show",
        description="A podcast for testing purposes",
        author="Test Author",
        feed_url="https://example.com/feed.xml",
        artwork_url="https://example.com/artwork.jpg",
    )
