"""rss feed parsing and episode extraction."""

import hashlib
import re
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import AsyncIterator

import feedparser
import httpx
from pydantic import BaseModel, Field

from podcast_downloader.errors import FeedParseError, NetworkError, categorize_http_error
from podcast_downloader.filename_utils import sanitize_dirname, sanitize_filename


class Episode(BaseModel):
    """represents a podcast episode."""

    id: str = Field(description="unique episode identifier")
    title: str
    audio_url: str
    published: datetime | None = None
    duration: int | None = Field(default=None, description="duration in seconds")
    description: str | None = None
    episode_number: int | None = None
    season_number: int | None = None
    file_size: int | None = Field(default=None, description="file size in bytes")
    mime_type: str | None = None

    @property
    def safe_filename(self) -> str:
        """generate a safe filename for this episode."""
        return sanitize_filename(self.title, max_length=100, extension=".mp3")


class Podcast(BaseModel):
    """represents a podcast feed."""

    title: str
    feed_url: str
    description: str | None = None
    author: str | None = None
    artwork_url: str | None = None
    website: str | None = None
    language: str | None = None
    episode_count: int = 0

    @property
    def safe_dirname(self) -> str:
        """generate a safe directory name for this podcast."""
        return sanitize_dirname(self.title, max_length=50)


def _generate_episode_id(entry: dict, feed_url: str) -> str:
    """generate a unique id for an episode."""
    # prefer guid if available.
    if entry.get("id"):
        return entry["id"]

    # fallback to hash of url + title.
    content = f"{feed_url}:{entry.get('title', '')}:{entry.get('link', '')}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _parse_duration(duration_str: str | None) -> int | None:
    """parse duration string to seconds."""
    if not duration_str:
        return None

    try:
        # handle HH:MM:SS or MM:SS format.
        parts = duration_str.split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        else:
            return int(duration_str)
    except (ValueError, TypeError):
        return None


def _parse_published_date(entry: dict) -> datetime | None:
    """parse published date from feed entry."""
    date_str = entry.get("published") or entry.get("updated")
    if not date_str:
        return None

    try:
        return parsedate_to_datetime(date_str)
    except (ValueError, TypeError):
        return None


def _extract_audio_url(entry: dict) -> str | None:
    """extract audio url from feed entry enclosures."""
    enclosures = entry.get("enclosures", [])

    for enclosure in enclosures:
        url = enclosure.get("href") or enclosure.get("url")
        mime_type = enclosure.get("type", "")

        # check for audio mime types.
        if url and ("audio" in mime_type or url.endswith((".mp3", ".m4a", ".ogg", ".wav"))):
            return url

    # fallback to links.
    for link in entry.get("links", []):
        if link.get("type", "").startswith("audio/"):
            return link.get("href")

    return None


def _extract_file_size(entry: dict) -> int | None:
    """extract file size from enclosure."""
    for enclosure in entry.get("enclosures", []):
        length = enclosure.get("length")
        if length:
            try:
                return int(length)
            except (ValueError, TypeError):
                pass
    return None


def _extract_mime_type(entry: dict) -> str | None:
    """extract mime type from enclosure."""
    for enclosure in entry.get("enclosures", []):
        if enclosure.get("type"):
            return enclosure["type"]
    return None


def parse_feed_content(content: str, feed_url: str) -> tuple[Podcast, list[Episode]]:
    """parse rss feed content into podcast and episodes."""
    feed = feedparser.parse(content)

    if feed.bozo and not feed.entries:
        raise FeedParseError(f"failed to parse feed: {feed.bozo_exception}", feed_url)

    feed_info = feed.feed

    podcast = Podcast(
        title=feed_info.get("title", "Unknown Podcast"),
        feed_url=feed_url,
        description=feed_info.get("description") or feed_info.get("subtitle"),
        author=feed_info.get("author") or feed_info.get("itunes_author"),
        artwork_url=feed_info.get("image", {}).get("href"),
        website=feed_info.get("link"),
        language=feed_info.get("language"),
    )

    episodes = []
    for entry in feed.entries:
        audio_url = _extract_audio_url(entry)
        if not audio_url:
            continue  # skip entries without audio.

        episode = Episode(
            id=_generate_episode_id(entry, feed_url),
            title=entry.get("title", "Untitled Episode"),
            audio_url=audio_url,
            published=_parse_published_date(entry),
            duration=_parse_duration(entry.get("itunes_duration")),
            description=entry.get("summary") or entry.get("description"),
            episode_number=entry.get("itunes_episode"),
            season_number=entry.get("itunes_season"),
            file_size=_extract_file_size(entry),
            mime_type=_extract_mime_type(entry),
        )
        episodes.append(episode)

    podcast.episode_count = len(episodes)

    # sort by published date, newest first.
    episodes.sort(key=lambda e: e.published or datetime.min, reverse=True)

    return podcast, episodes


async def parse_feed(
    feed_url: str,
    client: httpx.AsyncClient | None = None,
) -> tuple[Podcast, list[Episode]]:
    """fetch and parse an rss feed."""
    should_close = client is None
    if client is None:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=15.0, read=60.0, write=30.0, pool=30.0),
            follow_redirects=True,
        )

    try:
        response = await client.get(
            feed_url,
            headers={"User-Agent": "PodcastDownloader/1.0"},
        )

        if response.status_code != 200:
            raise categorize_http_error(response.status_code, feed_url)

        return parse_feed_content(response.text, feed_url)

    except httpx.TimeoutException as e:
        raise NetworkError(f"timeout fetching feed: {e}", feed_url)
    except httpx.RequestError as e:
        raise NetworkError(f"network error fetching feed: {e}", feed_url)
    finally:
        if should_close:
            await client.aclose()


async def fetch_episodes(
    feed_url: str,
    client: httpx.AsyncClient | None = None,
) -> AsyncIterator[Episode]:
    """fetch episodes from a feed as an async iterator."""
    _, episodes = await parse_feed(feed_url, client)
    for episode in episodes:
        yield episode
