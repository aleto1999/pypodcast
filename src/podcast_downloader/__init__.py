"""podcast downloader package for fetching podcast episodes from rss feeds and youtube."""

__version__ = "0.1.0"

from podcast_downloader.config import DownloadConfig, EpisodeFilter
from podcast_downloader.discovery import discover_feeds
from podcast_downloader.downloader import download_podcast, download_podcasts
from podcast_downloader.feed_parser import Episode, Podcast, parse_feed

__all__ = [
    "DownloadConfig",
    "EpisodeFilter",
    "download_podcast",
    "download_podcasts",
    "discover_feeds",
    "parse_feed",
    "Episode",
    "Podcast",
]
