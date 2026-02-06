"""rss/http download provider for direct audio file downloads."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from podcast_downloader.errors import (
    NetworkError,
    categorize_http_error,
)
from podcast_downloader.feed_parser import Episode
from podcast_downloader.providers import DownloadResult, register_provider
from podcast_downloader.retry import retry_with_backoff


class RSSDownloadProvider:
    """downloads audio files directly from rss feed enclosure urls."""

    def __init__(
        self,
        connect_timeout: float = 15.0,
        read_timeout: float = 300.0,
        chunk_size: int = 8192,
    ):
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.chunk_size = chunk_size

    def can_handle(self, episode: Episode) -> bool:
        """check if this provider can handle the episode."""
        url = episode.audio_url.lower()
        # handle direct http(s) urls to audio files.
        return url.startswith(("http://", "https://")) and not self._is_youtube_url(url)

    def _is_youtube_url(self, url: str) -> bool:
        """check if url is a youtube url."""
        return any(domain in url for domain in ["youtube.com", "youtu.be"])

    async def download(
        self,
        episode: Episode,
        output_dir: Path,
        progress_callback: Callable[[int, int], Any] | None = None,
    ) -> DownloadResult:
        """download an episode to the output directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / episode.safe_filename

        async def _download() -> DownloadResult:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=self.connect_timeout,
                    read=self.read_timeout,
                    write=30.0,
                    pool=30.0,
                ),
                follow_redirects=True,
            ) as client:
                try:
                    async with client.stream(
                        "GET",
                        episode.audio_url,
                        headers={"User-Agent": "PodcastDownloader/1.0"},
                    ) as response:
                        if response.status_code != 200:
                            raise categorize_http_error(
                                response.status_code,
                                episode.audio_url,
                            )

                        total_size = int(response.headers.get("content-length", 0))
                        downloaded = 0

                        with open(output_path, "wb") as f:
                            async for chunk in response.aiter_bytes(self.chunk_size):
                                f.write(chunk)
                                downloaded += len(chunk)
                                if progress_callback:
                                    progress_callback(downloaded, total_size)

                        return DownloadResult(
                            episode=episode,
                            success=True,
                            file_path=output_path,
                            bytes_downloaded=downloaded,
                        )

                except httpx.TimeoutException as e:
                    raise NetworkError(f"timeout downloading: {e}", episode.audio_url)
                except httpx.RequestError as e:
                    raise NetworkError(f"network error: {e}", episode.audio_url)

        try:
            return await retry_with_backoff(_download)
        except Exception as e:
            # clean up partial file on failure.
            if output_path.exists():
                output_path.unlink()

            return DownloadResult(
                episode=episode,
                success=False,
                error=str(e),
            )


# create singleton and register.
rss_provider = RSSDownloadProvider()
register_provider(rss_provider)
