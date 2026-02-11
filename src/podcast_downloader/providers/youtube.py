"""youtube download provider using yt-dlp."""

import asyncio
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from podcast_downloader.feed_parser import Episode
from podcast_downloader.providers import DownloadResult, register_provider


class YouTubeDownloadProvider:
    """downloads audio from youtube urls using yt-dlp."""

    def __init__(self, audio_quality: str = "192"):
        self.audio_quality = audio_quality
        self._yt_dlp_available: bool | None = None

    def can_handle(self, episode: Episode) -> bool:
        """check if this provider can handle the episode."""
        url = episode.audio_url.lower()
        return any(domain in url for domain in ["youtube.com", "youtu.be"])

    def _check_yt_dlp(self) -> bool:
        """check if yt-dlp is available."""
        if self._yt_dlp_available is None:
            self._yt_dlp_available = shutil.which("yt-dlp") is not None
        return self._yt_dlp_available

    async def download(
        self,
        episode: Episode,
        output_dir: Path,
        progress_callback: Callable[[int, int], Any] | None = None,
    ) -> DownloadResult:
        """download audio from youtube using yt-dlp."""
        if not self._check_yt_dlp():
            return DownloadResult(
                episode=episode,
                success=False,
                error="yt-dlp not installed. install with: pip install yt-dlp",
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        output_template = str(output_dir / episode.safe_filename.replace(".mp3", ""))

        # yt-dlp command for audio extraction.
        cmd = [
            "yt-dlp",
            "--extract-audio",
            "--audio-format", "mp3",
            "--audio-quality", self.audio_quality,
            "--output", f"{output_template}.%(ext)s",
            "--no-playlist",
            "--quiet",
            "--no-warnings",
            episode.audio_url,
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "unknown yt-dlp error"

                # categorize error.
                if "Video unavailable" in error_msg or "not available" in error_msg.lower():
                    return DownloadResult(
                        episode=episode,
                        success=False,
                        error=f"video not available: {episode.audio_url}",
                    )

                return DownloadResult(
                    episode=episode,
                    success=False,
                    error=f"yt-dlp error: {error_msg}",
                )

            # find the output file.
            output_path = output_dir / f"{episode.safe_filename.replace('.mp3', '')}.mp3"
            if not output_path.exists():
                # try finding any mp3 file with the base name.
                matches = list(output_dir.glob(f"{episode.safe_filename.replace('.mp3', '')}*"))
                if matches:
                    output_path = matches[0]

            if output_path.exists():
                file_size = output_path.stat().st_size
                return DownloadResult(
                    episode=episode,
                    success=True,
                    file_path=output_path,
                    bytes_downloaded=file_size,
                )
            else:
                return DownloadResult(
                    episode=episode,
                    success=False,
                    error="output file not found after yt-dlp completed",
                )

        except Exception as e:
            return DownloadResult(
                episode=episode,
                success=False,
                error=f"failed to run yt-dlp: {e}",
            )


# create singleton and register.
youtube_provider = YouTubeDownloadProvider()
register_provider(youtube_provider)
