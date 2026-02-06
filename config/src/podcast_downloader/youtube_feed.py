"""youtube channel/playlist parsing to extract episodes."""

import asyncio
import json
import shutil
from datetime import datetime

from podcast_downloader.feed_parser import Episode, Podcast
from podcast_downloader.filename_utils import sanitize_filename


def is_youtube_url(url: str) -> bool:
    """check if url is a youtube url."""
    return any(domain in url.lower() for domain in ["youtube.com", "youtu.be"])


async def parse_youtube_feed(
    url: str,
    max_episodes: int = 50,
) -> tuple[Podcast, list[Episode]]:
    """
    parse a youtube channel or playlist url to extract episodes.

    uses yt-dlp to fetch video metadata from the channel/playlist.

    args:
        url: youtube channel, playlist, or video url.
        max_episodes: maximum number of episodes to fetch.

    returns:
        tuple of (podcast, episodes).
    """
    if not shutil.which("yt-dlp"):
        raise RuntimeError("yt-dlp not installed. install with: pip install yt-dlp")

    # build yt-dlp command to extract metadata.
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--dump-json",
        "--no-warnings",
        "--quiet",
        "--playlist-end", str(max_episodes),
        url,
    ]

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60.0)

    except asyncio.TimeoutError:
        raise RuntimeError(f"timeout fetching youtube feed: {url}")

    if process.returncode != 0:
        error_msg = stderr.decode() if stderr else "unknown yt-dlp error"
        raise RuntimeError(f"yt-dlp error: {error_msg}")

    # parse json output (one json object per line).
    episodes: list[Episode] = []
    channel_name = "YouTube Channel"
    channel_id = None

    for line in stdout.decode().strip().split("\n"):
        if not line:
            continue

        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue

        video_id = item.get("id")
        if not video_id:
            continue

        title = item.get("title", "Untitled")
        duration = item.get("duration")  # in seconds.
        upload_date = item.get("upload_date")  # YYYYMMDD format.

        # extract channel info.
        if not channel_id:
            channel_name = item.get("channel") or item.get("uploader") or "YouTube Channel"
            channel_id = item.get("channel_id") or item.get("uploader_id")

        # parse upload date.
        published = None
        if upload_date and len(upload_date) == 8:
            try:
                published = datetime.strptime(upload_date, "%Y%m%d")
            except ValueError:
                pass

        # skip very short videos (likely not podcast episodes).
        if duration and duration < 180:  # less than 3 minutes.
            continue

        video_url = f"https://www.youtube.com/watch?v={video_id}"

        episode = Episode(
            id=video_id,
            title=title,
            audio_url=video_url,
            published=published,
            duration=duration,
            description=item.get("description"),
        )
        episodes.append(episode)

    # create podcast object.
    podcast = Podcast(
        title=channel_name,
        feed_url=url,
        author=channel_name,
        description=f"YouTube channel: {channel_name}",
        episode_count=len(episodes),
    )

    # sort by published date, newest first.
    episodes.sort(key=lambda e: e.published or datetime.min, reverse=True)

    return podcast, episodes


async def get_youtube_video_metadata(video_url: str) -> dict | None:
    """
    get metadata for a single youtube video.

    returns dict with video info or None if failed.
    """
    if not shutil.which("yt-dlp"):
        return None

    cmd = [
        "yt-dlp",
        "--dump-json",
        "--no-warnings",
        "--quiet",
        "--no-playlist",
        video_url,
    ]

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=30.0)

        if process.returncode == 0 and stdout:
            return json.loads(stdout.decode())

    except (asyncio.TimeoutError, json.JSONDecodeError):
        pass

    return None
