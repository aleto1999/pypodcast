"""youtube fallback for failed rss downloads.

when an rss download fails, this module can search youtube for the
episode and download it as a fallback. uses confidence scoring to
find the best match.
"""

import asyncio
import re
import shutil
from pathlib import Path
from typing import NamedTuple

from podcast_conversations.naming import sanitize_filename
from podcast_downloader.feed_parser import Episode
from podcast_downloader.providers import DownloadResult


class YouTubeSearchMatch(NamedTuple):
    """a youtube search result with confidence score."""

    video_id: str
    title: str
    channel: str
    duration: int | None  # seconds
    url: str
    confidence_score: float  # 0.0 to 1.0


class ErrorCategory(NamedTuple):
    """categorized error with retry information."""

    category: str
    message: str
    should_retry: bool
    max_retries: int


# error category definitions.
ERROR_CATEGORIES = {
    "network": {
        "keywords": ["timeout", "connection", "ssl", "certificate", "network"],
        "max_retries": 3,
    },
    "not_found": {
        "keywords": ["404", "not found", "unavailable", "private", "removed", "deleted"],
        "max_retries": 0,
    },
    "format": {
        "keywords": ["no suitable formats", "unsupported format", "format"],
        "max_retries": 1,
    },
    "permission": {
        "keywords": ["403", "forbidden", "blocked", "age", "signin"],
        "max_retries": 0,
    },
    "rate_limit": {
        "keywords": ["rate limit", "too many requests", "429", "quota"],
        "max_retries": 2,
    },
}


def categorize_error(error_message: str) -> ErrorCategory:
    """categorize an error message for appropriate handling.

    args:
        error_message: the error message to categorize.

    returns:
        error category with retry information.
    """
    error_lower = error_message.lower()

    for category, config in ERROR_CATEGORIES.items():
        if any(keyword in error_lower for keyword in config["keywords"]):
            return ErrorCategory(
                category=category,
                message=error_message,
                should_retry=config["max_retries"] > 0,
                max_retries=config["max_retries"],
            )

    return ErrorCategory(
        category="unknown",
        message=error_message,
        should_retry=True,
        max_retries=1,
    )


def calculate_confidence_score(
    episode: Episode,
    show_name: str,
    video_title: str,
    video_channel: str,
    video_duration: int | None,
) -> float:
    """calculate confidence score for a youtube video match.

    scoring factors:
    - show name in title or channel: 0.4
    - episode title word matches: 0.4
    - duration match (within 10%): 0.2

    args:
        episode: the episode we're looking for.
        show_name: the podcast show name.
        video_title: youtube video title.
        video_channel: youtube channel name.
        video_duration: video duration in seconds.

    returns:
        confidence score from 0.0 to 1.0.
    """
    score = 0.0

    title_lower = video_title.lower()
    channel_lower = video_channel.lower()
    show_lower = show_name.lower()
    episode_title_lower = episode.title.lower()

    # show name in title or channel (0.4).
    if show_lower in title_lower or show_lower in channel_lower:
        score += 0.4
    elif any(word in title_lower for word in show_lower.split() if len(word) > 3):
        score += 0.2  # partial match.

    # episode title word matches (0.4).
    episode_words = [w for w in episode_title_lower.split() if len(w) > 2]
    title_words = title_lower.split()

    if episode_words:
        word_matches = sum(1 for word in episode_words if word in title_words)
        word_ratio = word_matches / len(episode_words)
        score += 0.4 * word_ratio

    # duration match (0.2).
    if episode.duration and video_duration:
        duration_diff = abs(episode.duration - video_duration)
        if duration_diff <= episode.duration * 0.1:
            score += 0.2
        elif duration_diff <= episode.duration * 0.25:
            score += 0.1

    return min(score, 1.0)


async def search_youtube_for_episode(
    episode: Episode,
    show_name: str,
    max_results: int = 5,
) -> list[YouTubeSearchMatch]:
    """search youtube for a podcast episode.

    args:
        episode: the episode to search for.
        show_name: the podcast show name.
        max_results: maximum number of results to return.

    returns:
        list of youtube matches sorted by confidence score.
    """
    if not shutil.which("yt-dlp"):
        return []

    # build search query.
    search_query = f"{show_name} {episode.title}"
    search_query = re.sub(r"[^\w\s-]", "", search_query)
    search_query = " ".join(search_query.split())

    try:
        cmd = [
            "yt-dlp",
            "--flat-playlist",
            "--dump-json",
            "--no-warnings",
            "--quiet",
            f"ytsearch{max_results}:{search_query}",
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(
            process.communicate(),
            timeout=30.0,
        )

        if not stdout:
            return []

        import json
        results: list[YouTubeSearchMatch] = []

        for line in stdout.decode().strip().split("\n"):
            if not line:
                continue

            try:
                item = json.loads(line)
                video_id = item.get("id")
                title = item.get("title", "")
                channel = item.get("channel") or item.get("uploader") or ""
                duration = item.get("duration")

                if not video_id:
                    continue

                # skip very short videos (likely not podcast episodes).
                if duration and duration < 180:  # less than 3 minutes.
                    continue

                confidence = calculate_confidence_score(
                    episode,
                    show_name,
                    title,
                    channel,
                    duration,
                )

                results.append(
                    YouTubeSearchMatch(
                        video_id=video_id,
                        title=title,
                        channel=channel,
                        duration=duration,
                        url=f"https://www.youtube.com/watch?v={video_id}",
                        confidence_score=confidence,
                    )
                )

            except json.JSONDecodeError:
                continue

        # sort by confidence score.
        results.sort(key=lambda x: x.confidence_score, reverse=True)
        return results

    except asyncio.TimeoutError:
        return []
    except Exception:
        return []


async def download_from_youtube(
    match: YouTubeSearchMatch,
    episode: Episode,
    output_dir: Path,
    audio_quality: str = "192",
) -> DownloadResult:
    """download a youtube video as audio.

    args:
        match: the youtube match to download.
        episode: the original episode (for filename).
        output_dir: directory to save the file.
        audio_quality: audio quality in kbps.

    returns:
        download result.
    """
    if not shutil.which("yt-dlp"):
        return DownloadResult(
            episode=episode,
            success=False,
            error="yt-dlp not installed",
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # use episode title for filename, sanitized.
    safe_title = sanitize_filename(episode.title).replace(".mp3", "")
    output_template = str(output_dir / safe_title)

    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", audio_quality,
        "--output", f"{output_template}.%(ext)s",
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        match.url,
    ]

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=600.0,  # 10 minute timeout for long episodes.
        )

        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "unknown yt-dlp error"
            error_cat = categorize_error(error_msg)
            return DownloadResult(
                episode=episode,
                success=False,
                error=f"youtube fallback failed ({error_cat.category}): {error_msg}",
            )

        # find the output file.
        output_path = output_dir / f"{safe_title}.mp3"
        if not output_path.exists():
            # try finding any audio file with the base name.
            matches = list(output_dir.glob(f"{safe_title}*"))
            audio_matches = [m for m in matches if m.suffix in [".mp3", ".m4a", ".webm", ".opus"]]
            if audio_matches:
                output_path = audio_matches[0]

        if output_path.exists():
            file_size = output_path.stat().st_size
            return DownloadResult(
                episode=episode,
                success=True,
                file_path=output_path,
                bytes_downloaded=file_size,
            )

        return DownloadResult(
            episode=episode,
            success=False,
            error="output file not found after youtube download",
        )

    except asyncio.TimeoutError:
        return DownloadResult(
            episode=episode,
            success=False,
            error="youtube download timed out",
        )
    except Exception as e:
        return DownloadResult(
            episode=episode,
            success=False,
            error=f"youtube download failed: {e}",
        )


async def fallback_download_episode(
    episode: Episode,
    show_name: str,
    output_dir: Path,
    min_confidence: float = 0.5,
    max_retries: int = 2,
) -> DownloadResult:
    """attempt to download episode from youtube as fallback.

    args:
        episode: the episode that failed to download via rss.
        show_name: the podcast show name.
        output_dir: directory to save the file.
        min_confidence: minimum confidence score to attempt download.
        max_retries: maximum retry attempts per video.

    returns:
        download result.
    """
    # search for matching videos.
    matches = await search_youtube_for_episode(episode, show_name)

    if not matches:
        return DownloadResult(
            episode=episode,
            success=False,
            error="no matching videos found on youtube",
        )

    # filter by minimum confidence.
    matches = [m for m in matches if m.confidence_score >= min_confidence]

    if not matches:
        return DownloadResult(
            episode=episode,
            success=False,
            error=f"no youtube matches above confidence threshold ({min_confidence})",
        )

    # try each match in order of confidence.
    last_error = None
    for match in matches[:3]:  # try top 3 matches.
        for attempt in range(max_retries + 1):
            result = await download_from_youtube(match, episode, output_dir)

            if result.success:
                return result

            # check if we should retry.
            if result.error:
                error_cat = categorize_error(result.error)
                if not error_cat.should_retry or attempt >= error_cat.max_retries:
                    last_error = result.error
                    break

                # wait before retry.
                await asyncio.sleep(2 ** attempt)
            else:
                break

        last_error = result.error

    return DownloadResult(
        episode=episode,
        success=False,
        error=f"all youtube fallback attempts failed: {last_error}",
    )
