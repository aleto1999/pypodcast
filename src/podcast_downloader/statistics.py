"""statistics collection for download sessions.

tracks download performance, timing, and errors for monitoring
and debugging purposes.
"""

import time
from collections import deque
from typing import Any, NamedTuple


class DownloadMetrics(NamedTuple):
    """metrics for a single download."""

    episode_title: str
    podcast_name: str
    success: bool
    bytes_downloaded: int
    duration_seconds: float
    error_message: str | None = None
    download_speed_mbps: float = 0.0


class SessionMetrics(NamedTuple):
    """aggregate metrics for a session."""

    duration_seconds: float
    podcasts_processed: int
    episodes_found: int
    episodes_downloaded: int
    episodes_failed: int
    episodes_skipped: int
    total_bytes: int
    success_rate: float
    average_speed_mbps: float
    peak_concurrent: int
    errors_by_category: dict[str, int]


class StatisticsCollector:
    """collects and manages download statistics.

    tracks:
    - download counts (success, failure, skip)
    - download speeds and timing
    - error categorization
    - concurrent download peaks
    - session totals
    """

    def __init__(self) -> None:
        """initialize statistics collector."""
        self._session_start: float | None = None
        self._download_metrics: deque[DownloadMetrics] = deque(maxlen=1000)
        self._download_speeds: deque[float] = deque(maxlen=100)
        self._processing_times: deque[float] = deque(maxlen=100)

        # counters.
        self._podcasts_processed = 0
        self._episodes_found = 0
        self._episodes_downloaded = 0
        self._episodes_failed = 0
        self._episodes_skipped = 0
        self._total_bytes = 0

        # concurrency tracking.
        self._concurrent_downloads = 0
        self._peak_concurrent = 0

        # error tracking.
        self._errors_by_category: dict[str, int] = {}
        self._error_log: list[dict[str, Any]] = []

        # timing.
        self._download_start_times: dict[str, float] = {}

    def start_session(self) -> None:
        """start a new statistics session."""
        self._session_start = time.time()
        self.reset_counters()

    def reset_counters(self) -> None:
        """reset all counters but keep timing data."""
        self._podcasts_processed = 0
        self._episodes_found = 0
        self._episodes_downloaded = 0
        self._episodes_failed = 0
        self._episodes_skipped = 0
        self._total_bytes = 0
        self._concurrent_downloads = 0
        self._peak_concurrent = 0
        self._errors_by_category.clear()
        self._error_log.clear()
        self._download_start_times.clear()

    def record_podcast_processed(self, podcast_name: str, episode_count: int) -> None:
        """record a processed podcast.

        args:
            podcast_name: name of the podcast.
            episode_count: number of episodes found.
        """
        self._podcasts_processed += 1
        self._episodes_found += episode_count

    def record_episodes_skipped(self, count: int) -> None:
        """record skipped episodes.

        args:
            count: number of episodes skipped.
        """
        self._episodes_skipped += count

    def start_download(self, episode_id: str) -> None:
        """mark the start of a download.

        args:
            episode_id: unique identifier for the episode.
        """
        self._download_start_times[episode_id] = time.time()
        self._concurrent_downloads += 1
        self._peak_concurrent = max(self._peak_concurrent, self._concurrent_downloads)

    def end_download(
        self,
        episode_id: str,
        episode_title: str,
        podcast_name: str,
        success: bool,
        bytes_downloaded: int = 0,
        error_message: str | None = None,
    ) -> None:
        """record the end of a download.

        args:
            episode_id: unique identifier for the episode.
            episode_title: title of the episode.
            podcast_name: name of the podcast.
            success: whether the download succeeded.
            bytes_downloaded: number of bytes downloaded.
            error_message: error message if failed.
        """
        self._concurrent_downloads = max(0, self._concurrent_downloads - 1)

        # calculate duration.
        start_time = self._download_start_times.pop(episode_id, None)
        duration = time.time() - start_time if start_time else 0.0

        # calculate speed.
        speed_mbps = 0.0
        if duration > 0 and bytes_downloaded > 0:
            speed_mbps = (bytes_downloaded * 8) / (duration * 1_000_000)
            self._download_speeds.append(speed_mbps)

        self._processing_times.append(duration)

        # update counters.
        if success:
            self._episodes_downloaded += 1
            self._total_bytes += bytes_downloaded
        else:
            self._episodes_failed += 1
            if error_message:
                category = self._categorize_error(error_message)
                self._errors_by_category[category] = self._errors_by_category.get(category, 0) + 1
                self._error_log.append({
                    "episode": episode_title,
                    "podcast": podcast_name,
                    "category": category,
                    "message": error_message,
                    "timestamp": time.time(),
                })

        # record metrics.
        metrics = DownloadMetrics(
            episode_title=episode_title,
            podcast_name=podcast_name,
            success=success,
            bytes_downloaded=bytes_downloaded,
            duration_seconds=duration,
            error_message=error_message,
            download_speed_mbps=speed_mbps,
        )
        self._download_metrics.append(metrics)

    def _categorize_error(self, error_message: str) -> str:
        """categorize an error message.

        args:
            error_message: the error message.

        returns:
            error category string.
        """
        error_lower = error_message.lower()

        if any(kw in error_lower for kw in ["timeout", "timed out"]):
            return "timeout"
        if any(kw in error_lower for kw in ["connection", "network", "ssl"]):
            return "network"
        if any(kw in error_lower for kw in ["404", "not found"]):
            return "not_found"
        if any(kw in error_lower for kw in ["403", "forbidden", "blocked"]):
            return "permission"
        if any(kw in error_lower for kw in ["validation", "invalid", "corrupt"]):
            return "validation"
        if any(kw in error_lower for kw in ["rate limit", "429", "quota"]):
            return "rate_limit"

        return "unknown"

    def get_session_metrics(self) -> SessionMetrics:
        """get aggregate metrics for the current session.

        returns:
            session metrics.
        """
        duration = time.time() - self._session_start if self._session_start else 0.0
        total_attempts = self._episodes_downloaded + self._episodes_failed

        return SessionMetrics(
            duration_seconds=duration,
            podcasts_processed=self._podcasts_processed,
            episodes_found=self._episodes_found,
            episodes_downloaded=self._episodes_downloaded,
            episodes_failed=self._episodes_failed,
            episodes_skipped=self._episodes_skipped,
            total_bytes=self._total_bytes,
            success_rate=self._episodes_downloaded / total_attempts if total_attempts > 0 else 0.0,
            average_speed_mbps=sum(self._download_speeds) / len(self._download_speeds) if self._download_speeds else 0.0,
            peak_concurrent=self._peak_concurrent,
            errors_by_category=dict(self._errors_by_category),
        )

    def get_download_rate(self) -> float:
        """get current download rate in episodes per minute.

        returns:
            episodes per minute.
        """
        if not self._session_start:
            return 0.0

        elapsed = time.time() - self._session_start
        if elapsed <= 0:
            return 0.0

        return (self._episodes_downloaded / elapsed) * 60

    def get_estimated_completion(self, remaining_episodes: int) -> float | None:
        """estimate completion time for remaining episodes.

        args:
            remaining_episodes: number of episodes left to download.

        returns:
            estimated seconds until completion, or none if unknown.
        """
        rate = self.get_download_rate()
        if rate <= 0:
            return None

        return (remaining_episodes / rate) * 60

    def get_recent_errors(self, limit: int = 10) -> list[dict[str, Any]]:
        """get recent error log entries.

        args:
            limit: maximum number of entries to return.

        returns:
            list of error log entries.
        """
        return self._error_log[-limit:]

    def get_error_summary(self) -> dict[str, int]:
        """get summary of errors by category.

        returns:
            dict mapping category to count.
        """
        return dict(self._errors_by_category)

    def format_summary(self) -> str:
        """format a human-readable summary of the session.

        returns:
            formatted summary string.
        """
        metrics = self.get_session_metrics()

        lines = [
            "download session summary",
            "=" * 40,
            f"duration: {metrics.duration_seconds:.1f}s",
            f"podcasts processed: {metrics.podcasts_processed}",
            f"episodes found: {metrics.episodes_found}",
            f"episodes downloaded: {metrics.episodes_downloaded}",
            f"episodes skipped: {metrics.episodes_skipped}",
            f"episodes failed: {metrics.episodes_failed}",
            f"success rate: {metrics.success_rate * 100:.1f}%",
            f"total size: {metrics.total_bytes / (1024 * 1024):.1f}mb",
            f"average speed: {metrics.average_speed_mbps:.2f}mbps",
            f"peak concurrent: {metrics.peak_concurrent}",
        ]

        if metrics.errors_by_category:
            lines.append("")
            lines.append("errors by category:")
            for category, count in sorted(metrics.errors_by_category.items()):
                lines.append(f"  {category}: {count}")

        return "\n".join(lines)


# global collector instance for convenience.
_global_collector: StatisticsCollector | None = None


def get_collector() -> StatisticsCollector:
    """get the global statistics collector.

    returns:
        the global statistics collector instance.
    """
    global _global_collector
    if _global_collector is None:
        _global_collector = StatisticsCollector()
    return _global_collector


def reset_collector() -> None:
    """reset the global statistics collector."""
    global _global_collector
    _global_collector = None
