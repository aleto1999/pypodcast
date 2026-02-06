"""tests for statistics module."""

import time

import pytest

from podcast_downloader.statistics import (
    DownloadMetrics,
    SessionMetrics,
    StatisticsCollector,
    get_collector,
    reset_collector,
)


class TestDownloadMetrics:
    """tests for DownloadMetrics namedtuple."""

    def test_create_success_metrics(self):
        """should create metrics for successful download."""
        metrics = DownloadMetrics(
            episode_title="Test Episode",
            podcast_name="Test Podcast",
            success=True,
            bytes_downloaded=1000000,
            duration_seconds=10.5,
            download_speed_mbps=0.76,
        )
        assert metrics.success is True
        assert metrics.bytes_downloaded == 1000000
        assert metrics.error_message is None

    def test_create_failure_metrics(self):
        """should create metrics for failed download."""
        metrics = DownloadMetrics(
            episode_title="Test Episode",
            podcast_name="Test Podcast",
            success=False,
            bytes_downloaded=0,
            duration_seconds=5.0,
            error_message="connection timeout",
        )
        assert metrics.success is False
        assert metrics.error_message == "connection timeout"


class TestSessionMetrics:
    """tests for SessionMetrics namedtuple."""

    def test_create_session_metrics(self):
        """should create session metrics."""
        metrics = SessionMetrics(
            duration_seconds=120.5,
            podcasts_processed=3,
            episodes_found=50,
            episodes_downloaded=45,
            episodes_failed=5,
            episodes_skipped=10,
            total_bytes=100000000,
            success_rate=0.9,
            average_speed_mbps=5.5,
            peak_concurrent=4,
            errors_by_category={"timeout": 3, "network": 2},
        )
        assert metrics.success_rate == 0.9
        assert metrics.peak_concurrent == 4


class TestStatisticsCollector:
    """tests for StatisticsCollector class."""

    def test_initialization(self):
        """should initialize with zero values."""
        collector = StatisticsCollector()
        metrics = collector.get_session_metrics()
        assert metrics.episodes_downloaded == 0
        assert metrics.episodes_failed == 0

    def test_start_session(self):
        """should start session timer."""
        collector = StatisticsCollector()
        collector.start_session()
        metrics = collector.get_session_metrics()
        assert metrics.duration_seconds >= 0

    def test_record_podcast_processed(self):
        """should record podcast processing."""
        collector = StatisticsCollector()
        collector.start_session()
        collector.record_podcast_processed("Test Podcast", 10)
        metrics = collector.get_session_metrics()
        assert metrics.podcasts_processed == 1
        assert metrics.episodes_found == 10

    def test_record_episodes_skipped(self):
        """should record skipped episodes."""
        collector = StatisticsCollector()
        collector.start_session()
        collector.record_episodes_skipped(5)
        metrics = collector.get_session_metrics()
        assert metrics.episodes_skipped == 5

    def test_start_and_end_download(self):
        """should track download start and end."""
        collector = StatisticsCollector()
        collector.start_session()

        collector.start_download("ep1")
        collector.end_download(
            episode_id="ep1",
            episode_title="Episode 1",
            podcast_name="Test Podcast",
            success=True,
            bytes_downloaded=1000000,
        )

        metrics = collector.get_session_metrics()
        assert metrics.episodes_downloaded == 1
        assert metrics.total_bytes == 1000000

    def test_track_failed_download(self):
        """should track failed downloads."""
        collector = StatisticsCollector()
        collector.start_session()

        collector.start_download("ep1")
        collector.end_download(
            episode_id="ep1",
            episode_title="Episode 1",
            podcast_name="Test Podcast",
            success=False,
            bytes_downloaded=0,
            error_message="network timeout",
        )

        metrics = collector.get_session_metrics()
        assert metrics.episodes_failed == 1
        assert "timeout" in metrics.errors_by_category

    def test_concurrent_tracking(self):
        """should track concurrent downloads."""
        collector = StatisticsCollector()
        collector.start_session()

        # start multiple downloads
        collector.start_download("ep1")
        collector.start_download("ep2")
        collector.start_download("ep3")

        metrics = collector.get_session_metrics()
        assert metrics.peak_concurrent == 3

        # end some
        collector.end_download("ep1", "Ep1", "Podcast", True, 1000)
        collector.end_download("ep2", "Ep2", "Podcast", True, 1000)

        # peak should still be 3
        metrics = collector.get_session_metrics()
        assert metrics.peak_concurrent == 3

    def test_error_categorization(self):
        """should categorize errors correctly."""
        collector = StatisticsCollector()
        collector.start_session()

        test_cases = [
            ("connection timeout", "timeout"),
            ("network error: ssl failed", "network"),
            ("404 not found", "not_found"),
            ("403 forbidden", "permission"),
            ("file validation failed", "validation"),
            ("rate limit exceeded", "rate_limit"),
            ("unknown error xyz", "unknown"),
        ]

        for i, (error_msg, expected_category) in enumerate(test_cases):
            collector.start_download(f"ep{i}")
            collector.end_download(
                episode_id=f"ep{i}",
                episode_title=f"Episode {i}",
                podcast_name="Test",
                success=False,
                error_message=error_msg,
            )

        metrics = collector.get_session_metrics()
        assert "timeout" in metrics.errors_by_category
        assert "network" in metrics.errors_by_category
        assert "not_found" in metrics.errors_by_category

    def test_success_rate_calculation(self):
        """should calculate success rate correctly."""
        collector = StatisticsCollector()
        collector.start_session()

        # 8 successes, 2 failures = 80% success rate
        for i in range(8):
            collector.start_download(f"success{i}")
            collector.end_download(f"success{i}", f"Ep{i}", "Podcast", True, 1000)

        for i in range(2):
            collector.start_download(f"fail{i}")
            collector.end_download(f"fail{i}", f"Fail{i}", "Podcast", False, 0, "error")

        metrics = collector.get_session_metrics()
        assert abs(metrics.success_rate - 0.8) < 0.01

    def test_download_rate(self):
        """should calculate download rate."""
        collector = StatisticsCollector()
        collector.start_session()

        # simulate some downloads
        for i in range(5):
            collector.start_download(f"ep{i}")
            collector.end_download(f"ep{i}", f"Ep{i}", "Podcast", True, 1000)

        rate = collector.get_download_rate()
        assert rate > 0  # should be positive after downloads

    def test_estimated_completion(self):
        """should estimate completion time."""
        collector = StatisticsCollector()
        collector.start_session()

        # simulate some downloads to establish rate
        for i in range(5):
            collector.start_download(f"ep{i}")
            collector.end_download(f"ep{i}", f"Ep{i}", "Podcast", True, 1000)
            time.sleep(0.01)  # small delay

        estimate = collector.get_estimated_completion(remaining_episodes=10)
        # should return a number or None
        assert estimate is None or estimate >= 0

    def test_get_recent_errors(self):
        """should return recent errors."""
        collector = StatisticsCollector()
        collector.start_session()

        for i in range(15):
            collector.start_download(f"ep{i}")
            collector.end_download(f"ep{i}", f"Ep{i}", "Podcast", False, 0, f"error {i}")

        errors = collector.get_recent_errors(limit=10)
        assert len(errors) == 10
        assert errors[-1]["message"] == "error 14"  # most recent

    def test_get_error_summary(self):
        """should return error summary."""
        collector = StatisticsCollector()
        collector.start_session()

        for i in range(3):
            collector.start_download(f"timeout{i}")
            collector.end_download(f"timeout{i}", f"T{i}", "Podcast", False, 0, "timeout error")

        for i in range(2):
            collector.start_download(f"network{i}")
            collector.end_download(f"network{i}", f"N{i}", "Podcast", False, 0, "network error")

        summary = collector.get_error_summary()
        assert summary["timeout"] == 3
        assert summary["network"] == 2

    def test_format_summary(self):
        """should format human-readable summary."""
        collector = StatisticsCollector()
        collector.start_session()
        collector.record_podcast_processed("Test Podcast", 10)

        collector.start_download("ep1")
        collector.end_download("ep1", "Ep1", "Podcast", True, 1000000)

        summary = collector.format_summary()
        assert "download session summary" in summary.lower()
        assert "episodes downloaded: 1" in summary.lower()

    def test_reset_counters(self):
        """should reset counters."""
        collector = StatisticsCollector()
        collector.start_session()
        collector.record_podcast_processed("Test", 10)

        collector.start_download("ep1")
        collector.end_download("ep1", "Ep1", "Podcast", True, 1000)

        collector.reset_counters()
        metrics = collector.get_session_metrics()
        assert metrics.episodes_downloaded == 0
        assert metrics.podcasts_processed == 0


class TestGlobalCollector:
    """tests for global collector functions."""

    def test_get_collector(self):
        """should return global collector."""
        reset_collector()  # ensure clean state
        collector1 = get_collector()
        collector2 = get_collector()
        assert collector1 is collector2

    def test_reset_collector(self):
        """should reset global collector."""
        collector1 = get_collector()
        collector1.start_session()
        collector1.record_podcast_processed("Test", 10)

        reset_collector()
        collector2 = get_collector()

        assert collector1 is not collector2
        metrics = collector2.get_session_metrics()
        assert metrics.podcasts_processed == 0


class TestEdgeCases:
    """edge case tests."""

    def test_end_download_without_start(self):
        """should handle end without start gracefully."""
        collector = StatisticsCollector()
        collector.start_session()

        # this shouldn't crash
        collector.end_download("nonexistent", "Ep", "Podcast", True, 1000)
        metrics = collector.get_session_metrics()
        assert metrics.episodes_downloaded == 1

    def test_multiple_sessions(self):
        """should handle multiple session starts."""
        collector = StatisticsCollector()

        collector.start_session()
        collector.record_podcast_processed("Test1", 5)

        collector.start_session()  # start new session
        metrics = collector.get_session_metrics()
        assert metrics.podcasts_processed == 0  # should be reset

    def test_zero_duration_rate(self):
        """should handle zero duration for rate calculation."""
        collector = StatisticsCollector()
        # don't start session
        rate = collector.get_download_rate()
        assert rate == 0.0
