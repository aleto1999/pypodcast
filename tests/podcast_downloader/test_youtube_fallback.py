"""tests for youtube_fallback module."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from podcast_downloader.youtube_fallback import (
    ERROR_CATEGORIES,
    ErrorCategory,
    YouTubeSearchMatch,
    calculate_confidence_score,
    categorize_error,
    download_from_youtube,
    fallback_download_episode,
    search_youtube_for_episode,
)


class TestErrorCategory:
    """tests for ErrorCategory namedtuple."""

    def test_create_error_category(self):
        """should create error category."""
        category = ErrorCategory(
            category="network",
            message="connection timeout",
            should_retry=True,
            max_retries=3,
        )
        assert category.category == "network"
        assert category.should_retry is True


class TestCategorizeError:
    """tests for error categorization."""

    def test_network_errors(self):
        """should categorize network errors."""
        test_cases = [
            "connection timeout",
            "SSL certificate error",
            "network unreachable",
        ]
        for msg in test_cases:
            result = categorize_error(msg)
            assert result.category == "network"
            assert result.should_retry is True

    def test_not_found_errors(self):
        """should categorize not found errors."""
        test_cases = [
            "404 not found",
            "video unavailable",
            "this video is private",
            "video has been removed",
        ]
        for msg in test_cases:
            result = categorize_error(msg)
            assert result.category == "not_found"
            assert result.should_retry is False

    def test_permission_errors(self):
        """should categorize permission errors."""
        test_cases = [
            "403 forbidden",
            "video blocked in your country",
            "age verification required",
        ]
        for msg in test_cases:
            result = categorize_error(msg)
            assert result.category == "permission"
            assert result.should_retry is False

    def test_rate_limit_errors(self):
        """should categorize rate limit errors."""
        test_cases = [
            "rate limit exceeded",
            "429 too many requests",
            "quota exceeded",
        ]
        for msg in test_cases:
            result = categorize_error(msg)
            assert result.category == "rate_limit"
            assert result.should_retry is True

    def test_format_errors(self):
        """should categorize format errors."""
        test_cases = [
            "no suitable formats found",
            "unsupported format",
        ]
        for msg in test_cases:
            result = categorize_error(msg)
            assert result.category == "format"

    def test_unknown_errors(self):
        """should categorize unknown errors."""
        result = categorize_error("some random error xyz")
        assert result.category == "unknown"
        assert result.should_retry is True
        assert result.max_retries == 1


class TestYouTubeSearchMatch:
    """tests for YouTubeSearchMatch namedtuple."""

    def test_create_search_match(self):
        """should create search match."""
        match = YouTubeSearchMatch(
            video_id="abc123",
            title="Test Episode",
            channel="Test Channel",
            duration=3600,
            url="https://youtube.com/watch?v=abc123",
            confidence_score=0.85,
        )
        assert match.video_id == "abc123"
        assert match.confidence_score == 0.85


class TestCalculateConfidenceScore:
    """tests for confidence score calculation."""

    def test_exact_match(self, sample_episode):
        """should return high score for exact match."""
        score = calculate_confidence_score(
            episode=sample_episode,
            show_name="Test Show",
            video_title="Test Show: Test Episode",
            video_channel="Test Show",
            video_duration=3600,
        )
        # score should be reasonably high for good matches
        assert score >= 0.6

    def test_show_name_in_title(self, sample_episode):
        """should boost score for show name in title."""
        score = calculate_confidence_score(
            episode=sample_episode,
            show_name="Test Show",
            video_title="Test Show presents something",
            video_channel="Other Channel",
            video_duration=None,
        )
        assert score >= 0.4

    def test_show_name_in_channel(self, sample_episode):
        """should boost score for show name in channel."""
        score = calculate_confidence_score(
            episode=sample_episode,
            show_name="Test Show",
            video_title="Random Title",
            video_channel="Test Show Official",
            video_duration=None,
        )
        assert score >= 0.4

    def test_episode_title_words(self, sample_episode):
        """should match episode title words."""
        score = calculate_confidence_score(
            episode=sample_episode,
            show_name="Other Show",
            video_title="Test Episode: Special Characters",
            video_channel="Other Channel",
            video_duration=None,
        )
        assert score > 0.2

    def test_duration_match(self, sample_episode):
        """should boost score for matching duration."""
        # sample_episode has duration=3600
        score_match = calculate_confidence_score(
            episode=sample_episode,
            show_name="Test",
            video_title="Test",
            video_channel="Test",
            video_duration=3600,  # exact match
        )
        score_no_match = calculate_confidence_score(
            episode=sample_episode,
            show_name="Test",
            video_title="Test",
            video_channel="Test",
            video_duration=1800,  # 50% off
        )
        assert score_match > score_no_match

    def test_duration_tolerance(self, sample_episode):
        """should accept duration within 10%."""
        score = calculate_confidence_score(
            episode=sample_episode,
            show_name="Test Show",
            video_title="Test Show",
            video_channel="Test Show",
            video_duration=3500,  # ~3% off from 3600
        )
        assert score > 0.5

    def test_no_duration(self, sample_episode):
        """should handle missing duration."""
        score = calculate_confidence_score(
            episode=sample_episode,
            show_name="Test Show",
            video_title="Test Show Episode",
            video_channel="Test",
            video_duration=None,
        )
        assert 0 <= score <= 1

    def test_max_score_capped(self, sample_episode):
        """should cap score at 1.0."""
        score = calculate_confidence_score(
            episode=sample_episode,
            show_name="test",
            video_title="test test test test test",
            video_channel="test",
            video_duration=3600,
        )
        assert score <= 1.0


class TestSearchYoutubeForEpisode:
    """tests for youtube search functionality."""

    @pytest.mark.asyncio
    async def test_no_yt_dlp(self, sample_episode):
        """should return empty list if yt-dlp not available."""
        with patch("shutil.which", return_value=None):
            results = await search_youtube_for_episode(sample_episode, "Test Show")
            assert results == []

    @pytest.mark.asyncio
    async def test_search_timeout(self, sample_episode):
        """should handle timeout gracefully."""
        with patch("shutil.which", return_value="/usr/bin/yt-dlp"):
            with patch("asyncio.create_subprocess_exec") as mock_exec:
                mock_process = MagicMock()
                mock_process.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
                mock_exec.return_value = mock_process

                results = await search_youtube_for_episode(sample_episode, "Test Show")
                assert results == []

    @pytest.mark.asyncio
    async def test_search_returns_results(self, sample_episode):
        """should return sorted results."""
        mock_output = b'''{"id": "vid1", "title": "Test Show Episode 1", "channel": "Test Show", "duration": 3600}
{"id": "vid2", "title": "Other Video", "channel": "Other", "duration": 1800}'''

        with patch("shutil.which", return_value="/usr/bin/yt-dlp"):
            with patch("asyncio.create_subprocess_exec") as mock_exec:
                mock_process = MagicMock()
                mock_process.communicate = AsyncMock(return_value=(mock_output, b""))
                mock_exec.return_value = mock_process

                results = await search_youtube_for_episode(sample_episode, "Test Show")
                assert len(results) == 2
                # should be sorted by confidence
                assert results[0].confidence_score >= results[1].confidence_score

    @pytest.mark.asyncio
    async def test_filter_short_videos(self, sample_episode):
        """should filter out videos shorter than 3 minutes."""
        mock_output = b'''{"id": "vid1", "title": "Short Video", "channel": "Test", "duration": 60}
{"id": "vid2", "title": "Long Episode", "channel": "Test", "duration": 3600}'''

        with patch("shutil.which", return_value="/usr/bin/yt-dlp"):
            with patch("asyncio.create_subprocess_exec") as mock_exec:
                mock_process = MagicMock()
                mock_process.communicate = AsyncMock(return_value=(mock_output, b""))
                mock_exec.return_value = mock_process

                results = await search_youtube_for_episode(sample_episode, "Test Show")
                assert len(results) == 1
                assert results[0].video_id == "vid2"


class TestDownloadFromYoutube:
    """tests for youtube download functionality."""

    @pytest.mark.asyncio
    async def test_no_yt_dlp(self, sample_episode, temp_dir):
        """should fail if yt-dlp not available."""
        match = YouTubeSearchMatch(
            video_id="test",
            title="Test",
            channel="Test",
            duration=3600,
            url="https://youtube.com/watch?v=test",
            confidence_score=0.9,
        )

        with patch("shutil.which", return_value=None):
            result = await download_from_youtube(match, sample_episode, temp_dir)
            assert result.success is False
            assert "yt-dlp not installed" in result.error


class TestFallbackDownloadEpisode:
    """tests for fallback download functionality."""

    @pytest.mark.asyncio
    async def test_no_matches_found(self, sample_episode, temp_dir):
        """should fail if no youtube matches found."""
        with patch("podcast_downloader.youtube_fallback.search_youtube_for_episode") as mock_search:
            mock_search.return_value = []

            result = await fallback_download_episode(
                episode=sample_episode,
                show_name="Test Show",
                output_dir=temp_dir,
            )
            assert result.success is False
            assert "no matching videos" in result.error.lower()

    @pytest.mark.asyncio
    async def test_below_confidence_threshold(self, sample_episode, temp_dir):
        """should fail if matches below confidence threshold."""
        low_confidence_match = YouTubeSearchMatch(
            video_id="test",
            title="Unrelated Video",
            channel="Other",
            duration=100,
            url="https://youtube.com/watch?v=test",
            confidence_score=0.2,
        )

        with patch("podcast_downloader.youtube_fallback.search_youtube_for_episode") as mock_search:
            mock_search.return_value = [low_confidence_match]

            result = await fallback_download_episode(
                episode=sample_episode,
                show_name="Test Show",
                output_dir=temp_dir,
                min_confidence=0.5,
            )
            assert result.success is False
            assert "confidence threshold" in result.error.lower()


class TestRealWorldScenarios:
    """real-world scenario tests."""

    def test_confidence_for_real_podcasts(self, sample_episode):
        """should calculate reasonable confidence for real podcast scenarios."""
        # NPR podcast episode
        from podcast_downloader.feed_parser import Episode
        from datetime import datetime

        npr_episode = Episode(
            id="npr-1",
            title="Morning Edition: Today's Top Stories",
            description="NPR news",
            published=datetime(2024, 1, 15),
            duration=3600,
            audio_url="https://example.com/episode.mp3",
        )

        score = calculate_confidence_score(
            episode=npr_episode,
            show_name="NPR Morning Edition",
            video_title="NPR Morning Edition: Today's Top Stories - Full Episode",
            video_channel="NPR",
            video_duration=3580,
        )
        assert score > 0.7

    def test_error_category_max_retries(self):
        """should have appropriate max retries per category."""
        # network errors should have more retries
        network_cat = categorize_error("connection timeout")
        not_found_cat = categorize_error("video not found")

        assert network_cat.max_retries > not_found_cat.max_retries
