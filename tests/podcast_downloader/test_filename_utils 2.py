"""tests for filename_utils module."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest

from podcast_downloader.filename_utils import sanitize_dirname, sanitize_filename


class TestSanitizeFilename:
    """tests for sanitize_filename function."""

    def test_lowercase_conversion(self):
        """should convert all characters to lowercase."""
        result = sanitize_filename("UPPERCASE.mp3")
        assert result == "uppercase.mp3"

    def test_mixed_case_conversion(self):
        """should handle mixed case."""
        result = sanitize_filename("MixedCase Episode.mp3")
        assert result == "mixedcase_episode.mp3"

    def test_whitespace_replacement(self):
        """should replace whitespace with underscores."""
        result = sanitize_filename("episode with spaces.mp3")
        assert result == "episode_with_spaces.mp3"

    def test_multiple_whitespace_replacement(self):
        """should replace multiple spaces with single underscore."""
        result = sanitize_filename("episode   with   multiple   spaces.mp3")
        assert result == "episode_with_multiple_spaces.mp3"

    def test_special_characters_replacement(self):
        """should replace special characters with underscores."""
        result = sanitize_filename("episode: part 1 & 2!.mp3")
        assert result == "episode_part_1_2.mp3"

    def test_consecutive_underscores(self):
        """should replace consecutive underscores with single underscore."""
        result = sanitize_filename("episode___test.mp3")
        assert result == "episode_test.mp3"

    def test_leading_trailing_underscores(self):
        """should strip leading and trailing underscores."""
        result = sanitize_filename("___episode___")
        assert result == "episode"

    def test_preserve_extension(self):
        """should preserve file extension."""
        result = sanitize_filename("Test Episode.MP3")
        assert result == "test_episode.mp3"

    def test_max_length_truncation(self):
        """should truncate to max_length."""
        long_name = "a" * 150 + ".mp3"
        result = sanitize_filename(long_name, max_length=50)
        assert len(result) <= 50

    def test_numeric_characters_preserved(self):
        """should preserve numeric characters."""
        result = sanitize_filename("Episode 123.mp3")
        assert result == "episode_123.mp3"

    def test_empty_string(self):
        """should handle empty string."""
        result = sanitize_filename("")
        assert result == ""

    def test_only_special_characters(self):
        """should handle string with only special characters."""
        result = sanitize_filename("!@#$%^&*()")
        assert result == ""

    def test_unicode_characters(self):
        """should handle unicode characters."""
        result = sanitize_filename("épisode café.mp3")
        assert "pisode" in result or "episode" in result.lower()

    def test_real_world_example(self):
        """should handle real podcast episode names."""
        result = sanitize_filename("The Daily: Biden's New Policy & What It Means.mp3")
        assert result == "the_daily_biden_s_new_policy_what_it_means.mp3"


class TestSanitizeDirname:
    """tests for sanitize_dirname function."""

    def test_lowercase_conversion(self):
        """should convert to lowercase."""
        result = sanitize_dirname("THE DAILY")
        assert result == "the_daily"

    def test_whitespace_replacement(self):
        """should replace whitespace."""
        result = sanitize_dirname("podcast name here")
        assert result == "podcast_name_here"

    def test_special_characters(self):
        """should replace special characters."""
        result = sanitize_dirname("Podcast: The Show!")
        assert result == "podcast_the_show"

    def test_shorter_max_length(self):
        """should use shorter default max_length than filename."""
        long_name = "a" * 100
        result = sanitize_dirname(long_name)
        assert len(result) <= 50  # default max_length for dirname

    def test_real_world_podcast_name(self):
        """should handle real podcast names."""
        result = sanitize_dirname("The New York Times: The Daily")
        assert result == "the_new_york_times_the_daily"


class TestFilenameEdgeCases:
    """edge case tests for filename sanitization."""

    def test_dots_in_filename(self):
        """should handle multiple dots."""
        result = sanitize_filename("episode.v2.final.mp3")
        assert result == "episode.v2.final.mp3"

    def test_hyphen_preservation(self):
        """should preserve hyphens as word separators."""
        result = sanitize_filename("episode-1-intro.mp3")
        # hyphens may be preserved or converted to underscores
        assert "episode" in result and "intro" in result

    def test_date_in_filename(self):
        """should handle dates in filename."""
        result = sanitize_filename("2024-01-15 Episode.mp3")
        assert "2024" in result and "episode" in result

    def test_parentheses(self):
        """should handle parentheses."""
        result = sanitize_filename("Episode (Remastered).mp3")
        assert "episode" in result and "remastered" in result
