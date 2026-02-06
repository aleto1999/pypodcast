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
        result = sanitize_filename("UPPERCASE", extension=".mp3")
        assert result == "uppercase.mp3"

    def test_mixed_case_conversion(self):
        """should handle mixed case."""
        result = sanitize_filename("MixedCase Episode", extension=".mp3")
        assert result == "mixedcase_episode.mp3"

    def test_whitespace_replacement(self):
        """should replace whitespace with underscores."""
        result = sanitize_filename("episode with spaces", extension=".mp3")
        assert result == "episode_with_spaces.mp3"

    def test_multiple_whitespace_replacement(self):
        """should replace multiple spaces with single underscore."""
        result = sanitize_filename("episode   with   multiple   spaces")
        assert result == "episode_with_multiple_spaces"

    def test_special_characters_replacement(self):
        """should replace special characters with underscores."""
        result = sanitize_filename("episode: part 1 & 2!")
        assert result == "episode_part_1_2"

    def test_consecutive_underscores(self):
        """should replace consecutive underscores with single underscore."""
        result = sanitize_filename("episode___test")
        assert result == "episode_test"

    def test_leading_trailing_underscores(self):
        """should strip leading and trailing underscores."""
        result = sanitize_filename("___episode___")
        assert result == "episode"

    def test_extension_parameter(self):
        """should append extension parameter."""
        result = sanitize_filename("Test Episode", extension=".mp3")
        assert result == "test_episode.mp3"

    def test_max_length_truncation(self):
        """should truncate to max_length."""
        long_name = "a" * 150
        result = sanitize_filename(long_name, max_length=50)
        assert len(result) <= 50

    def test_numeric_characters_preserved(self):
        """should preserve numeric characters."""
        result = sanitize_filename("Episode 123")
        assert result == "episode_123"

    def test_empty_string(self):
        """should return untitled for empty string."""
        result = sanitize_filename("")
        assert result == "untitled"

    def test_empty_with_extension(self):
        """should return untitled with extension for empty string."""
        result = sanitize_filename("", extension=".mp3")
        assert result == "untitled.mp3"

    def test_only_special_characters(self):
        """should return untitled for string with only special characters."""
        result = sanitize_filename("!@#$%^&*()")
        assert result == "untitled"

    def test_dots_replaced(self):
        """dots should be replaced with underscores per naming convention."""
        result = sanitize_filename("episode.v2.final")
        assert result == "episode_v2_final"

    def test_real_world_example(self):
        """should handle real podcast episode names."""
        result = sanitize_filename("The Daily: Biden's New Policy & What It Means")
        assert result == "the_daily_biden_s_new_policy_what_it_means"


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

    def test_hyphen_replaced(self):
        """hyphens should be replaced with underscores."""
        result = sanitize_filename("episode-1-intro")
        assert result == "episode_1_intro"

    def test_date_in_filename(self):
        """should handle dates in filename."""
        result = sanitize_filename("2024-01-15 Episode")
        assert result == "2024_01_15_episode"

    def test_parentheses_replaced(self):
        """parentheses should be replaced with underscores."""
        result = sanitize_filename("Episode (Remastered)")
        assert result == "episode_remastered"

    def test_apostrophe_handling(self):
        """apostrophes should be replaced."""
        result = sanitize_filename("Biden's Speech")
        assert result == "biden_s_speech"

    def test_ampersand_handling(self):
        """ampersands should be replaced."""
        result = sanitize_filename("Tom & Jerry")
        assert result == "tom_jerry"

    def test_colon_handling(self):
        """colons should be replaced."""
        result = sanitize_filename("Episode: Part 1")
        assert result == "episode_part_1"

    def test_question_mark_handling(self):
        """question marks should be replaced."""
        result = sanitize_filename("What Is This?")
        assert result == "what_is_this"

    def test_slash_handling(self):
        """slashes should be replaced."""
        result = sanitize_filename("Episode 1/2")
        assert result == "episode_1_2"

    def test_unicode_accents(self):
        """unicode accents should be replaced."""
        result = sanitize_filename("café résumé")
        assert result == "caf_r_sum"

    def test_extension_with_special_chars(self):
        """extension should be added after sanitization."""
        result = sanitize_filename("Test: Episode!", extension=".mp3")
        assert result == "test_episode.mp3"

    def test_max_length_does_not_cut_extension(self):
        """extension should be added after truncation."""
        long_name = "a" * 100
        result = sanitize_filename(long_name, max_length=10, extension=".mp3")
        assert result.endswith(".mp3")
        # base name should be 10 chars, extension adds 4
        assert len(result) == 14

    def test_whitespace_only(self):
        """whitespace only should return untitled."""
        result = sanitize_filename("   ")
        assert result == "untitled"

    def test_tabs_and_newlines(self):
        """tabs and newlines should be handled as whitespace."""
        result = sanitize_filename("test\t\nepisode")
        assert result == "test_episode"
