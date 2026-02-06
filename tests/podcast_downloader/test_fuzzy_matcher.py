"""tests for fuzzy_matcher module."""

import pytest

from podcast_downloader.fuzzy_matcher import (
    MatchConfig,
    extract_words,
    find_best_match,
    fuzzy_match,
    generate_query_variations,
    levenshtein_ratio,
    normalize_text,
    phonetic_match_score,
    soundex,
    word_match_score,
    word_position_score,
)


class TestSoundex:
    """tests for soundex phonetic encoding."""

    def test_basic_encoding(self):
        """should encode basic words correctly."""
        assert soundex("Robert") == "R163"
        assert soundex("Rupert") == "R163"

    def test_similar_sounding_words(self):
        """similar sounding words should have same code."""
        assert soundex("Smith") == soundex("Smyth")

    def test_empty_string(self):
        """should handle empty string."""
        assert soundex("") == "0000"

    def test_single_letter(self):
        """should handle single letter."""
        result = soundex("A")
        assert len(result) == 4

    def test_numbers_ignored(self):
        """should ignore numbers."""
        assert soundex("Test123") == soundex("Test")


class TestNormalizeText:
    """tests for text normalization."""

    def test_lowercase(self):
        """should convert to lowercase."""
        result = normalize_text("HELLO WORLD")
        assert result == "hello world"

    def test_special_char_removal(self):
        """should remove special characters."""
        result = normalize_text("hello! world?")
        assert result == "hello world"

    def test_abbreviation_expansion(self):
        """should expand abbreviations."""
        result = normalize_text("NPR podcast", expand_abbreviations=True)
        assert "national public radio" in result

    def test_abbreviation_disabled(self):
        """should not expand when disabled."""
        result = normalize_text("NPR podcast", expand_abbreviations=False)
        assert "npr" in result
        assert "national" not in result

    def test_collapse_whitespace(self):
        """should collapse multiple spaces."""
        result = normalize_text("hello     world")
        assert result == "hello world"


class TestExtractWords:
    """tests for word extraction."""

    def test_basic_extraction(self):
        """should extract significant words."""
        words = extract_words("The Daily Podcast Show")
        assert "daily" in words
        assert "the" not in words  # stopword

    def test_min_length_filter(self):
        """should filter short words."""
        words = extract_words("a to be or not", min_length=3)
        assert "a" not in words
        assert "to" not in words

    def test_stopword_removal(self):
        """should remove stopwords."""
        words = extract_words("the and or but")
        assert len(words) == 0


class TestWordMatchScore:
    """tests for word matching score."""

    def test_perfect_match(self):
        """should return 1.0 for identical word lists."""
        score = word_match_score(["hello", "world"], ["hello", "world"])
        assert score == 1.0

    def test_partial_match(self):
        """should return partial score for partial match."""
        score = word_match_score(["hello", "world"], ["hello", "there"])
        assert 0 < score < 1

    def test_no_match(self):
        """should return 0 for no matches."""
        score = word_match_score(["hello"], ["world"])
        assert score == 0.0

    def test_empty_lists(self):
        """should handle empty lists."""
        assert word_match_score([], ["hello"]) == 0.0
        assert word_match_score(["hello"], []) == 0.0


class TestWordPositionScore:
    """tests for position-aware word matching."""

    def test_same_position_bonus(self):
        """words at same position should score higher."""
        score1 = word_position_score(["hello", "world"], ["hello", "world"])
        score2 = word_position_score(["hello", "world"], ["world", "hello"])
        assert score1 > score2

    def test_empty_lists(self):
        """should handle empty lists."""
        assert word_position_score([], ["hello"]) == 0.0


class TestLevenshteinRatio:
    """tests for levenshtein similarity."""

    def test_identical_strings(self):
        """should return 1.0 for identical strings."""
        assert levenshtein_ratio("hello", "hello") == 1.0

    def test_completely_different(self):
        """should return low score for different strings."""
        score = levenshtein_ratio("abc", "xyz")
        assert score < 0.5

    def test_similar_strings(self):
        """should return high score for similar strings."""
        score = levenshtein_ratio("hello", "hallo")
        assert score > 0.7

    def test_empty_strings(self):
        """should handle empty strings."""
        assert levenshtein_ratio("", "") == 0.0
        assert levenshtein_ratio("hello", "") == 0.0


class TestPhoneticMatchScore:
    """tests for phonetic matching."""

    def test_similar_sounding(self):
        """similar sounding words should score high."""
        score = phonetic_match_score(["smith"], ["smyth"])
        assert score == 1.0

    def test_different_sounding(self):
        """different sounding words should score low."""
        score = phonetic_match_score(["hello"], ["world"])
        assert score < 0.5


class TestFuzzyMatch:
    """tests for main fuzzy_match function."""

    def test_exact_match(self):
        """should return high score for exact match."""
        result = fuzzy_match("The Daily", "The Daily")
        assert result.score > 0.9

    def test_partial_match(self):
        """should return medium score for partial match."""
        result = fuzzy_match("The Daily", "The Daily Show by NYT")
        assert 0.5 < result.score < 1.0

    def test_no_match(self):
        """should return low score for no match."""
        result = fuzzy_match("The Daily", "Science Friday")
        assert result.score < 0.5

    def test_explanation_provided(self):
        """should provide explanation."""
        result = fuzzy_match("The Daily", "The Daily Show")
        assert result.explanation is not None

    def test_custom_config(self):
        """should respect custom config."""
        config = MatchConfig(word_weight=1.0, char_weight=0.0, phonetic_weight=0.0, position_weight=0.0)
        result = fuzzy_match("hello", "hello world", config)
        assert result.score > 0

    def test_case_insensitive(self):
        """should be case insensitive."""
        result1 = fuzzy_match("THE DAILY", "the daily")
        result2 = fuzzy_match("the daily", "THE DAILY")
        assert abs(result1.score - result2.score) < 0.01


class TestGenerateQueryVariations:
    """tests for query variation generation."""

    def test_includes_original(self):
        """should include original query."""
        variations = generate_query_variations("NPR")
        assert "NPR" in variations

    def test_adds_podcast_suffix(self):
        """should add podcast suffix."""
        variations = generate_query_variations("Science Friday")
        assert any("podcast" in v.lower() for v in variations)

    def test_removes_podcast(self):
        """should create variation without podcast."""
        variations = generate_query_variations("Science Friday Podcast")
        assert any("podcast" not in v.lower() for v in variations)

    def test_removes_leading_the(self):
        """should create variation without leading 'the'."""
        variations = generate_query_variations("The Daily")
        assert "Daily" in variations

    def test_expands_abbreviations(self):
        """should expand abbreviations."""
        variations = generate_query_variations("NPR")
        assert any("national public radio" in v.lower() for v in variations)


class TestFindBestMatch:
    """tests for find_best_match function."""

    def test_finds_best(self):
        """should find best matching candidate."""
        candidates = ["The Daily Show", "Science Friday", "The Daily by NYT"]
        result = find_best_match("The Daily", candidates)
        assert result is not None
        assert "Daily" in result[0]

    def test_respects_min_score(self):
        """should return None if below min_score."""
        candidates = ["Completely Different Podcast"]
        result = find_best_match("The Daily", candidates, min_score=0.9)
        assert result is None

    def test_empty_candidates(self):
        """should handle empty candidates."""
        result = find_best_match("The Daily", [])
        assert result is None

    def test_returns_match_result(self):
        """should return MatchResult with score."""
        candidates = ["The Daily"]
        result = find_best_match("The Daily", candidates)
        assert result is not None
        assert result[1].score > 0


class TestRealWorldScenarios:
    """real-world matching scenarios."""

    def test_npr_variations(self):
        """should match NPR variations."""
        result = fuzzy_match("NPR", "National Public Radio")
        assert result.score > 0.5

    def test_nyt_daily(self):
        """should match NYT Daily variations."""
        result = fuzzy_match("NYT Daily", "The Daily by New York Times")
        assert result.score > 0.4

    def test_podcast_suffix_matching(self):
        """should match with/without podcast suffix."""
        result = fuzzy_match("Science Friday", "Science Friday Podcast")
        assert result.score > 0.7

    def test_common_podcast_names(self):
        """should handle common podcast name patterns."""
        test_cases = [
            ("This American Life", "This American Life from WBEZ"),
            ("Serial", "Serial: A True Crime Podcast"),
            ("Radiolab", "Radiolab from WNYC"),
        ]
        for query, target in test_cases:
            result = fuzzy_match(query, target)
            assert result.score > 0.5, f"Failed for {query} -> {target}"
