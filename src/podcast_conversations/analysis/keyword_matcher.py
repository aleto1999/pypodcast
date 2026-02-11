"""keyword matching with context extraction."""

import logging
import re
from dataclasses import dataclass
from typing import Any

from podcast_conversations.analysis.config_loader import AnalysisConfig, KeywordAnalysisSettings
from podcast_conversations.analysis.transcript_reader import TranscriptSegment

logger = logging.getLogger(__name__)


@dataclass
class KeywordMatch:
    """single keyword match with context."""

    keyword: str
    matched_text: str
    context_before: str
    context_after: str
    position: int
    segment_id: int | None = None
    speaker: str | None = None
    start_time: float | None = None
    end_time: float | None = None
    category: str | None = None
    confidence_tier: str | None = None


class KeywordMatcher:
    """match keywords in transcript text with context extraction."""

    def __init__(
        self,
        keywords: list[str],
        settings: KeywordAnalysisSettings,
        config: AnalysisConfig | None = None,
    ):
        """
        initialize keyword matcher.

        Args:
            keywords: list of keywords to search for.
            settings: keyword analysis settings.
            config: optional full config for category lookups.
        """
        self.keywords = keywords
        self.settings = settings
        self.config = config

        # compile regex patterns for keywords.
        self.patterns: dict[str, re.Pattern] = {}
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """compile regex patterns for each keyword."""
        for keyword in self.keywords:
            if len(keyword) < self.settings.min_word_length:
                continue

            # escape special regex characters.
            escaped = re.escape(keyword)

            # build pattern based on settings.
            if self.settings.partial_matches:
                # match as substring.
                pattern_str = escaped
            else:
                # match whole word only.
                pattern_str = rf"\b{escaped}\b"

            # compile with appropriate flags.
            flags = 0 if self.settings.case_sensitive else re.IGNORECASE
            self.patterns[keyword] = re.compile(pattern_str, flags)

    def _extract_context(
        self, text: str, match_pos: int, match_len: int
    ) -> tuple[str, str]:
        """
        extract context before and after keyword match.

        Args:
            text: full text containing the match.
            match_pos: position of match start in text.
            match_len: length of matched text.

        Returns:
            tuple of (context_before, context_after).
        """
        window = self.settings.context_window

        # extract context before match.
        context_start = max(0, match_pos - window)
        context_before = text[context_start:match_pos]

        # extract context after match.
        match_end = match_pos + match_len
        context_end = min(len(text), match_end + window)
        context_after = text[match_end:context_end]

        return context_before, context_after

    def search_text(self, text: str, segment: TranscriptSegment | None = None) -> list[KeywordMatch]:
        """
        search for keywords in text.

        Args:
            text: text to search.
            segment: optional transcript segment for metadata.

        Returns:
            list of keyword matches found.
        """
        matches: list[KeywordMatch] = []

        for keyword, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                context_before, context_after = self._extract_context(
                    text, match.start(), len(match.group())
                )

                # look up category and confidence if config is available.
                category = None
                confidence_tier = None
                if self.config:
                    category = self.config.get_keyword_category(keyword)
                    confidence_tier = self.config.get_keyword_confidence(keyword)

                keyword_match = KeywordMatch(
                    keyword=keyword,
                    matched_text=match.group(),
                    context_before=context_before,
                    context_after=context_after,
                    position=match.start(),
                    segment_id=segment.segment_id if segment else None,
                    speaker=segment.speaker if segment else None,
                    start_time=segment.start_time if segment else None,
                    end_time=segment.end_time if segment else None,
                    category=category,
                    confidence_tier=confidence_tier,
                )
                matches.append(keyword_match)

        return matches

    def search_segments(self, segments: list[TranscriptSegment]) -> list[KeywordMatch]:
        """
        search for keywords across multiple transcript segments.

        Args:
            segments: list of transcript segments to search.

        Returns:
            list of all keyword matches found across segments.
        """
        all_matches: list[KeywordMatch] = []

        for segment in segments:
            if not segment.text:
                continue

            matches = self.search_text(segment.text, segment)
            all_matches.extend(matches)

        logger.info(f"found {len(all_matches)} keyword matches across {len(segments)} segments")
        return all_matches

    def get_match_statistics(self, matches: list[KeywordMatch]) -> dict[str, Any]:
        """
        calculate statistics about keyword matches.

        Args:
            matches: list of keyword matches.

        Returns:
            dictionary with match statistics.
        """
        stats: dict[str, Any] = {
            "total_matches": len(matches),
            "unique_keywords": len(set(m.keyword for m in matches)),
            "keyword_counts": {},
            "speaker_counts": {},
            "category_counts": {},
            "confidence_tier_counts": {
                "high_confidence": 0,
                "exploratory": 0,
            },
        }

        # count by keyword.
        for match in matches:
            keyword = match.keyword
            stats["keyword_counts"][keyword] = stats["keyword_counts"].get(keyword, 0) + 1

        # count by speaker (if available).
        if self.settings.include_speaker_info:
            for match in matches:
                if match.speaker:
                    speaker = match.speaker
                    stats["speaker_counts"][speaker] = (
                        stats["speaker_counts"].get(speaker, 0) + 1
                    )

        # count by category and confidence tier.
        for match in matches:
            if match.category:
                stats["category_counts"][match.category] = (
                    stats["category_counts"].get(match.category, 0) + 1
                )
            if match.confidence_tier:
                stats["confidence_tier_counts"][match.confidence_tier] = (
                    stats["confidence_tier_counts"].get(match.confidence_tier, 0) + 1
                )

        return stats

        return stats
