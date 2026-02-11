"""Keyword analysis types with runtime validation.

This module defines types for keyword analysis results:
- KeywordMatch: A single keyword occurrence with context
- KeywordCategory: A category of related keywords
- KeywordAnalysisResult: Complete analysis for a transcript
- CategoryStats: Aggregated statistics for a category

Used by keyword analysis pipeline and terminal data visualizer.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, computed_field


class ConfidenceTier(str, Enum):
    """Confidence level for keyword categories.

    Values:
        HIGH: High confidence keywords with clear meaning.
        MEDIUM: Moderate confidence, may have multiple meanings.
        LOW: Low confidence, context-dependent.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class KeywordMatch(BaseModel):
    """A single keyword occurrence in a transcript.

    Captures where a keyword was found along with surrounding
    context for review.

    Attributes:
        keyword: The matched keyword text.
        text: The segment text containing the keyword.
        speaker: Speaker who said this segment.
        start_time: Start time of the segment in seconds.
        end_time: End time of the segment in seconds.
        context_before: Text before the keyword (for context).
        context_after: Text after the keyword (for context).
        segment_index: Index of the segment in the transcript.
    """

    keyword: str = Field(description="The matched keyword")
    text: str = Field(description="Segment text containing keyword")
    speaker: str = Field(default="UNKNOWN", description="Speaker identifier")
    start_time: float = Field(ge=0, default=0.0, description="Segment start time")
    end_time: float = Field(ge=0, default=0.0, description="Segment end time")
    context_before: str = Field(default="", description="Text before keyword")
    context_after: str = Field(default="", description="Text after keyword")
    segment_index: int = Field(ge=0, default=0, description="Segment index")

    @computed_field
    @property
    def duration(self) -> float:
        """Duration of the segment in seconds."""
        return self.end_time - self.start_time


class KeywordCategory(BaseModel):
    """A category of related keywords.

    Groups keywords by semantic meaning or function,
    with optional confidence scoring.

    Attributes:
        name: Category name (e.g., "question_words", "hedges").
        description: Human-readable description of the category.
        keywords: List of keywords in this category.
        confidence: Confidence tier for this category.
        metadata: Additional category metadata.
    """

    name: str = Field(description="Category name")
    description: str = Field(default="", description="Category description")
    keywords: list[str] = Field(
        default_factory=list,
        description="Keywords in this category",
    )
    confidence: ConfidenceTier = Field(
        default=ConfidenceTier.MEDIUM,
        description="Confidence tier",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    @computed_field
    @property
    def keyword_count(self) -> int:
        """Number of keywords in this category."""
        return len(self.keywords)


class CategoryStats(BaseModel):
    """Statistics for a keyword category in a transcript.

    Aggregates match counts and provides summary metrics
    for a single category.

    Attributes:
        category_name: Name of the category.
        total_matches: Total keyword occurrences.
        unique_keywords: Count of unique keywords matched.
        matches_by_keyword: Occurrence count per keyword.
        matches_by_speaker: Occurrence count per speaker.
        sample_matches: Example matches for review.
    """

    category_name: str = Field(description="Category name")
    total_matches: int = Field(ge=0, default=0, description="Total occurrences")
    unique_keywords: int = Field(ge=0, default=0, description="Unique keywords found")
    matches_by_keyword: dict[str, int] = Field(
        default_factory=dict,
        description="Count per keyword",
    )
    matches_by_speaker: dict[str, int] = Field(
        default_factory=dict,
        description="Count per speaker",
    )
    sample_matches: list[KeywordMatch] = Field(
        default_factory=list,
        description="Example matches",
    )

    @computed_field
    @property
    def keyword_density(self) -> float:
        """Average matches per unique keyword."""
        if self.unique_keywords == 0:
            return 0.0
        return self.total_matches / self.unique_keywords

    def top_keywords(self, n: int = 5) -> list[tuple[str, int]]:
        """Get the top N most frequent keywords.

        Args:
            n: Number of keywords to return.

        Returns:
            List of (keyword, count) tuples sorted by count.
        """
        sorted_items = sorted(
            self.matches_by_keyword.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_items[:n]


class KeywordAnalysisResult(BaseModel):
    """Complete keyword analysis results for a transcript.

    Contains all category statistics and match details for
    a single transcript file.

    Attributes:
        file_name: Name of the analyzed transcript.
        file_path: Full path to the transcript.
        total_words: Total word count in transcript.
        total_matches: Total keyword matches across all categories.
        categories: Statistics for each keyword category.
        all_matches: All keyword matches found.
        metadata: Additional analysis metadata.
    """

    file_name: str = Field(description="Transcript filename")
    file_path: str = Field(description="Full path to transcript")
    total_words: int = Field(ge=0, default=0, description="Word count")
    total_matches: int = Field(ge=0, default=0, description="Total matches")
    categories: dict[str, CategoryStats] = Field(
        default_factory=dict,
        description="Stats by category",
    )
    all_matches: list[KeywordMatch] = Field(
        default_factory=list,
        description="All keyword matches",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Analysis metadata",
    )

    @computed_field
    @property
    def match_density(self) -> float:
        """Matches per 1000 words."""
        if self.total_words == 0:
            return 0.0
        return (self.total_matches / self.total_words) * 1000

    @computed_field
    @property
    def category_count(self) -> int:
        """Number of categories with matches."""
        return len([c for c in self.categories.values() if c.total_matches > 0])

    def get_category_summary(self) -> dict[str, int]:
        """Get match counts by category.

        Returns:
            Dictionary mapping category name to match count.
        """
        return {
            name: stats.total_matches
            for name, stats in self.categories.items()
        }

    def get_top_categories(self, n: int = 5) -> list[tuple[str, int]]:
        """Get categories with most matches.

        Args:
            n: Number of categories to return.

        Returns:
            List of (category_name, match_count) tuples.
        """
        summary = self.get_category_summary()
        sorted_items = sorted(summary.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n]


class SearchResult(BaseModel):
    """Result of a keyword search across multiple files.

    Attributes:
        query: The search query.
        total_files_searched: Number of files searched.
        files_with_matches: Number of files with matches.
        total_matches: Total matches found.
        matches_by_file: Matches grouped by file.
    """

    query: str = Field(description="Search query")
    total_files_searched: int = Field(ge=0, description="Files searched")
    files_with_matches: int = Field(ge=0, default=0, description="Files with matches")
    total_matches: int = Field(ge=0, default=0, description="Total matches")
    matches_by_file: dict[str, list[KeywordMatch]] = Field(
        default_factory=dict,
        description="Matches by file path",
    )

    @computed_field
    @property
    def match_rate(self) -> float:
        """Proportion of files with matches (0.0-1.0)."""
        if self.total_files_searched == 0:
            return 0.0
        return self.files_with_matches / self.total_files_searched
