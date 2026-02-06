"""Feature extraction types with runtime validation.

This module defines types for conversational feature statistics:
- QuestionStats: Statistics about questions in transcripts
- TurnTakingStats: Statistics about speaker turn patterns
- VocabularyStats: Statistics about vocabulary diversity
- PolitenessStats: Statistics about politeness markers
- TranscriptFeatures: Combined features for a transcript

These types consolidate the feature extraction output into
validated, documented structures.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, computed_field


class QuestionStats(BaseModel):
    """Statistics about questions detected in a transcript.

    Tracks both the count and types of questions, useful for
    analyzing interview dynamics and conversation patterns.

    Attributes:
        total_questions: Total number of questions detected.
        wh_questions: Count of wh-questions (who, what, where, etc.).
        yes_no_questions: Count of yes/no questions.
        rhetorical_questions: Count of likely rhetorical questions.
        question_rate: Questions per 100 words.
        questions_by_speaker: Question counts per speaker.
    """

    total_questions: int = Field(ge=0, default=0, description="Total questions detected")
    wh_questions: int = Field(ge=0, default=0, description="Wh-question count")
    yes_no_questions: int = Field(ge=0, default=0, description="Yes/no question count")
    rhetorical_questions: int = Field(ge=0, default=0, description="Rhetorical question count")
    question_rate: float = Field(ge=0, default=0.0, description="Questions per 100 words")
    questions_by_speaker: dict[str, int] = Field(
        default_factory=dict,
        description="Questions count per speaker",
    )

    @computed_field
    @property
    def classified_questions(self) -> int:
        """Total questions with a specific classification."""
        return self.wh_questions + self.yes_no_questions + self.rhetorical_questions


class TurnTakingStats(BaseModel):
    """Statistics about speaker turn-taking patterns.

    Captures dynamics of conversation flow including who dominates
    the conversation and how often speakers switch.

    Attributes:
        total_turns: Total number of speaker turns.
        unique_speakers: Number of unique speakers.
        turn_switch_count: Number of times speaker changed.
        turn_switch_rate: Switches per minute of audio.
        speaker_dominance: Proportion of turns by each speaker.
        avg_turn_duration: Average turn length in seconds.
        max_turn_duration: Longest turn in seconds.
        speaker_word_counts: Word counts per speaker.
    """

    total_turns: int = Field(ge=0, default=0, description="Total speaker turns")
    unique_speakers: int = Field(ge=0, default=0, description="Unique speaker count")
    turn_switch_count: int = Field(ge=0, default=0, description="Speaker switch count")
    turn_switch_rate: float = Field(ge=0, default=0.0, description="Switches per minute")
    speaker_dominance: dict[str, float] = Field(
        default_factory=dict,
        description="Proportion of turns by speaker (0.0-1.0)",
    )
    avg_turn_duration: float = Field(ge=0, default=0.0, description="Average turn seconds")
    max_turn_duration: float = Field(ge=0, default=0.0, description="Longest turn seconds")
    speaker_word_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Word counts per speaker",
    )

    @computed_field
    @property
    def balance_score(self) -> float:
        """Score indicating conversation balance (0=unbalanced, 1=equal).

        Calculated as 1 minus the standard deviation of speaker dominance values.
        """
        if not self.speaker_dominance:
            return 0.0
        values = list(self.speaker_dominance.values())
        if len(values) < 2:
            return 1.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std_dev = variance ** 0.5
        # normalize: max std_dev for 2 speakers is 0.5.
        return max(0.0, 1.0 - (std_dev * 2))


class VocabularyStats(BaseModel):
    """Statistics about vocabulary diversity and complexity.

    Measures lexical richness of the conversation using standard
    metrics from computational linguistics.

    Attributes:
        total_words: Total word count in transcript.
        unique_words: Count of unique word forms.
        type_token_ratio: Ratio of unique words to total (0.0-1.0).
        avg_word_length: Average word length in characters.
        vocabulary_by_speaker: Unique words per speaker.
        hapax_legomena: Words appearing only once.
        most_common_words: Top N most frequent words.
    """

    total_words: int = Field(ge=0, default=0, description="Total word count")
    unique_words: int = Field(ge=0, default=0, description="Unique word count")
    type_token_ratio: float = Field(
        ge=0,
        le=1,
        default=0.0,
        description="Type-token ratio (lexical diversity)",
    )
    avg_word_length: float = Field(ge=0, default=0.0, description="Average word length")
    vocabulary_by_speaker: dict[str, int] = Field(
        default_factory=dict,
        description="Unique words per speaker",
    )
    hapax_legomena: int = Field(ge=0, default=0, description="Words appearing once")
    most_common_words: list[tuple[str, int]] = Field(
        default_factory=list,
        description="Most frequent words with counts",
    )

    @computed_field
    @property
    def hapax_ratio(self) -> float:
        """Ratio of hapax legomena to total unique words."""
        if self.unique_words == 0:
            return 0.0
        return self.hapax_legomena / self.unique_words


class PolitenessStats(BaseModel):
    """Statistics about politeness markers in conversation.

    Tracks linguistic markers of politeness and formality
    in the conversation.

    Attributes:
        hedges_count: Count of hedge words (maybe, perhaps, etc.).
        boosters_count: Count of booster words (definitely, certainly, etc.).
        gratitude_count: Count of gratitude expressions.
        apology_count: Count of apology expressions.
        please_count: Count of "please" usage.
        formal_markers: Count of formal address markers.
        politeness_score: Overall politeness score (0.0-1.0).
        markers_by_speaker: Politeness marker counts per speaker.
    """

    hedges_count: int = Field(ge=0, default=0, description="Hedge word count")
    boosters_count: int = Field(ge=0, default=0, description="Booster word count")
    gratitude_count: int = Field(ge=0, default=0, description="Gratitude expressions")
    apology_count: int = Field(ge=0, default=0, description="Apology expressions")
    please_count: int = Field(ge=0, default=0, description="Please usage count")
    formal_markers: int = Field(ge=0, default=0, description="Formal address markers")
    politeness_score: float = Field(
        ge=0,
        le=1,
        default=0.0,
        description="Overall politeness score (0.0-1.0)",
    )
    markers_by_speaker: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description="Marker counts per speaker",
    )

    @computed_field
    @property
    def total_markers(self) -> int:
        """Total count of all politeness markers."""
        return (
            self.hedges_count
            + self.boosters_count
            + self.gratitude_count
            + self.apology_count
            + self.please_count
            + self.formal_markers
        )


class TranscriptFeatures(BaseModel):
    """Combined conversational features for a transcript.

    This is the primary output type from feature extraction,
    aggregating all feature categories with source file info.

    Attributes:
        file_name: Name of the source transcript file.
        file_path: Full path to the source file.
        questions: Question detection statistics.
        turn_taking: Turn-taking pattern statistics.
        vocabulary: Vocabulary diversity statistics.
        politeness: Politeness marker statistics.
        metadata: Additional extraction metadata.

    Example:
        >>> features = TranscriptFeatures.model_validate(data)
        >>> print(f"Questions: {features.questions.total_questions}")
        >>> print(f"TTR: {features.vocabulary.type_token_ratio:.2f}")
    """

    file_name: str = Field(description="Source transcript filename")
    file_path: str = Field(description="Full path to source file")
    questions: QuestionStats = Field(
        default_factory=QuestionStats,
        description="Question statistics",
    )
    turn_taking: TurnTakingStats = Field(
        default_factory=TurnTakingStats,
        description="Turn-taking statistics",
    )
    vocabulary: VocabularyStats = Field(
        default_factory=VocabularyStats,
        description="Vocabulary statistics",
    )
    politeness: PolitenessStats = Field(
        default_factory=PolitenessStats,
        description="Politeness statistics",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    def to_flat_dict(self) -> dict[str, Any]:
        """Convert to flat dictionary for CSV export.

        Flattens nested structures into a single-level dictionary
        with prefixed keys (e.g., "questions_total", "vocabulary_ttr").

        Returns:
            Flat dictionary suitable for CSV writer.
        """
        flat = {
            "file_name": self.file_name,
            "file_path": self.file_path,
        }

        # flatten questions.
        flat["total_questions"] = self.questions.total_questions
        flat["wh_questions"] = self.questions.wh_questions
        flat["yes_no_questions"] = self.questions.yes_no_questions
        flat["rhetorical_questions"] = self.questions.rhetorical_questions
        flat["question_rate"] = self.questions.question_rate

        # flatten turn-taking.
        flat["total_turns"] = self.turn_taking.total_turns
        flat["unique_speakers"] = self.turn_taking.unique_speakers
        flat["turn_switch_count"] = self.turn_taking.turn_switch_count
        flat["turn_switch_rate"] = self.turn_taking.turn_switch_rate
        flat["avg_turn_duration"] = self.turn_taking.avg_turn_duration
        flat["balance_score"] = self.turn_taking.balance_score

        # flatten vocabulary.
        flat["total_words"] = self.vocabulary.total_words
        flat["unique_words"] = self.vocabulary.unique_words
        flat["type_token_ratio"] = self.vocabulary.type_token_ratio
        flat["avg_word_length"] = self.vocabulary.avg_word_length
        flat["hapax_legomena"] = self.vocabulary.hapax_legomena

        # flatten politeness.
        flat["hedges_count"] = self.politeness.hedges_count
        flat["boosters_count"] = self.politeness.boosters_count
        flat["gratitude_count"] = self.politeness.gratitude_count
        flat["politeness_score"] = self.politeness.politeness_score
        flat["total_politeness_markers"] = self.politeness.total_markers

        return flat
