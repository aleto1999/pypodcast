"""Vocabulary diversity analysis for podcast conversations."""

import math
import re
from dataclasses import dataclass, field

# common english stopwords.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can", "need",
    "dare", "ought", "used", "it", "its", "this", "that", "these", "those",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "they", "them", "their", "theirs",
    "themselves", "what", "which", "who", "whom", "when", "where", "why",
    "how", "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "just", "also", "now", "here", "there", "then", "once",
    "if", "because", "until", "while", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below",
    "up", "down", "out", "off", "over", "under", "again", "further",
    "any", "being", "having", "doing", "am", "going", "get", "got",
    "like", "know", "think", "yeah", "yes", "no", "okay", "ok", "um",
    "uh", "well", "right", "mean", "really", "actually", "basically",
    "literally", "thing", "things", "something", "anything", "nothing",
    "everything", "someone", "anyone", "everyone", "nobody", "everybody",
}


@dataclass
class VocabularyStats:
    """Statistics about vocabulary diversity in a conversation."""

    total_words: int = 0
    total_words_no_stopwords: int = 0
    unique_words: int = 0
    unique_words_no_stopwords: int = 0
    type_token_ratio: float = 0.0
    root_type_token_ratio: float = 0.0
    words_by_speaker: dict = field(default_factory=dict)
    unique_words_by_speaker: dict = field(default_factory=dict)
    ttr_by_speaker: dict = field(default_factory=dict)


class VocabularyAnalyzer:
    """Analyzes vocabulary diversity in podcast transcripts."""

    def __init__(self, stopwords: set[str] | None = None):
        """Initialize the vocabulary analyzer.

        Args:
            stopwords: Set of stopwords to exclude. Defaults to English stopwords.
        """
        self.stopwords = stopwords if stopwords is not None else STOPWORDS

    def extract_words(self, text: str) -> list[str]:
        """Extract words from text."""
        words = re.findall(r"\b[a-zA-Z']+\b", text.lower())
        return words

    def extract_words_no_stopwords(self, text: str) -> list[str]:
        """Extract words from text, excluding stopwords."""
        words = self.extract_words(text)
        return [w for w in words if w not in self.stopwords]

    def get_unique_words(self, text: str) -> set[str]:
        """Get unique words from text."""
        return set(self.extract_words(text))

    def get_unique_words_no_stopwords(self, text: str) -> set[str]:
        """Get unique words from text, excluding stopwords."""
        return set(self.extract_words_no_stopwords(text))

    def calculate_ttr(self, total_words: int, unique_words: int) -> float:
        """Calculate Type-Token Ratio."""
        if total_words == 0:
            return 0.0
        return unique_words / total_words

    def calculate_rttr(self, total_words: int, unique_words: int) -> float:
        """Calculate Root Type-Token Ratio (more stable for varying text lengths)."""
        if total_words == 0:
            return 0.0
        return unique_words / math.sqrt(total_words)

    def analyze(self, segments: list[dict]) -> VocabularyStats:
        """Analyze vocabulary diversity from transcript segments.

        Args:
            segments: List of transcript segments with 'text' and 'speaker' keys.

        Returns:
            VocabularyStats with vocabulary analysis results.
        """
        stats = VocabularyStats()

        # aggregate text by speaker.
        text_by_speaker: dict[str, str] = {}
        all_text = ""

        for seg in segments:
            text = seg.get("text", "")
            speaker = seg.get("speaker", "UNKNOWN")
            all_text += " " + text

            if speaker not in text_by_speaker:
                text_by_speaker[speaker] = ""
            text_by_speaker[speaker] += " " + text

        # overall stats.
        all_words = self.extract_words(all_text)
        all_words_no_stop = self.extract_words_no_stopwords(all_text)
        all_unique = self.get_unique_words(all_text)
        all_unique_no_stop = self.get_unique_words_no_stopwords(all_text)

        stats.total_words = len(all_words)
        stats.total_words_no_stopwords = len(all_words_no_stop)
        stats.unique_words = len(all_unique)
        stats.unique_words_no_stopwords = len(all_unique_no_stop)
        stats.type_token_ratio = self.calculate_ttr(
            stats.total_words, stats.unique_words
        )
        stats.root_type_token_ratio = self.calculate_rttr(
            stats.total_words, stats.unique_words
        )

        # per-speaker stats.
        for speaker, text in text_by_speaker.items():
            words = self.extract_words(text)
            unique = self.get_unique_words(text)

            stats.words_by_speaker[speaker] = len(words)
            stats.unique_words_by_speaker[speaker] = len(unique)
            stats.ttr_by_speaker[speaker] = self.calculate_ttr(
                len(words), len(unique)
            )

        return stats

    def to_dict(self, stats: VocabularyStats) -> dict:
        """Convert VocabularyStats to dictionary for JSON serialization."""
        return {
            "total_words": stats.total_words,
            "total_words_no_stopwords": stats.total_words_no_stopwords,
            "unique_words": stats.unique_words,
            "unique_words_no_stopwords": stats.unique_words_no_stopwords,
            "type_token_ratio": round(stats.type_token_ratio, 4),
            "root_type_token_ratio": round(stats.root_type_token_ratio, 4),
            "words_by_speaker": stats.words_by_speaker,
            "unique_words_by_speaker": stats.unique_words_by_speaker,
            "ttr_by_speaker": {
                speaker: round(ttr, 4)
                for speaker, ttr in stats.ttr_by_speaker.items()
            },
        }

    def to_flat_dict(self, stats: VocabularyStats) -> dict:
        """Convert VocabularyStats to flat dictionary for CSV export."""
        flat = {
            "total_words": stats.total_words,
            "total_words_no_stopwords": stats.total_words_no_stopwords,
            "unique_words": stats.unique_words,
            "unique_words_no_stopwords": stats.unique_words_no_stopwords,
            "type_token_ratio": round(stats.type_token_ratio, 4),
            "root_type_token_ratio": round(stats.root_type_token_ratio, 4),
        }

        # flatten speaker-specific metrics.
        for speaker, count in stats.words_by_speaker.items():
            flat[f"words_{speaker}"] = count

        for speaker, count in stats.unique_words_by_speaker.items():
            flat[f"unique_words_{speaker}"] = count

        for speaker, ttr in stats.ttr_by_speaker.items():
            flat[f"ttr_{speaker}"] = round(ttr, 4)

        return flat
