"""Question detection and classification utilities."""

import re
from dataclasses import dataclass, field
from typing import Optional

# regex pattern for detecting direct address.
DIRECT_ADDRESS_RE = re.compile(
    r"""
    (
        # second-person pronouns.
        \b(
            you|your|yours|yourself|
            you['']re|youre|
            you['']ll|youll|
            you['']d|youd|
            you['']ve|youve
        )\b

      | # interrogatives + "you".
        \b(why|what|how|when|where|who|can|could|do|did|will|would|
           are|were|should|have|had|won['']?t|don['']?t|is)\b
        [\s,;:—-]+you\b

      | # imperatives: verb followed by anything.
        ^\s*(tell|explain|look|listen|imagine|remember|consider|try|check|answer)\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


@dataclass
class QuestionStats:
    """Statistics about questions in a transcript."""

    total_sentences: int = 0
    total_questions: int = 0
    question_ratio: float = 0.0
    questions_per_minute: float = 0.0
    questions_by_speaker: dict = field(default_factory=dict)
    question_texts: list = field(default_factory=list)
    direct_address_count: int = 0
    direct_address_ratio: float = 0.0


class QuestionAnalyzer:
    """Analyzes questions in podcast transcripts."""

    def __init__(self, nlp=None):
        """Initialize the question analyzer.

        Args:
            nlp: Optional spacy nlp model for sentence splitting.
                 If None, will use regex-based splitting.
        """
        self.nlp = nlp

    def split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        if self.nlp is not None:
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents]
        else:
            # fallback regex-based splitting.
            sentences = re.split(r"(?<=[.!?])\s+", text)
            return [s.strip() for s in sentences if s.strip()]

    def is_question(self, sentence: str) -> bool:
        """Check if a sentence is a question using regex."""
        # check for question mark.
        if re.search(r"\?\s*$", sentence):
            return True
        # check for question starters.
        question_starters = r"^(who|what|when|where|why|how|is|are|was|were|do|does|did|can|could|will|would|should|have|has|had)\b"
        if re.search(question_starters, sentence.strip(), re.IGNORECASE):
            return True
        return False

    def is_direct_address(self, text: str) -> bool:
        """Check if text contains direct address (second-person reference)."""
        return bool(DIRECT_ADDRESS_RE.search(text))

    def normalize_text(self, text: str) -> str:
        """Normalize text for matching."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def get_best_segment_match(
        self, sentence: str, segments: list[dict]
    ) -> tuple[Optional[dict], int]:
        """Find the best matching segment for a sentence."""
        q_words = set(self.normalize_text(sentence).split())
        best_score = 0
        best_seg = None

        for seg in segments:
            seg_words = set(self.normalize_text(seg.get("text", "")).split())
            score = len(q_words & seg_words)

            if score > best_score:
                best_score = score
                best_seg = seg

        return best_seg, best_score

    def analyze(
        self,
        segments: list[dict],
        full_text: Optional[str] = None,
        total_duration_seconds: float = 0.0,
    ) -> QuestionStats:
        """Analyze questions in transcript segments.

        Args:
            segments: List of transcript segments with 'text' and 'speaker' keys.
            full_text: Optional full transcript text. If not provided, will
                       concatenate segment texts.
            total_duration_seconds: Total duration for questions-per-minute calc.

        Returns:
            QuestionStats with question analysis results.
        """
        if full_text is None:
            full_text = " ".join(seg.get("text", "") for seg in segments)

        sentences = self.split_into_sentences(full_text)
        stats = QuestionStats(total_sentences=len(sentences))

        questions_by_speaker: dict[str, list[str]] = {}
        question_texts: list[str] = []
        direct_address_count = 0

        for sentence in sentences:
            if self.is_question(sentence):
                question_texts.append(sentence)

                # find speaker for this question.
                seg, _ = self.get_best_segment_match(sentence, segments)
                if seg:
                    speaker = seg.get("speaker", "UNKNOWN")
                    if speaker not in questions_by_speaker:
                        questions_by_speaker[speaker] = []
                    questions_by_speaker[speaker].append(sentence)

                # check for direct address.
                if self.is_direct_address(sentence):
                    direct_address_count += 1

        stats.total_questions = len(question_texts)
        stats.question_texts = question_texts
        stats.questions_by_speaker = questions_by_speaker

        if stats.total_sentences > 0:
            stats.question_ratio = stats.total_questions / stats.total_sentences

        if total_duration_seconds > 0:
            total_minutes = total_duration_seconds / 60.0
            stats.questions_per_minute = stats.total_questions / total_minutes

        stats.direct_address_count = direct_address_count
        if stats.total_questions > 0:
            stats.direct_address_ratio = direct_address_count / stats.total_questions

        return stats

    def to_dict(self, stats: QuestionStats) -> dict:
        """Convert QuestionStats to dictionary for JSON serialization."""
        return {
            "total_sentences": stats.total_sentences,
            "total_questions": stats.total_questions,
            "question_ratio": round(stats.question_ratio, 4),
            "questions_per_minute": round(stats.questions_per_minute, 4),
            "questions_by_speaker": {
                speaker: len(questions)
                for speaker, questions in stats.questions_by_speaker.items()
            },
            "direct_address_count": stats.direct_address_count,
            "direct_address_ratio": round(stats.direct_address_ratio, 4),
        }
