"""Main feature extractor orchestrating all feature extraction modules."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .questions import QuestionAnalyzer, QuestionStats
from .turn_taking import TurnTakingAnalyzer, TurnTakingStats
from .vocabulary import VocabularyAnalyzer, VocabularyStats
from .politeness import PolitenessAnalyzer, PolitenessStats

logger = logging.getLogger(__name__)


@dataclass
class TranscriptFeatures:
    """All extracted features from a transcript."""

    file_path: str = ""
    file_name: str = ""
    questions: QuestionStats = field(default_factory=QuestionStats)
    turn_taking: TurnTakingStats = field(default_factory=TurnTakingStats)
    vocabulary: VocabularyStats = field(default_factory=VocabularyStats)
    politeness: PolitenessStats = field(default_factory=PolitenessStats)
    metadata: dict = field(default_factory=dict)


class FeatureExtractor:
    """Extracts conversation features from podcast transcripts."""

    def __init__(
        self,
        use_convokit: bool = False,
        nlp=None,
    ):
        """Initialize the feature extractor.

        Args:
            use_convokit: Whether to use convokit for politeness analysis.
            nlp: Optional spacy nlp model for sentence splitting.
        """
        self.question_analyzer = QuestionAnalyzer(nlp=nlp)
        self.turn_taking_analyzer = TurnTakingAnalyzer()
        self.vocabulary_analyzer = VocabularyAnalyzer()
        self.politeness_analyzer = PolitenessAnalyzer(use_convokit=use_convokit)
        self.nlp = nlp

    def extract(self, transcript_data: dict, file_path: str = "") -> TranscriptFeatures:
        """Extract all features from a transcript.

        Args:
            transcript_data: Parsed transcript JSON with 'segments' key.
            file_path: Optional file path for metadata.

        Returns:
            TranscriptFeatures with all extracted features.
        """
        features = TranscriptFeatures(
            file_path=file_path,
            file_name=Path(file_path).stem if file_path else "",
        )

        # get segments.
        segments = transcript_data.get("segments", [])
        if not segments:
            # try nested structure.
            transcription = transcript_data.get("transcription", {})
            segments = transcription.get("segments", [])

        if not segments:
            logger.warning(f"No segments found in {file_path}")
            return features

        # get full text if available.
        full_text = transcript_data.get("full_text")
        if not full_text:
            transcription = transcript_data.get("transcription", {})
            full_text = transcription.get("full_text")

        # calculate total duration.
        total_duration = 0.0
        if segments:
            total_duration = segments[-1].get("end", 0.0)

        # extract metadata.
        features.metadata = transcript_data.get("metadata", {})

        # extract features.
        features.questions = self.question_analyzer.analyze(
            segments=segments,
            full_text=full_text,
            total_duration_seconds=total_duration,
        )

        features.turn_taking = self.turn_taking_analyzer.analyze(segments)
        features.vocabulary = self.vocabulary_analyzer.analyze(segments)
        features.politeness = self.politeness_analyzer.analyze(segments)

        return features

    def to_dict(self, features: TranscriptFeatures) -> dict:
        """Convert TranscriptFeatures to nested dictionary."""
        return {
            "file_path": features.file_path,
            "file_name": features.file_name,
            "metadata": features.metadata,
            "questions": self.question_analyzer.to_dict(features.questions),
            "turn_taking": self.turn_taking_analyzer.to_dict(features.turn_taking),
            "vocabulary": self.vocabulary_analyzer.to_dict(features.vocabulary),
            "politeness": self.politeness_analyzer.to_dict(features.politeness),
        }

    def to_flat_dict(self, features: TranscriptFeatures) -> dict:
        """Convert TranscriptFeatures to flat dictionary for CSV export."""
        flat = {
            "file_path": features.file_path,
            "file_name": features.file_name,
        }

        # add question features (already flat-ish).
        q = self.question_analyzer.to_dict(features.questions)
        flat["total_sentences"] = q["total_sentences"]
        flat["total_questions"] = q["total_questions"]
        flat["question_ratio"] = q["question_ratio"]
        flat["questions_per_minute"] = q["questions_per_minute"]
        flat["direct_address_count"] = q["direct_address_count"]
        flat["direct_address_ratio"] = q["direct_address_ratio"]

        # flatten questions by speaker.
        for speaker, count in q["questions_by_speaker"].items():
            flat[f"questions_{speaker}"] = count

        # add turn-taking features (use flat method).
        flat.update(self.turn_taking_analyzer.to_flat_dict(features.turn_taking))

        # add vocabulary features (use flat method).
        flat.update(self.vocabulary_analyzer.to_flat_dict(features.vocabulary))

        # add politeness features (use flat method).
        flat.update(self.politeness_analyzer.to_flat_dict(features.politeness))

        return flat


def extract_features_from_transcript(
    file_path: Path | str,
    use_convokit: bool = False,
) -> TranscriptFeatures:
    """Extract features from a single transcript file.

    Args:
        file_path: Path to the transcript JSON file.
        use_convokit: Whether to use convokit for politeness analysis.

    Returns:
        TranscriptFeatures with all extracted features.
    """
    file_path = Path(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)

    extractor = FeatureExtractor(use_convokit=use_convokit)
    return extractor.extract(transcript_data, str(file_path))


def features_to_csv_row(features: TranscriptFeatures) -> dict[str, Any]:
    """Convert features to a flat dictionary suitable for CSV export."""
    extractor = FeatureExtractor()
    return extractor.to_flat_dict(features)
