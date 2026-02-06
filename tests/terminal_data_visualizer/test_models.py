"""tests for terminal_data_visualizer.models module."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest

from terminal_data_visualizer.models import (
    Classification,
    Match,
    Segment,
)


class TestSegment:
    """tests for Segment dataclass."""

    def test_create_segment(self):
        """should create a segment with required fields."""
        seg = Segment(text="hello world", start=0.0, end=1.0, speaker="SPEAKER_01")
        assert seg.text == "hello world"
        assert seg.start == 0.0
        assert seg.end == 1.0
        assert seg.speaker == "SPEAKER_01"

    def test_segment_defaults(self):
        """should have correct default values."""
        seg = Segment(text="test", start=0.0, end=1.0, speaker="SPEAKER_01")
        assert seg.confidence is None
        assert seg.words is None

    def test_segment_from_dict(self):
        """should create segment from dictionary."""
        data = {
            "text": "hello world",
            "start": 0.0,
            "end": 2.5,
            "speaker": "SPEAKER_02",
            "confidence": 0.95,
        }
        seg = Segment.from_dict(data)
        assert seg.text == "hello world"
        assert seg.speaker == "SPEAKER_02"
        assert seg.confidence == 0.95

    def test_segment_from_dict_missing_fields(self):
        """should handle missing fields with defaults."""
        seg = Segment.from_dict({})
        assert seg.text == ""
        assert seg.start == 0.0
        assert seg.end == 0.0
        assert seg.speaker == "UNKNOWN"

    def test_segment_duration(self):
        """should calculate duration correctly."""
        seg = Segment(text="test", start=1.5, end=3.0, speaker="SPEAKER_01")
        assert seg.duration == 1.5

    def test_segment_word_count(self):
        """should count words correctly."""
        seg = Segment(
            text="this is a test with seven words",
            start=0.0,
            end=1.0,
            speaker="SPEAKER_01",
        )
        assert seg.word_count == 7

    def test_segment_word_count_empty(self):
        """should handle empty text."""
        seg = Segment(text="", start=0.0, end=1.0, speaker="SPEAKER_01")
        assert seg.word_count == 0


class TestMatch:
    """tests for Match dataclass."""

    def test_create_match(self):
        """should create a match with required fields."""
        match = Match(
            keyword="dehumanization",
            category="negative",
            speaker="SPEAKER_01",
            start_time=10.5,
            end_time=12.0,
        )
        assert match.keyword == "dehumanization"
        assert match.category == "negative"
        assert match.speaker == "SPEAKER_01"

    def test_match_defaults(self):
        """should have correct default values."""
        match = Match(
            keyword="test",
            category="neutral",
            speaker=None,
            start_time=None,
            end_time=None,
        )
        assert match.context is None
        assert match.confidence_tier is None
        assert match.episode_id is None

    def test_match_from_dict(self):
        """should create match from dictionary."""
        data = {
            "keyword": "example",
            "category": "positive",
            "speaker": "SPEAKER_02",
            "start_time": 5.0,
            "end_time": 6.0,
            "context": "surrounding text",
            "confidence_tier": "high",
        }
        match = Match.from_dict(data)
        assert match.keyword == "example"
        assert match.category == "positive"
        assert match.context == "surrounding text"
        assert match.confidence_tier == "high"

    def test_match_from_dict_missing_fields(self):
        """should handle missing fields with defaults."""
        match = Match.from_dict({})
        assert match.keyword == ""
        assert match.category == ""
        assert match.speaker is None


class TestClassification:
    """tests for Classification dataclass."""

    def test_create_classification(self):
        """should create a classification with required fields."""
        clf = Classification(
            model_name="sentiment_model", label="positive", confidence=0.92
        )
        assert clf.model_name == "sentiment_model"
        assert clf.label == "positive"
        assert clf.confidence == 0.92

    def test_classification_defaults(self):
        """should have correct default values."""
        clf = Classification(model_name="test", label="neutral")
        assert clf.confidence is None

    def test_classification_from_dict(self):
        """should create classification from dictionary."""
        data = {
            "model_name": "bert_classifier",
            "label": "negative",
            "confidence": 0.85,
        }
        clf = Classification.from_dict(data)
        assert clf.model_name == "bert_classifier"
        assert clf.label == "negative"
        assert clf.confidence == 0.85

    def test_classification_from_dict_missing_fields(self):
        """should handle missing fields with defaults."""
        clf = Classification.from_dict({})
        assert clf.model_name == ""
        assert clf.label == ""
        assert clf.confidence is None
