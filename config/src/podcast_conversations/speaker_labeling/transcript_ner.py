"""Stage 2: Transcript-based name extraction using NER and pattern matching.

Extracts person names from transcript text using:
- spaCy NER for PERSON entities
- Pattern matching for introduction phrases
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from podcast_conversations.speaker_labeling.types import (
    ExtractedName,
    ExtractionMethod,
)

logger = logging.getLogger(__name__)


# introduction patterns that indicate speaker names.
INTRODUCTION_PATTERNS = [
    # self-introductions.
    r"(?:I'm|I am|This is|My name is)\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+)",
    # guest introductions.
    r"(?:Welcome|Welcoming|Please welcome)\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+)",
    r"(?:Our guest|Today's guest|Joining us|Joined by)\s+(?:is\s+)?([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+)",
    r"(?:Talking to|Speaking with|Interviewing)\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+)",
    # role markers.
    r"[Hh]ost[,:\s]+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+)",
    r"[Gg]uest[,:\s]+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+)",
    # presence markers.
    r"([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+)(?:\s+here|,\s+here)",
    r"([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+)\s+joins?\s+(?:us|the show|me)",
]


class TranscriptNERExtractor:
    """Extracts person names from transcript text using NER and patterns."""

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize NER extractor.

        Args:
            spacy_model: spaCy model to use (en_core_web_sm, en_core_web_lg, etc.)
        """
        self.spacy_model = spacy_model
        self._nlp = None
        self._patterns = [re.compile(p) for p in INTRODUCTION_PATTERNS]

    @property
    def nlp(self):
        """Lazy load spaCy model."""
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load(self.spacy_model)
                logger.info(f"Loaded spaCy model: {self.spacy_model}")
            except OSError:
                # try to download and load.
                import spacy
                from spacy.cli import download
                logger.info(f"Downloading spaCy model: {self.spacy_model}")
                download(self.spacy_model)
                self._nlp = spacy.load(self.spacy_model)
        return self._nlp

    def extract_from_text(
        self,
        text: str,
        max_words: int = 500,
        source_speaker: str | None = None,
    ) -> list[ExtractedName]:
        """
        Extract person names from text.

        Args:
            text: Text to extract names from.
            max_words: Maximum words to analyze (focus on intro).
            source_speaker: Speaker ID who spoke this text.

        Returns:
            List of extracted names.
        """
        # truncate to intro portion.
        words = text.split()
        intro_text = " ".join(words[:max_words])

        names: list[ExtractedName] = []

        # method 1: spaCy NER.
        try:
            names.extend(self._extract_ner(intro_text, source_speaker))
        except Exception as e:
            logger.warning(f"NER extraction failed: {e}")

        # method 2: pattern matching.
        names.extend(self._extract_patterns(intro_text, source_speaker))

        # deduplicate and merge confidence.
        return self._deduplicate_names(names)

    def extract_from_transcript(
        self,
        transcript_data: dict[str, Any],
        max_words: int = 500,
        include_full_transcript: bool = False,
    ) -> list[ExtractedName]:
        """
        Extract person names from transcript data structure.

        Args:
            transcript_data: Transcript JSON data with segments.
            max_words: Words to analyze from start.
            include_full_transcript: Also scan full transcript (slower).

        Returns:
            List of extracted names.
        """
        segments = transcript_data.get("segments", [])
        if not segments:
            return []

        # build text from segments.
        full_text = " ".join(seg.get("text", "") for seg in segments)

        return self.extract_from_text(
            text=full_text,
            max_words=max_words if not include_full_transcript else len(full_text.split()),
        )

    def extract_names_by_speaker(
        self,
        transcript_data: dict[str, Any],
    ) -> dict[str, list[ExtractedName]]:
        """
        Extract names and group by which speaker mentioned them.

        Args:
            transcript_data: Transcript JSON data with segments.

        Returns:
            Dict mapping speaker_id -> list of names they mentioned.
        """
        segments = transcript_data.get("segments", [])
        speaker_names: dict[str, list[ExtractedName]] = {}

        for segment in segments:
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment.get("text", "")

            if speaker not in speaker_names:
                speaker_names[speaker] = []

            # extract from this segment.
            names = self.extract_from_text(
                text=text,
                max_words=len(text.split()),  # full segment.
                source_speaker=speaker,
            )
            speaker_names[speaker].extend(names)

        # deduplicate per speaker.
        return {
            speaker: self._deduplicate_names(names)
            for speaker, names in speaker_names.items()
        }

    def _extract_ner(
        self,
        text: str,
        source_speaker: str | None = None,
    ) -> list[ExtractedName]:
        """Extract names using spaCy NER."""
        doc = self.nlp(text)
        names = []

        for ent in doc.ents:
            if ent.label_ == "PERSON":
                # get context.
                start = max(0, ent.start_char - 50)
                end = min(len(text), ent.end_char + 50)
                context = text[start:end]

                # higher confidence for multi-word names.
                word_count = len(ent.text.split())
                confidence = 0.7 if word_count >= 2 else 0.4

                names.append(ExtractedName(
                    name=ent.text,
                    context=context,
                    position=ent.start_char,
                    extraction_method=ExtractionMethod.NER,
                    confidence=confidence,
                    source_speaker=source_speaker,
                ))

        return names

    def _extract_patterns(
        self,
        text: str,
        source_speaker: str | None = None,
    ) -> list[ExtractedName]:
        """Extract names using introduction patterns."""
        names = []

        for pattern in self._patterns:
            for match in pattern.finditer(text):
                name = match.group(1).strip()

                # validate name.
                if not self._is_valid_name(name):
                    continue

                # get context.
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 20)
                context = text[start:end]

                names.append(ExtractedName(
                    name=name,
                    context=context,
                    position=match.start(),
                    extraction_method=ExtractionMethod.PATTERN,
                    confidence=0.85,  # higher confidence for pattern matches.
                    source_speaker=source_speaker,
                ))

        return names

    def _deduplicate_names(self, names: list[ExtractedName]) -> list[ExtractedName]:
        """Merge duplicate names, combining confidence scores."""
        name_map: dict[str, ExtractedName] = {}

        for extracted in names:
            normalized = extracted.name.lower().strip()

            if normalized in name_map:
                existing = name_map[normalized]
                # boost confidence if found by multiple methods.
                if existing.extraction_method != extracted.extraction_method:
                    existing.extraction_method = ExtractionMethod.COMBINED
                    existing.confidence = min(0.95, existing.confidence + 0.15)
                # keep earliest position.
                if extracted.position < existing.position:
                    existing.position = extracted.position
                    existing.context = extracted.context
            else:
                name_map[normalized] = extracted

        # sort by position (earlier mentions first).
        return sorted(name_map.values(), key=lambda x: x.position)

    def _is_valid_name(self, name: str) -> bool:
        """Check if a string looks like a valid person name."""
        words = name.split()
        if len(words) < 2 or len(words) > 5:
            return False

        # check that words are capitalized.
        for word in words:
            # allow initials like "J." or middle initials.
            if len(word) == 2 and word[1] == ".":
                continue
            if len(word) < 2:
                return False
            if not word[0].isupper():
                return False

        # filter common false positives.
        lower_name = name.lower()
        false_positives = {
            "new york", "los angeles", "san francisco", "united states",
            "the show", "this week", "last week", "next week",
        }
        if lower_name in false_positives:
            return False

        return True


def create_ner_extractor(model: str = "en_core_web_sm") -> TranscriptNERExtractor:
    """Factory function to create NER extractor.

    Args:
        model: spaCy model to use.

    Returns:
        Configured TranscriptNERExtractor.
    """
    return TranscriptNERExtractor(spacy_model=model)
