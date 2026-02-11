"""Stage 5: Voice-name assignment (speaker mapping).

Maps extracted names to speaker IDs using multiple heuristics:
1. Direct mention heuristic: "first voice which says the host's name"
2. Cross-episode consistency: hosts have consistent speaking time rank
3. Introduction patterns: "I'm [name]" or "my name is [name]"
4. Metadata-informed priors: RSS data, episode titles
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from podcast_conversations.speaker_labeling.types import (
    SpeakerMapping,
    RoleClassification,
    SpeakerRole,
    AssignmentMethod,
)

logger = logging.getLogger(__name__)


# patterns for self-introduction.
SELF_INTRO_PATTERNS = [
    r"(?:i'm|i am|my name is|this is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+here",
    r"welcome.*(?:i'm|i am)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
]


@dataclass
class MentionContext:
    """Context around a name mention in transcript."""

    name: str
    speaker_id: str
    segment_index: int
    segment_start: float
    text: str


class VoiceNameMapper:
    """Maps names to speaker IDs using heuristics."""

    def __init__(self, intro_window: int = 30, use_cross_episode: bool = True):
        """
        Initialize mapper.

        Args:
            intro_window: Number of initial segments to search for introductions.
            use_cross_episode: Whether to use cross-episode patterns.
        """
        self.intro_window = intro_window
        self.use_cross_episode = use_cross_episode

    def map_names_to_speakers(
        self,
        transcript_data: dict[str, Any],
        role_classifications: list[RoleClassification],
        host_rank: int | None = None,
    ) -> list[SpeakerMapping]:
        """
        Map classified names to speaker IDs.

        Args:
            transcript_data: Full transcript JSON data.
            role_classifications: Role classifications from stage 4.
            host_rank: Expected host rank from cross-episode analysis (0=most speaking).

        Returns:
            List of speaker mappings.
        """
        segments = transcript_data.get("segments", [])
        if not segments:
            return []

        mappings = []
        assigned_speakers = set()
        assigned_names = set()

        # get unique speaker IDs.
        speaker_ids = list({seg.get("speaker", "UNKNOWN") for seg in segments})

        # sort roles by priority: HOST first, then GUEST, then NEITHER.
        sorted_roles = sorted(
            role_classifications,
            key=lambda r: (
                0 if r.role == SpeakerRole.HOST else
                1 if r.role == SpeakerRole.GUEST else 2,
                -r.confidence,
            ),
        )

        # calculate speaker ranks by speaking time.
        speaker_times = self._calculate_speaking_times(segments)
        speaker_ranks = {
            speaker: rank
            for rank, (speaker, _) in enumerate(
                sorted(speaker_times.items(), key=lambda x: x[1], reverse=True)
            )
        }

        for role in sorted_roles:
            if role.name in assigned_names:
                continue

            mapping = self._assign_speaker(
                role=role,
                segments=segments,
                speaker_ids=speaker_ids,
                assigned_speakers=assigned_speakers,
                speaker_ranks=speaker_ranks,
                host_rank=host_rank,
            )

            if mapping:
                mappings.append(mapping)
                assigned_speakers.add(mapping.speaker_id)
                assigned_names.add(mapping.name)

        return mappings

    def _assign_speaker(
        self,
        role: RoleClassification,
        segments: list[dict[str, Any]],
        speaker_ids: list[str],
        assigned_speakers: set[str],
        speaker_ranks: dict[str, int],
        host_rank: int | None,
    ) -> SpeakerMapping | None:
        """Assign a single name to a speaker ID."""
        available_speakers = [s for s in speaker_ids if s not in assigned_speakers]
        if not available_speakers:
            return None

        name = role.name
        name_lower = name.lower()

        # heuristic 1: self-introduction in first segments.
        for i, seg in enumerate(segments[: self.intro_window]):
            text = seg.get("text", "").lower()

            # check for self-introduction patterns.
            for pattern in SELF_INTRO_PATTERNS:
                match = re.search(pattern, seg.get("text", ""), re.IGNORECASE)
                if match and match.group(1).lower() == name_lower:
                    speaker = seg.get("speaker")
                    if speaker in available_speakers:
                        return SpeakerMapping(
                            speaker_id=speaker,
                            name=name,
                            role=role.role,
                            confidence=min(0.9, role.confidence + 0.2),
                            assignment_method=AssignmentMethod.SELF_INTRODUCTION,
                            evidence={
                                "pattern": "self_introduction",
                                "segment_index": i,
                                "text_snippet": seg.get("text", "")[:100],
                            },
                        )

        # heuristic 2: direct mention (first speaker to say the name).
        for i, seg in enumerate(segments[: self.intro_window * 2]):
            text = seg.get("text", "").lower()
            speaker = seg.get("speaker")

            if name_lower in text and speaker in available_speakers:
                # another speaker saying this name = that speaker is introducing them.
                intro_speaker = speaker
                introduced_speaker = self._get_next_different_speaker(
                    segments, i, intro_speaker
                )

                if introduced_speaker and introduced_speaker in available_speakers:
                    return SpeakerMapping(
                        speaker_id=introduced_speaker,
                        name=name,
                        role=role.role,
                        confidence=min(0.85, role.confidence + 0.1),
                        assignment_method=AssignmentMethod.DIRECT_MENTION,
                        evidence={
                            "pattern": "introduced_by_other",
                            "introducer": intro_speaker,
                            "segment_index": i,
                        },
                    )

        # heuristic 3: cross-episode host rank pattern.
        if (
            role.role == SpeakerRole.HOST
            and host_rank is not None
            and self.use_cross_episode
        ):
            for speaker, rank in speaker_ranks.items():
                if rank == host_rank and speaker in available_speakers:
                    return SpeakerMapping(
                        speaker_id=speaker,
                        name=name,
                        role=role.role,
                        confidence=min(0.75, role.confidence),
                        assignment_method=AssignmentMethod.CROSS_EPISODE_PATTERN,
                        evidence={
                            "pattern": "cross_episode_rank",
                            "expected_rank": host_rank,
                            "actual_rank": rank,
                        },
                    )

        # heuristic 4: role-based position.
        if role.role == SpeakerRole.HOST:
            # host is usually the speaker with most speaking time.
            for speaker, rank in speaker_ranks.items():
                if rank == 0 and speaker in available_speakers:
                    return SpeakerMapping(
                        speaker_id=speaker,
                        name=name,
                        role=role.role,
                        confidence=min(0.6, role.confidence),
                        assignment_method=AssignmentMethod.SPEAKING_TIME_RANK,
                        evidence={
                            "pattern": "highest_speaking_time",
                            "rank": rank,
                        },
                    )

        elif role.role == SpeakerRole.GUEST:
            # guest is usually second in speaking time.
            for speaker, rank in speaker_ranks.items():
                if rank == 1 and speaker in available_speakers:
                    return SpeakerMapping(
                        speaker_id=speaker,
                        name=name,
                        role=role.role,
                        confidence=min(0.5, role.confidence),
                        assignment_method=AssignmentMethod.SPEAKING_TIME_RANK,
                        evidence={
                            "pattern": "second_speaking_time",
                            "rank": rank,
                        },
                    )

        # fallback: assign to first available speaker.
        if available_speakers:
            return SpeakerMapping(
                speaker_id=available_speakers[0],
                name=name,
                role=role.role,
                confidence=0.3,
                assignment_method=AssignmentMethod.FALLBACK,
                evidence={"pattern": "fallback_first_available"},
            )

        return None

    def _calculate_speaking_times(
        self,
        segments: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Calculate total speaking time per speaker."""
        times: dict[str, float] = {}

        for seg in segments:
            speaker = seg.get("speaker", "UNKNOWN")
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            duration = max(0, end - start)
            times[speaker] = times.get(speaker, 0) + duration

        return times

    def _get_next_different_speaker(
        self,
        segments: list[dict[str, Any]],
        current_index: int,
        current_speaker: str,
    ) -> str | None:
        """Get the next speaker that's different from current."""
        for seg in segments[current_index + 1 : current_index + 5]:
            speaker = seg.get("speaker")
            if speaker and speaker != current_speaker:
                return speaker
        return None


class TranscriptLabeler:
    """Applies speaker mappings to create labeled transcript."""

    def apply_labels(
        self,
        transcript_data: dict[str, Any],
        mappings: list[SpeakerMapping],
    ) -> dict[str, Any]:
        """
        Apply speaker name mappings to transcript.

        Args:
            transcript_data: Original transcript data.
            mappings: Speaker mappings from voice-name mapper.

        Returns:
            New transcript data with speaker labels.
        """
        # create speaker_id -> name lookup.
        name_map = {m.speaker_id: m.name for m in mappings}

        # deep copy transcript.
        labeled = json.loads(json.dumps(transcript_data))

        # apply labels.
        for segment in labeled.get("segments", []):
            speaker_id = segment.get("speaker")
            if speaker_id in name_map:
                segment["speaker_name"] = name_map[speaker_id]
            else:
                segment["speaker_name"] = speaker_id

        # add metadata.
        labeled["speaker_mappings"] = [
            {
                "speaker_id": m.speaker_id,
                "name": m.name,
                "role": m.role.value,
                "confidence": m.confidence,
                "method": m.assignment_method.value,
                "evidence": m.evidence,
            }
            for m in mappings
        ]

        return labeled


def create_voice_name_mapper(
    intro_window: int = 30,
    use_cross_episode: bool = True,
) -> VoiceNameMapper:
    """Factory function to create voice-name mapper.

    Args:
        intro_window: Segments to search for introductions.
        use_cross_episode: Use cross-episode patterns.

    Returns:
        Configured VoiceNameMapper.
    """
    return VoiceNameMapper(intro_window=intro_window, use_cross_episode=use_cross_episode)


def create_transcript_labeler() -> TranscriptLabeler:
    """Factory function to create transcript labeler."""
    return TranscriptLabeler()
