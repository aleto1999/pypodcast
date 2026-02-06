"""Turn-taking pattern analysis for podcast conversations."""

from dataclasses import dataclass, field


@dataclass
class TurnTakingStats:
    """Statistics about turn-taking patterns in a conversation."""

    total_turns: int = 0
    turns_by_speaker: dict = field(default_factory=dict)
    speaking_time_by_speaker: dict = field(default_factory=dict)
    average_turn_time_by_speaker: dict = field(default_factory=dict)
    average_switch_time: float = 0.0
    dominance_index: float = 0.0
    total_duration_seconds: float = 0.0


class TurnTakingAnalyzer:
    """Analyzes turn-taking patterns in podcast transcripts."""

    def analyze(self, segments: list[dict]) -> TurnTakingStats:
        """Analyze turn-taking patterns from transcript segments.

        Args:
            segments: List of transcript segments with 'text', 'speaker',
                     'start', and 'end' keys.

        Returns:
            TurnTakingStats with turn-taking analysis results.
        """
        if not segments:
            return TurnTakingStats()

        stats = TurnTakingStats()

        # calculate total duration.
        if segments:
            stats.total_duration_seconds = segments[-1].get("end", 0.0)

        # count turns and speaking time.
        turns_by_speaker: dict[str, int] = {}
        speaking_time_by_speaker: dict[str, float] = {}

        current_speaker = segments[0].get("speaker", "UNKNOWN")
        turns_by_speaker[current_speaker] = 1
        total_turns = 1

        for seg in segments:
            speaker = seg.get("speaker", "UNKNOWN")
            duration = seg.get("end", 0.0) - seg.get("start", 0.0)

            # accumulate speaking time.
            if speaker not in speaking_time_by_speaker:
                speaking_time_by_speaker[speaker] = 0.0
            speaking_time_by_speaker[speaker] += duration

            # count turn changes.
            if speaker != current_speaker:
                total_turns += 1
                if speaker not in turns_by_speaker:
                    turns_by_speaker[speaker] = 0
                turns_by_speaker[speaker] += 1
                current_speaker = speaker

        stats.total_turns = total_turns
        stats.turns_by_speaker = turns_by_speaker
        stats.speaking_time_by_speaker = speaking_time_by_speaker

        # calculate average turn time per speaker.
        for speaker, turns in turns_by_speaker.items():
            if turns > 0 and speaker in speaking_time_by_speaker:
                stats.average_turn_time_by_speaker[speaker] = (
                    speaking_time_by_speaker[speaker] / turns
                )

        # calculate average switch time.
        if total_turns > 0 and stats.total_duration_seconds > 0:
            total_minutes = stats.total_duration_seconds / 60.0
            stats.average_switch_time = total_minutes / total_turns

        # calculate dominance index (speaking time imbalance).
        if speaking_time_by_speaker:
            sorted_times = sorted(
                speaking_time_by_speaker.values(), reverse=True
            )
            if len(sorted_times) >= 2 and stats.total_duration_seconds > 0:
                stats.dominance_index = (
                    (sorted_times[0] - sorted_times[1])
                    / stats.total_duration_seconds
                )
            elif len(sorted_times) == 1:
                stats.dominance_index = 1.0

        return stats

    def to_dict(self, stats: TurnTakingStats) -> dict:
        """Convert TurnTakingStats to dictionary for JSON/CSV serialization."""
        return {
            "total_turns": stats.total_turns,
            "total_duration_minutes": round(
                stats.total_duration_seconds / 60.0, 2
            ),
            "average_switch_time_minutes": round(stats.average_switch_time, 4),
            "dominance_index": round(stats.dominance_index, 4),
            "turns_by_speaker": stats.turns_by_speaker,
            "speaking_time_by_speaker": {
                speaker: round(time, 2)
                for speaker, time in stats.speaking_time_by_speaker.items()
            },
            "average_turn_time_by_speaker": {
                speaker: round(time, 2)
                for speaker, time in stats.average_turn_time_by_speaker.items()
            },
        }

    def to_flat_dict(self, stats: TurnTakingStats) -> dict:
        """Convert TurnTakingStats to flat dictionary for CSV export.

        Flattens nested speaker dictionaries into individual columns.
        """
        flat = {
            "total_turns": stats.total_turns,
            "total_duration_minutes": round(
                stats.total_duration_seconds / 60.0, 2
            ),
            "average_switch_time_minutes": round(stats.average_switch_time, 4),
            "dominance_index": round(stats.dominance_index, 4),
        }

        # flatten speaker-specific metrics.
        for speaker, turns in stats.turns_by_speaker.items():
            flat[f"turns_{speaker}"] = turns

        for speaker, time in stats.speaking_time_by_speaker.items():
            flat[f"speaking_time_{speaker}"] = round(time, 2)

        for speaker, time in stats.average_turn_time_by_speaker.items():
            flat[f"avg_turn_time_{speaker}"] = round(time, 2)

        return flat
