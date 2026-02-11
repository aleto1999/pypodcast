"""Stage 3: Cross-episode host consistency analysis.

Identifies hosts by analyzing speaker patterns across all episodes of a podcast:
- Presence: hosts appear in most/all episodes
- Speaking time: hosts typically have consistent speaking time
- Position: hosts usually speak first and last
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from podcast_conversations.speaker_labeling.types import (
    SpeakerStats,
    HostCandidate,
)

logger = logging.getLogger(__name__)


class CrossEpisodeAnalyzer:
    """Analyzes speaker patterns across episodes to identify hosts."""

    def __init__(self, transcripts_dir: Path):
        """
        Initialize analyzer.

        Args:
            transcripts_dir: Directory containing podcast transcript subdirectories.
        """
        self.transcripts_dir = transcripts_dir

    def analyze_podcast(self, podcast_name: str) -> HostCandidate | None:
        """
        Analyze all episodes of a podcast to identify likely host.

        The host is typically:
        1. Present in most/all episodes
        2. Has significant speaking time in each episode
        3. Consistent speaking time proportion across episodes

        Args:
            podcast_name: Name of the podcast (folder name).

        Returns:
            HostCandidate if pattern identified, None otherwise.
        """
        podcast_dir = self.transcripts_dir / podcast_name
        if not podcast_dir.exists():
            return None

        episode_files = list(podcast_dir.glob("*.json"))
        if not episode_files:
            return None

        # track stats by speaking time rank.
        # rank 0 = speaker with most speaking time in that episode.
        position_stats: dict[int, SpeakerStats] = defaultdict(SpeakerStats)

        for episode_file in episode_files:
            episode_speakers = self._analyze_episode(episode_file)

            # sort speakers by speaking time (descending).
            sorted_speakers = sorted(
                episode_speakers.items(),
                key=lambda x: x[1]["total_time"],
                reverse=True,
            )

            for rank, (speaker_id, stats) in enumerate(sorted_speakers):
                if rank > 3:  # only track top 4 speakers.
                    break

                pos_stats = position_stats[rank]
                pos_stats.total_speaking_time += stats["total_time"]
                pos_stats.total_segments += stats["segment_count"]
                pos_stats.episodes_appeared += 1
                pos_stats.episode_speaking_times.append(stats["total_time"])

        total_episodes = len(episode_files)
        best_candidate = None
        best_score = 0.0

        for rank, stats in position_stats.items():
            if rank > 2:  # only consider top 3 speakers.
                continue

            # scoring: presence * consistency * speaking_proportion.
            presence_score = stats.episodes_appeared / total_episodes

            # penalize if appears in less than 70% of episodes.
            if presence_score < 0.7:
                presence_score *= 0.5

            # consistency score (1 = perfectly consistent).
            consistency = stats.speaking_time_consistency

            # combine scores.
            score = presence_score * (0.5 + 0.5 * max(0, consistency))

            if score > best_score and stats.episodes_appeared >= min(3, total_episodes):
                best_score = score
                best_candidate = HostCandidate(
                    speaker_pattern=f"rank_{rank}_by_speaking_time",
                    confidence=score,
                    evidence={
                        "episodes_appeared": stats.episodes_appeared,
                        "total_episodes": total_episodes,
                        "presence_ratio": presence_score,
                        "consistency_score": consistency,
                        "avg_speaking_time": stats.avg_speaking_time_per_episode,
                        "total_speaking_time": stats.total_speaking_time,
                    },
                )

        return best_candidate

    def analyze_episode(self, episode_file: Path) -> dict[str, SpeakerStats]:
        """
        Analyze speaker distribution in a single episode.

        Args:
            episode_file: Path to transcript JSON file.

        Returns:
            Dict mapping speaker_id -> SpeakerStats.
        """
        raw_stats = self._analyze_episode(episode_file)

        return {
            speaker_id: SpeakerStats(
                total_speaking_time=stats["total_time"],
                total_segments=stats["segment_count"],
                episodes_appeared=1,
                episode_speaking_times=[stats["total_time"]],
            )
            for speaker_id, stats in raw_stats.items()
        }

    def _analyze_episode(self, episode_file: Path) -> dict[str, dict[str, Any]]:
        """Analyze speaker distribution in a single episode."""
        try:
            with open(episode_file, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load {episode_file}: {e}")
            return {}

        speaker_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"total_time": 0.0, "segment_count": 0, "first_segment": None}
        )

        for segment in data.get("segments", []):
            speaker = segment.get("speaker", "UNKNOWN")
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            duration = max(0, end - start)

            speaker_stats[speaker]["total_time"] += duration
            speaker_stats[speaker]["segment_count"] += 1

            if speaker_stats[speaker]["first_segment"] is None:
                speaker_stats[speaker]["first_segment"] = start

        return dict(speaker_stats)

    def get_speaker_rank(
        self,
        transcript_data: dict[str, Any],
    ) -> dict[str, int]:
        """
        Get speaker ranks by speaking time for a single transcript.

        Args:
            transcript_data: Transcript JSON data.

        Returns:
            Dict mapping speaker_id -> rank (0 = most speaking time).
        """
        stats = self._analyze_episode_data(transcript_data)

        sorted_speakers = sorted(
            stats.items(),
            key=lambda x: x[1]["total_time"],
            reverse=True,
        )

        return {speaker: rank for rank, (speaker, _) in enumerate(sorted_speakers)}

    def _analyze_episode_data(
        self,
        transcript_data: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """Analyze speaker distribution from transcript data dict."""
        speaker_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"total_time": 0.0, "segment_count": 0, "first_segment": None}
        )

        for segment in transcript_data.get("segments", []):
            speaker = segment.get("speaker", "UNKNOWN")
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            duration = max(0, end - start)

            speaker_stats[speaker]["total_time"] += duration
            speaker_stats[speaker]["segment_count"] += 1

            if speaker_stats[speaker]["first_segment"] is None:
                speaker_stats[speaker]["first_segment"] = start

        return dict(speaker_stats)

    def find_hosts_for_all_podcasts(self) -> dict[str, HostCandidate]:
        """
        Analyze all podcasts and identify hosts.

        Returns:
            Dict mapping podcast_name -> HostCandidate.
        """
        results = {}

        for podcast_dir in self.transcripts_dir.iterdir():
            if podcast_dir.is_dir():
                candidate = self.analyze_podcast(podcast_dir.name)
                if candidate:
                    results[podcast_dir.name] = candidate

        return results

    def get_host_rank_for_podcast(self, podcast_name: str) -> int | None:
        """
        Get the expected host rank for a podcast.

        Args:
            podcast_name: Podcast folder name.

        Returns:
            Expected rank (0 = most speaking time) or None.
        """
        candidate = self.analyze_podcast(podcast_name)
        if not candidate:
            return None

        # extract rank from pattern.
        if "rank_" in candidate.speaker_pattern:
            try:
                return int(candidate.speaker_pattern.split("_")[1])
            except (IndexError, ValueError):
                pass

        return None


def create_cross_episode_analyzer(transcripts_dir: Path) -> CrossEpisodeAnalyzer:
    """Factory function to create cross-episode analyzer.

    Args:
        transcripts_dir: Directory containing transcripts.

    Returns:
        Configured CrossEpisodeAnalyzer.
    """
    return CrossEpisodeAnalyzer(transcripts_dir)
