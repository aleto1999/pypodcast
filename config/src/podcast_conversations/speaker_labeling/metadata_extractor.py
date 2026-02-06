"""Stage 1: Metadata extraction from RSS feeds, filenames, and folder names.

Extracts speaker name hints from multiple metadata sources:
- RSS feed metadata (highest confidence)
- Episode filenames
- Podcast folder names (for host identification)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from podcast_conversations.speaker_labeling.types import (
    EpisodeMetadata,
    PodcastMetadata,
    ExtractedName,
    ExtractionMethod,
)

logger = logging.getLogger(__name__)


# mapping of podcast folder names to known hosts.
KNOWN_HOSTS: dict[str, str | list[str]] = {
    "the_joe_rogan_experience": "Joe Rogan",
    "the_ezra_klein_show": "Ezra Klein",
    "the_ben_shapiro_show": "Ben Shapiro",
    "the_charlie_kirk_show": "Charlie Kirk",
    "lex_fridman_podcast": "Lex Fridman",
    "the_jordan_b_peterson_podcast": "Jordan Peterson",
    "the_megyn_kelly_show": "Megyn Kelly",
    "the_tucker_carlson_show": "Tucker Carlson",
    "candace": "Candace Owens",
    "pod_save_america": ["Jon Favreau", "Jon Lovett", "Dan Pfeiffer", "Tommy Vietor"],
    "the_daily": "Michael Barbaro",
    "a_bit_fruity_with_matt_bernstein": "Matt Bernstein",
    "next_question_with_katie_couric": "Katie Couric",
    "deadline_white_house": "Nicolle Wallace",
    "the_lincoln_project": "Rick Wilson",
    "the_meidastouch_podcast": "Ben Meiselas",
    "the_npr_politics_podcast": ["Tamara Keith", "Domenico Montanaro"],
    "bannon_s_war_room": "Steve Bannon",
}


class MetadataExtractor:
    """Extracts speaker names from metadata sources."""

    def __init__(
        self,
        rss_metadata_dir: Path | None = None,
        known_hosts: dict[str, str | list[str]] | None = None,
    ):
        """
        Initialize metadata extractor.

        Args:
            rss_metadata_dir: Directory containing RSS metadata JSON files.
            known_hosts: Additional mapping of folder names to known hosts.
        """
        self.rss_metadata_dir = rss_metadata_dir
        self.known_hosts = {**KNOWN_HOSTS, **(known_hosts or {})}
        self._rss_cache: dict[str, dict] = {}

    def load_rss_metadata(self, podcast_folder: str) -> dict | None:
        """Load RSS metadata for a podcast if available."""
        if podcast_folder in self._rss_cache:
            return self._rss_cache[podcast_folder]

        if not self.rss_metadata_dir:
            return None

        rss_file = self.rss_metadata_dir / podcast_folder / "rss_metadata.json"
        if not rss_file.exists():
            self._rss_cache[podcast_folder] = None
            return None

        try:
            with open(rss_file, encoding="utf-8") as f:
                data = json.load(f)
            self._rss_cache[podcast_folder] = data
            return data
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load RSS metadata for {podcast_folder}: {e}")
            self._rss_cache[podcast_folder] = None
            return None

    def extract_from_filename(self, filename: str) -> list[ExtractedName]:
        """
        Extract potential names from episode filename.

        Examples:
            "1_brian_redban.json" -> ["Brian Redban"]
            "interview_with_elon_musk_and_sam_altman.json" -> ["Elon Musk", "Sam Altman"]
        """
        names = []

        # remove extension and episode number prefix.
        stem = Path(filename).stem
        stem = re.sub(r"^\d+_?", "", stem)

        # skip patterns that aren't names.
        skip_words = {
            "episode", "part", "interview", "with", "and", "the", "a", "an",
            "special", "bonus", "live", "recap", "preview", "rerun",
        }

        # split on underscores and common separators.
        parts = re.split(r"[_\-\s]+", stem)

        current_name: list[str] = []
        for part in parts:
            if part.lower() in skip_words:
                if current_name:
                    name = " ".join(current_name).title()
                    if self._is_valid_name(name):
                        names.append(ExtractedName(
                            name=name,
                            extraction_method=ExtractionMethod.FILENAME,
                            confidence=0.6,
                        ))
                    current_name = []
            else:
                current_name.append(part)

        # handle remaining name.
        if current_name:
            name = " ".join(current_name).title()
            if self._is_valid_name(name):
                names.append(ExtractedName(
                    name=name,
                    extraction_method=ExtractionMethod.FILENAME,
                    confidence=0.6,
                ))

        return names

    def extract_from_podcast_name(self, podcast_folder: str) -> list[ExtractedName]:
        """Get known host(s) from podcast folder name."""
        hosts = self.known_hosts.get(podcast_folder)
        if not hosts:
            return []

        if isinstance(hosts, str):
            hosts = [hosts]

        return [
            ExtractedName(
                name=host,
                extraction_method=ExtractionMethod.FOLDER_NAME,
                confidence=0.95,
            )
            for host in hosts
        ]

    def extract_from_rss(
        self,
        podcast_folder: str,
        episode_filename: str,
    ) -> tuple[list[ExtractedName], str | None, list[str]]:
        """
        Extract names from RSS metadata.

        Returns:
            Tuple of (extracted_names, inferred_host, inferred_guests)
        """
        rss_data = self.load_rss_metadata(podcast_folder)
        if not rss_data:
            return [], None, []

        names: list[ExtractedName] = []
        inferred_host = rss_data.get("podcast", {}).get("inferred_host")
        inferred_guests: list[str] = []

        if inferred_host:
            names.append(ExtractedName(
                name=inferred_host,
                extraction_method=ExtractionMethod.RSS_METADATA,
                confidence=0.9,
            ))

        # find matching episode in RSS data.
        episode_stem = Path(episode_filename).stem.lower()
        for ep in rss_data.get("episodes", []):
            ep_title = ep.get("title", "").lower()
            # fuzzy match on title.
            if self._titles_match(ep_title, episode_stem):
                hints = ep.get("speaker_hints", {})
                for host in hints.get("hosts", []):
                    if host and host != inferred_host:
                        names.append(ExtractedName(
                            name=host,
                            extraction_method=ExtractionMethod.RSS_METADATA,
                            confidence=0.85,
                        ))
                for guest in hints.get("guests", []):
                    if guest:
                        names.append(ExtractedName(
                            name=guest,
                            extraction_method=ExtractionMethod.RSS_METADATA,
                            confidence=0.85,
                        ))
                        inferred_guests.append(guest)
                break

        return names, inferred_host, inferred_guests

    def process_episode(
        self,
        podcast_folder: str,
        episode_filename: str,
    ) -> tuple[EpisodeMetadata, list[ExtractedName]]:
        """
        Extract all available metadata for an episode.

        Returns:
            Tuple of (episode_metadata, extracted_names)
        """
        all_names: list[ExtractedName] = []

        # priority 1: RSS metadata (highest confidence).
        rss_names, rss_host, rss_guests = self.extract_from_rss(podcast_folder, episode_filename)
        all_names.extend(rss_names)

        # priority 2: known hosts from folder name.
        folder_names = self.extract_from_podcast_name(podcast_folder)
        for fn in folder_names:
            if not any(n.name.lower() == fn.name.lower() for n in all_names):
                all_names.append(fn)

        # priority 3: names from filename.
        filename_names = self.extract_from_filename(episode_filename)
        for fn in filename_names:
            if not any(n.name.lower() == fn.name.lower() for n in all_names):
                all_names.append(fn)

        # determine inferred host.
        inferred_host = rss_host
        if not inferred_host:
            hosts_from_folder = self.known_hosts.get(podcast_folder)
            if isinstance(hosts_from_folder, str):
                inferred_host = hosts_from_folder
            elif isinstance(hosts_from_folder, list) and hosts_from_folder:
                inferred_host = hosts_from_folder[0]

        # inferred guests from filename (exclude host).
        inferred_guests = list(rss_guests)
        for fn in filename_names:
            if fn.name.lower() != (inferred_host or "").lower():
                if fn.name not in inferred_guests:
                    inferred_guests.append(fn.name)

        # calculate confidence.
        confidence = 0.3
        if rss_names:
            confidence = 0.9
        elif inferred_host:
            confidence = 0.8
        elif filename_names:
            confidence = 0.5

        metadata = EpisodeMetadata(
            podcast_name=podcast_folder,
            episode_filename=episode_filename,
            inferred_host=inferred_host,
            inferred_guests=inferred_guests,
            confidence=confidence,
        )

        return metadata, all_names

    def process_podcast(self, podcast_folder: str) -> PodcastMetadata:
        """Extract podcast-level metadata."""
        rss_data = self.load_rss_metadata(podcast_folder)

        if rss_data:
            podcast_info = rss_data.get("podcast", {})
            return PodcastMetadata(
                title=podcast_info.get("title", podcast_folder),
                feed_url=podcast_info.get("feed_url"),
                author=podcast_info.get("author"),
                description=podcast_info.get("description"),
                inferred_host=podcast_info.get("inferred_host"),
                frequent_guests=podcast_info.get("frequent_guests", []),
                episode_count=podcast_info.get("episode_count", 0),
            )

        # fallback to known hosts.
        hosts = self.known_hosts.get(podcast_folder)
        inferred_host = hosts if isinstance(hosts, str) else (hosts[0] if hosts else None)

        return PodcastMetadata(
            title=podcast_folder.replace("_", " ").title(),
            inferred_host=inferred_host,
        )

    def _is_valid_name(self, name: str) -> bool:
        """Check if a string looks like a valid person name."""
        words = name.split()
        if len(words) < 2:
            return False
        # check that words are capitalized and reasonable length.
        for word in words:
            if len(word) < 2 or not word[0].isupper():
                return False
        return True

    def _titles_match(self, title1: str, title2: str) -> bool:
        """Check if two titles are likely referring to the same episode."""
        # normalize.
        t1 = re.sub(r"[^\w\s]", "", title1.lower())
        t2 = re.sub(r"[^\w\s]", "", title2.lower())

        # check for substantial overlap.
        words1 = set(t1.split())
        words2 = set(t2.split())

        if not words1 or not words2:
            return False

        overlap = len(words1 & words2)
        min_len = min(len(words1), len(words2))

        return overlap / min_len > 0.5 if min_len > 0 else False
