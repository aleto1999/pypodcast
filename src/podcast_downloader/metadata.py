"""json-based metadata storage for tracking downloads."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from podcast_conversations.naming import sanitize_filename
from podcast_downloader.feed_parser import Episode, Podcast


class EpisodeMetadata(BaseModel):
    """stored metadata for a downloaded episode."""

    episode_id: str
    title: str
    audio_url: str
    file_path: str
    episode_date: datetime
    downloaded_at: datetime = Field(default_factory=datetime.now)
    file_size: int = 0
    duration: int | None = None


class PodcastMetadata(BaseModel):
    """stored metadata for a podcast."""

    title: str
    feed_url: str
    author: str | None = None
    last_updated: datetime = Field(default_factory=datetime.now)
    episodes: dict[str, EpisodeMetadata] = Field(default_factory=dict)


class MetadataStore:
    """manages metadata storage for downloaded podcasts."""

    def __init__(self, metadata_dir: Path):
        self.metadata_dir = metadata_dir
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def _podcast_path(self, podcast_title: str) -> Path:
        """get the metadata file path for a podcast."""
        safe_name = sanitize_filename(podcast_title, max_length=50, extension=".json")
        return self.metadata_dir / safe_name

    def load_podcast(self, podcast_title: str) -> PodcastMetadata | None:
        """load metadata for a podcast."""
        path = self._podcast_path(podcast_title)
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text())
            return PodcastMetadata.model_validate(data)
        except (json.JSONDecodeError, ValueError):
            return None

    def save_podcast(self, metadata: PodcastMetadata) -> None:
        """save metadata for a podcast."""
        path = self._podcast_path(metadata.title)
        path.write_text(metadata.model_dump_json(indent=2))

    def is_downloaded(self, podcast_title: str, episode_id: str) -> bool:
        """check if an episode has already been downloaded."""
        metadata = self.load_podcast(podcast_title)
        if metadata is None:
            return False
        return episode_id in metadata.episodes

    def mark_downloaded(
        self,
        podcast: Podcast,
        episode: Episode,
        file_path: Path,
        file_size: int = 0,
    ) -> None:
        """mark an episode as downloaded."""
        metadata = self.load_podcast(podcast.title)

        if metadata is None:
            metadata = PodcastMetadata(
                title=podcast.title,
                feed_url=podcast.feed_url,
                author=podcast.author,
            )

        metadata.episodes[episode.id] = EpisodeMetadata(
            episode_id=episode.id,
            title=episode.title,
            audio_url=episode.audio_url,
            file_path=str(file_path),
            file_size=file_size,
            duration=episode.duration,
            episode_date=episode.published
        )
        metadata.last_updated = datetime.now()

        self.save_podcast(metadata)

    def get_downloaded_episodes(self, podcast_title: str) -> list[EpisodeMetadata]:
        """get list of downloaded episodes for a podcast."""
        metadata = self.load_podcast(podcast_title)
        if metadata is None:
            return []
        return list(metadata.episodes.values())

    def list_podcasts(self) -> list[PodcastMetadata]:
        """list all podcasts with metadata."""
        podcasts = []
        for path in self.metadata_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                podcasts.append(PodcastMetadata.model_validate(data))
            except (json.JSONDecodeError, ValueError):
                continue
        return podcasts

    def get_stats(self) -> dict[str, Any]:
        """get overall download statistics."""
        podcasts = self.list_podcasts()
        total_episodes = sum(len(p.episodes) for p in podcasts)
        total_size = sum(
            ep.file_size for p in podcasts for ep in p.episodes.values()
        )

        return {
            "total_podcasts": len(podcasts),
            "total_episodes": total_episodes,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
        }
