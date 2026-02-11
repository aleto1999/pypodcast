"""file discovery and path mapping for diarization pipeline."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FileMapping:
    """mapping of transcript, audio, and output paths."""

    transcript_path: Path
    audio_path: Path
    output_path: Path


class FileDiscovery:
    """discover and map transcript files to audio files and output paths."""

    def __init__(
        self,
        transcripts_dir: Path,
        audio_base_dir: Path,
        output_dir: Path,
    ):
        """
        initialize file discovery.

        Args:
            transcripts_dir: root directory containing JSON transcript files.
            audio_base_dir: root directory containing audio files.
            output_dir: output directory for RTTM files.
        """
        self.transcripts_dir = Path(transcripts_dir)
        self.audio_base_dir = Path(audio_base_dir)
        self.output_dir = Path(output_dir)

        if not self.transcripts_dir.exists():
            raise FileNotFoundError(
                f"transcripts directory not found: {self.transcripts_dir}"
            )
        if not self.audio_base_dir.exists():
            raise FileNotFoundError(
                f"audio base directory not found: {self.audio_base_dir}"
            )

    def discover_files(self) -> list[FileMapping]:
        """
        discover all transcript-audio-output mappings.

        Returns:
            list of FileMapping objects.

        Raises:
            FileNotFoundError: if an audio file referenced in a transcript is not found.
        """
        mappings = []

        for transcript_path in self.transcripts_dir.rglob("*.json"):
            try:
                with open(transcript_path, encoding="utf-8") as f:
                    transcript_data = json.load(f)

                audio_path = self._resolve_audio_path(transcript_data, transcript_path)
                output_path = self._create_output_path(transcript_path)

                mappings.append(
                    FileMapping(
                        transcript_path=transcript_path,
                        audio_path=audio_path,
                        output_path=output_path,
                    )
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(
                    f"skipping invalid transcript file {transcript_path}: {e}"
                )
                continue
            except FileNotFoundError as e:
                logger.warning(f"skipping transcript {transcript_path}: {e}")
                continue

        logger.info(f"discovered {len(mappings)} transcript-audio pairs")
        return mappings

    def _get_relative_path(self, transcript_path: Path) -> Path:
        """
        get relative path from transcript path to transcripts directory.

        Args:
            transcript_path: path to the transcript file.

        Returns:
            relative path from transcripts_dir to transcript's parent directory.
        """
        return transcript_path.parent.relative_to(self.transcripts_dir)

    def _resolve_audio_path(self, transcript_json: dict, transcript_path: Path) -> Path:
        """
        resolve audio file path from transcript json.

        Args:
            transcript_json: parsed json transcript data.
            transcript_path: path to the transcript file.

        Returns:
            path to the audio file.

        Raises:
            KeyError: if audio_file field is missing from the json.
            FileNotFoundError: if audio file does not exist.

        Example:
            transcript_path: /transcripts/show_name/episode.json
            audio_file: "episode.mp3"
            audio_base_dir: /downloads/
            result: /downloads/show_name/episode.mp3
        """
        audio_filename = transcript_json["audio_file"]

        rel_path = self._get_relative_path(transcript_path)

        audio_path = self.audio_base_dir / rel_path / audio_filename

        if not audio_path.exists():
            raise FileNotFoundError(f"audio file not found: {audio_path}")

        return audio_path

    def _create_output_path(self, transcript_path: Path) -> Path:
        """
        create mirrored output path for rttm file.

        Args:
            transcript_path: path to the transcript file.

        Returns:
            path where the rttm output should be written.

        Example:
            transcript_path: /transcripts/show_name/episode.json
            output_dir: /diarizations/
            result: /diarizations/show_name/episode.rttm
        """
        rel_path = self._get_relative_path(transcript_path)

        output_path = self.output_dir / rel_path / (transcript_path.stem + ".rttm")

        return output_path
