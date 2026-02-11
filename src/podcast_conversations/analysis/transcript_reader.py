"""transcript file reader for analysis."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    """single segment of transcript text."""

    text: str
    start_time: float | None = None
    end_time: float | None = None
    speaker: str | None = None
    segment_id: int | None = None


class TranscriptReader:
    """read and parse transcript JSON files."""

    def __init__(self, transcript_path: Path):
        """
        initialize transcript reader.

        Args:
            transcript_path: path to JSON transcript file.
        """
        self.transcript_path = Path(transcript_path)
        if not self.transcript_path.exists():
            raise FileNotFoundError(f"transcript file not found: {transcript_path}")

        self.metadata: dict = {}
        self.segments: list[TranscriptSegment] = []

        self._load_transcript()

    def _load_transcript(self) -> None:
        """load and parse JSON transcript file."""
        try:
            with open(self.transcript_path, encoding="utf-8") as f:
                data = json.load(f)

            # extract metadata.
            self.metadata = {
                "title": data.get("title", ""),
                "show_name": data.get("show_name", ""),
                "audio_file": data.get("audio_file", ""),
                "transcript_file": data.get("transcript_file", ""),
            }

            # extract segments.
            if "segments" in data and isinstance(data["segments"], list):
                for idx, segment in enumerate(data["segments"]):
                    self.segments.append(
                        TranscriptSegment(
                            text=segment.get("text", ""),
                            start_time=segment.get("start"),
                            end_time=segment.get("end"),
                            speaker=segment.get("speaker"),
                            segment_id=idx,
                        )
                    )
            elif "text" in data:
                # single text field (no segments).
                self.segments.append(
                    TranscriptSegment(
                        text=data["text"],
                        segment_id=0,
                    )
                )
            else:
                logger.warning(f"no segments or text found in {self.transcript_path}")

            logger.info(
                f"loaded {len(self.segments)} segments from {self.transcript_path.name}"
            )

        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"failed to parse JSON transcript {self.transcript_path}: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"failed to load transcript {self.transcript_path}: {e}"
            ) from e

    def get_full_text(self) -> str:
        """
        get complete transcript text.

        Returns:
            concatenated text from all segments.
        """
        return " ".join(segment.text for segment in self.segments if segment.text)

    def get_segments(self) -> list[TranscriptSegment]:
        """
        get list of transcript segments.

        Returns:
            list of transcript segments.
        """
        return self.segments

    def get_metadata(self) -> dict:
        """
        get transcript metadata.

        Returns:
            metadata dictionary.
        """
        return self.metadata
