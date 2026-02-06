"""rttm format writer for speaker diarization output."""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class RTTMWriter:
    """write speaker diarization annotations to rttm format."""

    def write(self, annotation: Any, output_path: Path, audio_filename: str) -> None:
        """
        write annotation to rttm file.

        Args:
            annotation: pyannote.core.Annotation or pyannote DiarizeOutput object.
            output_path: path where the RTTM file should be written.
            audio_filename: name of the audio file (used as file_id in RTTM).

        RTTM format specification:
            SPEAKER <file> 1 <start> <duration> <NA> <NA> <speaker> <NA> <NA>
            - file: audio filename without extension
            - channel: always 1
            - start: start time in seconds (3 decimal places)
            - duration: duration in seconds (3 decimal places)
            - speaker: speaker label
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        temp_path = output_path.with_suffix(".rttm.tmp")

        file_id = Path(audio_filename).stem

        # handle both pyannote.core.Annotation and pyannote 3.1+ DiarizeOutput objects.
        annotation_data = annotation

        # if it's a DiarizeOutput object, extract the actual annotation.
        if not hasattr(annotation, "itertracks"):
            # pyannote 3.1+ returns DiarizeOutput with speaker_diarization attribute.
            if hasattr(annotation, "speaker_diarization"):
                annotation_data = annotation.speaker_diarization
                logger.debug("extracted annotation from .speaker_diarization attribute")
            elif hasattr(annotation, "diarization"):
                # fallback for other versions.
                annotation_data = annotation.diarization
                logger.debug("extracted annotation from .diarization attribute")
            elif hasattr(annotation, "segments"):
                # some versions use .segments.
                annotation_data = annotation.segments
                logger.debug("extracted annotation from .segments attribute")
            elif isinstance(annotation, dict):
                # might be a dict directly.
                annotation_data = annotation
                logger.debug("annotation is a dict")
            else:
                raise RuntimeError(
                    f"unsupported annotation type: {type(annotation)}. "
                    f"available attributes: {[a for a in dir(annotation) if not a.startswith('_')]}"
                )

        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                # handle different iteration methods.
                if hasattr(annotation_data, "itertracks"):
                    # standard pyannote.core.Annotation interface.
                    for segment, _, speaker in annotation_data.itertracks(
                        yield_label=True
                    ):
                        line = self._format_rttm_line(
                            file_id=file_id,
                            start=segment.start,
                            duration=segment.end - segment.start,
                            speaker=speaker,
                        )
                        f.write(line)
                else:
                    # DiarizeOutput.segments interface (dict-like with speaker labels as keys).
                    for speaker, timeline in annotation_data.items():
                        for segment in timeline:
                            line = self._format_rttm_line(
                                file_id=file_id,
                                start=segment.start,
                                duration=segment.end - segment.start,
                                speaker=speaker,
                            )
                            f.write(line)

            temp_path.rename(output_path)
            logger.info(f"wrote RTTM file: {output_path}")

        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise RuntimeError(f"failed to write RTTM file {output_path}: {e}") from e

    def _format_rttm_line(
        self,
        file_id: str,
        start: float,
        duration: float,
        speaker: str,
    ) -> str:
        """
        format single RTTM line.

        Args:
            file_id: audio file identifier.
            start: start time in seconds.
            duration: duration in seconds.
            speaker: speaker label.

        Returns:
            formatted RTTM line with newline.

        Format:
            SPEAKER <file> 1 <start> <duration> <NA> <NA> <speaker> <NA> <NA>
        """
        return f"SPEAKER {file_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n"
