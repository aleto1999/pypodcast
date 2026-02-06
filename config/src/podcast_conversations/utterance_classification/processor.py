"""transcript processor for utterance classification."""

import json
import logging
from pathlib import Path

from podcast_conversations.utterance_classification.classifier import UtteranceClassifier

logger = logging.getLogger(__name__)


def check_classifications_exist(output_path: Path, expected_models: set[str] | None = None) -> bool:
    """
    check if classification file exists and has valid classifications.

    Args:
        output_path: path to the output file to check.
        expected_models: set of model names that should have classified each utterance.
                        if None, only checks that classifications field exists.

    Returns:
        True if file exists and contains complete classifications for all segments.
    """
    if not output_path.exists():
        return False

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        segments = data.get("segments", [])

        if not segments:
            return False

        # check if all segments have classifications.
        for segment in segments:
            if "classifications" not in segment:
                return False
            # verify classifications is a list (can be empty for empty text segments).
            if not isinstance(segment["classifications"], list):
                return False
            
            # if expected_models is provided, verify all models have classified this segment.
            if expected_models:
                text = segment.get("text", "").strip()
                # skip validation for empty text segments.
                if not text:
                    continue
                
                # extract model names from classifications.
                classified_models = {
                    c.get("model_name") for c in segment["classifications"] 
                    if isinstance(c, dict) and c.get("model_name")
                }
                
                # check if all expected models are present.
                missing_models = expected_models - classified_models
                if missing_models:
                    return False

        return True

    except Exception:
        return False


class TranscriptProcessor:
    """processes transcript files and applies utterance classification."""

    def __init__(self, classifier: UtteranceClassifier):
        """
        initialize transcript processor.

        Args:
            classifier: utterance classifier to use.
        """
        self.classifier = classifier

    def process_file(
        self,
        input_path: Path,
        output_path: Path,
        skip_existing: bool = True,
    ) -> dict:
        """
        process a single transcript file and add utterance classifications.

        Args:
            input_path: path to input JSON transcript file.
            output_path: path to output JSON file.
            skip_existing: if True, skip files that already have complete classifications.

        Returns:
            dict with processing statistics (includes 'skipped' key).

        Raises:
            FileNotFoundError: if input file doesn't exist.
            ValueError: if JSON format is invalid.
        """
        if not input_path.exists():
            raise FileNotFoundError(f"input file not found: {input_path}")

        # get expected model names.
        expected_model_names = set(self.classifier.get_loaded_models())
        
        # check if we should skip this file (all models have classified all utterances).
        if skip_existing and check_classifications_exist(output_path, expected_model_names):
            return {"utterances_classified": 0, "models_applied": 0, "skipped": True}

        # load existing output if it exists (to preserve partial classifications).
        transcript_data = None
        if output_path.exists():
            try:
                with open(output_path, encoding="utf-8") as f:
                    transcript_data = json.load(f)
                logger.info(f"found existing partial classifications in {output_path.name}")
            except (json.JSONDecodeError, IOError):
                logger.warning(f"failed to load existing output {output_path.name}, starting fresh")
        
        # if no existing output, load from input.
        if transcript_data is None:
            try:
                with open(input_path, encoding="utf-8") as f:
                    transcript_data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"invalid JSON in {input_path}: {e}") from e

        # get segments/utterances.
        segments = transcript_data.get("segments", [])
        if not segments:
            logger.warning(f"no segments found in {input_path.name}")
            # still write output file even if no segments.
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(transcript_data, f, indent=2, ensure_ascii=False)
            return {"utterances_classified": 0, "models_applied": 0, "skipped": False}

        # classify each segment.
        utterances_classified = 0
        for segment in segments:
            text = segment.get("text", "").strip()

            if not text:
                # add empty classifications for segments without text.
                if "classifications" not in segment:
                    segment["classifications"] = []
                continue

            # check existing classifications.
            existing_classifications = segment.get("classifications", [])
            existing_models = {
                c.get("model_name") for c in existing_classifications 
                if isinstance(c, dict) and c.get("model_name")
            }
            
            # determine which models need to classify this utterance.
            missing_models = expected_model_names - existing_models
            
            if not missing_models:
                # all models have already classified this utterance.
                continue

            # classify utterance with all models (easier than partial).
            results = self.classifier.classify_utterance(text)

            # merge with existing classifications (keep existing, add new).
            # use dict to deduplicate by model_name.
            all_classifications = {c.get("model_name"): c for c in existing_classifications}
            for result in results:
                all_classifications[result.model_name] = {
                    "model_name": result.model_name,
                    "label": result.label,
                    "confidence": result.confidence,
                }
            
            # update segment with merged classifications.
            segment["classifications"] = list(all_classifications.values())
            utterances_classified += 1

        # write output file.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(transcript_data, f, indent=2, ensure_ascii=False)

        logger.info(
            f"processed {input_path.name}: {utterances_classified} utterances classified"
        )

        return {
            "utterances_classified": utterances_classified,
            "models_applied": len(self.classifier.get_loaded_models()),
            "skipped": False,
        }

    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
    ) -> dict:
        """
        process all transcript files in a directory, preserving structure.

        Args:
            input_dir: root directory containing transcript files.
            output_dir: root directory for output files.

        Returns:
            dict with overall processing statistics.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        if not input_dir.exists():
            raise FileNotFoundError(f"input directory not found: {input_dir}")

        # discover all JSON files.
        json_files = sorted(input_dir.rglob("*.json"))

        if not json_files:
            logger.warning(f"no JSON files found in {input_dir}")
            return {
                "files_processed": 0,
                "files_failed": 0,
                "total_utterances": 0,
            }

        logger.info(f"found {len(json_files)} JSON files to process")

        # process each file.
        stats = {
            "files_processed": 0,
            "files_failed": 0,
            "total_utterances": 0,
        }

        for input_path in json_files:
            # compute relative path from input_dir.
            relative_path = input_path.relative_to(input_dir)

            # create mirrored output path.
            output_path = output_dir / relative_path

            try:
                file_stats = self.process_file(input_path, output_path)
                stats["files_processed"] += 1
                stats["total_utterances"] += file_stats["utterances_classified"]

            except Exception as e:
                logger.error(f"failed to process {input_path.name}: {e}")
                stats["files_failed"] += 1

        return stats
