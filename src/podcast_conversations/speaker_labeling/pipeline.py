"""Main speaker labeling pipeline with parallelization.

Orchestrates all 5 stages:
1. Metadata extraction
2. Transcript NER
3. Cross-episode host analysis
4. Role classification
5. Voice-name mapping

Supports parallel processing, dynamic batch allocation, and resource optimization.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator
import multiprocessing

from podcast_conversations.speaker_labeling.types import (
    PipelineConfig,
    EpisodeResult,
    PodcastResult,
    PipelineSummary,
    ExtractedName,
    RoleClassification,
    SpeakerMapping,
    SpeakerRole,
)
from podcast_conversations.speaker_labeling.metadata_extractor import MetadataExtractor
from podcast_conversations.speaker_labeling.transcript_ner import TranscriptNERExtractor
from podcast_conversations.speaker_labeling.cross_episode_analyzer import CrossEpisodeAnalyzer
from podcast_conversations.speaker_labeling.role_classifier import (
    HeuristicRoleClassifier,
    LLMRoleClassifier,
    LocalLLMRoleClassifier,
    create_role_classifier,
)
from podcast_conversations.speaker_labeling.voice_name_mapper import VoiceNameMapper, TranscriptLabeler

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Dynamic batch configuration based on available resources."""

    cpu_count: int = field(default_factory=lambda: multiprocessing.cpu_count())
    max_workers: int = field(default_factory=lambda: min(8, multiprocessing.cpu_count()))
    io_workers: int = field(default_factory=lambda: min(16, multiprocessing.cpu_count() * 2))
    batch_size: int = 10
    memory_limit_mb: int = 4096

    def __post_init__(self):
        """Adjust batch size based on available memory."""
        try:
            import psutil
            available_mb = psutil.virtual_memory().available / (1024 * 1024)
            # allocate ~100MB per batch item.
            self.batch_size = max(5, min(50, int(available_mb / 100)))
        except ImportError:
            pass


class SpeakerLabelingPipeline:
    """Main pipeline for speaker labeling."""

    def __init__(
        self,
        config: PipelineConfig,
        batch_config: BatchConfig | None = None,
    ):
        """
        Initialize pipeline.

        Args:
            config: Pipeline configuration.
            batch_config: Batch processing configuration.
        """
        self.config = config
        self.batch_config = batch_config or BatchConfig()

        # initialize components.
        self.metadata_extractor = MetadataExtractor(
            rss_metadata_dir=config.rss_metadata_dir,
        )
        self.ner_extractor = TranscriptNERExtractor()
        self.cross_episode_analyzer = CrossEpisodeAnalyzer(
            transcripts_dir=config.input_dir,
        )

        # initialize role classifier based on config.
        self.role_classifier = create_role_classifier(
            use_llm=config.use_llm,
            model=config.llm_model,
            use_local=config.use_local_llm,
            local_model=config.local_model,
            load_in_4bit=config.load_in_4bit,
            load_in_8bit=config.load_in_8bit,
        )

        self.voice_mapper = VoiceNameMapper()
        self.labeler = TranscriptLabeler()

        # ensure output directory exists.
        config.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> PipelineSummary:
        """
        Run the complete pipeline.

        Returns:
            Pipeline summary with statistics.
        """
        logger.info("Starting speaker labeling pipeline")
        logger.info(f"Input: {self.config.input_dir}")
        logger.info(f"Output: {self.config.output_dir}")

        # discover podcasts.
        podcasts = self._discover_podcasts()
        logger.info(f"Found {len(podcasts)} podcasts")

        # stage 3: run cross-episode analysis first (needs all episodes).
        host_ranks = self._run_cross_episode_analysis(podcasts)

        # process podcasts with parallelization.
        podcast_results = []
        with ThreadPoolExecutor(max_workers=self.batch_config.max_workers) as executor:
            futures = {
                executor.submit(self._process_podcast, podcast, host_ranks.get(podcast)): podcast
                for podcast in podcasts
            }

            for future in as_completed(futures):
                podcast = futures[future]
                try:
                    result = future.result()
                    podcast_results.append(result)
                    logger.info(f"Completed: {podcast} ({result.episodes_processed} episodes)")
                except Exception as e:
                    logger.error(f"Failed to process {podcast}: {e}")
                    podcast_results.append(PodcastResult(
                        podcast_name=podcast,
                        episodes_processed=0,
                        episodes_failed=1,
                        episode_results=[],
                    ))

        # generate summary.
        summary = self._generate_summary(podcast_results)

        logger.info(f"Pipeline complete: {summary.total_episodes_processed} episodes processed")
        return summary

    def _discover_podcasts(self) -> list[str]:
        """Discover podcast directories in input folder."""
        podcasts = []
        for item in self.config.input_dir.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                # check for transcript files.
                if list(item.glob("*.json")):
                    podcasts.append(item.name)
        return sorted(podcasts)

    def _run_cross_episode_analysis(self, podcasts: list[str]) -> dict[str, int]:
        """Run cross-episode analysis to identify host patterns."""
        logger.info("Running cross-episode host analysis...")
        host_ranks = {}

        for podcast in podcasts:
            candidate = self.cross_episode_analyzer.analyze_podcast(podcast)
            if candidate and candidate.confidence > 0.5:
                # extract rank from pattern.
                if "rank_" in candidate.speaker_pattern:
                    try:
                        rank = int(candidate.speaker_pattern.split("_")[1])
                        host_ranks[podcast] = rank
                        logger.debug(f"{podcast}: host rank = {rank} (confidence={candidate.confidence:.2f})")
                    except (IndexError, ValueError):
                        pass

        logger.info(f"Identified host patterns for {len(host_ranks)} podcasts")
        return host_ranks

    def _process_podcast(self, podcast_name: str, host_rank: int | None) -> PodcastResult:
        """Process all episodes of a single podcast."""
        podcast_dir = self.config.input_dir / podcast_name
        output_dir = self.config.output_dir / podcast_name
        output_dir.mkdir(parents=True, exist_ok=True)

        episode_files = list(podcast_dir.glob("*.json"))
        episode_results = []
        episodes_failed = 0

        # get metadata-based host info.
        metadata_names = self.metadata_extractor.extract_from_podcast_name(podcast_name)
        known_host = None
        if metadata_names:
            known_host = metadata_names[0].name

        # process episodes in batches.
        for batch in self._batch_iterator(episode_files, self.batch_config.batch_size):
            batch_results = self._process_episode_batch(
                batch, podcast_name, output_dir, host_rank, known_host
            )
            episode_results.extend(batch_results)
            episodes_failed += sum(1 for r in batch_results if not r.success)

        return PodcastResult(
            podcast_name=podcast_name,
            episodes_processed=len(episode_results),
            episodes_failed=episodes_failed,
            episode_results=episode_results,
        )

    def _process_episode_batch(
        self,
        episode_files: list[Path],
        podcast_name: str,
        output_dir: Path,
        host_rank: int | None,
        known_host: str | None,
    ) -> list[EpisodeResult]:
        """Process a batch of episodes."""
        results = []

        for episode_file in episode_files:
            try:
                result = self._process_episode(
                    episode_file, podcast_name, output_dir, host_rank, known_host
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {episode_file}: {e}")
                results.append(EpisodeResult(
                    episode_file=str(episode_file),
                    success=False,
                    error=str(e),
                    speakers_labeled=0,
                    speakers_total=0,
                    mappings=[],
                ))

        return results

    def _process_episode(
        self,
        episode_file: Path,
        podcast_name: str,
        output_dir: Path,
        host_rank: int | None,
        known_host: str | None,
    ) -> EpisodeResult:
        """Process a single episode through all stages."""
        # load transcript.
        with open(episode_file, encoding="utf-8") as f:
            transcript_data = json.load(f)

        segments = transcript_data.get("segments", [])
        unique_speakers = set(seg.get("speaker") for seg in segments if seg.get("speaker"))

        # stage 1: extract metadata names.
        episode_title = transcript_data.get("title", episode_file.stem)
        metadata_names = self.metadata_extractor.extract_from_filename(episode_file.name)
        metadata_names.extend(self.metadata_extractor.extract_from_podcast_name(podcast_name))

        # load RSS metadata if available.
        rss_names = []
        if self.config.rss_metadata_dir:
            rss_result = self.metadata_extractor.extract_from_rss(podcast_name, episode_file.name)
            rss_names = rss_result[0]  # tuple returns (names, host, guests).

        all_metadata_names = metadata_names + rss_names

        # stage 2: NER extraction.
        ner_names = self.ner_extractor.extract_from_transcript(transcript_data)

        # combine all names.
        all_names = list({n.name: n for n in all_metadata_names + ner_names}.values())

        # stage 4: role classification.
        transcript_intro = " ".join(
            seg.get("text", "") for seg in segments[:30]
        )

        if isinstance(self.role_classifier, HeuristicRoleClassifier):
            role_classifications = self.role_classifier.classify_roles(
                podcast_name=podcast_name,
                episode_title=episode_title,
                extracted_names=all_names,
                transcript_intro=transcript_intro,
                known_host=known_host,
            )
        else:
            role_classifications = self.role_classifier.classify_roles(
                podcast_name=podcast_name,
                episode_title=episode_title,
                extracted_names=[n.name for n in all_names],
                transcript_intro=transcript_intro,
            )

        # stage 5: voice-name mapping.
        mappings = self.voice_mapper.map_names_to_speakers(
            transcript_data=transcript_data,
            role_classifications=role_classifications,
            host_rank=host_rank,
        )

        # apply labels.
        labeled_transcript = self.labeler.apply_labels(transcript_data, mappings)

        # save output.
        output_file = output_dir / episode_file.name
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(labeled_transcript, f, indent=2, ensure_ascii=False)

        return EpisodeResult(
            episode_file=str(episode_file),
            success=True,
            speakers_labeled=len(mappings),
            speakers_total=len(unique_speakers),
            mappings=mappings,
        )

    def _batch_iterator(self, items: list, batch_size: int) -> Iterator[list]:
        """Yield batches of items."""
        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]

    def _generate_summary(self, podcast_results: list[PodcastResult]) -> PipelineSummary:
        """Generate pipeline summary statistics."""
        total_episodes = sum(r.episodes_processed for r in podcast_results)
        failed_episodes = sum(r.episodes_failed for r in podcast_results)
        total_speakers_labeled = sum(
            sum(e.speakers_labeled for e in r.episode_results)
            for r in podcast_results
        )
        total_speakers = sum(
            sum(e.speakers_total for e in r.episode_results)
            for r in podcast_results
        )

        # calculate confidence distribution.
        all_confidences = []
        method_counts: dict[str, int] = {}
        role_counts: dict[str, int] = {}

        for podcast_result in podcast_results:
            for ep_result in podcast_result.episode_results:
                for mapping in ep_result.mappings:
                    all_confidences.append(mapping.confidence)
                    method = mapping.assignment_method.value
                    method_counts[method] = method_counts.get(method, 0) + 1
                    role = mapping.role.value
                    role_counts[role] = role_counts.get(role, 0) + 1

        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0

        return PipelineSummary(
            total_podcasts=len(podcast_results),
            total_episodes_processed=total_episodes,
            total_episodes_failed=failed_episodes,
            total_speakers_labeled=total_speakers_labeled,
            total_speakers_found=total_speakers,
            average_confidence=avg_confidence,
            method_distribution=method_counts,
            role_distribution=role_counts,
            podcast_results=podcast_results,
        )


def create_pipeline(
    input_dir: Path | str,
    output_dir: Path | str,
    rss_dir: Path | str | None = None,
    use_llm: bool = False,
    llm_model: str = "gpt-4o-mini",
    use_local: bool = False,
    local_model: str | None = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
) -> SpeakerLabelingPipeline:
    """
    Factory function to create speaker labeling pipeline.

    Args:
        input_dir: Input transcripts directory.
        output_dir: Output directory for labeled transcripts.
        rss_dir: RSS metadata directory.
        use_llm: Whether to use OpenAI API for role classification.
        llm_model: OpenAI model to use.
        use_local: Whether to use local CUDA model.
        local_model: HuggingFace model for local inference.
        load_in_4bit: Use 4-bit quantization.
        load_in_8bit: Use 8-bit quantization.

    Returns:
        Configured pipeline.
    """
    config = PipelineConfig(
        input_dir=Path(input_dir),
        output_dir=Path(output_dir),
        rss_metadata_dir=Path(rss_dir) if rss_dir else None,
        use_llm=use_llm,
        llm_model=llm_model,
        use_local_llm=use_local,
        local_model=local_model,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )

    return SpeakerLabelingPipeline(config)
