"""Parallel processing utilities for LLM annotation pipeline.

Implements:
- Concurrent file I/O with GPU compute overlap
- Cross-file segment batching for optimal GPU utilization
- Prefetching pipeline to minimize GPU idle time
- Multi-GPU support via tensor parallelism
- Throughput tracking and adaptive batch sizing
- Segment filtering based on classification labels
"""

import json
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


# label mappings for determining negative labels.
# maps model labels to whether they are negative (True) or positive (False).
NEGATIVE_LABEL_MAPPINGS = {
    "hate_speech_detection": {
        "HATE": True,
        "NOT-HATE": False,
    },
    "fine_grained_hate_speech_detection": {
        "LABEL_0": False,
        "LABEL_1": True,
        "LABEL_2": True,
        "LABEL_3": True,
    },
    "hostile_content": {
        "OFFENSIVE": True,
        "NOT-OFFENSIVE": False,
    },
}

# threshold for hate_against_minorities model (confidence-based).
HATE_MINORITIES_THRESHOLD = 0.5


def is_ad_segment(segment: dict) -> bool:
    """Check if a segment is classified as an advertisement.

    Args:
        segment: Segment dict with classifications.

    Returns:
        True if the segment is classified as an ad.
    """
    classifications = segment.get("classifications", [])
    for clf in classifications:
        if clf.get("model_name") == "ad_content_detection":
            if clf.get("label") == "LABEL_1":
                return True
    return False


def count_negative_labels(segment: dict) -> int:
    """Count the number of negative labels for a segment.

    Args:
        segment: Segment dict with classifications.

    Returns:
        Number of negative labels from classification models.
    """
    classifications = segment.get("classifications", [])
    negative_count = 0

    for clf in classifications:
        model_name = clf.get("model_name", "")
        label = clf.get("label", "")
        confidence = clf.get("confidence", 0.0)

        # skip ad detection model - used only for filtering.
        if model_name == "ad_content_detection":
            continue

        # special handling for hate_against_minorities (confidence-based).
        if model_name == "hate_against_minorities":
            if label == "toxic" and confidence >= HATE_MINORITIES_THRESHOLD:
                negative_count += 1
            continue

        # standard label mapping for other models.
        if model_name in NEGATIVE_LABEL_MAPPINGS:
            mapping = NEGATIVE_LABEL_MAPPINGS[model_name]
            if label in mapping and mapping[label]:
                negative_count += 1

    return negative_count


def should_annotate_segment(segment: dict, min_negative_labels: int = 2) -> bool:
    """Determine if a segment should be annotated by the LLM.

    A segment should be annotated if:
    1. It is not classified as an advertisement
    2. It has at least min_negative_labels negative labels from classifiers

    Args:
        segment: Segment dict with classifications.
        min_negative_labels: Minimum number of negative labels required.

    Returns:
        True if the segment should be annotated.
    """
    # skip ads.
    if is_ad_segment(segment):
        return False

    # check if segment has classifications.
    if not segment.get("classifications"):
        # no classifications = no filtering possible, skip annotation.
        return False

    # count negative labels.
    negative_count = count_negative_labels(segment)

    return negative_count >= min_negative_labels


def get_skipped_annotation() -> dict[str, Any]:
    """Return annotation dict for segments that were skipped.

    Returns:
        Annotation dict indicating the segment was skipped.
    """
    return {
        "has_hate_speech": None,
        "has_advertisement": None,
        "target_group": None,
        "hate_speech_type": None,
        "main_topic": None,
        "skipped": True,
        "skip_reason": "segment did not meet annotation criteria (< 2 negative labels or is ad)",
    }


@dataclass
class FileTask:
    """Represents a file to be annotated."""

    input_path: Path
    output_path: Path
    segments: list[dict] | None = None
    transcript_data: dict | None = None
    error: str | None = None


@dataclass
class SegmentBatch:
    """A batch of segments from potentially multiple files."""

    segments: list[dict]
    texts: list[str]
    file_indices: list[int]  # maps each segment to its file index
    segment_indices: list[int]  # maps each segment to its index within file


@dataclass
class ThroughputMetrics:
    """Tracks throughput statistics."""

    total_segments: int = 0
    total_files: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_time_seconds: float = 0.0
    batch_times: list[float] = field(default_factory=list)

    @property
    def segments_per_second(self) -> float:
        if self.total_time_seconds == 0:
            return 0.0
        return self.total_segments / self.total_time_seconds

    @property
    def files_per_minute(self) -> float:
        if self.total_time_seconds == 0:
            return 0.0
        return (self.total_files / self.total_time_seconds) * 60

    @property
    def avg_batch_time_ms(self) -> float:
        if not self.batch_times:
            return 0.0
        return (sum(self.batch_times) / len(self.batch_times)) * 1000


def load_file_async(task: FileTask) -> FileTask:
    """Load a transcript file (for use in thread pool)."""
    try:
        with open(task.input_path, encoding="utf-8") as f:
            task.transcript_data = json.load(f)
        task.segments = task.transcript_data.get("segments", [])
    except Exception as e:
        task.error = str(e)
        logger.error(f"Failed to load {task.input_path}: {e}")
    return task


def save_file_async(task: FileTask) -> FileTask:
    """Save an annotated transcript file (for use in thread pool)."""
    try:
        task.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(task.output_path, "w", encoding="utf-8") as f:
            json.dump(task.transcript_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        task.error = str(e)
        logger.error(f"Failed to save {task.output_path}: {e}")
    return task


class PrefetchingFileLoader:
    """
    Loads files ahead of processing to minimize I/O wait time.

    Uses a thread pool to load files in the background while the GPU
    processes the current batch.
    """

    def __init__(
        self,
        file_tasks: list[FileTask],
        prefetch_count: int = 4,
        num_workers: int = 2,
    ):
        """
        Initialize prefetching loader.

        Args:
            file_tasks: List of file tasks to process
            prefetch_count: Number of files to prefetch ahead
            num_workers: Number of I/O threads
        """
        self.file_tasks = file_tasks
        self.prefetch_count = prefetch_count
        self.num_workers = num_workers
        self._queue: queue.Queue[FileTask] = queue.Queue(maxsize=prefetch_count)
        self._executor: ThreadPoolExecutor | None = None
        self._stop_event = threading.Event()
        self._loader_thread: threading.Thread | None = None
        self._current_index = 0

    def _loader_worker(self) -> None:
        """Background thread that prefetches files."""
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for task in self.file_tasks:
                if self._stop_event.is_set():
                    break

                # submit load task
                future = executor.submit(load_file_async, task)
                loaded_task = future.result()

                # block until queue has space
                while not self._stop_event.is_set():
                    try:
                        self._queue.put(loaded_task, timeout=0.1)
                        break
                    except queue.Full:
                        continue

    def start(self) -> None:
        """Start the prefetching background thread."""
        self._stop_event.clear()
        self._loader_thread = threading.Thread(target=self._loader_worker, daemon=True)
        self._loader_thread.start()

    def stop(self) -> None:
        """Stop the prefetching background thread."""
        self._stop_event.set()
        if self._loader_thread:
            self._loader_thread.join(timeout=5.0)

    def __iter__(self) -> Iterator[FileTask]:
        """Iterate over loaded file tasks."""
        for _ in range(len(self.file_tasks)):
            try:
                task = self._queue.get(timeout=60.0)
                yield task
            except queue.Empty:
                logger.warning("Timeout waiting for prefetched file")
                break

    def __enter__(self) -> "PrefetchingFileLoader":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()


class CrossFileBatcher:
    """
    Batches segments across multiple files for optimal GPU utilization.

    Instead of processing one file at a time, this collects segments from
    multiple files to form optimal batch sizes, maximizing GPU throughput.

    Supports filtering segments based on classification labels to only
    annotate segments that meet certain criteria (e.g., 2+ negative labels).
    """

    def __init__(
        self,
        batch_size: int,
        max_files_in_batch: int = 8,
        filter_segments: bool = True,
        min_negative_labels: int = 2,
    ):
        """
        Initialize cross-file batcher.

        Args:
            batch_size: Target batch size for GPU
            max_files_in_batch: Maximum number of files to batch together
            filter_segments: Whether to filter segments based on classification labels
            min_negative_labels: Minimum negative labels required for annotation
        """
        self.batch_size = batch_size
        self.max_files_in_batch = max_files_in_batch
        self.filter_segments = filter_segments
        self.min_negative_labels = min_negative_labels
        self.segments_skipped = 0
        self.segments_to_annotate = 0

    def create_batches(
        self,
        file_tasks: list[FileTask],
    ) -> Iterator[tuple[SegmentBatch, list[FileTask]]]:
        """
        Create batches of segments across files.

        Segments are filtered based on classification labels if filter_segments=True.
        Skipped segments are marked with a placeholder annotation.

        Yields:
            Tuple of (SegmentBatch, list of FileTask involved in batch)
        """
        # reset counters.
        self.segments_skipped = 0
        self.segments_to_annotate = 0

        # process one file at a time but batch segments up to batch_size.
        # this ensures segment indices remain valid per file.
        current_segments: list[dict] = []
        current_texts: list[str] = []
        current_file_indices: list[int] = []
        current_segment_indices: list[int] = []
        current_files: list[FileTask] = []

        for task in file_tasks:
            if task.error or not task.segments:
                # yield empty batch for this file so it gets saved.
                if task.transcript_data is not None:
                    yield SegmentBatch([], [], [], []), [task]
                continue

            # add file to current batch.
            batch_file_idx = len(current_files)
            current_files.append(task)

            for seg_idx, segment in enumerate(task.segments):
                # filter segments if enabled.
                if self.filter_segments:
                    if not should_annotate_segment(segment, self.min_negative_labels):
                        # mark segment as skipped with placeholder annotation.
                        segment["llm_annotation"] = get_skipped_annotation()
                        self.segments_skipped += 1
                        continue

                self.segments_to_annotate += 1
                text = segment.get("text", "")
                current_segments.append(segment)
                current_texts.append(text)
                current_file_indices.append(batch_file_idx)
                current_segment_indices.append(seg_idx)

                # yield batch if full, but keep the current file for continuity.
                if len(current_texts) >= self.batch_size:
                    yield SegmentBatch(
                        segments=current_segments,
                        texts=current_texts,
                        file_indices=current_file_indices,
                        segment_indices=current_segment_indices,
                    ), current_files

                    # reset segments but keep files - we'll only clear files after
                    # yielding the final batch for tracking purposes.
                    current_segments = []
                    current_texts = []
                    current_file_indices = []
                    current_segment_indices = []

            # after processing all segments of this file, check if we should yield.
            if len(current_files) >= self.max_files_in_batch:
                if current_texts:
                    yield SegmentBatch(
                        segments=current_segments,
                        texts=current_texts,
                        file_indices=current_file_indices,
                        segment_indices=current_segment_indices,
                    ), current_files

                # now reset everything including files.
                current_segments = []
                current_texts = []
                current_file_indices = []
                current_segment_indices = []
                current_files = []

        # yield remaining.
        if current_texts:
            yield SegmentBatch(
                segments=current_segments,
                texts=current_texts,
                file_indices=current_file_indices,
                segment_indices=current_segment_indices,
            ), current_files
        elif current_files:
            # files with no remaining segments to batch - still need to yield them.
            yield SegmentBatch([], [], [], []), current_files


class ParallelAnnotationPipeline:
    """
    High-throughput annotation pipeline with parallel I/O and GPU compute.

    Features:
    - Prefetches files while GPU processes current batch
    - Batches segments across files for optimal GPU utilization
    - Async file saving to overlap I/O with compute
    - Throughput tracking and adaptive optimization
    - Segment filtering based on classification labels
    """

    def __init__(
        self,
        annotator: Any,  # LLMAnnotator
        prefetch_count: int = 4,
        io_workers: int = 2,
        save_workers: int = 2,
        cross_file_batching: bool = True,
        max_files_per_batch: int = 8,
        filter_segments: bool = True,
        min_negative_labels: int = 2,
    ):
        """
        Initialize parallel pipeline.

        Args:
            annotator: LLMAnnotator instance
            prefetch_count: Number of files to prefetch
            io_workers: Number of file I/O threads
            save_workers: Number of file save threads
            cross_file_batching: Whether to batch segments across files
            max_files_per_batch: Max files to combine in one batch
            filter_segments: Whether to filter segments based on classification labels
            min_negative_labels: Minimum negative labels required for annotation
        """
        self.annotator = annotator
        self.prefetch_count = prefetch_count
        self.io_workers = io_workers
        self.save_workers = save_workers
        self.cross_file_batching = cross_file_batching
        self.max_files_per_batch = max_files_per_batch
        self.filter_segments = filter_segments
        self.min_negative_labels = min_negative_labels

        self.metrics = ThroughputMetrics()
        self.segments_skipped = 0
        self._save_executor: ThreadPoolExecutor | None = None
        self._save_futures: list = []

    def _apply_annotations(
        self,
        batch: SegmentBatch,
        annotations: list[dict[str, Any]],
        files: list[FileTask],
    ) -> None:
        """Apply annotations back to file tasks."""
        for i, annotation in enumerate(annotations):
            file_idx = batch.file_indices[i]
            seg_idx = batch.segment_indices[i]

            if file_idx < len(files) and files[file_idx].segments:
                files[file_idx].segments[seg_idx]["llm_annotation"] = annotation

    def process_files(
        self,
        file_tasks: list[FileTask],
        progress_callback: Any | None = None,
    ) -> tuple[int, int, int]:
        """
        Process files with parallel I/O and GPU compute.

        Args:
            file_tasks: List of FileTask objects to process
            progress_callback: Optional callback(processed, total) for progress

        Returns:
            Tuple of (files_processed, files_failed, total_segments)
        """
        if not file_tasks:
            return 0, 0, 0

        start_time = time.time()
        files_processed = 0
        files_failed = 0
        total_segments = 0
        files_saved = set()

        # start save executor
        self._save_executor = ThreadPoolExecutor(max_workers=self.save_workers)
        self._save_futures = []

        try:
            if self.cross_file_batching:
                # use cross-file batching for better GPU utilization
                files_processed, files_failed, total_segments = self._process_cross_file(
                    file_tasks, progress_callback, files_saved
                )
            else:
                # process file by file with prefetching
                files_processed, files_failed, total_segments = self._process_sequential(
                    file_tasks, progress_callback, files_saved
                )

            # wait for all saves to complete
            for future in self._save_futures:
                try:
                    future.result(timeout=30.0)
                except Exception as e:
                    logger.error(f"Save failed: {e}")

        finally:
            if self._save_executor:
                self._save_executor.shutdown(wait=True)

        # update metrics
        self.metrics.total_time_seconds = time.time() - start_time
        self.metrics.total_files = files_processed
        self.metrics.total_segments = total_segments

        return files_processed, files_failed, total_segments

    def _process_sequential(
        self,
        file_tasks: list[FileTask],
        progress_callback: Any | None,
        files_saved: set,
    ) -> tuple[int, int, int]:
        """Process files sequentially with prefetching."""
        files_processed = 0
        files_failed = 0
        total_segments = 0
        segments_skipped = 0

        with PrefetchingFileLoader(
            file_tasks,
            prefetch_count=self.prefetch_count,
            num_workers=self.io_workers,
        ) as loader:
            for task in loader:
                if task.error:
                    files_failed += 1
                    if progress_callback:
                        progress_callback(files_processed + files_failed, len(file_tasks))
                    continue

                if not task.segments:
                    # empty file - just save
                    self._save_futures.append(
                        self._save_executor.submit(save_file_async, task)
                    )
                    files_processed += 1
                    if progress_callback:
                        progress_callback(files_processed + files_failed, len(file_tasks))
                    continue

                # filter segments if enabled.
                segments_to_annotate = []
                segment_indices = []

                for seg_idx, segment in enumerate(task.segments):
                    if self.filter_segments:
                        if not should_annotate_segment(segment, self.min_negative_labels):
                            # mark segment as skipped.
                            segment["llm_annotation"] = get_skipped_annotation()
                            segments_skipped += 1
                            continue

                    segments_to_annotate.append(segment)
                    segment_indices.append(seg_idx)

                if not segments_to_annotate:
                    # all segments were filtered out - just save.
                    self._save_futures.append(
                        self._save_executor.submit(save_file_async, task)
                    )
                    files_processed += 1
                    if progress_callback:
                        progress_callback(files_processed + files_failed, len(file_tasks))
                    continue

                # annotate filtered segments.
                batch_start = time.time()
                texts = [s.get("text", "") for s in segments_to_annotate]
                annotations = self.annotator.annotate_batch(texts)
                batch_time = time.time() - batch_start
                self.metrics.batch_times.append(batch_time)

                # apply annotations to the correct segments.
                for seg, ann in zip(segments_to_annotate, annotations):
                    seg["llm_annotation"] = ann

                total_segments += len(segments_to_annotate)

                # async save
                self._save_futures.append(
                    self._save_executor.submit(save_file_async, task)
                )

                files_processed += 1
                if progress_callback:
                    progress_callback(files_processed + files_failed, len(file_tasks))

        # track skipped segments.
        self.segments_skipped = segments_skipped

        return files_processed, files_failed, total_segments

    def _process_cross_file(
        self,
        file_tasks: list[FileTask],
        progress_callback: Any | None,
        files_saved: set,
    ) -> tuple[int, int, int]:
        """Process with cross-file batching."""
        files_processed = 0
        files_failed = 0
        total_segments = 0

        # first, load all files (with threading)
        with ThreadPoolExecutor(max_workers=self.io_workers) as executor:
            loaded_tasks = list(executor.map(load_file_async, file_tasks))

        # create batcher with filtering options.
        batcher = CrossFileBatcher(
            batch_size=self.annotator.batch_size,
            max_files_in_batch=self.max_files_per_batch,
            filter_segments=self.filter_segments,
            min_negative_labels=self.min_negative_labels,
        )

        # track which files have been completed
        completed_files: dict[int, FileTask] = {}

        for batch, batch_files in batcher.create_batches(loaded_tasks):
            if not batch.texts:
                # empty batch - save files directly
                for task in batch_files:
                    if task.error:
                        files_failed += 1
                    else:
                        self._save_futures.append(
                            self._save_executor.submit(save_file_async, task)
                        )
                        files_processed += 1
                    if progress_callback:
                        progress_callback(
                            files_processed + files_failed, len(file_tasks)
                        )
                continue

            # annotate batch
            batch_start = time.time()
            annotations = self.annotator.annotate_batch(batch.texts)
            batch_time = time.time() - batch_start
            self.metrics.batch_times.append(batch_time)

            # apply annotations
            self._apply_annotations(batch, annotations, batch_files)
            total_segments += len(batch.texts)

            # save completed files
            for task in batch_files:
                task_id = id(task)
                if task_id not in completed_files:
                    completed_files[task_id] = task
                    self._save_futures.append(
                        self._save_executor.submit(save_file_async, task)
                    )
                    files_processed += 1
                    if progress_callback:
                        progress_callback(
                            files_processed + files_failed, len(file_tasks)
                        )

        # track skipped segments from the batcher.
        self.segments_skipped = batcher.segments_skipped

        return files_processed, files_failed, total_segments

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get throughput metrics summary."""
        return {
            "total_segments": self.metrics.total_segments,
            "total_files": self.metrics.total_files,
            "total_time_seconds": self.metrics.total_time_seconds,
            "segments_per_second": self.metrics.segments_per_second,
            "files_per_minute": self.metrics.files_per_minute,
            "avg_batch_time_ms": self.metrics.avg_batch_time_ms,
            "segments_skipped": self.segments_skipped,
        }
