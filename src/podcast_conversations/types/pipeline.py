"""Pipeline task and batch types with runtime validation.

This module defines types for managing processing pipelines:
- FileTask: A single file to be processed with paths and metadata
- SegmentBatch: A batch of segments for parallel processing
- ProcessingResult: Result of processing a single item
- BatchResult: Aggregated results from batch processing

Used across transcription, diarization, annotation, and analysis pipelines.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, computed_field


class TaskStatus(str, Enum):
    """Status of a processing task.

    Values:
        PENDING: Task is queued but not started.
        IN_PROGRESS: Task is currently being processed.
        COMPLETED: Task finished successfully.
        FAILED: Task failed with an error.
        SKIPPED: Task was skipped (e.g., output already exists).
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class FileTask(BaseModel):
    """A file processing task with input/output paths.

    Represents a single unit of work in a processing pipeline,
    tracking the input file, expected output, and processing state.

    Attributes:
        input_path: Path to the input file.
        output_path: Path where output will be written.
        relative_path: Path relative to input directory (for display).
        status: Current processing status.
        error_message: Error details if task failed.
        metadata: Additional task-specific data.
    """

    input_path: str = Field(description="Path to input file")
    output_path: str = Field(description="Path for output file")
    relative_path: str = Field(
        default="",
        description="Relative path for display",
    )
    status: TaskStatus = Field(
        default=TaskStatus.PENDING,
        description="Current task status",
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if failed",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional task metadata",
    )

    @classmethod
    def from_paths(
        cls,
        input_path: Path,
        output_path: Path,
        base_dir: Path | None = None,
    ) -> FileTask:
        """Create a FileTask from Path objects.

        Args:
            input_path: Input file path.
            output_path: Output file path.
            base_dir: Base directory for computing relative path.

        Returns:
            New FileTask instance.
        """
        relative = ""
        if base_dir:
            try:
                relative = str(input_path.relative_to(base_dir))
            except ValueError:
                relative = input_path.name

        return cls(
            input_path=str(input_path),
            output_path=str(output_path),
            relative_path=relative or input_path.name,
        )

    @property
    def input_path_obj(self) -> Path:
        """Input path as Path object."""
        return Path(self.input_path)

    @property
    def output_path_obj(self) -> Path:
        """Output path as Path object."""
        return Path(self.output_path)

    def mark_completed(self) -> None:
        """Mark the task as completed."""
        self.status = TaskStatus.COMPLETED
        self.error_message = None

    def mark_failed(self, error: str) -> None:
        """Mark the task as failed with error message."""
        self.status = TaskStatus.FAILED
        self.error_message = error

    def mark_skipped(self, reason: str = "Output already exists") -> None:
        """Mark the task as skipped."""
        self.status = TaskStatus.SKIPPED
        self.error_message = reason


class SegmentBatch(BaseModel):
    """A batch of segments for parallel processing.

    Groups segments from one or more files for efficient
    batch processing (e.g., LLM inference).

    Attributes:
        batch_id: Unique identifier for this batch.
        segments: List of segment texts to process.
        source_files: Files these segments came from.
        segment_indices: Original indices for result mapping.
        metadata: Additional batch metadata.
    """

    batch_id: int = Field(ge=0, description="Unique batch identifier")
    segments: list[str] = Field(
        default_factory=list,
        description="Segment texts to process",
    )
    source_files: list[str] = Field(
        default_factory=list,
        description="Source file paths",
    )
    segment_indices: list[tuple[int, int]] = Field(
        default_factory=list,
        description="(file_idx, segment_idx) for each segment",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional batch metadata",
    )

    @computed_field
    @property
    def size(self) -> int:
        """Number of segments in this batch."""
        return len(self.segments)

    @computed_field
    @property
    def unique_files(self) -> int:
        """Number of unique source files."""
        return len(set(self.source_files))


class ProcessingResult(BaseModel):
    """Result of processing a single item.

    Generic result type that can hold success/failure status
    along with output data or error information.

    Attributes:
        success: Whether processing succeeded.
        file_path: Path to the processed file.
        output_path: Path where output was written.
        error_message: Error details if failed.
        processing_time: Time taken in seconds.
        output_data: Processing output (type varies by pipeline).
        metrics: Performance or quality metrics.
    """

    success: bool = Field(description="Whether processing succeeded")
    file_path: str = Field(description="Source file path")
    output_path: str | None = Field(
        default=None,
        description="Output file path",
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if failed",
    )
    processing_time: float | None = Field(
        default=None,
        ge=0,
        description="Processing time in seconds",
    )
    output_data: dict[str, Any] | None = Field(
        default=None,
        description="Processing output data",
    )
    metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Performance/quality metrics",
    )

    @classmethod
    def success_result(
        cls,
        file_path: str,
        output_path: str,
        processing_time: float | None = None,
        output_data: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> ProcessingResult:
        """Create a successful result.

        Args:
            file_path: Source file path.
            output_path: Output file path.
            processing_time: Time taken in seconds.
            output_data: Processing output.
            metrics: Performance metrics.

        Returns:
            ProcessingResult with success=True.
        """
        return cls(
            success=True,
            file_path=file_path,
            output_path=output_path,
            processing_time=processing_time,
            output_data=output_data,
            metrics=metrics or {},
        )

    @classmethod
    def failure_result(
        cls,
        file_path: str,
        error_message: str,
        processing_time: float | None = None,
    ) -> ProcessingResult:
        """Create a failed result.

        Args:
            file_path: Source file path.
            error_message: Error description.
            processing_time: Time before failure.

        Returns:
            ProcessingResult with success=False.
        """
        return cls(
            success=False,
            file_path=file_path,
            error_message=error_message,
            processing_time=processing_time,
        )


class BatchResult(BaseModel):
    """Aggregated results from batch processing.

    Collects results from multiple items and computes
    summary statistics.

    Attributes:
        total_items: Total items in the batch.
        successful: Count of successful items.
        failed: Count of failed items.
        skipped: Count of skipped items.
        results: Individual processing results.
        total_time: Total processing time.
        started_at: When processing started.
        completed_at: When processing completed.
    """

    total_items: int = Field(ge=0, description="Total items processed")
    successful: int = Field(ge=0, default=0, description="Successful count")
    failed: int = Field(ge=0, default=0, description="Failed count")
    skipped: int = Field(ge=0, default=0, description="Skipped count")
    results: list[ProcessingResult] = Field(
        default_factory=list,
        description="Individual results",
    )
    total_time: float | None = Field(
        default=None,
        ge=0,
        description="Total processing time",
    )
    started_at: datetime | None = Field(
        default=None,
        description="Processing start time",
    )
    completed_at: datetime | None = Field(
        default=None,
        description="Processing completion time",
    )

    @computed_field
    @property
    def success_rate(self) -> float:
        """Proportion of successful items (0.0-1.0)."""
        if self.total_items == 0:
            return 0.0
        return self.successful / self.total_items

    @computed_field
    @property
    def items_per_second(self) -> float:
        """Processing throughput."""
        if not self.total_time or self.total_time == 0:
            return 0.0
        return self.total_items / self.total_time

    def add_result(self, result: ProcessingResult) -> None:
        """Add a processing result and update counts.

        Args:
            result: The result to add.
        """
        self.results.append(result)
        if result.success:
            self.successful += 1
        else:
            self.failed += 1

    def get_errors(self) -> list[tuple[str, str]]:
        """Get list of (file_path, error_message) for failed items.

        Returns:
            List of tuples with file path and error message.
        """
        return [
            (r.file_path, r.error_message or "Unknown error")
            for r in self.results
            if not r.success and r.error_message
        ]
