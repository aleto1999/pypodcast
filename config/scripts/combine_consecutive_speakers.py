#!/usr/bin/env python3
"""combine consecutive utterances from the same speaker in transcript files."""

import json
import logging
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click

# add src to path for imports.
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
src_dir = project_root / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from podcast_conversations.monitoring import ResourceDisplay, print_system_info
from utils.rich_utils import console, setup_logging


def check_postprocessed_exists(output_path: Path) -> bool:
    """
    check if postprocessed transcript file exists and has valid combined segments.

    returns True if file exists and appears to have been postprocessed.
    """
    if not output_path.exists():
        return False

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # verify it's a valid transcript with segments.
        segments = data.get("segments", [])
        if not segments:
            return False

        # check that segments have expected fields.
        for segment in segments:
            if "text" not in segment or "speaker" not in segment:
                return False

        return True

    except Exception:
        return False


def combine_consecutive_speakers(segments: list[dict]) -> list[dict]:
    """
    combine consecutive utterances from the same speaker.

    args:
        segments: list of segment dicts with 'text', 'start', 'end', 'speaker', 'words'

    returns:
        list of combined segments
    """
    if not segments:
        return []

    combined = []
    current_segment = segments[0].copy()

    for i in range(1, len(segments)):
        segment = segments[i]

        # if same speaker as current segment, combine them.
        if segment["speaker"] == current_segment["speaker"]:
            # concatenate text with space.
            current_segment["text"] = current_segment["text"] + " " + segment["text"]
            # extend end time to include this segment.
            current_segment["end"] = segment["end"]
            # merge words arrays (if they have content).
            if segment.get("words"):
                current_segment["words"] = current_segment.get("words", []) + segment["words"]
        else:
            # different speaker, save current segment and start new one.
            combined.append(current_segment)
            current_segment = segment.copy()

    # don't forget the last segment.
    combined.append(current_segment)

    return combined


def generate_text_with_speakers(segments: list[dict]) -> str:
    """
    generate formatted text with speaker labels.

    args:
        segments: list of combined segments with 'speaker' and 'text' fields.

    returns:
        formatted string with speaker labels, e.g.:
        [SPEAKER_00]: Hello and welcome.

        [SPEAKER_01]: Thank you for having me.
    """
    lines = []
    for segment in segments:
        speaker = segment.get("speaker", "UNKNOWN")
        text = segment.get("text", "").strip()
        if text:
            lines.append(f"[{speaker}]: {text}")
    return "\n\n".join(lines)


def process_transcript_file(
    input_path: Path,
    output_path: Path
) -> dict:
    """
    process a single transcript file and combine consecutive speakers.

    args:
        input_path: path to input JSON transcript file
        output_path: path to output JSON file

    returns:
        dict with processing statistics
    """
    logger = logging.getLogger(__name__)

    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")

    # load transcript.
    try:
        with open(input_path, encoding="utf-8") as f:
            transcript_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON in {input_path}: {e}") from e

    # get segments.
    original_segments = transcript_data.get("segments", [])
    if not original_segments:
        logger.warning(f"no segments found in {input_path.name}")
        # still write output file even if no segments.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(transcript_data, f, indent=2, ensure_ascii=False)
        return {"original_segments": 0, "combined_segments": 0, "reduction": 0}

    # combine consecutive speakers.
    combined_segments = combine_consecutive_speakers(original_segments)

    # update transcript data with combined segments.
    transcript_data["segments"] = combined_segments

    # generate text_with_speakers formatted output.
    transcript_data["text_with_speakers"] = generate_text_with_speakers(combined_segments)

    # write output file.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, indent=2, ensure_ascii=False)

    reduction = len(original_segments) - len(combined_segments)
    logger.info(
        f"processed {input_path.name}: "
        f"{len(original_segments)} → {len(combined_segments)} segments (reduced by {reduction})"
    )

    return {
        "original_segments": len(original_segments),
        "combined_segments": len(combined_segments),
        "reduction": reduction
    }


def process_single_file_worker(
    input_path: Path,
    output_path: Path,
    relative_path: Path,
) -> tuple[Path, dict | None, str | None]:
    """
    Worker function for parallel processing.

    Returns:
        Tuple of (relative_path, stats_dict, error_message)
    """
    try:
        stats = process_transcript_file(input_path, output_path)
        return (relative_path, stats, None)
    except Exception as e:
        return (relative_path, None, str(e))


def detect_optimal_workers() -> int:
    """Detect optimal number of workers based on CPU cores."""
    import os

    # check for slurm.
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return min(int(os.environ["SLURM_CPUS_PER_TASK"]), 16)

    # check for pbs.
    if "PBS_NUM_PPN" in os.environ:
        return min(int(os.environ["PBS_NUM_PPN"]), 16)

    # default to 75% of cpu count.
    cpu_count = multiprocessing.cpu_count()
    return max(1, min(int(cpu_count * 0.75), 16))


@click.command()
@click.option(
    "--transcripts-dir",
    type=click.Path(exists=True, path_type=Path),
    default="outputs/transcripts_with_diarization_labels",
    help="input directory containing transcript JSON files (default: outputs/transcripts_with_diarization_labels)",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default="outputs/transcripts_with_diarization_labels_postprocessed",
    help="output directory for processed transcripts (default: outputs/transcripts_with_diarization_labels_postprocessed)",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Logging level",
)
@click.option(
    "--force",
    is_flag=True,
    help="Force reprocessing of files with existing outputs",
)
@click.option(
    "--workers",
    type=int,
    default=None,
    help="Number of parallel workers (auto-detected if not set)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be processed without running",
)
def combine_speakers(
    transcripts_dir: Path,
    output_dir: Path,
    log_level: str,
    force: bool,
    workers: int | None,
    dry_run: bool,
) -> None:
    """
    combine consecutive utterances from the same speaker in transcript files.

    processes all JSON files in transcripts-dir, combines consecutive segments from
    the same speaker, and outputs the processed transcripts to output-dir with
    the same directory structure.

    example:
        python scripts/combine_consecutive_speakers.py \\
            --transcripts-dir outputs/transcripts_with_diarization_labels \\
            --output-dir outputs/transcripts_with_diarization_labels_postprocessed
    """
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    console.print("\n[bold cyan]🔗 combining consecutive speaker utterances[/bold cyan]\n")

    # discover all JSON files recursively.
    console.print("[bold blue]discovering transcript files...[/bold blue]")
    json_files = sorted(transcripts_dir.rglob("*.json"))

    if not json_files:
        console.print(f"[yellow]no JSON files found in {transcripts_dir}[/yellow]")
        sys.exit(0)

    console.print(f"[green]found {len(json_files)} files to process[/green]")

    # filter out files that already exist (unless --force).
    files_to_process = []
    files_skipped = 0

    for input_path in json_files:
        relative_path = input_path.relative_to(transcripts_dir)
        output_path = output_dir / relative_path

        if not force and check_postprocessed_exists(output_path):
            files_skipped += 1
            logger.debug(f"skipping {relative_path} (already exists)")
        else:
            files_to_process.append((input_path, output_path, relative_path))

    if files_skipped > 0:
        console.print(f"[yellow]skipping {files_skipped} files with existing outputs[/yellow]")

    console.print(f"[green]will process {len(files_to_process)} new files[/green]\n")

    if not files_to_process:
        console.print("[yellow]all files already processed - nothing to do[/yellow]")
        console.print("  use --force to reprocess\n")
        sys.exit(0)

    # dry-run mode: show what would be processed.
    if dry_run:
        console.print("[bold yellow]dry run mode - files that would be processed:[/bold yellow]\n")
        for i, (input_path, output_path, relative_path) in enumerate(files_to_process[:10], 1):
            console.print(f"  {i}. {relative_path}")
        if len(files_to_process) > 10:
            console.print(f"  ... and {len(files_to_process) - 10} more files")
        console.print()
        return

    # determine number of workers.
    if workers is None:
        workers = detect_optimal_workers()
    console.print(f"[dim]using {workers} workers[/dim]\n")

    # display system info before starting.
    print_system_info(console)

    # process files with progress bar.
    stats = {
        "files_processed": 0,
        "files_skipped": files_skipped,
        "files_failed": 0,
        "total_original_segments": 0,
        "total_combined_segments": 0,
        "total_reduction": 0,
    }

    def update_progress(progress, task, stats):
        """Helper to update progress bar with current stats."""
        if stats["total_original_segments"] > 0:
            reduction_pct = (
                stats["total_reduction"] / stats["total_original_segments"] * 100
            )
        else:
            reduction_pct = 0

        progress.update(
            task,
            description=(
                f"[cyan]merging | "
                f"{stats['total_original_segments']:,} → "
                f"{stats['total_combined_segments']:,} segments "
                f"({reduction_pct:.0f}% reduction)"
            ),
            advance=1,
        )

    with ResourceDisplay(console=console) as display:
        task = display.progress.add_task(
            "[cyan]merging | starting...",
            total=len(files_to_process),
        )

        if workers == 1:
            # sequential processing.
            for input_path, output_path, relative_path in files_to_process:
                show_name = relative_path.parts[0] if relative_path.parts else "unknown"
                file_name = input_path.stem

                try:
                    file_stats = process_transcript_file(input_path, output_path)
                    stats["files_processed"] += 1
                    stats["total_original_segments"] += file_stats["original_segments"]
                    stats["total_combined_segments"] += file_stats["combined_segments"]
                    stats["total_reduction"] += file_stats["reduction"]
                except Exception as e:
                    stats["files_failed"] += 1
                    logger.error(f"✗ {show_name}/{file_name}: {e}")

                update_progress(display.progress, task, stats)
        else:
            # parallel processing.
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        process_single_file_worker, input_path, output_path, relative_path
                    ): relative_path
                    for input_path, output_path, relative_path in files_to_process
                }

                for future in as_completed(futures):
                    relative_path, file_stats, error = future.result()
                    show_name = relative_path.parts[0] if relative_path.parts else "unknown"
                    file_name = relative_path.stem

                    if error:
                        stats["files_failed"] += 1
                        logger.error(f"✗ {show_name}/{file_name}: {error}")
                    else:
                        stats["files_processed"] += 1
                        stats["total_original_segments"] += file_stats["original_segments"]
                        stats["total_combined_segments"] += file_stats["combined_segments"]
                        stats["total_reduction"] += file_stats["reduction"]

                    update_progress(display.progress, task, stats)

        # print resource usage summary.
        display.print_summary()

    # display final statistics.
    console.print("\n[bold green]✓ processing complete![/bold green]\n")
    console.print("[bold]statistics:[/bold]")
    console.print(f"  • files processed: {stats['files_processed']}")
    if stats['files_skipped'] > 0:
        console.print(f"  • files skipped: {stats['files_skipped']}")
    console.print(f"  • files failed: {stats['files_failed']}")
    if stats['total_original_segments'] > 0:
        console.print(f"  • original segments: {stats['total_original_segments']}")
        console.print(f"  • combined segments: {stats['total_combined_segments']}")
        console.print(f"  • segments reduced: {stats['total_reduction']} ({stats['total_reduction'] / stats['total_original_segments'] * 100:.1f}%)")
    console.print(f"\n[bold green]output saved to: {output_dir}[/bold green]\n")


if __name__ == "__main__":
    combine_speakers()
