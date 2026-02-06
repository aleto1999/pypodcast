#!/usr/bin/env python3
"""combine transcripts with speaker diarization from RTTM files."""

import json
import logging
import multiprocessing
import sys
from collections import defaultdict
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


def parse_rttm_file(rttm_path: Path) -> list[dict]:
    """
    parse RTTM file and extract speaker segments.

    RTTM format: SPEAKER <file_id> <channel> <start> <duration> <NA> <NA> <speaker_id> <NA> <NA>

    returns:
        list of dicts with start_time, end_time, and speaker_id
    """
    speakers = []

    try:
        with open(rttm_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8 and parts[0] == "SPEAKER":
                    start_time = float(parts[3])
                    duration = float(parts[4])
                    speaker_id = parts[7]

                    speakers.append({
                        "start_time": start_time,
                        "end_time": start_time + duration,
                        "speaker_id": speaker_id
                    })

        # sort by start time for efficient matching.
        speakers.sort(key=lambda x: x["start_time"])

    except Exception as e:
        logging.error(f"error parsing RTTM file {rttm_path}: {e}")
        return []

    return speakers


def find_speaker_for_segment(segment_start: float, segment_end: float, speakers: list[dict]) -> str:
    """
    find the speaker for a given segment based on maximum overlap.

    args:
        segment_start: start time of transcript segment
        segment_end: end time of transcript segment
        speakers: list of speaker dicts from RTTM

    returns:
        speaker_id with maximum overlap, or "UNKNOWN" if no overlap found
    """
    max_overlap = 0
    best_speaker = "UNKNOWN"

    segment_duration = segment_end - segment_start

    for speaker in speakers:
        # calculate overlap between segment and speaker timing.
        overlap_start = max(segment_start, speaker["start_time"])
        overlap_end = min(segment_end, speaker["end_time"])
        overlap_duration = max(0, overlap_end - overlap_start)

        if overlap_duration > max_overlap:
            max_overlap = overlap_duration
            best_speaker = speaker["speaker_id"]

    # if overlap is less than 10% of segment duration, mark as unknown.
    if max_overlap < segment_duration * 0.1:
        return "UNKNOWN"

    return best_speaker


def combine_transcript_with_speakers(
    transcript_path: Path,
    diarization_path: Path
) -> dict | None:
    """
    combine transcript JSON with speaker labels from RTTM.

    args:
        transcript_path: path to transcript JSON file
        diarization_path: path to RTTM diarization file

    returns:
        combined dict with speaker labels added to segments, or None if error
    """
    logger = logging.getLogger(__name__)

    try:
        # read transcript JSON.
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript = json.load(f)

        # parse RTTM file.
        speakers = parse_rttm_file(diarization_path)

        if not speakers:
            logger.warning(f"no speakers found in {diarization_path.name}")
            return None

        # add speaker labels to each segment.
        for segment in transcript.get("segments", []):
            start = segment.get("start", 0)
            end = segment.get("end", 0)

            speaker = find_speaker_for_segment(start, end, speakers)
            segment["speaker"] = speaker

            # ensure words field exists (empty list if not present).
            if "words" not in segment:
                segment["words"] = []

        # add metadata about diarization.
        if "metadata" not in transcript:
            transcript["metadata"] = {}

        transcript["metadata"]["diarization_file"] = str(diarization_path.name)
        transcript["metadata"]["diarization_applied"] = True

        return transcript

    except Exception as e:
        logger.error(f"error combining {transcript_path.name} with speakers: {e}")
        return None


def discover_show_pairs(
    transcripts_dir: Path,
    diarizations_dir: Path
) -> dict[str, list[tuple[Path, Path]]]:
    """
    discover and pair transcript and diarization files by show.

    args:
        transcripts_dir: root directory containing transcript JSON files
        diarizations_dir: root directory containing RTTM files

    returns:
        dict mapping show names to list of (transcript_path, rttm_path) tuples
    """
    logger = logging.getLogger(__name__)

    show_pairs = defaultdict(list)

    # traverse each show folder in transcripts directory.
    for show_dir in transcripts_dir.iterdir():
        if not show_dir.is_dir():
            continue

        show_name = show_dir.name

        # check if corresponding diarization folder exists.
        diarization_show_dir = diarizations_dir / show_name

        if not diarization_show_dir.exists() or not diarization_show_dir.is_dir():
            logger.warning(f"no diarization folder found for show: {show_name}")
            continue

        # find all transcript JSON files in this show.
        for transcript_file in show_dir.glob("*.json"):
            # construct expected RTTM filename (same stem, .rttm extension).
            rttm_file = diarization_show_dir / f"{transcript_file.stem}.rttm"

            if rttm_file.exists():
                show_pairs[show_name].append((transcript_file, rttm_file))
            else:
                logger.warning(
                    f"no diarization found for {show_name}/{transcript_file.name}"
                )

    return show_pairs


def check_combined_exists(output_path: Path) -> bool:
    """
    check if combined transcript file exists and has valid speaker labels.

    returns True if file exists and contains speaker labels in segments.
    """
    if not output_path.exists():
        return False

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        segments = data.get("segments", [])

        if not segments:
            return False

        # check if all segments have speaker labels.
        for segment in segments:
            if "speaker" not in segment:
                return False

        # check metadata indicates diarization was applied.
        metadata = data.get("metadata", {})
        if not metadata.get("diarization_applied", False):
            return False

        return True

    except Exception:
        return False


def process_single_file(
    show_name: str,
    transcript_path: Path,
    rttm_path: Path,
    output_dir: Path,
) -> tuple[str, Path, int, str | None]:
    """
    Process a single transcript-diarization pair.

    Returns:
        Tuple of (show_name, transcript_path, segments_labeled, error_message)
    """
    try:
        combined = combine_transcript_with_speakers(transcript_path, rttm_path)

        if combined is None:
            return (show_name, transcript_path, 0, "failed to combine")

        # count segments with speaker labels.
        segments = combined.get("segments", [])
        labeled = sum(1 for s in segments if s.get("speaker"))

        # write combined JSON to output directory.
        show_output_dir = output_dir / show_name
        show_output_dir.mkdir(parents=True, exist_ok=True)
        output_file = show_output_dir / transcript_path.name

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)

        return (show_name, transcript_path, labeled, None)

    except Exception as e:
        return (show_name, transcript_path, 0, str(e))


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
    default="outputs/transcripts",
    help="directory containing transcript JSON files",
)
@click.option(
    "--diarizations-dir",
    type=click.Path(exists=True, path_type=Path),
    default="outputs/diarizations",
    help="directory containing RTTM diarization files",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default="outputs/transcripts_with_diarization_labels",
    help="output directory for combined transcripts (default: outputs/transcripts_with_diarization_labels)",
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
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Logging level",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be processed without running",
)
def main(
    transcripts_dir: Path,
    diarizations_dir: Path,
    output_dir: Path,
    force: bool,
    workers: int | None,
    log_level: str,
    dry_run: bool,
):
    """combine transcripts with speaker diarization from RTTM files."""
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    console.print("\n[bold cyan]🎤 Combining Transcripts with Speaker Diarization[/bold cyan]\n")

    # discover show pairs.
    console.print("[dim]discovering transcript-diarization pairs...[/dim]")
    show_pairs = discover_show_pairs(transcripts_dir, diarizations_dir)

    if not show_pairs:
        console.print("[bold red]no matching transcript-diarization pairs found![/bold red]")
        sys.exit(1)

    total_pairs = sum(len(pairs) for pairs in show_pairs.values())
    console.print(f"[green]found {len(show_pairs)} shows with {total_pairs} transcript-diarization pairs[/green]")

    # filter out files that already have combined outputs (unless --force).
    files_to_process = []
    files_skipped = 0

    for show_name, pairs in show_pairs.items():
        for transcript_path, rttm_path in pairs:
            output_file = output_dir / show_name / transcript_path.name

            if not force and check_combined_exists(output_file):
                files_skipped += 1
                logger.debug(f"skipping {show_name}/{transcript_path.name} (already combined)")
            else:
                files_to_process.append((show_name, transcript_path, rttm_path))

    if files_skipped > 0:
        console.print(f"[yellow]skipping {files_skipped} files with existing speaker labels[/yellow]")

    console.print(f"[green]will process {len(files_to_process)} new files[/green]\n")

    if not files_to_process:
        console.print("[yellow]all files already have speaker labels - nothing to process[/yellow]")
        console.print("  use --force to reprocess\n")
        sys.exit(0)

    # dry-run mode: show what would be processed.
    if dry_run:
        console.print("[bold yellow]dry run mode - files that would be processed:[/bold yellow]\n")
        for i, (show_name, transcript_path, rttm_path) in enumerate(files_to_process[:10], 1):
            console.print(f"  {i}. {show_name}/{transcript_path.name}")
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

    # process files.
    processed_count = 0
    error_count = 0
    total_segments = 0

    with ResourceDisplay(console=console) as display:
        task = display.progress.add_task(
            "[cyan]combining | starting...",
            total=len(files_to_process),
        )

        if workers == 1:
            # sequential processing.
            for show_name, transcript_path, rttm_path in files_to_process:
                show_name, transcript_path, labeled, error = process_single_file(
                    show_name, transcript_path, rttm_path, output_dir
                )

                if error:
                    logger.warning(f"error processing {show_name}/{transcript_path.name}: {error}")
                    error_count += 1
                else:
                    processed_count += 1
                    total_segments += labeled

                display.progress.update(
                    task,
                    description=(
                        f"[cyan]combining | "
                        f"{total_segments:,} segments labeled in {processed_count} files"
                    ),
                    advance=1,
                )
        else:
            # parallel processing.
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        process_single_file, show_name, transcript_path, rttm_path, output_dir
                    ): (show_name, transcript_path)
                    for show_name, transcript_path, rttm_path in files_to_process
                }

                for future in as_completed(futures):
                    show_name, transcript_path, labeled, error = future.result()

                    if error:
                        logger.warning(f"error processing {show_name}/{transcript_path.name}: {error}")
                        error_count += 1
                    else:
                        processed_count += 1
                        total_segments += labeled

                    display.progress.update(
                        task,
                        description=(
                            f"[cyan]combining | "
                            f"{total_segments:,} segments labeled in {processed_count} files"
                        ),
                        advance=1,
                    )

        # print resource usage summary.
        display.print_summary()

    # print summary.
    console.print()
    console.print("[bold]Summary:[/bold]")
    console.print(f"  ✓ Processed: {processed_count} files")
    if files_skipped > 0:
        console.print(f"  ⊘ Skipped: {files_skipped} files (already exist)")
    console.print(f"  ✗ Errors: {error_count} files")
    console.print(f"\n[bold green]results saved to: {output_dir}[/bold green]\n")


if __name__ == "__main__":
    main()
