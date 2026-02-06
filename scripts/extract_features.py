#!/usr/bin/env python3
"""Extract conversational features from podcast transcripts.

This script processes transcript JSON files and extracts:
- Question detection and classification
- Turn-taking patterns (dominance, switch rate)
- Vocabulary diversity (TTR, unique words)
- Politeness features

Output is saved as CSV file(s) for analysis.
"""

import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# add src and scripts to path for imports.
script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent / "src"
if src_dir.exists():
    sys.path.insert(0, str(src_dir))
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

import click
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from podcast_conversations.feature_extraction import (
    FeatureExtractor,
    extract_features_from_transcript,
)
from utils.rich_utils import console, setup_logging


def get_transcript_files(transcripts_dir: Path) -> list[Path]:
    """Get all JSON transcript files from directory."""
    files = []
    for json_file in transcripts_dir.rglob("*.json"):
        files.append(json_file)
    return sorted(files)


def process_single_file(
    file_path: Path,
    use_convokit: bool = False,
) -> tuple[Path, dict | None, str | None]:
    """Process a single transcript file.

    Returns:
        Tuple of (file_path, features_dict, error_message)
    """
    try:
        features = extract_features_from_transcript(file_path, use_convokit=use_convokit)
        extractor = FeatureExtractor(use_convokit=use_convokit)
        flat_dict = extractor.to_flat_dict(features)
        return file_path, flat_dict, None
    except Exception as e:
        return file_path, None, str(e)


def detect_optimal_workers() -> int:
    """Detect optimal number of workers for HPC environments."""
    import os

    # check for slurm.
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return min(int(os.environ["SLURM_CPUS_PER_TASK"]), 32)

    # check for pbs.
    if "PBS_NUM_PPN" in os.environ:
        return min(int(os.environ["PBS_NUM_PPN"]), 32)

    # default to cpu count.
    import multiprocessing

    return min(multiprocessing.cpu_count(), 8)


@click.command()
@click.option(
    "--transcripts-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing transcript JSON files",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("outputs/analysis/features"),
    help="Output directory for CSV files",
)
@click.option(
    "--output-file",
    type=str,
    default="features.csv",
    help="Output CSV filename",
)
@click.option(
    "--use-convokit",
    is_flag=True,
    default=False,
    help="Use convokit for advanced politeness analysis (requires convokit + spacy)",
)
@click.option(
    "--workers",
    type=int,
    default=None,
    help="Number of parallel workers (auto-detected if not set)",
)
@click.option(
    "--also-json",
    is_flag=True,
    default=False,
    help="Also output detailed JSON file with nested features",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Force reprocessing even if output file already exists",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Show what would be processed without running",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Logging level",
)
def main(
    transcripts_dir: Path,
    output_dir: Path,
    output_file: str,
    use_convokit: bool,
    workers: int | None,
    also_json: bool,
    force: bool,
    dry_run: bool,
    log_level: str,
) -> None:
    """Extract conversational features from podcast transcripts.

    Processes transcript JSON files and extracts question patterns,
    turn-taking dynamics, vocabulary diversity, and politeness features.

    Output is saved as a CSV file with one row per transcript.

    Example:
        python scripts/extract_features.py \\
            --transcripts-dir outputs/transcripts_with_diarization_labels_postprocessed \\
            --output-dir outputs/analysis/features
    """
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    # create output directory.
    output_dir.mkdir(parents=True, exist_ok=True)

    # check if output already exists.
    csv_path = output_dir / output_file
    if csv_path.exists() and not force:
        console.print(f"[yellow]Output file already exists: {csv_path}[/yellow]")
        console.print("[yellow]Use --force to reprocess[/yellow]")
        sys.exit(0)

    # get transcript files.
    transcript_files = get_transcript_files(transcripts_dir)

    if not transcript_files:
        console.print("[red]No JSON files found in transcripts directory[/red]")
        sys.exit(1)

    console.print(f"[cyan]Found {len(transcript_files)} transcript files[/cyan]")

    if dry_run:
        console.print("\n[yellow]Dry run - would process:[/yellow]")
        for f in transcript_files[:10]:
            console.print(f"  {f.name}")
        if len(transcript_files) > 10:
            console.print(f"  ... and {len(transcript_files) - 10} more")
        return

    # determine workers.
    if workers is None:
        workers = detect_optimal_workers()
    console.print(f"[cyan]Using {workers} workers[/cyan]")

    # process files.
    all_features: list[dict] = []
    all_features_nested: list[dict] = []
    errors: list[tuple[Path, str]] = []
    processed = 0
    total_questions = 0
    total_turns = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=4,
    ) as progress:
        task = progress.add_task(
            "[cyan]extracting features...",
            total=len(transcript_files),
        )

        if workers == 1:
            # single-threaded processing.
            for file_path in transcript_files:
                file_path, features_dict, error = process_single_file(
                    file_path, use_convokit=use_convokit
                )

                if error:
                    errors.append((file_path, error))
                    logger.warning(f"Error processing {file_path.name}: {error}")
                else:
                    all_features.append(features_dict)
                    total_questions += features_dict.get("total_questions", 0)
                    total_turns += features_dict.get("total_turns", 0)

                    if also_json:
                        # get nested version too.
                        try:
                            features = extract_features_from_transcript(
                                file_path, use_convokit=use_convokit
                            )
                            extractor = FeatureExtractor(use_convokit=use_convokit)
                            all_features_nested.append(extractor.to_dict(features))
                        except Exception:
                            pass

                processed += 1
                progress.update(
                    task,
                    description=(
                        f"[cyan]features | {total_questions:,} questions, "
                        f"{total_turns:,} turns in {processed} files"
                    ),
                    advance=1,
                )
        else:
            # parallel processing.
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        process_single_file, f, use_convokit
                    ): f
                    for f in transcript_files
                }

                for future in as_completed(futures):
                    file_path, features_dict, error = future.result()

                    if error:
                        errors.append((file_path, error))
                        logger.warning(f"Error processing {file_path.name}: {error}")
                    else:
                        all_features.append(features_dict)
                        total_questions += features_dict.get("total_questions", 0)
                        total_turns += features_dict.get("total_turns", 0)

                    processed += 1
                    progress.update(
                        task,
                        description=(
                            f"[cyan]features | {total_questions:,} questions, "
                            f"{total_turns:,} turns in {processed} files"
                        ),
                        advance=1,
                    )

    # write csv output.
    if all_features:
        import csv

        # get all column names (union of all keys).
        all_columns = set()
        for f in all_features:
            all_columns.update(f.keys())

        # sort columns for consistent output.
        columns = ["file_name", "file_path"]
        columns.extend(
            sorted(c for c in all_columns if c not in ["file_name", "file_path"])
        )

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for features in all_features:
                writer.writerow(features)

        console.print(f"\n[green]✓ Saved CSV: {csv_path}[/green]")

        # optionally write json.
        if also_json and all_features_nested:
            json_path = output_dir / output_file.replace(".csv", ".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(all_features_nested, f, indent=2, ensure_ascii=False)
            console.print(f"[green]✓ Saved JSON: {json_path}[/green]")

    # summary.
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Files processed: {len(all_features)}")
    console.print(f"  Total questions detected: {total_questions:,}")
    console.print(f"  Total turns detected: {total_turns:,}")
    if errors:
        console.print(f"  [red]Errors: {len(errors)}[/red]")


if __name__ == "__main__":
    main()
