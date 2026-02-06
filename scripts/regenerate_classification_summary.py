#!/usr/bin/env python3
"""regenerate classification summary files from existing labeled transcripts.

this script scans already classified transcripts in the outputs directory and
regenerates the summary files (TXT format) without re-running the classification
pipeline. useful for when summary files are missing or need to be updated.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import click
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

# add src and scripts to path for imports.
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
src_dir = project_root / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from utils.rich_utils import console, setup_logging


def generate_summary(
    input_dir: Path,
    start_time: datetime,
    end_time: datetime,
    total_files: int,
    label_stats: dict,
    show_stats: dict,
    episode_stats: dict,
    total_utterances: int,
) -> str:
    """generate a summary report from classification statistics."""
    duration = end_time - start_time
    hours, remainder = divmod(int(duration.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)

    summary = f"""
================================================================================
           UTTERANCE CLASSIFICATION SUMMARY - REGENERATED FROM EXISTING DATA
================================================================================

Generation Date: {start_time.strftime('%Y-%m-%d %H:%M:%S')}
Duration: {hours}h {minutes}m {seconds}s

--------------------------------------------------------------------------------
                                    INPUT
--------------------------------------------------------------------------------
Classified Transcripts Directory:  {input_dir}
Total Files Analyzed:               {total_files}
Total Utterances:                   {total_utterances:,}

"""

    # add label statistics per model.
    if label_stats:
        summary += """
================================================================================
                         LABEL STATISTICS BY MODEL
================================================================================

"""
        for model_name in sorted(label_stats.keys()):
            labels = label_stats[model_name]
            total_labels = sum(labels.values())
            summary += f"\n{model_name}:\n"
            summary += f"  Total Classifications: {total_labels:,}\n"
            summary += "  Label Distribution:\n"
            # sort by count descending.
            for label, count in sorted(labels.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_labels * 100) if total_labels > 0 else 0
                summary += f"    {label:30s}: {count:8,} ({percentage:5.2f}%)\n"

    # add show statistics.
    if show_stats:
        summary += """
================================================================================
                           STATISTICS BY SHOW
================================================================================

"""
        for show_name in sorted(show_stats.keys()):
            show_data = show_stats[show_name]
            summary += f"\n{show_name}:\n"
            summary += f"  Episodes Processed: {show_data['episodes']:,}\n"
            summary += f"  Total Utterances:   {show_data['utterances']:,}\n"
            if show_data.get("labels"):
                summary += "  Label Distribution by Model:\n"
                for model_name in sorted(show_data["labels"].keys()):
                    labels = show_data["labels"][model_name]
                    total_model_labels = sum(labels.values())
                    summary += f"    {model_name}:\n"
                    for label, count in sorted(labels.items(), key=lambda x: x[1], reverse=True):
                        pct = (count / total_model_labels * 100) if total_model_labels > 0 else 0
                        summary += f"      {label:28s}: {count:8,} ({pct:5.2f}%)\n"

    # add episode statistics.
    if episode_stats:
        summary += """
================================================================================
                          STATISTICS BY EPISODE
================================================================================

"""
        # group episodes by show.
        episodes_by_show = {}
        for episode_key, episode_data in episode_stats.items():
            show_name = episode_data.get("show", "unknown")
            if show_name not in episodes_by_show:
                episodes_by_show[show_name] = {}
            episodes_by_show[show_name][episode_key] = episode_data

        for show_name in sorted(episodes_by_show.keys()):
            summary += f"\n{show_name}:\n"
            episodes = episodes_by_show[show_name]
            for episode_key in sorted(episodes.keys()):
                episode_data = episodes[episode_key]
                episode_name = episode_data.get("episode", episode_key)
                summary += f"\n  Episode: {episode_name}\n"
                summary += f"    Utterances: {episode_data['utterances']:,}\n"
                if episode_data.get("labels"):
                    summary += "    Label Distribution:\n"
                    for model_name in sorted(episode_data["labels"].keys()):
                        labels = episode_data["labels"][model_name]
                        total_model_labels = sum(labels.values())
                        summary += f"      {model_name}:\n"
                        for label, count in sorted(
                            labels.items(), key=lambda x: x[1], reverse=True
                        ):
                            pct = (
                                (count / total_model_labels * 100)
                                if total_model_labels > 0
                                else 0
                            )
                            summary += f"        {label:26s}: {count:6,} ({pct:5.2f}%)\n"

    summary += """
================================================================================
                                  END REPORT
================================================================================
"""
    return summary


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, path_type=Path),
    default="outputs/transcripts_with_speaker_labels_postprocessed_with_classification_labels",
    help="directory containing classified transcript JSON files",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default="outputs/analysis",
    help="output directory for summary file (default: outputs/analysis)",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="logging level",
)
def regenerate_summary(
    input_dir: Path,
    output_dir: Path,
    log_level: str,
) -> None:
    """regenerate classification summary from existing labeled transcripts."""
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    start_time = datetime.now()

    console.print("\n[bold cyan]📊 Classification Summary Regenerator[/bold cyan]\n")

    # discover classified transcript files.
    console.print("[bold blue]discovering classified transcript files...[/bold blue]")
    json_files = sorted(input_dir.rglob("*.json"))

    if not json_files:
        console.print(f"[yellow]no JSON files found in {input_dir}[/yellow]")
        sys.exit(0)

    console.print(f"[green]found {len(json_files)} classified transcript files[/green]\n")

    # initialize statistics.
    label_stats = {}  # global label counts by model.
    show_stats = {}  # per-show statistics.
    episode_stats = {}  # per-episode statistics.
    total_utterances = 0
    files_processed = 0
    files_skipped = 0

    # process files with progress bar.
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "[cyan]analyzing classified transcripts...",
            total=len(json_files),
        )

        for json_file in json_files:
            try:
                # extract show name and episode name from path.
                relative_path = json_file.relative_to(input_dir)
                show_name = relative_path.parts[0] if relative_path.parts else "unknown"
                episode_name = json_file.stem

                # load classified transcript.
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # check if file has classifications.
                segments = data.get("segments", [])
                has_classifications = any(
                    seg.get("classifications") for seg in segments
                )

                if not has_classifications:
                    files_skipped += 1
                    progress.advance(task)
                    continue

                # initialize show stats if needed.
                if show_name not in show_stats:
                    show_stats[show_name] = {
                        "episodes": 0,
                        "utterances": 0,
                        "labels": {},
                    }

                # initialize episode stats.
                episode_key = f"{show_name}/{episode_name}"
                episode_stats[episode_key] = {
                    "show": show_name,
                    "episode": episode_name,
                    "utterances": 0,
                    "labels": {},
                }

                # count classifications.
                for segment in segments:
                    classifications = segment.get("classifications", [])
                    if classifications:
                        episode_stats[episode_key]["utterances"] += 1
                        show_stats[show_name]["utterances"] += 1
                        total_utterances += 1

                        for classification in classifications:
                            model_name = classification.get("model_name")
                            label = classification.get("label")

                            if not model_name or not label:
                                continue

                            # update global label stats.
                            if model_name not in label_stats:
                                label_stats[model_name] = {}
                            label_stats[model_name][label] = (
                                label_stats[model_name].get(label, 0) + 1
                            )

                            # update show label stats.
                            if model_name not in show_stats[show_name]["labels"]:
                                show_stats[show_name]["labels"][model_name] = {}
                            show_stats[show_name]["labels"][model_name][label] = (
                                show_stats[show_name]["labels"][model_name].get(label, 0) + 1
                            )

                            # update episode label stats.
                            if model_name not in episode_stats[episode_key]["labels"]:
                                episode_stats[episode_key]["labels"][model_name] = {}
                            episode_stats[episode_key]["labels"][model_name][label] = (
                                episode_stats[episode_key]["labels"][model_name].get(label, 0) + 1
                            )

                # update episode count for show.
                show_stats[show_name]["episodes"] += 1
                files_processed += 1

            except Exception as e:
                logger.error(f"error processing {json_file}: {e}")

            progress.advance(task)

    # display statistics.
    console.print("\n[bold green]✓ analysis complete![/bold green]\n")
    console.print("[bold]statistics:[/bold]")
    console.print(f"  • files analyzed: {files_processed}")
    console.print(f"  • files skipped (no classifications): {files_skipped}")
    console.print(f"  • total utterances: {total_utterances:,}")
    console.print(f"  • shows: {len(show_stats)}")
    console.print(f"  • episodes: {len(episode_stats)}")

    # generate and save summary.
    end_time = datetime.now()
    summary = generate_summary(
        input_dir=input_dir,
        start_time=start_time,
        end_time=end_time,
        total_files=files_processed,
        label_stats=label_stats,
        show_stats=show_stats,
        episode_stats=episode_stats,
        total_utterances=total_utterances,
    )

    # save summary to file.
    summary_filename = f"classification_summary_{end_time.strftime('%Y_%m_%d_%H_%M_%S')}.txt"
    summary_path = output_dir / summary_filename

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    console.print(f"\n[bold green]summary saved to: {summary_path}[/bold green]\n")


if __name__ == "__main__":
    regenerate_summary()
