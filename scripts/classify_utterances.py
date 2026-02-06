#!/usr/bin/env python3
"""classify utterances in podcast transcripts using multiple models."""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# add src to path for imports.
# resolve symlinks and make absolute to ensure it works from any directory.
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
src_dir = project_root / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# add scripts to path for utils import.
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

import click
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from podcast_conversations.utterance_classification import (
    ClassifierConfig,
    TranscriptProcessor,
    UtteranceClassifier,
    check_classifications_exist,
    detect_classification_device,
    get_classification_device_info,
)
from utils.cli_utils import get_default_workers
from utils.rich_utils import console, setup_logging


def get_keyword_match_count(
    keywords_dir: Path, transcript_path: Path, transcripts_dir: Path
) -> int:
    """check if a transcript has keyword matches and return the count.

    looks for the corresponding keyword analysis file based on the transcript path.
    keyword files are named {transcript_stem}_keywords.json in mirrored directory structure.
    """
    relative_path = transcript_path.relative_to(transcripts_dir)
    # keyword files use same structure but with _keywords suffix.
    keyword_file = keywords_dir / relative_path.parent / f"{transcript_path.stem}_keywords.json"

    if not keyword_file.exists():
        return 0

    try:
        with open(keyword_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("match_count", 0)
    except (json.JSONDecodeError, IOError):
        return 0


def generate_summary(
    transcripts_dir: Path,
    output_dir: Path,
    config_path: Path,
    start_time: datetime,
    end_time: datetime,
    files_processed: int,
    files_skipped: int,
    files_failed: int,
    total_utterances: int,
    models_loaded: int,
    models_configured: int,
    batch_size: int,
    device: str,
    keywords_filter: bool,
    min_matches: int | None,
    label_stats: dict,
    show_stats: dict,
    episode_stats: dict,
) -> str:
    """Generate a summary report of the classification run."""
    duration = end_time - start_time
    hours, remainder = divmod(int(duration.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)

    # count output files.
    output_count = len(list(output_dir.rglob("*.json"))) if output_dir.exists() else 0

    # build filter info.
    filter_info = "None"
    if keywords_filter:
        filter_info = f"Keyword matches (min: {min_matches})"

    summary = f"""
================================================================================
                 UTTERANCE CLASSIFICATION PIPELINE - SUMMARY REPORT
================================================================================

Run Date: {start_time.strftime('%Y-%m-%d %H:%M:%S')}
Duration: {hours}h {minutes}m {seconds}s

--------------------------------------------------------------------------------
                                 CONFIGURATION
--------------------------------------------------------------------------------
Config File:        {config_path}
Device:             {device}
Batch Size:         {batch_size}
Models Loaded:      {models_loaded}/{models_configured}
Filter:             {filter_info}

--------------------------------------------------------------------------------
                                    INPUT
--------------------------------------------------------------------------------
Transcripts Directory:  {transcripts_dir}
Input Files Found:      {files_processed + files_skipped + files_failed}

--------------------------------------------------------------------------------
                              PROCESSING RESULTS
--------------------------------------------------------------------------------
Files Processed:        {files_processed}
Files Skipped:          {files_skipped} (already classified)
Files Failed:           {files_failed}
Total Utterances:       {total_utterances:,}

--------------------------------------------------------------------------------
                                   OUTPUT
--------------------------------------------------------------------------------
Output Directory:       {output_dir}
Output Files:           {output_count}

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
                            pct = (count / total_model_labels * 100) if total_model_labels > 0 else 0
                            summary += f"        {label:26s}: {count:6,} ({pct:5.2f}%)\n"

    summary += """
================================================================================
                                  END REPORT
================================================================================
"""
    return summary


@click.command()
@click.option(
    "--transcripts-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="directory containing transcript JSON files with speakers",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="output directory for classified transcripts (mirrored structure)",
)
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    default="config/classifiers.yaml",
    help="path to classifiers YAML configuration file",
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cuda", "mps", "mlx", "cpu"]),
    default="auto",
    help="device to use for inference",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="batch size for processing (auto-determined if not specified)",
)
@click.option(
    "--force",
    is_flag=True,
    help="Force reprocessing of files with existing outputs",
)
@click.option(
    "--hf-token",
    type=str,
    default=None,
    help="hugging face token for accessing gated models (or set HF_TOKEN env var)",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Logging level",
)
@click.option(
    "--workers",
    type=int,
    default=None,
    help="Number of parallel workers (default: auto-detect based on system resources)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be processed without running",
)
@click.option(
    "--keywords-dir",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="directory containing keyword analysis results; only process transcripts with matches",
)
@click.option(
    "--min-matches",
    type=int,
    default=1,
    help="minimum number of keyword matches required (default: 1)",
)
@click.option(
    "--shows",
    multiple=True,
    help="process only specific shows (can be specified multiple times); e.g., --shows the_daily",
)
def classify_utterances(
    transcripts_dir: Path,
    output_dir: Path,
    config: Path,
    device: str,
    batch_size: int | None,
    force: bool,
    hf_token: str | None,
    log_level: str,
    workers: int | None,  # noqa: ARG001  # reserved for future use
    dry_run: bool,
    keywords_dir: Path | None,
    min_matches: int,
    shows: tuple[str, ...],
) -> None:
    """
    classify utterances in podcast transcripts using transformer models.

    processes all JSON files in transcripts-dir, applies all configured
    classification models to each utterance, and outputs enriched transcripts
    to output-dir with the same directory structure.

    when --keywords-dir is specified, only transcripts that have keyword matches
    (from the keyword analysis pipeline) are processed. this allows focusing
    classification on transcripts containing mentions of target population groups.

    when --shows is specified, only transcripts from the specified show(s) are
    processed. this can be used multiple times to process multiple shows.

    Example:
        python scripts/classify_utterances.py \\
            --transcripts-dir outputs/transcripts_with_diarization_labels_postprocessed \\
            --output-dir outputs/transcripts_with_speaker_labels_postprocessed_with_classification_labels \\
            --config config/classifiers.yaml

    Example with keyword filtering:
        python scripts/classify_utterances.py \\
            --transcripts-dir outputs/transcripts_with_diarization_labels_postprocessed \\
            --output-dir outputs/classified \\
            --keywords-dir outputs/analysis/keyword_analysis \\
            --min-matches 5

    Example processing specific shows:
        python scripts/classify_utterances.py \\
            --transcripts-dir outputs/transcripts_with_diarization_labels_postprocessed \\
            --output-dir outputs/classified \\
            --shows the_daily --shows pod_save_america

    Example processing one show with keyword filtering:
        python scripts/classify_utterances.py \\
            --transcripts-dir outputs/transcripts_with_diarization_labels_postprocessed \\
            --output-dir outputs/classified \\
            --shows the_daily \\
            --keywords-dir outputs/analysis/keyword_analysis
    """
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    start_time = datetime.now()

    # auto-detect workers if not specified.
    if workers is None:
        workers = get_default_workers(memory_per_worker_gb=2.0, cpu_fraction=0.5)

    console.print("\n[bold cyan]🤖 Utterance Classification Pipeline[/bold cyan]\n")

    # resolve hugging face token (CLI flag takes precedence over env var).
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN")

    if hf_token:
        console.print("[dim]using hugging face token for authentication[/dim]\n")

    # load classifier configuration.
    console.print("[bold blue]loading classifier configuration...[/bold blue]")
    try:
        classifier_config = ClassifierConfig(config)
        models = classifier_config.get_models()
        console.print(f"[green]✓ loaded {len(models)} model configurations[/green]")

        # display models.
        console.print("\n[bold]models to apply:[/bold]")
        for model in models:
            console.print(f"  • {model.name} - {model.description}")
        console.print()

    except Exception as e:
        console.print(f"failed to load config: {e}", style="bold red", markup=False)
        sys.exit(1)

    # initialize classifier.
    console.print("[bold blue]initializing classification models...[/bold blue]")

    # resolve device and show info.
    if device == "auto":
        resolved_device = detect_classification_device()
        console.print(f"[dim]device: auto-detected → {resolved_device}[/dim]")
    else:
        resolved_device = device
        console.print(f"[dim]device: {device}[/dim]")

    # show device-specific info.
    device_info = get_classification_device_info()
    if resolved_device == "mps":
        console.print("[green]using MPS (Apple Silicon GPU)[/green]")
        console.print(
            f"[dim]system memory: {device_info.get('system_memory_gb', 0):.1f} GB unified[/dim]"
        )
    elif resolved_device == "cuda":
        console.print(f"[green]using CUDA ({device_info.get('cuda_device_name', 'GPU')})[/green]")
        console.print(f"[dim]GPU memory: {device_info.get('cuda_memory_gb', 0):.1f} GB[/dim]")
    else:
        console.print("[yellow]using CPU (slower than GPU)[/yellow]")

    try:
        classifier = UtteranceClassifier(
            models=models,
            device=device,
            batch_size=batch_size,
            hf_token=hf_token,
        )
        loaded_models = classifier.get_loaded_models()
        console.print(f"[green]✓ loaded {len(loaded_models)}/{len(models)} models[/green]")

        if len(loaded_models) < len(models):
            failed = set(m.name for m in models) - set(loaded_models)
            console.print(f"[yellow]⚠ failed to load: {', '.join(failed)}[/yellow]")

        console.print()

    except Exception as e:
        console.print(f"failed to initialize classifier: {e}", style="bold red", markup=False)
        sys.exit(1)

    # discover files to process.
    console.print("[bold blue]discovering transcript files...[/bold blue]")
    json_files = sorted(Path(transcripts_dir).rglob("*.json"))

    if not json_files:
        console.print(f"[yellow]no JSON files found in {transcripts_dir}[/yellow]")
        sys.exit(0)

    console.print(f"[green]found {len(json_files)} transcript files[/green]")

    # filter by specific shows if requested.
    if shows:
        console.print(f"[bold blue]filtering to specific shows: {', '.join(shows)}[/bold blue]")
        show_filtered_files = []
        for json_file in json_files:
            # extract show name from path (assumes structure: .../show_name/episode.json).
            try:
                relative_path = json_file.relative_to(transcripts_dir)
                show_name = relative_path.parts[0] if relative_path.parts else ""
                if show_name in shows:
                    show_filtered_files.append(json_file)
            except ValueError:
                # file not relative to transcripts_dir, skip.
                continue

        if not show_filtered_files:
            console.print(
                f"[yellow]no files found for shows: {', '.join(shows)}[/yellow]"
            )
            sys.exit(0)

        console.print(
            f"[green]✓ filtered to {len(show_filtered_files)} files "
            f"from {len(shows)} show(s)[/green]"
        )
        json_files = show_filtered_files

    # ensure output directory exists.
    output_dir.mkdir(parents=True, exist_ok=True)

    # filter by keyword matches if keywords-dir is specified.
    no_keywords_count = 0
    if keywords_dir:
        console.print(
            f"[bold blue]filtering by keyword matches (min: {min_matches})...[/bold blue]"
        )
        filtered_files = []
        for input_path in json_files:
            match_count = get_keyword_match_count(keywords_dir, input_path, transcripts_dir)
            if match_count >= min_matches:
                filtered_files.append(input_path)
            else:
                no_keywords_count += 1

        console.print(
            f"[green]✓ {len(filtered_files)} files have ≥{min_matches} keyword matches[/green]"
        )
        if no_keywords_count > 0:
            console.print(
                f"[yellow]skipping {no_keywords_count} files "
                f"without sufficient keyword matches[/yellow]"
            )
        json_files = filtered_files

    if not json_files:
        console.print("[yellow]no files match the keyword filter - nothing to process[/yellow]")
        sys.exit(0)

    # check for existing classifications if not forcing.
    files_to_process = []
    files_with_partial = []
    existing_count = 0
    
    # get expected model names from loaded models.
    expected_model_names = set(classifier.get_loaded_models())

    console.print(f"[bold blue]checking existing classifications (expecting {len(expected_model_names)} models per utterance)...[/bold blue]")
    
    for input_path in json_files:
        relative_path = input_path.relative_to(transcripts_dir)
        output_path = output_dir / relative_path
        
        if force:
            files_to_process.append(input_path)
        elif check_classifications_exist(output_path, expected_model_names):
            existing_count += 1
        else:
            # check if file has any classifications (partial or missing).
            has_any = check_classifications_exist(output_path, expected_models=None)
            if has_any:
                files_with_partial.append(input_path)
            files_to_process.append(input_path)

    if existing_count > 0:
        console.print(
            f"[green]✓ {existing_count} files have complete classifications ({len(expected_model_names)} models)[/green]"
        )
    
    if files_with_partial:
        console.print(
            f"[yellow]⚠ {len(files_with_partial)} files have partial classifications (will be updated)[/yellow]"
        )

    console.print(f"[green]will process {len(files_to_process)} files[/green]")

    if not files_to_process:
        console.print(
            "[yellow]all files already have classifications - nothing to process[/yellow]"
        )
        console.print("  use --force to reprocess\n")
        sys.exit(0)

    # dry-run mode: show what would be processed.
    if dry_run:
        console.print(
            "\n[bold yellow]dry run mode - files that would be processed:[/bold yellow]\n"
        )
        for i, input_path in enumerate(files_to_process[:10], 1):
            relative_path = input_path.relative_to(transcripts_dir)
            console.print(f"  {i}. {relative_path}")
        if len(files_to_process) > 10:
            console.print(f"  ... and {len(files_to_process) - 10} more files")
        console.print()
        return

    console.print()

    # process files with progress bar.
    processor = TranscriptProcessor(classifier)

    stats = {
        "files_processed": 0,
        "files_skipped": 0,
        "files_failed": 0,
        "total_utterances": 0,
    }

    # statistics for labels, shows, and episodes.
    label_stats = {}  # {model_name: {label: count}}
    show_stats = {}  # {show_name: {episodes, utterances, labels: {model: {label: count}}}}
    episode_stats = {}  # {episode_key: {show, episode, utterances, labels: {model: {label}}}}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=4,
    ) as progress:
        task = progress.add_task(
            "[cyan]classifying | starting...",
            total=len(files_to_process),
        )

        for input_path in files_to_process:
            # compute relative path and output path.
            relative_path = input_path.relative_to(transcripts_dir)
            output_path = output_dir / relative_path

            # show current file.
            show_name = relative_path.parts[0] if relative_path.parts else "unknown"
            file_name = input_path.stem

            try:
                file_stats = processor.process_file(input_path, output_path, skip_existing=False)

                if file_stats.get("skipped", False):
                    stats["files_skipped"] += 1
                else:
                    stats["files_processed"] += 1
                    stats["total_utterances"] += file_stats["utterances_classified"]

                    # collect label statistics from the output file.
                    if output_path.exists():
                        with open(output_path, "r", encoding="utf-8") as f:
                            classified_data = json.load(f)

                        # initialize show stats if needed.
                        if show_name not in show_stats:
                            show_stats[show_name] = {
                                "episodes": 0,
                                "utterances": 0,
                                "labels": {},
                            }

                        # initialize episode stats.
                        episode_key = f"{show_name}/{file_name}"
                        episode_stats[episode_key] = {
                            "show": show_name,
                            "episode": file_name,
                            "utterances": 0,
                            "labels": {},
                        }

                        # count classifications.
                        for segment in classified_data.get("segments", []):
                            classifications = segment.get("classifications", [])
                            if classifications:
                                episode_stats[episode_key]["utterances"] += 1
                                show_stats[show_name]["utterances"] += 1

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
                                        show_stats[show_name]["labels"][model_name].get(label, 0)
                                        + 1
                                    )

                                    # update episode label stats.
                                    if model_name not in episode_stats[episode_key]["labels"]:
                                        episode_stats[episode_key]["labels"][model_name] = {}
                                    episode_stats[episode_key]["labels"][model_name][label] = (
                                        episode_stats[episode_key]["labels"][model_name].get(label, 0)
                                        + 1
                                    )

                        # update episode count for show.
                        show_stats[show_name]["episodes"] += 1

                    logger.info(
                        f"✓ {show_name}/{file_name}: "
                        f"{file_stats['utterances_classified']} utterances"
                    )

            except Exception as e:
                stats["files_failed"] += 1
                logger.error(f"✗ {show_name}/{file_name}: {e}")

            # update progress with running statistics.
            progress.update(
                task,
                description=(
                    f"[cyan]classifying | "
                    f"{stats['total_utterances']:,} utterances in {stats['files_processed']} files"
                ),
                advance=1,
            )

    # display final statistics.
    console.print("\n[bold green]✓ classification complete![/bold green]\n")
    console.print("[bold]statistics:[/bold]")
    console.print(f"  • files processed: {stats['files_processed']}")
    if stats['files_skipped'] > 0:
        console.print(f"  • files skipped: {stats['files_skipped']}")
    console.print(f"  • files failed: {stats['files_failed']}")
    console.print(f"  • utterances classified: {stats['total_utterances']}")
    console.print(f"  • models applied per utterance: {len(loaded_models)}")
    if batch_size is None:
        console.print(f"  • batch size: auto-determined ({classifier.batch_size})")
    else:
        console.print(f"  • batch size: {batch_size}")

    # generate and save summary.
    end_time = datetime.now()
    summary = generate_summary(
        transcripts_dir=transcripts_dir,
        output_dir=output_dir,
        config_path=config,
        start_time=start_time,
        end_time=end_time,
        files_processed=stats["files_processed"],
        files_skipped=existing_count,
        files_failed=stats["files_failed"],
        total_utterances=stats["total_utterances"],
        models_loaded=len(loaded_models),
        models_configured=len(models),
        batch_size=classifier.batch_size,
        device=device,
        keywords_filter=keywords_dir is not None,
        min_matches=min_matches if keywords_dir else None,
        label_stats=label_stats,
        show_stats=show_stats,
        episode_stats=episode_stats,
    )

    # save summary to outputs/analysis/ directory.
    analysis_dir = Path("outputs/analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary_filename = f"classification_{end_time.strftime('%Y_%m_%d_%H_%M_%S')}.txt"
    summary_path = analysis_dir / summary_filename

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    console.print(f"\n[bold green]output saved to: {output_dir}[/bold green]")
    console.print(f"[bold green]summary saved to: {summary_path}[/bold green]\n")


if __name__ == "__main__":
    classify_utterances()
