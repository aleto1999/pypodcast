#!/usr/bin/env python3
"""semantic analysis of podcast transcripts using embeddings method."""
# testing github ?

import sys
from pathlib import Path
from typing import Optional

# add src and scripts to path for imports.
# resolve symlinks and make absolute to ensure it works from any directory.
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
src_dir = project_root / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

import logging

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
from rich.prompt import FloatPrompt, Prompt

from podcast_conversations.embeddings_method import (
    analyze_multiple_transcripts,
    check_embeddings_exist,
    detect_device,
    discover_shows,
    display_available_shows,
    generate_taxonomy_embeddings,
    get_mlx_status,
    get_optimal_batch_size,
    is_apple_silicon,
    load_embedding_model,
    load_taxonomy,
    process_transcript_file,
    save_analysis_results,
)
from utils.rich_utils import console, setup_logging


def prompt_show_selection(show_names: list[str]) -> str | None:
    """
    prompt user to select a show for analysis.

    returns:
        show name if single show selected, or None if "all shows" selected
    """
    console.print("\n[bold cyan]select a show to analyze:[/bold cyan]\n")

    console.print("  [0] [bold]All shows (batch process)[/bold]")
    console.print()

    for idx, show_name in enumerate(show_names, 1):
        console.print(f"  [{idx}] {show_name}")

    console.print()

    while True:
        selection = Prompt.ask(
            "enter show number or name (0 for all)", default="1", show_default=True
        )

        # check for "all shows" option.
        if selection == "0" or selection.lower() in ["all", "all shows"]:
            return None

        # try as index.
        try:
            idx = int(selection) - 1
            if 0 <= idx < len(show_names):
                return show_names[idx]
        except ValueError:
            pass

        # try as name.
        if selection in show_names:
            return selection

        console.print("[red]invalid selection. try again.[/red]")


def get_similarity_threshold() -> float:
    """prompt user for similarity threshold."""
    console.print("\n[bold cyan]🎯 semantic similarity configuration[/bold cyan]\n")
    console.print("set the similarity threshold:")
    console.print("  • 0.5-0.6: strict matching (high precision, few matches)")
    console.print("  • 0.3-0.4: balanced (recommended for general use)")
    console.print("  • 0.2-0.3: lenient matching (high recall, more false positives)")
    console.print()
    console.print("[dim]note: embeddings-based similarity typically yields lower scores than exact "
                  "text matching. scores of 0.3-0.4 indicate meaningful semantic similarity.[/dim]\n")

    threshold = FloatPrompt.ask(
        "enter similarity threshold", default=0.35, show_default=True
    )

    if not 0.0 <= threshold <= 1.0:
        console.print("[red]threshold must be between 0.0 and 1.0[/red]")
        return get_similarity_threshold()

    return threshold


def count_existing_embeddings(show_name: str, transcript_paths: list[Path]) -> tuple[int, int]:
    """
    count how many transcript files already have embeddings.

    returns:
        tuple of (files_with_embeddings, total_files)
    """
    embeddings_dir = Path(f"outputs/analysis/embeddings_method/{show_name}/embeddings")

    if not embeddings_dir.exists():
        return (0, len(transcript_paths))

    files_with_embeddings = 0

    for transcript_path in transcript_paths:
        embedding_file = embeddings_dir / transcript_path.name
        if check_embeddings_exist(embedding_file):
            files_with_embeddings += 1

    return (files_with_embeddings, len(transcript_paths))


def generate_transcript_embeddings(
    show_name: str,
    transcript_paths: list[Path],
    model,
    device: str,
    force: bool = False,
    batch_size_override: int | None = None,
) -> list[Path]:
    """generate or load embeddings for transcript files."""
    embeddings_dir = Path(f"outputs/analysis/embeddings_method/{show_name}/embeddings")
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # check how many files already have embeddings.
    existing_count, total_count = count_existing_embeddings(show_name, transcript_paths)

    if not force and existing_count > 0:
        console.print(
            f"\n[dim]found existing embeddings for {existing_count}/{total_count} files[/dim]"
        )
        if existing_count == total_count:
            console.print("[green]✓ all files already have embeddings (skipping generation)[/green]")
            console.print("  use --force-regenerate to reprocess")
            return [embeddings_dir / p.name for p in transcript_paths]

    # determine optimal batch size.
    if batch_size_override:
        batch_size = batch_size_override
        console.print(f"[dim]using batch size: {batch_size} (override)[/dim]")
    else:
        batch_size = get_optimal_batch_size(device, model)

    files_to_process = total_count if force else (total_count - existing_count)
    console.print(
        f"\n[cyan]generating embeddings for {files_to_process} files (batch size: {batch_size})...[/cyan]"
    )

    embedding_paths = []
    processed_count = 0
    skipped_count = 0
    total_utterances = 0

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
            "[cyan]embeddings | starting...",
            total=len(transcript_paths),
        )

        for transcript_path in transcript_paths:
            output_path = embeddings_dir / transcript_path.name

            try:
                was_processed = process_transcript_file(
                    transcript_path,
                    output_path,
                    model,
                    batch_size=batch_size,
                    skip_existing=not force,
                )
                embedding_paths.append(output_path)

                if was_processed:
                    processed_count += 1
                    # count utterances from the output file.
                    try:
                        with open(output_path, encoding="utf-8") as f:
                            import json
                            data = json.load(f)
                            total_utterances += len(data.get("segments", []))
                    except Exception:
                        pass
                else:
                    skipped_count += 1

            except Exception as e:
                console.print(f"[red]error processing {transcript_path.name}: {e}[/red]")

            # update progress with running statistics.
            progress.update(
                task,
                description=(
                    f"[cyan]embeddings | "
                    f"{total_utterances:,} utterances in {processed_count} files"
                ),
                advance=1,
            )

    if skipped_count > 0:
        console.print(f"✓ processed {processed_count} files, skipped {skipped_count} existing")
    else:
        console.print(f"✓ processed {processed_count} files")

    return embedding_paths


def process_all_shows(
    force_regenerate: bool = False,
    transcripts_dir: Optional[str] = None,
    threshold: Optional[float] = None,
    shows: Optional[dict] = None,
    dry_run: bool = False,
    device_override: Optional[str] = None,
    batch_size_override: Optional[int] = None,
    hf_token: Optional[str] = None,
) -> Optional[str]:
    """process all available shows in batch mode."""

    console.print("\n[bold cyan]🎤 semantic analysis using embeddings method (batch mode)[/bold cyan]\n")

    # step 1: discover shows (if not already provided).
    if shows is None:
        console.print("[dim]discovering shows...[/dim]")
        shows = discover_shows(base_dir=transcripts_dir)

        if not shows:
            console.print("[bold red]❌ no shows with transcripts found[/bold red]")
            if transcripts_dir:
                console.print(f"\nensure transcripts exist in: {transcripts_dir}")
            else:
                console.print("\nensure transcripts exist in:")
                console.print("  • outputs/transcripts_with_speaker_labels_postprocessed_with_classification_labels/ (with classifications)")
                console.print("  • outputs/transcripts_with_diarization_labels_postprocessed/ (fallback)")
            return None

        # show which directory is being used.
        first_show_files = list(shows.values())[0]
        if first_show_files:
            input_dir = first_show_files[0].parent.parent
            console.print(f"[dim]using transcripts from: {input_dir}[/dim]\n")

        display_available_shows(shows)

    console.print(f"\n[green]✓ processing all {len(shows)} shows[/green]\n")

    # dry-run mode: show what would be processed.
    if dry_run:
        console.print("[bold yellow]dry run mode - shows that would be processed:[/bold yellow]\n")
        total_files = 0
        for i, (show_name, files) in enumerate(list(shows.items())[:10], 1):
            console.print(f"  {i}. {show_name} ({len(files)} files)")
            total_files += len(files)
        if len(shows) > 10:
            remaining = list(shows.items())[10:]
            for show_name, files in remaining:
                total_files += len(files)
            console.print(f"  ... and {len(shows) - 10} more shows")
        console.print(f"\n  total files: {total_files}")
        console.print()
        return "DRY_RUN"

    # step 2: load taxonomy.
    console.print("[dim]loading taxonomy...[/dim]")
    try:
        taxonomy = load_taxonomy()
    except FileNotFoundError as e:
        console.print(f"[bold red]❌ {e}[/bold red]")
        console.print("\nensure taxonomy configuration exists at: config/taxonomy.yml")
        return None

    # step 3: initialize model.
    console.print("\n[dim]initializing embedding model...[/dim]")

    # show mlx status on Apple Silicon.
    if is_apple_silicon():
        status = get_mlx_status()
        if status["mlx_embeddings_available"]:
            console.print("[dim]mlx embeddings available for optimized execution[/dim]")
        else:
            cmd = status["install_command"]
            console.print(f"[dim]tip: install mlx for faster execution: {cmd}[/dim]")

    device = device_override if device_override else detect_device()
    if device_override:
        console.print(f"[dim]using device: {device} (override)[/dim]")
    model = load_embedding_model(device, hf_token=hf_token)

    # step 4: generate taxonomy embeddings.
    console.print("\n[dim]generating taxonomy embeddings...[/dim]")
    taxonomy_embeddings = generate_taxonomy_embeddings(taxonomy, model)
    console.print(f"✓ generated {len(taxonomy_embeddings)} taxonomy embeddings")

    # step 5: get threshold (prompt if not provided).
    if threshold is None:
        threshold = get_similarity_threshold()
    else:
        if not 0.0 <= threshold <= 1.0:
            console.print("[bold red]❌ threshold must be between 0.0 and 1.0[/bold red]")
            return None
        console.print(f"\n[cyan]using threshold: {threshold}[/cyan]\n")

    # step 6: process each show.
    overall_stats = {
        "shows_processed": 0,
        "shows_failed": 0,
        "total_files": 0,
        "total_matches": 0,
    }

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
        main_task = progress.add_task(
            "[cyan]analysis | starting...",
            total=len(shows),
        )

        for show_name, transcript_paths in shows.items():
            try:
                # generate embeddings for this show.
                embedding_paths = generate_transcript_embeddings(
                    show_name, transcript_paths, model, device, force=force_regenerate,
                    batch_size_override=batch_size_override
                )

                if not embedding_paths:
                    console.print(f"[yellow]⚠ skipped {show_name}: no embeddings generated[/yellow]")
                    overall_stats["shows_failed"] += 1
                    progress.update(
                        main_task,
                        description=(
                            f"[cyan]analysis | "
                            f"{overall_stats['total_files']} files, "
                            f"{overall_stats['total_matches']:,} matches"
                        ),
                        advance=1,
                    )
                    continue

                # perform analysis.
                results = analyze_multiple_transcripts(embedding_paths, taxonomy_embeddings, threshold)

                # save results.
                save_analysis_results(show_name, results, threshold, "all-MiniLM-L6-v2")

                # update stats.
                total_matches = sum(r["matches_found"] for r in results)
                overall_stats["shows_processed"] += 1
                overall_stats["total_files"] += len(results)
                overall_stats["total_matches"] += total_matches

                console.print(f"[green]✓ {show_name}: {len(results)} files, {total_matches} matches[/green]")

            except Exception as e:
                console.print(f"[red]✗ {show_name}: {e}[/red]")
                overall_stats["shows_failed"] += 1

            # update progress with running statistics.
            progress.update(
                main_task,
                description=(
                    f"[cyan]analysis | "
                    f"{overall_stats['total_files']} files, "
                    f"{overall_stats['total_matches']:,} matches"
                ),
                advance=1,
            )

    # step 7: display overall summary.
    console.print("\n[bold green]✓ batch analysis complete![/bold green]\n")
    console.print("[bold]overall statistics:[/bold]")
    console.print(f"  • shows processed: {overall_stats['shows_processed']}")
    console.print(f"  • shows failed: {overall_stats['shows_failed']}")
    console.print(f"  • total files analyzed: {overall_stats['total_files']}")
    console.print(f"  • total matches: {overall_stats['total_matches']}")
    console.print(f"  • similarity threshold: {threshold}")

    return "SUCCESS"


def run_embeddings_analysis(
    force_regenerate: bool = False,
    transcripts_dir: Optional[str] = None,
    dry_run: bool = False,
    device_override: Optional[str] = None,
    batch_size_override: Optional[int] = None,
    hf_token: Optional[str] = None,
) -> Optional[str]:
    """main orchestrator for embeddings analysis pipeline."""

    console.print("\n[bold cyan]🎤 semantic analysis using embeddings method[/bold cyan]\n")

    # step 1: discover shows.
    console.print("[dim]discovering shows...[/dim]")
    shows = discover_shows(base_dir=transcripts_dir)

    if not shows:
        console.print("[bold red]❌ no shows with transcripts found[/bold red]")
        if transcripts_dir:
            console.print(f"\nensure transcripts exist in: {transcripts_dir}")
        else:
            console.print("\nensure transcripts exist in:")
            console.print("  • outputs/transcripts_with_speaker_labels_postprocessed_with_classification_labels/ (with classifications)")
            console.print("  • outputs/transcripts_with_diarization_labels_postprocessed/ (fallback)")
        return None

    # show which directory is being used.
    if shows:
        first_show_files = list(shows.values())[0]
        if first_show_files:
            input_dir = first_show_files[0].parent.parent
            console.print(f"[dim]using transcripts from: {input_dir}[/dim]\n")

    display_available_shows(shows)

    # dry-run mode: show what would be processed.
    if dry_run:
        console.print("\n[bold yellow]dry run mode - shows that would be processed:[/bold yellow]\n")
        total_files = 0
        for i, (show_name, files) in enumerate(list(shows.items())[:10], 1):
            console.print(f"  {i}. {show_name} ({len(files)} files)")
            total_files += len(files)
        if len(shows) > 10:
            remaining = list(shows.items())[10:]
            for show_name, files in remaining:
                total_files += len(files)
            console.print(f"  ... and {len(shows) - 10} more shows")
        console.print(f"\n  total files: {total_files}")
        console.print()
        return "DRY_RUN"

    # step 2: user selects show (or all shows).
    show_name = prompt_show_selection(list(shows.keys()))

    # if user selected "all shows", redirect to batch processing.
    if show_name is None:
        console.print("\n[green]✓ selected: all shows[/green]")
        threshold = get_similarity_threshold()
        return process_all_shows(
            force_regenerate=force_regenerate,
            transcripts_dir=transcripts_dir,
            threshold=threshold,
            shows=shows,  # pass already-discovered shows to avoid rediscovering
            device_override=device_override,
            batch_size_override=batch_size_override,
            hf_token=hf_token,
        )

    console.print(f"\n[green]✓ selected: {show_name}[/green]")

    transcript_paths = shows[show_name]

    # step 3: load taxonomy.
    console.print("\n[dim]loading taxonomy...[/dim]")
    try:
        taxonomy = load_taxonomy()
    except FileNotFoundError as e:
        console.print(f"[bold red]❌ {e}[/bold red]")
        console.print(
            "\nensure taxonomy configuration exists at: config/taxonomy.yml"
        )
        return None

    # step 4: initialize model.
    console.print("\n[dim]initializing embedding model...[/dim]")

    # show mlx status on Apple Silicon.
    if is_apple_silicon():
        status = get_mlx_status()
        if status["mlx_embeddings_available"]:
            console.print("[dim]mlx embeddings available for optimized execution[/dim]")
        else:
            cmd = status["install_command"]
            console.print(f"[dim]tip: install mlx for faster execution: {cmd}[/dim]")

    device = device_override if device_override else detect_device()
    if device_override:
        console.print(f"[dim]using device: {device} (override)[/dim]")
    model = load_embedding_model(device, hf_token=hf_token)

    # step 5: generate taxonomy embeddings.
    console.print("\n[dim]generating taxonomy embeddings...[/dim]")
    taxonomy_embeddings = generate_taxonomy_embeddings(taxonomy, model)
    console.print(f"✓ generated {len(taxonomy_embeddings)} taxonomy embeddings")

    # step 6: process transcripts (generate utterance embeddings).
    embedding_paths = generate_transcript_embeddings(
        show_name, transcript_paths, model, device, force=force_regenerate,
        batch_size_override=batch_size_override
    )

    if not embedding_paths:
        console.print("[bold red]❌ no embeddings generated[/bold red]")
        return None

    # step 7: get threshold.
    threshold = get_similarity_threshold()

    # step 8: perform analysis.
    console.print(
        f"\n[cyan]analyzing {len(embedding_paths)} files with threshold {threshold}...[/cyan]\n"
    )
    results = analyze_multiple_transcripts(embedding_paths, taxonomy_embeddings, threshold)

    # step 9: save results.
    save_analysis_results(show_name, results, threshold, "all-MiniLM-L6-v2")

    # step 10: display summary.
    total_matches = sum(r["matches_found"] for r in results)
    console.print("\n[bold green]✓ analysis complete![/bold green]")
    console.print(f"  • files analyzed: {len(results)}")
    console.print(f"  • total matches: {total_matches}")

    return "SUCCESS"


@click.command()
@click.option(
    "--transcripts-dir",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="directory containing transcripts (defaults to transcripts_with_speaker_labels_postprocessed_with_classification_labels if available)",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default="outputs/analysis/embeddings_method",
    help="output directory for analysis results",
)
@click.option(
    "--threshold",
    type=float,
    default=None,
    help="similarity threshold (0.0-1.0). if not provided, prompts interactively",
)
@click.option(
    "--device",
    type=click.Choice(["auto", "mlx", "cuda", "mps", "cpu"]),
    default="auto",
    help="device for embeddings. 'mlx' uses Apple Silicon optimization (auto-detected)",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="batch size for embedding generation (auto-detected based on device memory if not specified)",
)
@click.option(
    "--hf-token",
    type=str,
    envvar="HF_TOKEN",
    help="HuggingFace token for gated embedding models (or set HF_TOKEN env var)",
)
@click.option(
    "--workers",
    type=int,
    default=None,
    help="Number of parallel workers (auto-detected if not set)",
)
@click.option(
    "--force",
    is_flag=True,
    help="Force reprocessing of files with existing outputs",
)
@click.option(
    "--process-all",
    is_flag=True,
    help="process all available shows instead of selecting one interactively",
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
    transcripts_dir: Path | None,
    output_dir: Path,
    threshold: float | None,
    device: str,
    batch_size: int | None,
    hf_token: str | None,
    workers: int | None,
    force: bool,
    process_all: bool,
    log_level: str,
    dry_run: bool,
) -> None:
    """
    semantic analysis of podcast transcripts using embeddings.

    uses sentence embeddings to find semantically similar utterances
    to a taxonomy of dehumanization strategies.

    Example:
        python scripts/analyze_embeddings.py --transcripts-dir outputs/transcripts_with_diarization_labels_postprocessed
        python scripts/analyze_embeddings.py --process-all --threshold 0.35
        python scripts/analyze_embeddings.py --device cuda --batch-size 64 --hf-token $HF_TOKEN
    """
    setup_logging(log_level)

    try:
        # route to batch or interactive mode based on --process-all flag.
        if process_all:
            result = process_all_shows(
                force_regenerate=force,
                transcripts_dir=str(transcripts_dir) if transcripts_dir else None,
                threshold=threshold,
                dry_run=dry_run,
                device_override=device if device != "auto" else None,
                batch_size_override=batch_size,
                hf_token=hf_token,
            )
        else:
            result = run_embeddings_analysis(
                force_regenerate=force,
                transcripts_dir=str(transcripts_dir) if transcripts_dir else None,
                dry_run=dry_run,
                device_override=device if device != "auto" else None,
                batch_size_override=batch_size,
                hf_token=hf_token,
            )

        if result is None:
            sys.exit(1)

    except KeyboardInterrupt:
        console.print("\n\n[yellow]⚠️  analysis cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        # use markup=False to prevent Rich from interpreting error message as markup.
        console.print(f"\n❌ error: {e}", style="bold red", markup=False)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
