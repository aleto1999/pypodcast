#!/usr/bin/env python3
"""
speaker statistics generation script for podcast transcripts.

generates both fine-grained per-episode speaker statistics and
a general summary file aggregating statistics across all podcasts.

usage:
    # run with default settings.
    uv run python scripts/generate_speaker_statistics.py

    # run with custom input/output directories.
    uv run python scripts/generate_speaker_statistics.py \
        --transcripts-dir outputs/transcripts_with_speaker_labels_postprocessed_with_classification_labels \
        --output-dir outputs/analysis/speaker_statistics

    # process specific shows only.
    uv run python scripts/generate_speaker_statistics.py --shows the_daily --shows latinus

    # force reprocessing of all files.
    uv run python scripts/generate_speaker_statistics.py --force
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

# add src to path for imports.
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
src_dir = project_root / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import click
from rich.panel import Panel
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
from rich.table import Table

# add scripts to path for utils import.
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from utils.rich_utils import console, setup_logging


def get_default_workers() -> int:
    """get default number of workers based on available resources."""
    # check for SLURM allocation first (HPC environments).
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            return max(1, int(slurm_cpus))
        except ValueError:
            pass

    # fallback to system detection.
    cpu_count = os.cpu_count() or 1

    # use 80% of available cores.
    return max(1, int(cpu_count * 0.8))


@dataclass
class SpeakerStats:
    """statistics for a single speaker in an episode."""

    speaker_id: str
    segment_count: int = 0
    word_count: int = 0
    total_duration_seconds: float = 0.0
    avg_words_per_segment: float = 0.0
    avg_segment_duration_seconds: float = 0.0
    percentage_of_words: float = 0.0
    percentage_of_duration: float = 0.0
    percentage_of_segments: float = 0.0
    first_appearance_seconds: float = 0.0
    last_appearance_seconds: float = 0.0


@dataclass
class EpisodeStats:
    """statistics for an entire episode."""

    episode_name: str
    show_name: str
    audio_file: str = ""
    language: str = ""
    total_segments: int = 0
    total_words: int = 0
    total_duration_seconds: float = 0.0
    unique_speaker_count: int = 0
    speakers: list[SpeakerStats] = field(default_factory=list)
    dominant_speaker: str = ""
    dominant_speaker_percentage: float = 0.0
    avg_segment_duration_seconds: float = 0.0
    avg_words_per_segment: float = 0.0
    processing_timestamp: str = ""


@dataclass
class ShowSummary:
    """summary statistics for a show (podcast)."""

    show_name: str
    episode_count: int = 0
    total_segments: int = 0
    total_words: int = 0
    total_duration_hours: float = 0.0
    avg_episode_duration_minutes: float = 0.0
    avg_speakers_per_episode: float = 0.0
    unique_speakers_total: int = 0
    most_common_speaker_count: int = 0  # most common number of speakers per episode.


@dataclass
class CorpusSummary:
    """summary statistics for the entire corpus."""

    total_shows: int = 0
    total_episodes: int = 0
    total_segments: int = 0
    total_words: int = 0
    total_duration_hours: float = 0.0
    avg_episode_duration_minutes: float = 0.0
    avg_speakers_per_episode: float = 0.0
    shows: list[ShowSummary] = field(default_factory=list)
    generation_timestamp: str = ""
    input_directory: str = ""
    output_directory: str = ""


def count_words(text: str) -> int:
    """count words in text."""
    if not text:
        return 0
    return len(text.split())


def process_single_transcript(transcript_path: Path) -> EpisodeStats | None:
    """
    process a single transcript file and extract speaker statistics.

    Args:
        transcript_path: path to transcript JSON file.

    Returns:
        EpisodeStats object or None on error.
    """
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        segments = data.get("segments", [])
        if not segments:
            return None

        # extract metadata.
        audio_file = data.get("audio_file", "")
        language = data.get("language", "")
        show_name = transcript_path.parent.name
        episode_name = transcript_path.stem

        # collect per-speaker statistics.
        speaker_data: dict[str, dict] = defaultdict(
            lambda: {
                "segment_count": 0,
                "word_count": 0,
                "total_duration": 0.0,
                "first_appearance": float("inf"),
                "last_appearance": 0.0,
            }
        )

        total_words = 0
        total_duration = 0.0

        for segment in segments:
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment.get("text", "")
            start = segment.get("start", 0.0)
            end = segment.get("end", 0.0)
            duration = max(0.0, end - start)

            words = count_words(text)
            total_words += words
            total_duration += duration

            speaker_data[speaker]["segment_count"] += 1
            speaker_data[speaker]["word_count"] += words
            speaker_data[speaker]["total_duration"] += duration
            speaker_data[speaker]["first_appearance"] = min(
                speaker_data[speaker]["first_appearance"], start
            )
            speaker_data[speaker]["last_appearance"] = max(
                speaker_data[speaker]["last_appearance"], end
            )

        # build speaker stats list.
        speakers_list: list[SpeakerStats] = []
        total_segments = len(segments)

        for speaker_id, stats in speaker_data.items():
            seg_count = stats["segment_count"]
            word_count = stats["word_count"]
            duration = stats["total_duration"]

            speaker_stats = SpeakerStats(
                speaker_id=speaker_id,
                segment_count=seg_count,
                word_count=word_count,
                total_duration_seconds=round(duration, 3),
                avg_words_per_segment=round(word_count / seg_count, 2) if seg_count > 0 else 0.0,
                avg_segment_duration_seconds=round(duration / seg_count, 3) if seg_count > 0 else 0.0,
                percentage_of_words=round((word_count / total_words) * 100, 2) if total_words > 0 else 0.0,
                percentage_of_duration=round((duration / total_duration) * 100, 2) if total_duration > 0 else 0.0,
                percentage_of_segments=round((seg_count / total_segments) * 100, 2) if total_segments > 0 else 0.0,
                first_appearance_seconds=round(stats["first_appearance"], 3) if stats["first_appearance"] != float("inf") else 0.0,
                last_appearance_seconds=round(stats["last_appearance"], 3),
            )
            speakers_list.append(speaker_stats)

        # sort speakers by word count (descending).
        speakers_list.sort(key=lambda s: s.word_count, reverse=True)

        # determine dominant speaker.
        dominant_speaker = speakers_list[0].speaker_id if speakers_list else ""
        dominant_percentage = speakers_list[0].percentage_of_words if speakers_list else 0.0

        episode_stats = EpisodeStats(
            episode_name=episode_name,
            show_name=show_name,
            audio_file=audio_file,
            language=language,
            total_segments=total_segments,
            total_words=total_words,
            total_duration_seconds=round(total_duration, 3),
            unique_speaker_count=len(speakers_list),
            speakers=speakers_list,
            dominant_speaker=dominant_speaker,
            dominant_speaker_percentage=dominant_percentage,
            avg_segment_duration_seconds=round(total_duration / total_segments, 3) if total_segments > 0 else 0.0,
            avg_words_per_segment=round(total_words / total_segments, 2) if total_segments > 0 else 0.0,
            processing_timestamp=datetime.now().isoformat(),
        )

        return episode_stats

    except (json.JSONDecodeError, OSError, KeyError) as e:
        logging.warning(f"error processing {transcript_path}: {e}")
        return None


def save_episode_stats(stats: EpisodeStats, output_path: Path) -> None:
    """save episode statistics to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # convert dataclass to dict, handling nested dataclasses.
    stats_dict = asdict(stats)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats_dict, f, indent=2, ensure_ascii=False)


def generate_corpus_summary(
    all_episode_stats: list[EpisodeStats],
    input_dir: Path,
    output_dir: Path,
) -> CorpusSummary:
    """generate summary statistics for the entire corpus."""
    # group by show.
    shows_data: dict[str, list[EpisodeStats]] = defaultdict(list)
    for episode in all_episode_stats:
        shows_data[episode.show_name].append(episode)

    show_summaries: list[ShowSummary] = []
    total_segments = 0
    total_words = 0
    total_duration_seconds = 0.0
    total_speakers = 0

    for show_name, episodes in sorted(shows_data.items()):
        show_segments = sum(ep.total_segments for ep in episodes)
        show_words = sum(ep.total_words for ep in episodes)
        show_duration = sum(ep.total_duration_seconds for ep in episodes)
        show_speakers = sum(ep.unique_speaker_count for ep in episodes)

        # count speaker distribution.
        speaker_counts = [ep.unique_speaker_count for ep in episodes]
        most_common_count = max(set(speaker_counts), key=speaker_counts.count) if speaker_counts else 0

        # unique speakers across show (approximate - by ID).
        all_speaker_ids = set()
        for ep in episodes:
            for speaker in ep.speakers:
                all_speaker_ids.add(speaker.speaker_id)

        show_summary = ShowSummary(
            show_name=show_name,
            episode_count=len(episodes),
            total_segments=show_segments,
            total_words=show_words,
            total_duration_hours=round(show_duration / 3600, 2),
            avg_episode_duration_minutes=round((show_duration / len(episodes)) / 60, 2) if episodes else 0.0,
            avg_speakers_per_episode=round(show_speakers / len(episodes), 2) if episodes else 0.0,
            unique_speakers_total=len(all_speaker_ids),
            most_common_speaker_count=most_common_count,
        )
        show_summaries.append(show_summary)

        total_segments += show_segments
        total_words += show_words
        total_duration_seconds += show_duration
        total_speakers += show_speakers

    total_episodes = len(all_episode_stats)

    corpus_summary = CorpusSummary(
        total_shows=len(shows_data),
        total_episodes=total_episodes,
        total_segments=total_segments,
        total_words=total_words,
        total_duration_hours=round(total_duration_seconds / 3600, 2),
        avg_episode_duration_minutes=round((total_duration_seconds / total_episodes) / 60, 2) if total_episodes > 0 else 0.0,
        avg_speakers_per_episode=round(total_speakers / total_episodes, 2) if total_episodes > 0 else 0.0,
        shows=show_summaries,
        generation_timestamp=datetime.now().isoformat(),
        input_directory=str(input_dir),
        output_directory=str(output_dir),
    )

    return corpus_summary


def check_output_exists(transcript_path: Path, output_dir: Path) -> bool:
    """check if speaker statistics output file exists for a transcript."""
    show_name = transcript_path.parent.name
    output_subdir = output_dir / show_name
    output_file = output_subdir / f"{transcript_path.stem}_speaker_stats.json"
    return output_file.exists()


def display_summary_table(summary: CorpusSummary) -> None:
    """display a summary table of the corpus statistics."""
    table = Table(title="Speaker Statistics Summary", show_header=True, header_style="bold magenta")
    table.add_column("Show", style="cyan")
    table.add_column("Episodes", justify="right")
    table.add_column("Duration (h)", justify="right")
    table.add_column("Words", justify="right")
    table.add_column("Avg Speakers", justify="right")

    for show in summary.shows:
        table.add_row(
            show.show_name[:30],
            str(show.episode_count),
            f"{show.total_duration_hours:.1f}",
            f"{show.total_words:,}",
            f"{show.avg_speakers_per_episode:.1f}",
        )

    # add totals row.
    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{summary.total_episodes}[/bold]",
        f"[bold]{summary.total_duration_hours:.1f}[/bold]",
        f"[bold]{summary.total_words:,}[/bold]",
        f"[bold]{summary.avg_speakers_per_episode:.1f}[/bold]",
        style="bold green",
    )

    console.print(table)


@click.command()
@click.option(
    "--transcripts-dir",
    type=click.Path(exists=True, path_type=Path),
    default=Path("outputs/transcripts_with_speaker_labels_postprocessed_with_classification_labels"),
    help="input directory containing transcript JSON files with speaker labels",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("outputs/analysis/speaker_statistics"),
    help="output directory for speaker statistics files",
)
@click.option(
    "--shows",
    multiple=True,
    help="process only specific shows (can be specified multiple times)",
)
@click.option(
    "--workers",
    type=int,
    default=None,
    help="number of parallel workers (auto-detect if not specified)",
)
@click.option(
    "--force",
    is_flag=True,
    help="force reprocessing of files with existing outputs",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="show what would be processed without running",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="logging level",
)
def generate_speaker_statistics(
    transcripts_dir: Path,
    output_dir: Path,
    shows: tuple[str, ...],
    workers: int | None,
    force: bool,
    dry_run: bool,
    log_level: str,
) -> None:
    """
    generate speaker statistics for podcast transcripts.

    creates fine-grained per-episode statistics files and a general
    summary file aggregating statistics across all podcasts.

    \b
    examples:
        # run with default settings.
        uv run python scripts/generate_speaker_statistics.py

        # process specific shows only.
        uv run python scripts/generate_speaker_statistics.py --shows the_daily

        # force reprocessing.
        uv run python scripts/generate_speaker_statistics.py --force
    """
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    console.print(Panel.fit(
        "[bold cyan]Speaker Statistics Generator[/bold cyan]\n\n"
        f"Input: {transcripts_dir}\n"
        f"Output: {output_dir}",
        border_style="cyan",
    ))

    # resolve paths.
    transcripts_dir = transcripts_dir.resolve()
    output_dir = output_dir.resolve()

    # discover transcript files.
    if shows:
        # process only specified shows.
        transcript_files = []
        for show in shows:
            show_dir = transcripts_dir / show
            if show_dir.exists():
                transcript_files.extend(sorted(show_dir.glob("*.json")))
            else:
                logger.warning(f"show directory not found: {show_dir}")
    else:
        # process all shows.
        transcript_files = sorted(transcripts_dir.rglob("*.json"))

    # filter out normalized files (if present).
    transcript_files = [f for f in transcript_files if "_normalized" not in f.stem]

    if not transcript_files:
        console.print("[yellow]no transcript files found[/yellow]")
        return

    console.print(f"[dim]found {len(transcript_files)} transcript files[/dim]\n")

    # filter out files that already have speaker statistics (unless --force).
    if not force:
        files_to_process = []
        files_skipped = 0

        for transcript_path in transcript_files:
            if check_output_exists(transcript_path, output_dir):
                files_skipped += 1
                logger.debug(f"skipping {transcript_path.name} (statistics already exist)")
            else:
                files_to_process.append(transcript_path)

        if files_skipped > 0:
            console.print(
                f"[yellow]skipping {files_skipped} files with existing statistics[/yellow]"
            )
            if files_to_process:
                console.print(
                    f"[dim]processing {len(files_to_process)} new files[/dim]"
                )
            else:
                console.print("[dim]no new files to process[/dim]")
            console.print("  use --force to reprocess\n")
    else:
        files_to_process = transcript_files

    if dry_run:
        console.print("[yellow][DRY RUN] would process:[/yellow]")
        for f in files_to_process[:20]:
            console.print(f"  {f.relative_to(transcripts_dir)}")
        if len(files_to_process) > 20:
            console.print(f"  ... and {len(files_to_process) - 20} more files")
        return

    if not files_to_process:
        # still generate summary if we have existing stats.
        existing_stats = list(output_dir.rglob("*_speaker_stats.json"))
        if existing_stats:
            console.print("[dim]generating summary from existing statistics...[/dim]")
            all_stats = []
            for stat_file in existing_stats:
                try:
                    with open(stat_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    # reconstruct EpisodeStats from dict.
                    speakers = [SpeakerStats(**s) for s in data.get("speakers", [])]
                    episode_stats = EpisodeStats(
                        episode_name=data["episode_name"],
                        show_name=data["show_name"],
                        audio_file=data.get("audio_file", ""),
                        language=data.get("language", ""),
                        total_segments=data["total_segments"],
                        total_words=data["total_words"],
                        total_duration_seconds=data["total_duration_seconds"],
                        unique_speaker_count=data["unique_speaker_count"],
                        speakers=speakers,
                        dominant_speaker=data.get("dominant_speaker", ""),
                        dominant_speaker_percentage=data.get("dominant_speaker_percentage", 0.0),
                        avg_segment_duration_seconds=data.get("avg_segment_duration_seconds", 0.0),
                        avg_words_per_segment=data.get("avg_words_per_segment", 0.0),
                        processing_timestamp=data.get("processing_timestamp", ""),
                    )
                    all_stats.append(episode_stats)
                except (json.JSONDecodeError, KeyError, OSError) as e:
                    logger.warning(f"error reading {stat_file}: {e}")

            if all_stats:
                corpus_summary = generate_corpus_summary(all_stats, transcripts_dir, output_dir)
                summary_path = output_dir / "speaker_statistics_summary.json"
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(asdict(corpus_summary), f, indent=2, ensure_ascii=False)
                console.print(f"\n[green]summary saved to: {summary_path}[/green]")
                display_summary_table(corpus_summary)
        return

    # determine number of workers.
    num_workers = workers or get_default_workers()
    console.print(f"[dim]using {num_workers} workers[/dim]\n")

    # ensure output directory exists.
    output_dir.mkdir(parents=True, exist_ok=True)

    # process files with progress bar.
    all_episode_stats: list[EpisodeStats] = []
    processed_count = 0
    failed_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "[cyan]processing transcripts...",
            total=len(files_to_process),
        )

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # submit all tasks.
            future_to_path = {
                executor.submit(process_single_transcript, path): path
                for path in files_to_process
            }

            for future in as_completed(future_to_path):
                transcript_path = future_to_path[future]

                try:
                    episode_stats = future.result()

                    if episode_stats:
                        # save per-episode statistics.
                        show_output_dir = output_dir / episode_stats.show_name
                        output_path = show_output_dir / f"{episode_stats.episode_name}_speaker_stats.json"
                        save_episode_stats(episode_stats, output_path)

                        all_episode_stats.append(episode_stats)
                        processed_count += 1
                    else:
                        failed_count += 1
                        logger.warning(f"failed to process: {transcript_path.name}")

                except Exception as e:
                    failed_count += 1
                    logger.error(f"error processing {transcript_path}: {e}")

                progress.advance(task)

    # also load existing stats for the summary.
    if not force:
        existing_stats_files = list(output_dir.rglob("*_speaker_stats.json"))
        existing_names = {s.episode_name for s in all_episode_stats}

        for stat_file in existing_stats_files:
            try:
                with open(stat_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if data["episode_name"] not in existing_names:
                    speakers = [SpeakerStats(**s) for s in data.get("speakers", [])]
                    episode_stats = EpisodeStats(
                        episode_name=data["episode_name"],
                        show_name=data["show_name"],
                        audio_file=data.get("audio_file", ""),
                        language=data.get("language", ""),
                        total_segments=data["total_segments"],
                        total_words=data["total_words"],
                        total_duration_seconds=data["total_duration_seconds"],
                        unique_speaker_count=data["unique_speaker_count"],
                        speakers=speakers,
                        dominant_speaker=data.get("dominant_speaker", ""),
                        dominant_speaker_percentage=data.get("dominant_speaker_percentage", 0.0),
                        avg_segment_duration_seconds=data.get("avg_segment_duration_seconds", 0.0),
                        avg_words_per_segment=data.get("avg_words_per_segment", 0.0),
                        processing_timestamp=data.get("processing_timestamp", ""),
                    )
                    all_episode_stats.append(episode_stats)
            except (json.JSONDecodeError, KeyError, OSError):
                pass

    # generate and save corpus summary.
    corpus_summary: CorpusSummary | None = None
    if all_episode_stats:
        corpus_summary = generate_corpus_summary(all_episode_stats, transcripts_dir, output_dir)

        summary_path = output_dir / "speaker_statistics_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(asdict(corpus_summary), f, indent=2, ensure_ascii=False)

        console.print(f"\n[green]summary saved to: {summary_path}[/green]")

    # display results.
    console.print("\n" + "=" * 60)
    console.print("[bold]Processing Complete[/bold]")
    console.print("=" * 60)
    console.print(f"[green]processed:[/green] {processed_count}")
    console.print(f"[yellow]skipped (already exists):[/yellow] {len(transcript_files) - len(files_to_process)}")
    console.print(f"[red]failed:[/red] {failed_count}")

    if corpus_summary is not None:
        console.print()
        display_summary_table(corpus_summary)

    console.print(f"\n[dim]output directory: {output_dir}[/dim]")


if __name__ == "__main__":
    generate_speaker_statistics()
