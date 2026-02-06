"""Batch processing utilities for analyzing multiple episodes."""

from __future__ import annotations

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from terminal_data_visualizer.cache import get_episode_stats
from terminal_data_visualizer.config import COLOR_PRIMARY
from terminal_data_visualizer.models import EpisodeStats
from terminal_data_visualizer.selectors import get_podcast_folders, get_transcript_files

console = Console(record=True)


def analyze_all_episodes_in_podcast(podcast_name: str) -> list[EpisodeStats]:
    """Analyze all episodes in a podcast with progress bar."""
    transcript_files = get_transcript_files(podcast_name)

    stats_list: list[EpisodeStats] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Analyzing {len(transcript_files)} episodes...", total=len(transcript_files)
        )

        for file in transcript_files:
            stats = get_episode_stats(file)
            if stats:
                stats_list.append(stats)
            progress.advance(task)

    return stats_list


def display_podcast_summary(stats_list: list[EpisodeStats], podcast_name: str) -> None:
    """Display aggregated statistics for a podcast."""
    if not stats_list:
        console.print("[yellow]No episode data available[/yellow]")
        return

    console.print(
        f"\n[bold {COLOR_PRIMARY}]📊 Podcast Summary: {podcast_name}[/bold {COLOR_PRIMARY}]\n"
    )

    # calculate aggregated metrics.
    total_episodes = len(stats_list)
    total_segments = sum(s.total_segments for s in stats_list)
    total_duration = sum(s.total_duration for s in stats_list)
    total_words = sum(s.total_words for s in stats_list)

    avg_speakers = sum(s.speaker_count for s in stats_list) / total_episodes
    avg_segments = total_segments / total_episodes
    avg_duration = total_duration / total_episodes / 60
    avg_words = total_words / total_episodes

    # display summary table.
    table = Table(show_header=True, header_style="bold magenta", title="Aggregated Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Total", justify="right", style="green")
    table.add_column("Average", justify="right", style="yellow")

    table.add_row("Episodes", str(total_episodes), "-")
    table.add_row("Segments", f"{total_segments:,}", f"{avg_segments:.1f}")
    table.add_row("Duration (hours)", f"{total_duration / 3600:.1f}h", f"{avg_duration:.1f}m")
    table.add_row("Words", f"{total_words:,}", f"{avg_words:.0f}")
    table.add_row("Speakers per Episode", "-", f"{avg_speakers:.1f}")

    console.print(table)

    # speaker distribution across all episodes.
    all_speakers: dict[str, int] = {}
    for stats in stats_list:
        for speaker in stats.speakers:
            all_speakers[speaker] = all_speakers.get(speaker, 0) + 1

    console.print("\n[bold]Speaker Distribution (Episodes Appeared):[/bold]\n")

    speaker_table = Table(show_header=True, header_style="bold magenta")
    speaker_table.add_column("Speaker", style="cyan")
    speaker_table.add_column("Episodes", justify="right", style="green")
    speaker_table.add_column("Percentage", justify="right", style="yellow")

    for speaker, count in sorted(all_speakers.items(), key=lambda x: x[1], reverse=True)[:15]:
        percentage = (count / total_episodes) * 100
        speaker_table.add_row(speaker, str(count), f"{percentage:.1f}%")

    console.print(speaker_table)


def analyze_all_podcasts() -> dict[str, list[EpisodeStats]]:
    """Analyze all podcasts and return aggregated data."""
    podcasts = get_podcast_folders()
    all_stats: dict[str, list[EpisodeStats]] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Analyzing {len(podcasts)} podcasts...", total=len(podcasts))

        for podcast in podcasts:
            stats_list = analyze_all_episodes_in_podcast(podcast.name)
            if stats_list:
                all_stats[podcast.name] = stats_list
            progress.advance(task)

    return all_stats


def display_all_podcasts_summary(all_stats: dict[str, list[EpisodeStats]]) -> None:
    """Display summary of all podcasts."""
    if not all_stats:
        console.print("[yellow]No podcast data available[/yellow]")
        return

    console.print(f"\n[bold {COLOR_PRIMARY}]📊 All Podcasts Summary[/bold {COLOR_PRIMARY}]\n")

    # create comparison table.
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Podcast", style="cyan", width=40)
    table.add_column("Episodes", justify="right", width=10)
    table.add_column("Total Hours", justify="right", width=12)
    table.add_column("Avg Duration", justify="right", width=12)
    table.add_column("Total Words", justify="right", width=12)

    # overall totals.
    total_episodes = 0
    total_duration = 0.0
    total_words = 0

    for podcast_name, stats_list in sorted(all_stats.items()):
        episodes = len(stats_list)
        duration = sum(s.total_duration for s in stats_list)
        words = sum(s.total_words for s in stats_list)
        avg_duration = duration / episodes / 60 if episodes > 0 else 0

        table.add_row(
            podcast_name[:38],
            str(episodes),
            f"{duration / 3600:.1f}h",
            f"{avg_duration:.1f}m",
            f"{words:,}",
        )

        total_episodes += episodes
        total_duration += duration
        total_words += words

    # add totals row.
    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{total_episodes}[/bold]",
        f"[bold]{total_duration / 3600:.1f}h[/bold]",
        f"[bold]{total_duration / total_episodes / 60:.1f}m[/bold]" if total_episodes > 0 else "-",
        f"[bold]{total_words:,}[/bold]",
        style="bold green",
    )

    console.print(table)
