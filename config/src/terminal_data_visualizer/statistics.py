"""Statistical analysis and terminal-based visualizations."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table

from terminal_data_visualizer.cache import get_episode_stats
from terminal_data_visualizer.config import BAR_WIDTH, COLOR_ERROR, COLOR_PRIMARY, TIMELINE_BINS
from terminal_data_visualizer.models import EpisodeStats

console = Console(record=True)


def calculate_episode_stats(transcript_file: Path) -> EpisodeStats | None:
    """Calculate statistics for an episode (with caching)."""
    return get_episode_stats(transcript_file)


def display_speaker_distribution(stats: EpisodeStats, max_width: int = BAR_WIDTH) -> None:
    """Display speaker participation as horizontal bar chart."""
    if not stats.speakers:
        console.print("[yellow]No speaker data available[/yellow]")
        return

    console.print(
        f"\n[bold {COLOR_PRIMARY}]Speaker Distribution - {stats.filename}[/bold {COLOR_PRIMARY}]\n"
    )

    # sort by segment count.
    sorted_speakers = sorted(stats.speakers.values(), key=lambda x: x.segment_count, reverse=True)

    # find max for scaling.
    max_segments = max(s.segment_count for s in sorted_speakers)

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Speaker", style="cyan", width=15)
    table.add_column("Segments", justify="right", width=8)
    table.add_column("Words", justify="right", width=8)
    table.add_column("Duration", justify="right", width=10)
    table.add_column("Chart", width=max_width)

    for speaker_stat in sorted_speakers:
        # calculate bar width.
        bar_width = int((speaker_stat.segment_count / max_segments) * max_width)
        bar = "█" * bar_width

        # calculate percentage.
        pct = (speaker_stat.segment_count / stats.total_segments) * 100

        # format duration.
        duration_min = speaker_stat.total_duration / 60

        table.add_row(
            speaker_stat.speaker,
            str(speaker_stat.segment_count),
            str(speaker_stat.total_words),
            f"{duration_min:.1f}m",
            f"[green]{bar}[/green] {pct:.1f}%",
        )

    console.print(table)


def display_category_distribution(keyword_file: Path, max_width: int = 50) -> None:
    """Display category distribution from keyword analysis."""
    try:
        with open(keyword_file, encoding="utf-8") as f:
            data = json.load(f)

        matches = data.get("matches", [])
        if not matches:
            console.print("[yellow]No matches found[/yellow]")
            return

        # count by category.
        category_counts: Counter[str] = Counter()
        for match in matches:
            category = match.get("category", "unknown")
            category_counts[category] += 1

        console.print(f"\n[bold cyan]Category Distribution - {keyword_file.stem}[/bold cyan]\n")

        # display as table with bars.
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Category", style="cyan", width=30)
        table.add_column("Matches", justify="right", width=8)
        table.add_column("Chart", width=max_width)

        max_count = max(category_counts.values()) if category_counts else 1

        for category, count in category_counts.most_common():
            bar_width = int((count / max_count) * max_width)
            bar = "█" * bar_width
            pct = (count / len(matches)) * 100

            table.add_row(category, str(count), f"[yellow]{bar}[/yellow] {pct:.1f}%")

        console.print(table)
        summary = (
            f"\n[dim]Total matches: {len(matches)} across {len(category_counts)} categories[/dim]"
        )
        console.print(summary)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


def display_keyword_timeline(keyword_file: Path, bins: int = TIMELINE_BINS) -> None:
    """Display when keywords appear throughout the episode (timeline)."""
    try:
        with open(keyword_file, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        console.print(f"[{COLOR_ERROR}]File not found: {keyword_file}[/{COLOR_ERROR}]")
        return
    except json.JSONDecodeError as e:
        console.print(f"[{COLOR_ERROR}]Invalid JSON in {keyword_file}: {e}[/{COLOR_ERROR}]")
        return

    matches = data.get("matches", [])
    if not matches:
        console.print("[yellow]No matches found[/yellow]")
        return

    # get min/max timestamps.
    timestamps = [m.get("start_time", 0) for m in matches if "start_time" in m]
    if not timestamps:
        console.print("[yellow]No timestamp data available[/yellow]")
        return

    min_time = min(timestamps)
    max_time = max(timestamps)
    duration = max_time - min_time

    if duration == 0:
        console.print("[yellow]Episode too short for timeline[/yellow]")
        return

    # create bins.
    bin_width = duration / bins
    bin_counts = [0] * bins

    for ts in timestamps:
        bin_idx = min(int((ts - min_time) / bin_width), bins - 1)
        bin_counts[bin_idx] += 1

    console.print(
        f"\n[bold {COLOR_PRIMARY}]Keyword Timeline - {keyword_file.stem}[/bold {COLOR_PRIMARY}]\n"
    )

    # display timeline.
    max_count = max(bin_counts) if bin_counts else 1
    bar_width = 40

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Time Range", style="cyan", width=20)
    table.add_column("Matches", justify="right", width=8)
    table.add_column("Distribution", width=bar_width)

    for i, count in enumerate(bin_counts):
        start_min = (min_time + i * bin_width) / 60
        end_min = (min_time + (i + 1) * bin_width) / 60

        bar_len = int((count / max_count) * bar_width) if max_count > 0 else 0
        bar = "▓" * bar_len

        table.add_row(
            f"{start_min:5.1f} - {end_min:5.1f}m",
            str(count),
            f"[blue]{bar}[/blue]",
        )

    console.print(table)
    summary = f"\n[dim]Total matches: {len(matches)}, Duration: {max_time / 60:.1f} minutes[/dim]"
    console.print(summary)


def compare_episodes(transcript_files: list[Path], metric: str = "segments") -> None:
    """Compare statistics across multiple episodes."""
    if not transcript_files:
        console.print("[yellow]No files to compare[/yellow]")
        return

    console.print(
        f"\n[bold {COLOR_PRIMARY}]Episode Comparison - By {metric.title()}[/bold {COLOR_PRIMARY}]\n"
    )

    # calculate stats for each file.
    episode_stats: list[EpisodeStats] = []
    for file_path in transcript_files[:10]:  # limit to 10 episodes.
        stats = calculate_episode_stats(file_path)
        if stats:
            episode_stats.append(stats)

    if not episode_stats:
        console.print("[yellow]No valid episode data found[/yellow]")
        return

    # display comparison table.
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Episode", style="cyan", width=30)
    table.add_column("Speakers", justify="right", width=8)
    table.add_column("Segments", justify="right", width=9)
    table.add_column("Words", justify="right", width=8)
    table.add_column("Duration", justify="right", width=10)

    for stats in episode_stats:
        duration_min = stats.total_duration / 60
        table.add_row(
            stats.filename[:28] if len(stats.filename) > 28 else stats.filename,
            str(stats.speaker_count),
            str(stats.total_segments),
            str(stats.total_words),
            f"{duration_min:.1f}m",
        )

    console.print(table)

    # summary statistics.
    avg_speakers = sum(s.speaker_count for s in episode_stats) / len(episode_stats)
    avg_duration = sum(s.total_duration for s in episode_stats) / len(episode_stats) / 60

    console.print("\n[bold]Summary Statistics:[/bold]")
    console.print(f"  Episodes analyzed: {len(episode_stats)}")
    console.print(f"  Average speakers: {avg_speakers:.1f}")
    console.print(f"  Average duration: {avg_duration:.1f} minutes")
