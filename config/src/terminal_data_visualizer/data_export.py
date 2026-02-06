"""Export utilities for statistics and data."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from rich.console import Console

from terminal_data_visualizer.config import COLOR_ERROR, COLOR_SUCCESS, OUTPUTS_PATH
from terminal_data_visualizer.models import EpisodeStats

console = Console()


def export_episode_stats_to_csv(stats_list: list[EpisodeStats], output_file: Path) -> bool:
    """Export episode statistics to CSV file."""
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # header.
            writer.writerow(
                [
                    "Episode",
                    "Total Segments",
                    "Total Duration (min)",
                    "Speaker Count",
                    "Total Words",
                ]
            )

            # data rows.
            for stats in stats_list:
                writer.writerow(
                    [
                        stats.filename,
                        stats.total_segments,
                        round(stats.total_duration / 60, 2),
                        stats.speaker_count,
                        stats.total_words,
                    ]
                )

        console.print(
            f"[{COLOR_SUCCESS}]✓ Exported {len(stats_list)} episodes to "
            f"{output_file}[/{COLOR_SUCCESS}]"
        )
        return True

    except OSError as e:
        console.print(f"[{COLOR_ERROR}]Failed to export CSV: {e}[/{COLOR_ERROR}]")
        return False


def export_episode_stats_to_json(stats_list: list[EpisodeStats], output_file: Path) -> bool:
    """Export episode statistics to JSON file."""
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)

        data = []
        for stats in stats_list:
            episode_data = {
                "filename": stats.filename,
                "total_segments": stats.total_segments,
                "total_duration_seconds": stats.total_duration,
                "total_duration_minutes": round(stats.total_duration / 60, 2),
                "speaker_count": stats.speaker_count,
                "total_words": stats.total_words,
                "speakers": {
                    speaker: {
                        "segment_count": s.segment_count,
                        "total_words": s.total_words,
                        "total_duration_seconds": s.total_duration,
                        "avg_segment_length": s.avg_segment_length,
                    }
                    for speaker, s in stats.speakers.items()
                },
            }
            data.append(episode_data)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        console.print(
            f"[{COLOR_SUCCESS}]✓ Exported {len(stats_list)} episodes to "
            f"{output_file}[/{COLOR_SUCCESS}]"
        )
        return True

    except OSError as e:
        console.print(f"[{COLOR_ERROR}]Failed to export JSON: {e}[/{COLOR_ERROR}]")
        return False


def export_category_distribution(
    category_data: dict[str, int], output_file: Path, format: str = "csv"
) -> bool:
    """Export category distribution to file."""
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if format == "csv":
            with open(output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Category", "Match Count"])
                for category, count in sorted(
                    category_data.items(), key=lambda x: x[1], reverse=True
                ):
                    writer.writerow([category, count])

        elif format == "json":
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(category_data, f, indent=2)

        console.print(
            f"[{COLOR_SUCCESS}]✓ Exported category data to {output_file}[/{COLOR_SUCCESS}]"
        )
        return True

    except OSError as e:
        console.print(f"[{COLOR_ERROR}]Failed to export: {e}[/{COLOR_ERROR}]")
        return False


def get_export_path(base_name: str, extension: str = "csv") -> Path:
    """Generate export file path with timestamp."""
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    filename = f"{base_name}_{timestamp}.{extension}"
    return OUTPUTS_PATH / "terminal_viewer" / "exports" / filename
