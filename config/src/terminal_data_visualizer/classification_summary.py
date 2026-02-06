"""Classification pipeline summary statistics viewer."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from terminal_data_visualizer.config import (
    COLOR_ERROR,
    COLOR_PRIMARY,
    COLOR_SUCCESS,
    COLOR_WARNING,
    KEY_BACK,
    OUTPUTS_PATH,
)
from terminal_data_visualizer.screen_capture import prompt_and_save

console = Console(record=True)

# use same config as main for transcript paths.
from terminal_data_visualizer.config import TRANSCRIPTS_DIR
CLASSIFIED_DIR = OUTPUTS_PATH / TRANSCRIPTS_DIR
ANALYSIS_DIR = OUTPUTS_PATH / "analysis"


def find_latest_summary_file() -> Path | None:
    """find the most recent classification summary file."""
    if not ANALYSIS_DIR.exists():
        return None
    
    summary_files = list(ANALYSIS_DIR.glob("classification_*.txt"))
    if not summary_files:
        return None
    
    # sort by modification time, most recent first.
    return max(summary_files, key=lambda p: p.stat().st_mtime)


def display_summary_file(summary_file: Path) -> None:
    """display a classification summary file."""
    console.clear()
    console.print(
        Panel.fit(
            f"[bold {COLOR_PRIMARY}]Classification Pipeline Summary[/bold {COLOR_PRIMARY}]\n"
            f"File: {summary_file.name}",
            border_style=COLOR_PRIMARY,
        )
    )
    console.print()
    
    with open(summary_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    console.print(content)
    console.print()
    
    prompt_and_save(console)
    Prompt.ask(f"\n[dim]Press Enter to continue[/dim]")


def calculate_corpus_statistics() -> dict:
    """calculate aggregate statistics across all classified episodes."""
    if not CLASSIFIED_DIR.exists():
        return {}
    
    stats = {
        "total_shows": 0,
        "total_episodes": 0,
        "total_utterances": 0,
        "models": set(),
        "labels_by_model": defaultdict(Counter),
        "shows": set(),
    }
    
    # scan all classified files.
    for show_dir in sorted(CLASSIFIED_DIR.iterdir()):
        if not show_dir.is_dir():
            continue
        
        stats["shows"].add(show_dir.name)
        episode_count = 0
        
        for json_file in show_dir.glob("*.json"):
            episode_count += 1
            
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # count utterances.
                segments = data.get("segments", [])
                stats["total_utterances"] += len(segments)
                
                # collect label statistics.
                for segment in segments:
                    for classification in segment.get("classifications", []):
                        model = classification.get("model_name")
                        label = classification.get("label")
                        
                        if model and label:
                            stats["models"].add(model)
                            stats["labels_by_model"][model][label] += 1
            
            except (json.JSONDecodeError, KeyError) as e:
                continue
        
        if episode_count > 0:
            stats["total_shows"] += 1
            stats["total_episodes"] += episode_count
    
    return stats


def display_corpus_statistics() -> None:
    """display aggregate statistics for the entire classified corpus."""
    console.clear()
    console.print(
        Panel.fit(
            f"[bold {COLOR_PRIMARY}]📊 Corpus-Wide Classification Statistics[/bold {COLOR_PRIMARY}]",
            border_style=COLOR_PRIMARY,
        )
    )
    console.print()
    
    console.print(f"[{COLOR_WARNING}]Scanning classified transcripts...[/{COLOR_WARNING}]")
    stats = calculate_corpus_statistics()
    
    if not stats or stats["total_episodes"] == 0:
        console.print(f"[{COLOR_ERROR}]No classified episodes found.[/{COLOR_ERROR}]")
        Prompt.ask(f"\n[dim]Press Enter to continue[/dim]")
        return
    
    console.clear()
    console.print(
        Panel.fit(
            f"[bold {COLOR_PRIMARY}]📊 Corpus-Wide Classification Statistics[/bold {COLOR_PRIMARY}]",
            border_style=COLOR_PRIMARY,
        )
    )
    console.print()
    
    # overview table.
    overview = Table(title="Overview", show_header=False, box=None)
    overview.add_column("Metric", style=COLOR_PRIMARY)
    overview.add_column("Value", style=COLOR_SUCCESS)
    
    overview.add_row("Total Shows", f"{stats['total_shows']:,}")
    overview.add_row("Total Episodes", f"{stats['total_episodes']:,}")
    overview.add_row("Total Utterances", f"{stats['total_utterances']:,}")
    overview.add_row("Models Applied", f"{len(stats['models'])}")
    
    console.print(overview)
    console.print()
    
    # models applied.
    console.print(f"[bold {COLOR_PRIMARY}]Models Applied:[/bold {COLOR_PRIMARY}]")
    for model in sorted(stats['models']):
        console.print(f"  • {model}")
    console.print()
    
    # label distribution by model.
    console.print(f"[bold {COLOR_PRIMARY}]Label Distribution by Model:[/bold {COLOR_PRIMARY}]\n")
    
    for model in sorted(stats['models']):
        label_counts = stats['labels_by_model'][model]
        total_for_model = sum(label_counts.values())
        
        console.print(f"[bold]{model}[/bold] ({total_for_model:,} classifications):")
        
        # sort by count descending.
        for label, count in label_counts.most_common(10):
            percentage = (count / total_for_model * 100) if total_for_model > 0 else 0
            bar_width = int(percentage / 2)  # scale to 50 chars max.
            bar = "█" * bar_width
            
            console.print(f"  {label:30s} : {count:8,} ({percentage:5.2f}%) {bar}")
        
        if len(label_counts) > 10:
            console.print(f"  [dim]... and {len(label_counts) - 10} more labels[/dim]")
        
        console.print()
    
    # show coverage.
    console.print(f"[bold {COLOR_PRIMARY}]Show Coverage:[/bold {COLOR_PRIMARY}]")
    for show in sorted(stats['shows']):
        console.print(f"  • {show}")
    console.print()
    
    prompt_and_save(console)
    Prompt.ask(f"\n[dim]Press Enter to continue[/dim]")


def display_per_show_statistics() -> None:
    """display classification statistics broken down by show."""
    console.clear()
    console.print(
        Panel.fit(
            f"[bold {COLOR_PRIMARY}]📊 Per-Show Classification Statistics[/bold {COLOR_PRIMARY}]",
            border_style=COLOR_PRIMARY,
        )
    )
    console.print()
    
    if not CLASSIFIED_DIR.exists():
        console.print(f"[{COLOR_ERROR}]Classified directory not found.[/{COLOR_ERROR}]")
        Prompt.ask(f"\n[dim]Press Enter to continue[/dim]")
        return
    
    # collect statistics per show.
    show_stats = []
    
    for show_dir in sorted(CLASSIFIED_DIR.iterdir()):
        if not show_dir.is_dir():
            continue
        
        episode_count = 0
        utterance_count = 0
        
        for json_file in show_dir.glob("*.json"):
            episode_count += 1
            
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                utterance_count += len(data.get("segments", []))
            
            except (json.JSONDecodeError, KeyError):
                continue
        
        if episode_count > 0:
            show_stats.append({
                "show": show_dir.name,
                "episodes": episode_count,
                "utterances": utterance_count,
                "avg_per_episode": utterance_count / episode_count if episode_count > 0 else 0,
            })
    
    if not show_stats:
        console.print(f"[{COLOR_ERROR}]No classified shows found.[/{COLOR_ERROR}]")
        Prompt.ask(f"\n[dim]Press Enter to continue[/dim]")
        return
    
    # display table.
    table = Table(title="Classification Coverage by Show")
    table.add_column("Show", style=COLOR_PRIMARY)
    table.add_column("Episodes", justify="right", style=COLOR_SUCCESS)
    table.add_column("Utterances", justify="right", style=COLOR_SUCCESS)
    table.add_column("Avg/Episode", justify="right", style=COLOR_WARNING)
    
    # sort by episode count descending.
    for stat in sorted(show_stats, key=lambda x: x["episodes"], reverse=True):
        table.add_row(
            stat["show"],
            f"{stat['episodes']:,}",
            f"{stat['utterances']:,}",
            f"{stat['avg_per_episode']:.1f}",
        )
    
    console.print(table)
    console.print()
    
    # summary.
    total_shows = len(show_stats)
    total_episodes = sum(s["episodes"] for s in show_stats)
    total_utterances = sum(s["utterances"] for s in show_stats)
    
    console.print(f"[bold {COLOR_PRIMARY}]Summary:[/bold {COLOR_PRIMARY}]")
    console.print(f"  Total Shows: {total_shows:,}")
    console.print(f"  Total Episodes: {total_episodes:,}")
    console.print(f"  Total Utterances: {total_utterances:,}")
    console.print()
    
    prompt_and_save(console)
    Prompt.ask(f"\n[dim]Press Enter to continue[/dim]")


def classification_summary_menu() -> None:
    """main menu for classification pipeline summary statistics."""
    while True:
        console.clear()
        console.print(
            Panel.fit(
                f"[bold {COLOR_PRIMARY}]📊 Classification Pipeline Summary[/bold {COLOR_PRIMARY}]",
                border_style=COLOR_PRIMARY,
            )
        )
        console.print()
        
        table = Table(show_header=False, box=None)
        table.add_column("Option", style=COLOR_WARNING)
        table.add_column("Description")
        
        table.add_row("1", "📄 View Latest Pipeline Summary File")
        table.add_row("2", "📊 Corpus-Wide Statistics")
        table.add_row("3", "📈 Per-Show Statistics")
        table.add_row(KEY_BACK, "⬅️  Back to Main Menu")
        
        console.print(table)
        console.print()
        
        choice = Prompt.ask(
            f"[{COLOR_WARNING}]Select option[/{COLOR_WARNING}]",
            default=KEY_BACK,
        )
        
        if choice == "1":
            summary_file = find_latest_summary_file()
            if summary_file:
                display_summary_file(summary_file)
            else:
                console.print(f"[{COLOR_ERROR}]No summary files found in {ANALYSIS_DIR}[/{COLOR_ERROR}]")
                Prompt.ask(f"\n[dim]Press Enter to continue[/dim]")
        
        elif choice == "2":
            display_corpus_statistics()
        
        elif choice == "3":
            display_per_show_statistics()
        
        elif choice.lower() == KEY_BACK:
            break
        
        else:
            console.print(f"[{COLOR_ERROR}]Invalid option[/{COLOR_ERROR}]")
            Prompt.ask(f"\n[dim]Press Enter to continue[/dim]")
