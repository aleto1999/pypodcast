"""Category heatmap visualization."""

from __future__ import annotations

import json
from collections import defaultdict

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from terminal_data_visualizer.config import (
    ANALYSIS_DIR,
    COLOR_ERROR,
    COLOR_PRIMARY,
    COLOR_WARNING,
    KEY_BACK,
    OUTPUTS_PATH,
    TRANSCRIPTS_DIR,
)
from terminal_data_visualizer.screen_capture import prompt_and_save

console = Console(record=True)


def get_category_counts_per_show() -> dict[str, dict[str, int]]:
    """Get keyword category counts per show."""
    result: dict[str, dict[str, int]] = {}

    analysis_path = OUTPUTS_PATH / ANALYSIS_DIR
    if not analysis_path.exists():
        return result

    for show_dir in analysis_path.iterdir():
        if not show_dir.is_dir():
            continue

        show_name = show_dir.name
        category_counts: dict[str, int] = defaultdict(int)

        for file in show_dir.glob("*.json"):
            try:
                with open(file, encoding="utf-8") as f:
                    data = json.load(f)

                for match in data.get("matches", []):
                    category = match.get("category", "Unknown")
                    category_counts[category] += 1

            except (json.JSONDecodeError, OSError):
                pass

        if category_counts:
            result[show_name] = dict(category_counts)

    return result


def get_all_categories(data: dict[str, dict[str, int]]) -> list[str]:
    """Get all unique categories across all shows."""
    categories: set[str] = set()
    for show_data in data.values():
        categories.update(show_data.keys())
    return sorted(categories)


def display_category_heatmap(
    data: dict[str, dict[str, int]] | None = None,
    max_shows: int = 10,
    max_categories: int = 10,
) -> None:
    """Display a category distribution heatmap across shows."""
    console.print(
        Panel(
            f"[bold {COLOR_PRIMARY}]📊 Category Heatmap[/bold {COLOR_PRIMARY}]",
            border_style="blue",
        )
    )

    if data is None:
        data = get_category_counts_per_show()

    if not data:
        console.print(f"[{COLOR_WARNING}]No data available[/{COLOR_WARNING}]")
        return

    # get shows and categories.
    shows = sorted(data.keys())[:max_shows]
    all_categories = get_all_categories(data)

    # sort categories by total count.
    category_totals = {
        cat: sum(data.get(show, {}).get(cat, 0) for show in shows) for cat in all_categories
    }
    top_categories = sorted(category_totals.keys(), key=lambda x: category_totals[x], reverse=True)[
        :max_categories
    ]

    # find max value for scaling.
    max_val = (
        max(data.get(show, {}).get(cat, 0) for show in shows for cat in top_categories)
        if shows and top_categories
        else 1
    )

    # create heatmap table.
    table = Table(show_header=True, header_style="bold magenta", title="Category Distribution")
    table.add_column("Category", style="cyan", width=18)

    for show in shows:
        # truncate show name.
        show_display = show[:10] if len(show) > 10 else show
        table.add_column(show_display, justify="center", width=12)

    # intensity characters for heatmap.
    intensity_chars = ["░", "▒", "▓", "█"]

    for category in top_categories:
        row = [category[:16]]

        for show in shows:
            count = data.get(show, {}).get(category, 0)

            if count == 0:
                cell = "[dim]·[/dim]"
            else:
                # calculate intensity (0-3).
                intensity = min(3, int((count / max_val) * 4))
                char = intensity_chars[intensity] * 3

                # color by intensity.
                if intensity >= 3:
                    cell = f"[red]{char}[/red]"
                elif intensity >= 2:
                    cell = f"[yellow]{char}[/yellow]"
                elif intensity >= 1:
                    cell = f"[green]{char}[/green]"
                else:
                    cell = f"[dim]{char}[/dim]"

            row.append(cell)

        table.add_row(*row)

    console.print(table)

    # legend.
    console.print(
        "\n[bold]Legend:[/bold] [dim]·[/dim] = 0  [dim]░░░[/dim] = Low  [green]▒▒▒[/green] = Medium  [yellow]▓▓▓[/yellow] = High  [red]███[/red] = Very High"
    )
    console.print(f"[dim]Max value: {max_val}[/dim]")


def display_numeric_heatmap(
    data: dict[str, dict[str, int]] | None = None,
    max_shows: int = 10,
    max_categories: int = 10,
) -> None:
    """Display a category heatmap with numeric values."""
    console.print(
        Panel(
            f"[bold {COLOR_PRIMARY}]📊 Category Distribution (Numeric)[/bold {COLOR_PRIMARY}]",
            border_style="blue",
        )
    )

    if data is None:
        data = get_category_counts_per_show()

    if not data:
        console.print(f"[{COLOR_WARNING}]No data available[/{COLOR_WARNING}]")
        return

    # get shows and categories.
    shows = sorted(data.keys())[:max_shows]
    all_categories = get_all_categories(data)

    # sort categories by total count.
    category_totals = {
        cat: sum(data.get(show, {}).get(cat, 0) for show in shows) for cat in all_categories
    }
    top_categories = sorted(category_totals.keys(), key=lambda x: category_totals[x], reverse=True)[
        :max_categories
    ]

    # create table.
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Category", style="cyan", width=18)

    for show in shows:
        show_display = show[:10] if len(show) > 10 else show
        table.add_column(show_display, justify="right", width=10)

    table.add_column("Total", justify="right", style="bold green", width=10)

    for category in top_categories:
        row = [category[:16]]
        total = 0

        for show in shows:
            count = data.get(show, {}).get(category, 0)
            total += count

            if count == 0:
                row.append("[dim]-[/dim]")
            else:
                row.append(str(count))

        row.append(str(total))
        table.add_row(*row)

    console.print(table)


def _get_available_shows() -> list[str]:
    """Get list of available shows."""
    transcripts_path = OUTPUTS_PATH / TRANSCRIPTS_DIR
    if not transcripts_path.exists():
        return []
    return sorted([d.name for d in transcripts_path.iterdir() if d.is_dir()])


def heatmap_menu() -> None:
    """Interactive heatmap visualization menu."""
    while True:
        console.clear()
        console.print(
            Panel(
                f"[bold {COLOR_PRIMARY}]📊 Heatmap Visualizations[/bold {COLOR_PRIMARY}]\n"
                "[dim]View category distributions across shows[/dim]",
                border_style="blue",
            )
        )

        table = Table(show_header=False, box=None)
        table.add_column("Option", style=COLOR_WARNING)
        table.add_column("Description")

        table.add_row("1", "🎨 Visual heatmap (all shows)")
        table.add_row("2", "📊 Numeric table (all shows)")
        table.add_row("3", "🔍 Filter by shows")
        table.add_row(KEY_BACK, "⬅️  Back")

        console.print(table)
        console.print()

        choice = Prompt.ask(f"[{COLOR_WARNING}]Select option[/{COLOR_WARNING}]", default=KEY_BACK)

        match choice.lower():
            case "1":
                console.clear()
                display_category_heatmap()
                prompt_and_save(console, screen_name="heatmap_visual")
            case "2":
                console.clear()
                display_numeric_heatmap()
                prompt_and_save(console, screen_name="heatmap_numeric")
            case "3":
                _filtered_heatmap_menu()
            case _ if choice.lower() == KEY_BACK:
                break


def _filtered_heatmap_menu() -> None:
    """Filtered heatmap menu."""
    console.clear()
    shows = _get_available_shows()

    if not shows:
        console.print(f"[{COLOR_WARNING}]No shows found[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    console.print(f"[bold {COLOR_PRIMARY}]Select Shows to Compare[/bold {COLOR_PRIMARY}]\n")
    for i, show in enumerate(shows, 1):
        console.print(f"  {i}. {show}")

    console.print("\n[dim]Enter numbers separated by commas (e.g., 1,2,3)[/dim]")

    choice = Prompt.ask(
        f"\n[{COLOR_WARNING}]Select shows (or '{KEY_BACK}' to go back)[/{COLOR_WARNING}]",
        default=KEY_BACK,
    )

    if choice.lower() == KEY_BACK:
        return

    try:
        indices = [int(x.strip()) - 1 for x in choice.split(",")]
        selected_shows = [shows[i] for i in indices if 0 <= i < len(shows)]

        if selected_shows:
            # filter data to selected shows.
            all_data = get_category_counts_per_show()
            filtered_data = {show: all_data.get(show, {}) for show in selected_shows}

            console.clear()
            display_category_heatmap(filtered_data)
            prompt_and_save(console, screen_name="heatmap_filtered")
        else:
            console.print(f"[{COLOR_ERROR}]No valid shows selected[/{COLOR_ERROR}]")

    except (ValueError, IndexError):
        console.print(f"[{COLOR_ERROR}]Invalid input[/{COLOR_ERROR}]")

    Prompt.ask("\n[dim]Press Enter to continue[/dim]")
