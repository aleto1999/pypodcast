"""Timeline visualizations for episode and content trends."""

from __future__ import annotations

import json

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from terminal_data_visualizer.config import (
    ANALYSIS_DIR,
    COLOR_ERROR,
    COLOR_PRIMARY,
    COLOR_SUCCESS,
    COLOR_WARNING,
    KEY_BACK,
    LLM_ANNOTATIONS_DIR,
    OUTPUTS_PATH,
    TRANSCRIPTS_DIR,
)
from terminal_data_visualizer.screen_capture import prompt_and_save

console = Console(record=True)


def get_episode_order(show_name: str) -> list[str]:
    """Get episodes in order (by filename)."""
    transcripts_path = OUTPUTS_PATH / TRANSCRIPTS_DIR / show_name
    if not transcripts_path.exists():
        return []

    episodes = sorted([f.stem for f in transcripts_path.glob("*.json")])
    return episodes


def get_hate_speech_by_episode(show_name: str) -> dict[str, int]:
    """Get hate speech count per episode."""
    counts: dict[str, int] = {}

    llm_path = OUTPUTS_PATH / LLM_ANNOTATIONS_DIR / show_name
    if not llm_path.exists():
        return counts

    for file in llm_path.glob("*.json"):
        episode_name = file.stem
        count = 0

        try:
            with open(file, encoding="utf-8") as f:
                data = json.load(f)

            for seg in data.get("segments", []):
                if seg.get("llm_annotation", {}).get("has_hate_speech"):
                    count += 1

            counts[episode_name] = count
        except (json.JSONDecodeError, OSError):
            counts[episode_name] = 0

    return counts


def get_keyword_matches_by_episode(show_name: str, category: str | None = None) -> dict[str, int]:
    """Get keyword match count per episode, optionally filtered by category."""
    counts: dict[str, int] = {}

    analysis_path = OUTPUTS_PATH / ANALYSIS_DIR / show_name
    if not analysis_path.exists():
        return counts

    for file in analysis_path.glob("*.json"):
        episode_name = file.stem.replace("_keywords", "")
        count = 0

        try:
            with open(file, encoding="utf-8") as f:
                data = json.load(f)

            for match in data.get("matches", []):
                if category is None or match.get("category", "").lower() == category.lower():
                    count += 1

            counts[episode_name] = count
        except (json.JSONDecodeError, OSError):
            counts[episode_name] = 0

    return counts


def display_episode_timeline(
    data_by_episode: dict[str, int],
    title: str = "Episode Timeline",
    episodes: list[str] | None = None,
    max_display: int = 20,
    bar_width: int = 40,
) -> None:
    """Display a timeline visualization of data across episodes."""
    console.print(
        Panel(
            f"[bold {COLOR_PRIMARY}]{title}[/bold {COLOR_PRIMARY}]",
            border_style="blue",
        )
    )

    if not data_by_episode:
        console.print(f"[{COLOR_WARNING}]No data available[/{COLOR_WARNING}]")
        return

    # use provided episode order or sort by episode name.
    episode_list = episodes if episodes else sorted(data_by_episode.keys())

    # limit display.
    if len(episode_list) > max_display:
        # sample evenly across episodes.
        step = len(episode_list) // max_display
        display_episodes = episode_list[::step][:max_display]
    else:
        display_episodes = episode_list

    # find max value for scaling.
    values = [data_by_episode.get(ep, 0) for ep in display_episodes]
    max_val = max(values) if values else 1

    # display timeline.
    console.print()

    for ep in display_episodes:
        value = data_by_episode.get(ep, 0)

        # scale bar.
        if max_val > 0:
            bar_len = int((value / max_val) * bar_width)
        else:
            bar_len = 0

        bar = "█" * bar_len + "░" * (bar_width - bar_len)

        # truncate episode name.
        ep_display = ep[:15] if len(ep) > 15 else ep.ljust(15)

        # color code by value.
        if value == 0:
            value_str = f"[dim]{value}[/dim]"
        elif value > max_val * 0.75:
            value_str = f"[{COLOR_ERROR}]{value}[/{COLOR_ERROR}]"
        elif value > max_val * 0.5:
            value_str = f"[{COLOR_WARNING}]{value}[/{COLOR_WARNING}]"
        else:
            value_str = f"[{COLOR_SUCCESS}]{value}[/{COLOR_SUCCESS}]"

        console.print(f"{ep_display} {bar} {value_str}")

    # scale legend.
    console.print(f"\n[dim]Scale: 0 {'─' * (bar_width - 8)} {max_val}[/dim]")

    # trend analysis.
    if len(values) >= 4:
        first_quarter = sum(values[: len(values) // 4])
        last_quarter = sum(values[-len(values) // 4 :])

        if first_quarter > 0:
            change = ((last_quarter - first_quarter) / first_quarter) * 100

            if change > 20:
                trend = f"[{COLOR_WARNING}]↑ Increasing (+{change:.0f}%)[/{COLOR_WARNING}]"
            elif change < -20:
                trend = f"[{COLOR_SUCCESS}]↓ Decreasing ({change:.0f}%)[/{COLOR_SUCCESS}]"
            else:
                trend = "[dim]→ Stable[/dim]"

            console.print(f"\n[bold]Trend:[/bold] {trend}")


def display_hate_speech_timeline(show_name: str) -> None:
    """Display hate speech timeline for a show."""
    episodes = get_episode_order(show_name)
    hate_speech_data = get_hate_speech_by_episode(show_name)

    display_episode_timeline(
        hate_speech_data,
        title=f"Hate Speech Timeline: {show_name}",
        episodes=episodes,
    )


def display_keyword_timeline_chart(show_name: str, category: str | None = None) -> None:
    """Display keyword matches timeline for a show."""
    episodes = get_episode_order(show_name)
    keyword_data = get_keyword_matches_by_episode(show_name, category)

    title = f"Keyword Timeline: {show_name}"
    if category:
        title += f" ({category})"

    display_episode_timeline(
        keyword_data,
        title=title,
        episodes=episodes,
    )


def _get_available_shows() -> list[str]:
    """Get list of available shows."""
    transcripts_path = OUTPUTS_PATH / TRANSCRIPTS_DIR
    if not transcripts_path.exists():
        return []
    return sorted([d.name for d in transcripts_path.iterdir() if d.is_dir()])


def timeline_menu() -> None:
    """Interactive timeline visualization menu."""
    while True:
        console.clear()
        console.print(
            Panel(
                f"[bold {COLOR_PRIMARY}]📈 Timeline Visualizations[/bold {COLOR_PRIMARY}]\n"
                "[dim]View content trends across episodes[/dim]",
                border_style="blue",
            )
        )

        table = Table(show_header=False, box=None)
        table.add_column("Option", style=COLOR_WARNING)
        table.add_column("Description")

        table.add_row("1", "⚠️  Hate speech timeline")
        table.add_row("2", "🔑 Keyword matches timeline")
        table.add_row("3", "📊 Custom metric timeline")
        table.add_row(KEY_BACK, "⬅️  Back")

        console.print(table)
        console.print()

        choice = Prompt.ask(f"[{COLOR_WARNING}]Select option[/{COLOR_WARNING}]", default=KEY_BACK)

        match choice.lower():
            case "1":
                _hate_speech_timeline_menu()
            case "2":
                _keyword_timeline_menu()
            case "3":
                _custom_timeline_menu()
            case _ if choice.lower() == KEY_BACK:
                break


def _hate_speech_timeline_menu() -> None:
    """Hate speech timeline menu."""
    console.clear()
    shows = _get_available_shows()

    if not shows:
        console.print(f"[{COLOR_WARNING}]No shows found[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    console.print(f"[bold {COLOR_PRIMARY}]Select Show[/bold {COLOR_PRIMARY}]\n")
    for i, show in enumerate(shows, 1):
        console.print(f"  {i}. {show}")

    choice = Prompt.ask(
        f"\n[{COLOR_WARNING}]Select show number (or '{KEY_BACK}' to go back)[/{COLOR_WARNING}]",
        default=KEY_BACK,
    )

    if choice.lower() == KEY_BACK:
        return

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(shows):
            console.clear()
            display_hate_speech_timeline(shows[idx])
            prompt_and_save(console, screen_name=f"timeline_hate_speech_{shows[idx]}")
        else:
            console.print(f"[{COLOR_ERROR}]Invalid selection[/{COLOR_ERROR}]")
    except ValueError:
        console.print(f"[{COLOR_ERROR}]Invalid input[/{COLOR_ERROR}]")

    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def _keyword_timeline_menu() -> None:
    """Keyword timeline menu."""
    console.clear()
    shows = _get_available_shows()

    if not shows:
        console.print(f"[{COLOR_WARNING}]No shows found[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    console.print(f"[bold {COLOR_PRIMARY}]Select Show[/bold {COLOR_PRIMARY}]\n")
    for i, show in enumerate(shows, 1):
        console.print(f"  {i}. {show}")

    choice = Prompt.ask(
        f"\n[{COLOR_WARNING}]Select show number (or '{KEY_BACK}' to go back)[/{COLOR_WARNING}]",
        default=KEY_BACK,
    )

    if choice.lower() == KEY_BACK:
        return

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(shows):
            show = shows[idx]

            # optionally filter by category.
            category = Prompt.ask(
                f"[{COLOR_WARNING}]Filter by category (leave blank for all)[/{COLOR_WARNING}]",
                default="",
            )

            console.clear()
            display_keyword_timeline_chart(show, category if category else None)
            prompt_and_save(console, screen_name=f"timeline_keywords_{show}")
        else:
            console.print(f"[{COLOR_ERROR}]Invalid selection[/{COLOR_ERROR}]")
    except ValueError:
        console.print(f"[{COLOR_ERROR}]Invalid input[/{COLOR_ERROR}]")

    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def _custom_timeline_menu() -> None:
    """Custom metric timeline placeholder."""
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]📊 Custom Timeline[/bold {COLOR_PRIMARY}]\n")
    console.print(f"[{COLOR_WARNING}]Custom metrics coming soon![/{COLOR_WARNING}]")
    console.print("\nAvailable in future versions:")
    console.print("  • Word count per episode")
    console.print("  • Speaker ratio trends")
    console.print("  • Segment density")
    console.print("  • Custom category combinations")

    Prompt.ask("\n[dim]Press Enter to continue[/dim]")
