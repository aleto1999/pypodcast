"""Reusable selection utilities for interactive menus."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt

from terminal_data_visualizer.config import (
    ANALYSIS_DIR,
    COLOR_ERROR,
    COLOR_PRIMARY,
    COLOR_WARNING,
    KEY_BACK,
    OUTPUTS_PATH,
    TRANSCRIPTS_DIR,
)
from terminal_data_visualizer.state import session_state

console = Console(record=True)


def get_podcast_folders() -> list[Path]:
    """Get all available podcast folders from analysis directory."""
    analysis_path = OUTPUTS_PATH / ANALYSIS_DIR
    if not analysis_path.exists():
        return []

    return sorted([d for d in analysis_path.iterdir() if d.is_dir()])


def get_transcript_files(podcast_name: str) -> list[Path]:
    """Get all transcript files for a podcast."""
    transcript_path = OUTPUTS_PATH / TRANSCRIPTS_DIR / podcast_name
    if not transcript_path.exists():
        return []

    return sorted(transcript_path.glob("*.json"))


def select_podcast(title: str = "Select Podcast", remember_last: bool = True) -> Path | None:
    """Interactive podcast selector with memory."""
    podcasts = get_podcast_folders()
    if not podcasts:
        console.print(f"[{COLOR_WARNING}]No podcast folders found[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return None

    # show available podcasts.
    console.print(f"[bold {COLOR_PRIMARY}]{title}[/bold {COLOR_PRIMARY}]\n")
    console.print(f"[{COLOR_PRIMARY}]Available podcasts:[/{COLOR_PRIMARY}]\n")

    # highlight last selection.
    last_podcast = session_state.get_last_podcast() if remember_last else None

    for i, podcast in enumerate(podcasts, 1):
        if remember_last and last_podcast and podcast.name == last_podcast:
            console.print(f"  {i}. {podcast.name} [dim](last selected)[/dim]")
        else:
            console.print(f"  {i}. {podcast.name}")

    # get user selection.
    choice = Prompt.ask(
        f"\n[{COLOR_WARNING}]Select podcast number (or '{KEY_BACK}' to go back)[/{COLOR_WARNING}]",
        default=KEY_BACK,
    )

    if choice.lower() == KEY_BACK:
        return None

    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(podcasts):
            console.print(f"[{COLOR_ERROR}]Invalid selection[/{COLOR_ERROR}]")
            Prompt.ask("\n[dim]Press Enter to continue[/dim]")
            return None

        selected = podcasts[idx]
        if remember_last:
            session_state.set_last_podcast(selected.name)
        return selected

    except ValueError:
        console.print(f"[{COLOR_ERROR}]Invalid input[/{COLOR_ERROR}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return None


def select_episode(
    podcast_name: str, title: str = "Select Episode", max_display: int = 20
) -> Path | None:
    """Interactive episode selector."""
    episodes = get_transcript_files(podcast_name)
    if not episodes:
        console.print(f"[{COLOR_WARNING}]No episodes found[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return None

    # show available episodes.
    console.print(f"\n[{COLOR_PRIMARY}]{title} from {podcast_name}:[/{COLOR_PRIMARY}]\n")

    display_count = min(len(episodes), max_display)
    for i, episode in enumerate(episodes[:display_count], 1):
        console.print(f"  {i}. {episode.stem}")

    if len(episodes) > max_display:
        console.print(f"\n[dim]... and {len(episodes) - max_display} more episodes[/dim]")

    # get user selection.
    choice = Prompt.ask(
        f"\n[{COLOR_WARNING}]Select episode number (or '{KEY_BACK}' to go back)[/{COLOR_WARNING}]",
        default=KEY_BACK,
    )

    if choice.lower() == KEY_BACK:
        return None

    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(episodes):
            console.print(f"[{COLOR_ERROR}]Invalid selection[/{COLOR_ERROR}]")
            Prompt.ask("\n[dim]Press Enter to continue[/dim]")
            return None

        selected = episodes[idx]
        session_state.set_last_episode(selected.stem)
        return selected

    except ValueError:
        console.print(f"[{COLOR_ERROR}]Invalid input[/{COLOR_ERROR}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return None


def select_podcast_and_episode() -> tuple[Path, Path] | None:
    """Combined podcast and episode selector."""
    podcast = select_podcast()
    if not podcast:
        return None

    console.clear()
    episode = select_episode(podcast.name)
    if not episode:
        return None

    return (podcast, episode)


def confirm_action(message: str, default: bool = False) -> bool:
    """Ask user to confirm an action."""
    default_str = "Y/n" if default else "y/N"
    response = Prompt.ask(f"[{COLOR_WARNING}]{message} ({default_str})[/{COLOR_WARNING}]")

    if not response:
        return default

    return response.lower() in ("y", "yes")
