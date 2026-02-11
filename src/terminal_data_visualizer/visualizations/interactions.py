"""Speaker interaction visualization."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from terminal_data_visualizer.config import (
    BAR_WIDTH,
    COLOR_ERROR,
    COLOR_PRIMARY,
    COLOR_SUCCESS,
    COLOR_WARNING,
    KEY_BACK,
    OUTPUTS_PATH,
    TRANSCRIPTS_DIR,
)
from terminal_data_visualizer.models import Segment
from terminal_data_visualizer.screen_capture import prompt_and_save

console = Console(record=True)


@dataclass
class SpeakerInteraction:
    """Statistics about speaker interactions in an episode."""

    episode_name: str
    speakers: dict[str, float]  # speaker -> percentage of words.
    turn_counts: dict[str, int]  # speaker -> number of turns.
    avg_turn_length: dict[str, float]  # speaker -> average words per turn.
    transitions: dict[str, dict[str, int]]  # from_speaker -> to_speaker -> count.
    total_turns: int
    total_words: int
    total_duration: float


def analyze_speaker_interactions(transcript_path: Path) -> SpeakerInteraction | None:
    """Analyze speaker interactions from a transcript."""
    try:
        with open(transcript_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    segments_data = data.get("segments", [])
    if not segments_data:
        return None

    segments = [Segment.from_dict(s) for s in segments_data]

    # calculate speaker statistics.
    speaker_words: dict[str, int] = defaultdict(int)
    speaker_turns: dict[str, int] = defaultdict(int)
    transitions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    total_words = 0
    total_duration = 0.0
    prev_speaker: str | None = None

    for seg in segments:
        speaker_words[seg.speaker] += seg.word_count
        speaker_turns[seg.speaker] += 1
        total_words += seg.word_count
        total_duration += seg.duration

        # track speaker transitions.
        if prev_speaker and prev_speaker != seg.speaker:
            transitions[prev_speaker][seg.speaker] += 1

        prev_speaker = seg.speaker

    # calculate percentages and averages.
    speaker_percentages = {
        speaker: (words / total_words * 100) if total_words > 0 else 0
        for speaker, words in speaker_words.items()
    }

    avg_turn_length = {
        speaker: speaker_words[speaker] / speaker_turns[speaker]
        if speaker_turns[speaker] > 0
        else 0
        for speaker in speaker_words
    }

    return SpeakerInteraction(
        episode_name=transcript_path.stem,
        speakers=speaker_percentages,
        turn_counts=dict(speaker_turns),
        avg_turn_length=avg_turn_length,
        transitions={k: dict(v) for k, v in transitions.items()},
        total_turns=sum(speaker_turns.values()),
        total_words=total_words,
        total_duration=total_duration,
    )


def display_speaker_interactions(interaction: SpeakerInteraction) -> None:
    """Display speaker interaction visualization."""
    console.print(
        Panel(
            f"[bold {COLOR_PRIMARY}]👥 Speaker Interactions: {interaction.episode_name}[/bold {COLOR_PRIMARY}]",
            border_style="blue",
        )
    )

    # overview stats.
    console.print("\n[bold]Episode Overview:[/bold]")
    console.print(f"  Total turns: {interaction.total_turns}")
    console.print(f"  Total words: {interaction.total_words:,}")
    console.print(f"  Duration: {interaction.total_duration / 60:.0f} minutes")
    console.print(f"  Speakers: {len(interaction.speakers)}")

    # speaker distribution visualization.
    console.print(f"\n[bold {COLOR_PRIMARY}]Speaker Distribution:[/bold {COLOR_PRIMARY}]\n")

    sorted_speakers = sorted(
        interaction.speakers.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    max_pct = max(interaction.speakers.values()) if interaction.speakers else 100

    for speaker, pct in sorted_speakers:
        bar_len = int((pct / max_pct) * BAR_WIDTH)
        bar = "═" * bar_len

        # color code by dominance.
        if pct > 60:
            color = COLOR_PRIMARY
        elif pct > 30:
            color = COLOR_SUCCESS
        else:
            color = COLOR_WARNING

        console.print(f"[{color}]{speaker}[/{color}] {bar} {pct:.1f}%")

    # turn statistics table.
    console.print(f"\n[bold {COLOR_PRIMARY}]Turn Statistics:[/bold {COLOR_PRIMARY}]")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Speaker", style="cyan", width=20)
    table.add_column("Turns", justify="right", width=10)
    table.add_column("Words", justify="right", width=10)
    table.add_column("Avg Turn", justify="right", width=12)

    for speaker, pct in sorted_speakers:
        turns = interaction.turn_counts.get(speaker, 0)
        words = int(interaction.total_words * pct / 100)
        avg_len = interaction.avg_turn_length.get(speaker, 0)

        table.add_row(
            speaker[:18],
            str(turns),
            str(words),
            f"{avg_len:.1f} words",
        )

    console.print(table)

    # speaker transitions visualization.
    if interaction.transitions:
        console.print(f"\n[bold {COLOR_PRIMARY}]Speaker Transitions:[/bold {COLOR_PRIMARY}]\n")

        for from_speaker, to_speakers in sorted(interaction.transitions.items()):
            total_transitions = sum(to_speakers.values())
            console.print(f"[{COLOR_PRIMARY}]{from_speaker}[/{COLOR_PRIMARY}] transitions to:")

            for to_speaker, count in sorted(to_speakers.items(), key=lambda x: x[1], reverse=True):
                pct = (count / total_transitions * 100) if total_transitions > 0 else 0
                arrows = "→" * min(10, max(1, int(pct / 10)))
                console.print(f"    {arrows} {to_speaker} ({count} times, {pct:.0f}%)")

            console.print()


def display_interaction_graph(interaction: SpeakerInteraction) -> None:
    """Display simplified speaker interaction graph (ASCII art)."""
    console.print(
        Panel(
            f"[bold {COLOR_PRIMARY}]Speaker Flow: {interaction.episode_name}[/bold {COLOR_PRIMARY}]",
            border_style="blue",
        )
    )

    sorted_speakers = sorted(
        interaction.speakers.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    if not sorted_speakers:
        console.print(f"[{COLOR_WARNING}]No speaker data available[/{COLOR_WARNING}]")
        return

    # main speaker at top.
    main_speaker, main_pct = sorted_speakers[0]
    bar_width = min(50, int(main_pct / 2))
    main_bar = "═" * bar_width

    console.print(f"\n{main_speaker} {main_bar} {main_pct:.1f}%")

    # draw connections to other speakers.
    for i, (speaker, pct) in enumerate(sorted_speakers[1:], 1):
        # connection line.
        indent = "   │" if i < len(sorted_speakers) - 1 else "   │"
        console.print(indent)

        # arrow and speaker.
        arrow = "   ├──► " if i < len(sorted_speakers) - 1 else "   └──► "

        bar_width = min(30, int(pct / 2))
        bar = "═" * bar_width

        console.print(f"{arrow}{speaker} {bar} {pct:.1f}%")

        # transition info.
        transitions_to = interaction.transitions.get(main_speaker, {}).get(speaker, 0)
        transitions_from = interaction.transitions.get(speaker, {}).get(main_speaker, 0)

        if transitions_to > 0 or transitions_from > 0:
            detail_indent = "   │       " if i < len(sorted_speakers) - 1 else "           "
            console.print(
                f"{detail_indent}[dim]↔ {transitions_to + transitions_from} exchanges[/dim]"
            )

    # avg turn length comparison.
    console.print("\n[bold]Avg turn length:[/bold]")
    for speaker, pct in sorted_speakers[:5]:
        avg = interaction.avg_turn_length.get(speaker, 0)
        console.print(f"  {speaker}: {avg:.0f} words")


def _get_available_shows() -> list[str]:
    """Get list of available shows."""
    transcripts_path = OUTPUTS_PATH / TRANSCRIPTS_DIR
    if not transcripts_path.exists():
        return []
    return sorted([d.name for d in transcripts_path.iterdir() if d.is_dir()])


def _get_episodes(show_name: str) -> list[Path]:
    """Get episode transcript paths for a show."""
    show_path = OUTPUTS_PATH / TRANSCRIPTS_DIR / show_name
    if not show_path.exists():
        return []
    return sorted(show_path.glob("*.json"))


def interactions_menu() -> None:
    """Interactive speaker interactions menu."""
    while True:
        console.clear()
        console.print(
            Panel(
                f"[bold {COLOR_PRIMARY}]👥 Speaker Interactions[/bold {COLOR_PRIMARY}]\n"
                "[dim]Analyze speaker dynamics in episodes[/dim]",
                border_style="blue",
            )
        )

        table = Table(show_header=False, box=None)
        table.add_column("Option", style=COLOR_WARNING)
        table.add_column("Description")

        table.add_row("1", "📊 Detailed interaction analysis")
        table.add_row("2", "🔀 Speaker flow graph")
        table.add_row("3", "📈 Compare episodes")
        table.add_row(KEY_BACK, "⬅️  Back")

        console.print(table)
        console.print()

        choice = Prompt.ask(f"[{COLOR_WARNING}]Select option[/{COLOR_WARNING}]", default=KEY_BACK)

        match choice.lower():
            case "1":
                _detailed_analysis_menu()
            case "2":
                _flow_graph_menu()
            case "3":
                _compare_episodes_menu()
            case _ if choice.lower() == KEY_BACK:
                break


def _detailed_analysis_menu() -> None:
    """Detailed interaction analysis menu."""
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
        if idx < 0 or idx >= len(shows):
            console.print(f"[{COLOR_ERROR}]Invalid selection[/{COLOR_ERROR}]")
            Prompt.ask("\n[dim]Press Enter to continue[/dim]")
            return

        show = shows[idx]
        episodes = _get_episodes(show)

        if not episodes:
            console.print(f"[{COLOR_WARNING}]No episodes found[/{COLOR_WARNING}]")
            Prompt.ask("\n[dim]Press Enter to continue[/dim]")
            return

        # select episode.
        console.clear()
        console.print(f"[bold {COLOR_PRIMARY}]Select Episode from {show}[/bold {COLOR_PRIMARY}]\n")

        for i, ep in enumerate(episodes[:30], 1):
            console.print(f"  {i}. {ep.stem}")

        if len(episodes) > 30:
            console.print(f"\n[dim]... and {len(episodes) - 30} more episodes[/dim]")

        ep_choice = Prompt.ask(
            f"\n[{COLOR_WARNING}]Select episode number (or '{KEY_BACK}' to go back)[/{COLOR_WARNING}]",
            default=KEY_BACK,
        )

        if ep_choice.lower() == KEY_BACK:
            return

        ep_idx = int(ep_choice) - 1
        if 0 <= ep_idx < len(episodes):
            interaction = analyze_speaker_interactions(episodes[ep_idx])

            if interaction:
                console.clear()
                display_speaker_interactions(interaction)
                prompt_and_save(console, screen_name=f"interactions_{show}_{episodes[ep_idx].stem}")
            else:
                console.print(f"[{COLOR_ERROR}]Could not analyze episode[/{COLOR_ERROR}]")
        else:
            console.print(f"[{COLOR_ERROR}]Invalid selection[/{COLOR_ERROR}]")

    except ValueError:
        console.print(f"[{COLOR_ERROR}]Invalid input[/{COLOR_ERROR}]")

    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def _flow_graph_menu() -> None:
    """Speaker flow graph menu."""
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
        if idx < 0 or idx >= len(shows):
            console.print(f"[{COLOR_ERROR}]Invalid selection[/{COLOR_ERROR}]")
            Prompt.ask("\n[dim]Press Enter to continue[/dim]")
            return

        show = shows[idx]
        episodes = _get_episodes(show)

        if not episodes:
            console.print(f"[{COLOR_WARNING}]No episodes found[/{COLOR_WARNING}]")
            Prompt.ask("\n[dim]Press Enter to continue[/dim]")
            return

        # select episode.
        console.clear()
        console.print(f"[bold {COLOR_PRIMARY}]Select Episode from {show}[/bold {COLOR_PRIMARY}]\n")

        for i, ep in enumerate(episodes[:30], 1):
            console.print(f"  {i}. {ep.stem}")

        ep_choice = Prompt.ask(
            f"\n[{COLOR_WARNING}]Select episode number (or '{KEY_BACK}' to go back)[/{COLOR_WARNING}]",
            default=KEY_BACK,
        )

        if ep_choice.lower() == KEY_BACK:
            return

        ep_idx = int(ep_choice) - 1
        if 0 <= ep_idx < len(episodes):
            interaction = analyze_speaker_interactions(episodes[ep_idx])

            if interaction:
                console.clear()
                display_interaction_graph(interaction)
                prompt_and_save(console, screen_name=f"flow_{show}_{episodes[ep_idx].stem}")
            else:
                console.print(f"[{COLOR_ERROR}]Could not analyze episode[/{COLOR_ERROR}]")
        else:
            console.print(f"[{COLOR_ERROR}]Invalid selection[/{COLOR_ERROR}]")

    except ValueError:
        console.print(f"[{COLOR_ERROR}]Invalid input[/{COLOR_ERROR}]")

    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def _compare_episodes_menu() -> None:
    """Compare speaker interactions across episodes."""
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]📈 Compare Episodes[/bold {COLOR_PRIMARY}]\n")
    console.print(f"[{COLOR_WARNING}]Episode comparison coming soon![/{COLOR_WARNING}]")
    console.print("\nPlanned features:")
    console.print("  • Compare speaker balance across episodes")
    console.print("  • Track speaker presence over time")
    console.print("  • Identify most interactive episodes")

    Prompt.ask("\n[dim]Press Enter to continue[/dim]")
