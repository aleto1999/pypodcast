"""Episode-centric view for exploring all outputs of a single episode."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    DOCUMENT_LABELS_DIR,
    KEY_BACK,
    LLM_ANNOTATIONS_DIR,
    OUTPUTS_PATH,
    TRANSCRIPTS_DIR,
)
from terminal_data_visualizer.models import EpisodeStats, Segment
from terminal_data_visualizer.screen_capture import prompt_and_save

console = Console(record=True)


@dataclass
class EpisodeOutput:
    """Information about a specific output type for an episode."""

    output_type: str
    exists: bool
    file_path: Path | None
    summary: dict[str, Any] | None = None


@dataclass
class EpisodeSummary:
    """Comprehensive summary of an episode across all output types."""

    show_name: str
    episode_name: str
    outputs: dict[str, EpisodeOutput]
    stats: EpisodeStats | None = None
    keyword_matches: int = 0
    hate_speech_flags: int = 0
    ad_detections: int = 0


def get_episode_base_name(episode_path: Path) -> str:
    """Extract base name from episode path (without extension)."""
    return episode_path.stem


def find_episode_outputs(show_name: str, episode_name: str) -> EpisodeSummary:
    """Find all outputs for a specific episode across all directories."""
    outputs: dict[str, EpisodeOutput] = {}

    # transcript file.
    transcript_path = OUTPUTS_PATH / TRANSCRIPTS_DIR / show_name / f"{episode_name}.json"
    outputs["Transcript"] = EpisodeOutput(
        output_type="Transcript",
        exists=transcript_path.exists(),
        file_path=transcript_path if transcript_path.exists() else None,
    )

    # keyword analysis file.
    keyword_path = OUTPUTS_PATH / ANALYSIS_DIR / show_name / f"{episode_name}_keywords.json"
    keyword_matches = 0
    if keyword_path.exists():
        try:
            with open(keyword_path, encoding="utf-8") as f:
                data = json.load(f)
                keyword_matches = len(data.get("matches", []))
        except (json.JSONDecodeError, OSError):
            pass

    outputs["Keyword Analysis"] = EpisodeOutput(
        output_type="Keyword Analysis",
        exists=keyword_path.exists(),
        file_path=keyword_path if keyword_path.exists() else None,
        summary={"matches": keyword_matches} if keyword_path.exists() else None,
    )

    # llm annotations file.
    llm_path = OUTPUTS_PATH / LLM_ANNOTATIONS_DIR / show_name / f"{episode_name}.json"
    hate_speech_flags = 0
    ad_detections = 0
    if llm_path.exists():
        try:
            with open(llm_path, encoding="utf-8") as f:
                data = json.load(f)
                for seg in data.get("segments", []):
                    ann = seg.get("llm_annotation", {})
                    if ann.get("has_hate_speech"):
                        hate_speech_flags += 1
                    if ann.get("has_advertisement"):
                        ad_detections += 1
        except (json.JSONDecodeError, OSError):
            pass

    outputs["LLM Annotations"] = EpisodeOutput(
        output_type="LLM Annotations",
        exists=llm_path.exists(),
        file_path=llm_path if llm_path.exists() else None,
        summary={
            "hate_speech": hate_speech_flags,
            "advertisements": ad_detections,
        }
        if llm_path.exists()
        else None,
    )

    # document labels file.
    labels_path = OUTPUTS_PATH / DOCUMENT_LABELS_DIR / show_name / f"{episode_name}.md"
    outputs["Document Labels"] = EpisodeOutput(
        output_type="Document Labels",
        exists=labels_path.exists(),
        file_path=labels_path if labels_path.exists() else None,
    )

    # calculate episode stats from transcript.
    stats = None
    if transcript_path.exists():
        try:
            with open(transcript_path, encoding="utf-8") as f:
                data = json.load(f)
            segments_data = data.get("segments", [])
            if segments_data:
                segments = [Segment.from_dict(s) for s in segments_data]
                speakers: dict[str, dict[str, Any]] = {}
                total_words = 0
                total_duration = 0.0

                for seg in segments:
                    total_words += seg.word_count
                    total_duration += seg.duration
                    if seg.speaker not in speakers:
                        speakers[seg.speaker] = {
                            "count": 0,
                            "words": 0,
                            "duration": 0.0,
                        }
                    speakers[seg.speaker]["count"] += 1
                    speakers[seg.speaker]["words"] += seg.word_count
                    speakers[seg.speaker]["duration"] += seg.duration

                stats = EpisodeStats(
                    filename=episode_name,
                    total_segments=len(segments),
                    total_duration=total_duration,
                    speaker_count=len(speakers),
                    total_words=total_words,
                    speakers={},  # simplified for display.
                )
        except (json.JSONDecodeError, OSError):
            pass

    return EpisodeSummary(
        show_name=show_name,
        episode_name=episode_name,
        outputs=outputs,
        stats=stats,
        keyword_matches=keyword_matches,
        hate_speech_flags=hate_speech_flags,
        ad_detections=ad_detections,
    )


def display_episode_summary(summary: EpisodeSummary) -> None:
    """Display comprehensive episode summary."""
    console.print(
        Panel(
            f"[bold {COLOR_PRIMARY}]📄 Episode: {summary.episode_name}[/bold {COLOR_PRIMARY}]",
            border_style="blue",
        )
    )

    # basic stats.
    if summary.stats:
        stats_text = (
            f"Duration: {summary.stats.total_duration / 60:.0f}m | "
            f"Segments: {summary.stats.total_segments:,} | "
            f"Words: {summary.stats.total_words:,}"
        )
        console.print(f"[dim]{stats_text}[/dim]\n")

    # outputs status table.
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Output Type", style="cyan", width=20)
    table.add_column("Status", width=12)
    table.add_column("Path / Summary", width=50)

    for output in summary.outputs.values():
        if output.exists:
            status = f"[{COLOR_SUCCESS}]✅ Available[/{COLOR_SUCCESS}]"
            if output.file_path:
                path_str = str(output.file_path.relative_to(OUTPUTS_PATH))
            else:
                path_str = ""
        else:
            status = f"[{COLOR_ERROR}]❌ Missing[/{COLOR_ERROR}]"
            path_str = "[dim]Not generated[/dim]"

        table.add_row(output.output_type, status, path_str)

    console.print(table)

    # quick stats section.
    console.print(f"\n[bold {COLOR_PRIMARY}]Quick Stats:[/bold {COLOR_PRIMARY}]")

    stats_table = Table(show_header=False, box=None)
    stats_table.add_column("Metric", style="cyan", width=25)
    stats_table.add_column("Value", style="green", width=20)

    if summary.stats:
        stats_table.add_row("Speakers", str(summary.stats.speaker_count))

    stats_table.add_row("Keyword Matches", str(summary.keyword_matches))

    if summary.hate_speech_flags > 0:
        stats_table.add_row(
            "Hate Speech Flags",
            f"[{COLOR_ERROR}]{summary.hate_speech_flags}[/{COLOR_ERROR}]",
        )
    else:
        stats_table.add_row("Hate Speech Flags", "0")

    if summary.ad_detections > 0:
        stats_table.add_row("Ad Detections", str(summary.ad_detections))
    else:
        stats_table.add_row("Ad Detections", "0")

    console.print(stats_table)


def episode_view_menu() -> None:
    """Interactive episode-centric view menu."""
    console.clear()
    console.print(
        Panel(
            f"[bold {COLOR_PRIMARY}]📄 Episode View[/bold {COLOR_PRIMARY}]\n"
            "[dim]View all outputs for a single episode[/dim]",
            border_style="blue",
        )
    )

    # select show first.
    transcripts_path = OUTPUTS_PATH / TRANSCRIPTS_DIR
    if not transcripts_path.exists():
        console.print(f"[{COLOR_WARNING}]No transcripts directory found[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    shows = sorted([d.name for d in transcripts_path.iterdir() if d.is_dir()])
    if not shows:
        console.print(f"[{COLOR_WARNING}]No shows found[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    console.print(f"[{COLOR_PRIMARY}]Available shows:[/{COLOR_PRIMARY}]\n")
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
        show_name = shows[idx]
    except ValueError:
        console.print(f"[{COLOR_ERROR}]Invalid input[/{COLOR_ERROR}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    # select episode.
    console.clear()
    show_path = transcripts_path / show_name
    episodes = sorted([f.stem for f in show_path.glob("*.json")])

    if not episodes:
        console.print(f"[{COLOR_WARNING}]No episodes found for {show_name}[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    console.print(f"[bold {COLOR_PRIMARY}]Episodes in {show_name}[/bold {COLOR_PRIMARY}]\n")

    # show first 30 episodes.
    display_count = min(30, len(episodes))
    for i, ep in enumerate(episodes[:display_count], 1):
        console.print(f"  {i}. {ep}")

    if len(episodes) > display_count:
        console.print(f"\n[dim]... and {len(episodes) - display_count} more episodes[/dim]")

    ep_choice = Prompt.ask(
        f"\n[{COLOR_WARNING}]Select episode number (or '{KEY_BACK}' to go back)[/{COLOR_WARNING}]",
        default=KEY_BACK,
    )

    if ep_choice.lower() == KEY_BACK:
        return

    try:
        ep_idx = int(ep_choice) - 1
        if ep_idx < 0 or ep_idx >= len(episodes):
            console.print(f"[{COLOR_ERROR}]Invalid selection[/{COLOR_ERROR}]")
            Prompt.ask("\n[dim]Press Enter to continue[/dim]")
            return
        episode_name = episodes[ep_idx]
    except ValueError:
        console.print(f"[{COLOR_ERROR}]Invalid input[/{COLOR_ERROR}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    # display episode summary.
    console.clear()
    summary = find_episode_outputs(show_name, episode_name)
    display_episode_summary(summary)

    # options for drilling down.
    console.print(f"\n[{COLOR_PRIMARY}]Options:[/{COLOR_PRIMARY}]")
    console.print("  1. View Transcript")
    console.print("  2. View Keywords")
    console.print("  3. View LLM Annotations")
    console.print("  4. View Document Labels")
    console.print(f"  {KEY_BACK}. Back")

    drill_choice = Prompt.ask(
        f"\n[{COLOR_WARNING}]Select option[/{COLOR_WARNING}]",
        default=KEY_BACK,
    )

    if drill_choice == "1" and summary.outputs["Transcript"].exists:
        _view_transcript_detail(summary.outputs["Transcript"].file_path)
    elif drill_choice == "2" and summary.outputs["Keyword Analysis"].exists:
        _view_keywords_detail(summary.outputs["Keyword Analysis"].file_path)
    elif drill_choice == "3" and summary.outputs["LLM Annotations"].exists:
        _view_llm_detail(summary.outputs["LLM Annotations"].file_path)
    elif drill_choice == "4" and summary.outputs["Document Labels"].exists:
        _view_labels_detail(summary.outputs["Document Labels"].file_path)

    prompt_and_save(console, screen_name=f"episode_{show_name}_{episode_name}")


def _view_transcript_detail(file_path: Path | None) -> None:
    """View transcript detail."""
    if not file_path:
        return

    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]📝 Transcript: {file_path.stem}[/bold {COLOR_PRIMARY}]\n")

    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        segments = data.get("segments", [])
        console.print(f"[dim]Total segments: {len(segments)}[/dim]\n")

        # show first 10 segments.
        for i, seg in enumerate(segments[:10], 1):
            speaker = seg.get("speaker", "UNKNOWN")
            text = seg.get("text", "")
            if len(text) > 150:
                text = text[:147] + "..."
            console.print(f"[{COLOR_PRIMARY}]{speaker}:[/{COLOR_PRIMARY}] {text}\n")

        if len(segments) > 10:
            console.print(f"[dim]... and {len(segments) - 10} more segments[/dim]")

    except Exception as e:
        console.print(f"[{COLOR_ERROR}]Error reading file: {e}[/{COLOR_ERROR}]")

    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def _view_keywords_detail(file_path: Path | None) -> None:
    """View keyword analysis detail."""
    if not file_path:
        return

    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]🔑 Keywords: {file_path.stem}[/bold {COLOR_PRIMARY}]\n")

    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        matches = data.get("matches", [])
        console.print(f"[dim]Total matches: {len(matches)}[/dim]\n")

        # group by category.
        categories: dict[str, list[dict]] = {}
        for match in matches:
            cat = match.get("category", "Unknown")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(match)

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Category", style="cyan", width=25)
        table.add_column("Matches", justify="right", width=10)

        for cat, cat_matches in sorted(categories.items(), key=lambda x: len(x[1]), reverse=True):
            table.add_row(cat, str(len(cat_matches)))

        console.print(table)

    except Exception as e:
        console.print(f"[{COLOR_ERROR}]Error reading file: {e}[/{COLOR_ERROR}]")

    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def _view_llm_detail(file_path: Path | None) -> None:
    """View LLM annotations detail."""
    if not file_path:
        return

    console.clear()
    console.print(
        f"[bold {COLOR_PRIMARY}]🤖 LLM Annotations: {file_path.stem}[/bold {COLOR_PRIMARY}]\n"
    )

    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        segments = data.get("segments", [])

        # count flags.
        hate_count = sum(1 for s in segments if s.get("llm_annotation", {}).get("has_hate_speech"))
        ad_count = sum(1 for s in segments if s.get("llm_annotation", {}).get("has_advertisement"))

        console.print(f"Total segments: {len(segments)}")
        console.print(f"Hate speech detections: {hate_count}")
        console.print(f"Advertisement detections: {ad_count}\n")

        # show flagged segments.
        flagged = [s for s in segments if s.get("llm_annotation", {}).get("has_hate_speech")]

        if flagged:
            console.print(f"[{COLOR_WARNING}]Flagged segments:[/{COLOR_WARNING}]\n")
            for seg in flagged[:5]:
                text = seg.get("text", "")
                if len(text) > 100:
                    text = text[:97] + "..."
                target = seg.get("llm_annotation", {}).get("target_group", "N/A")
                console.print(f"  • {text}")
                console.print(f"    [dim]Target: {target}[/dim]\n")

    except Exception as e:
        console.print(f"[{COLOR_ERROR}]Error reading file: {e}[/{COLOR_ERROR}]")

    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def _view_labels_detail(file_path: Path | None) -> None:
    """View document labels detail."""
    if not file_path:
        return

    console.clear()
    console.print(
        f"[bold {COLOR_PRIMARY}]🏷️ Document Labels: {file_path.stem}[/bold {COLOR_PRIMARY}]\n"
    )

    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        # show first 50 lines.
        lines = content.split("\n")
        display_lines = min(50, len(lines))

        from rich.markdown import Markdown

        md = Markdown("\n".join(lines[:display_lines]))
        console.print(md)

        if len(lines) > display_lines:
            console.print(f"\n[dim]... and {len(lines) - display_lines} more lines[/dim]")

    except Exception as e:
        console.print(f"[{COLOR_ERROR}]Error reading file: {e}[/{COLOR_ERROR}]")

    Prompt.ask("\n[dim]Press Enter to continue[/dim]")
