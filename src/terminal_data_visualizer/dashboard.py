"""Unified dashboard for outputs health and overview."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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
from terminal_data_visualizer.screen_capture import prompt_and_save

console = Console(record=True)


@dataclass
class OutputCoverage:
    """Coverage statistics for a specific output type."""

    output_type: str
    complete: int
    missing: int
    total: int

    @property
    def coverage_percent(self) -> float:
        """Calculate coverage percentage."""
        if self.total == 0:
            return 0.0
        return (self.complete / self.total) * 100

    def coverage_bar(self, width: int = 16) -> str:
        """Generate a visual progress bar."""
        filled = int((self.coverage_percent / 100) * width)
        empty = width - filled
        return "█" * filled + "░" * empty


@dataclass
class ShowHealth:
    """Health status for a podcast show."""

    show_name: str
    total_episodes: int
    coverages: dict[str, OutputCoverage]

    @property
    def overall_health(self) -> float:
        """Calculate overall health percentage."""
        if not self.coverages:
            return 0.0
        return sum(c.coverage_percent for c in self.coverages.values()) / len(self.coverages)


def get_all_shows() -> list[str]:
    """Get all available show names from transcripts directory."""
    transcripts_path = OUTPUTS_PATH / TRANSCRIPTS_DIR
    if not transcripts_path.exists():
        return []
    return sorted([d.name for d in transcripts_path.iterdir() if d.is_dir()])


def count_episodes_in_dir(directory: Path) -> int:
    """Count JSON files in a directory."""
    if not directory.exists():
        return 0
    return len(list(directory.glob("*.json")))


def count_markdown_files(directory: Path) -> int:
    """Count markdown files in a directory."""
    if not directory.exists():
        return 0
    return len(list(directory.glob("*.md")))


def count_utterances_in_transcripts(directory: Path) -> int:
    """Count total utterances/segments in transcript JSON files."""
    if not directory.exists():
        return 0
    
    total = 0
    for json_file in directory.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            segments = data.get("segments", [])
            total += len(segments)
        except (json.JSONDecodeError, KeyError, IOError):
            continue
    
    return total


def count_episodes_with_keywords(keywords_dir: Path) -> int:
    """Count episodes that have at least one keyword match.
    
    Args:
        keywords_dir: Directory containing keyword JSON files for the show
        
    Returns:
        int: Number of episodes with at least one keyword match
    """
    if not keywords_dir.exists():
        return 0
    
    episodes_with_keywords = 0
    
    for json_file in keywords_dir.glob("*_keywords.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            matches = data.get("matches", [])
            # if file has at least one match, count it.
            if len(matches) > 0:
                episodes_with_keywords += 1
                
        except (json.JSONDecodeError, KeyError, IOError):
            continue
    
    return episodes_with_keywords


def count_classification_coverage(directory: Path) -> tuple[int, int]:
    """Count utterances with complete classification coverage.
    
    Only counts non-empty segments (segments with text content).
    Empty segments are skipped as they cannot be meaningfully classified.
    
    Returns:
        tuple[int, int]: (utterances_with_all_8_models, total_non_empty_utterances)
    """
    if not directory.exists():
        return 0, 0
    
    total_utterances = 0
    complete_utterances = 0
    
    for json_file in directory.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            segments = data.get("segments", [])
            
            for seg in segments:
                # skip empty segments (consistent with classification pipeline).
                text = seg.get("text", "").strip()
                if not text:
                    continue
                
                total_utterances += 1
                
                classifications = seg.get("classifications", [])
                # check if this segment has all 8 classification models.
                if len(classifications) == 8:
                    complete_utterances += 1
        except (json.JSONDecodeError, KeyError, IOError):
            continue
    
    return complete_utterances, total_utterances


def scan_show_outputs(show_name: str) -> ShowHealth:
    """Scan all output directories for a specific show."""
    transcripts_path = OUTPUTS_PATH / TRANSCRIPTS_DIR / show_name
    
    # get baseline counts.
    total_episodes = count_episodes_in_dir(transcripts_path)
    
    # classification labels coverage (utterance-level).
    # this gives us both classification coverage AND total utterances.
    complete_utterances, total_utterances = count_classification_coverage(transcripts_path)
    
    coverages: dict[str, OutputCoverage] = {}

    # transcripts coverage (file-level).
    coverages["Transcripts"] = OutputCoverage(
        output_type="Transcripts",
        complete=total_episodes,
        missing=0,
        total=total_episodes,
    )

    # keyword analysis coverage (file-level).
    analysis_path = OUTPUTS_PATH / ANALYSIS_DIR / show_name
    episodes_with_keywords = count_episodes_with_keywords(analysis_path)
    coverages["Keyword Analysis"] = OutputCoverage(
        output_type="Keyword Analysis",
        complete=episodes_with_keywords,
        missing=max(0, total_episodes - episodes_with_keywords),
        total=total_episodes,
    )

    # llm annotations coverage (file-level - no data exists yet).
    llm_path = OUTPUTS_PATH / LLM_ANNOTATIONS_DIR / show_name
    llm_count = count_episodes_in_dir(llm_path)
    coverages["LLM Annotations"] = OutputCoverage(
        output_type="LLM Annotations",
        complete=llm_count,
        missing=max(0, total_episodes - llm_count),
        total=total_episodes,
    )

    # classification labels coverage (already calculated above).
    coverages["Classifications"] = OutputCoverage(
        output_type="Classifications",
        complete=complete_utterances,
        missing=max(0, total_utterances - complete_utterances),
        total=total_utterances,
    )

    return ShowHealth(
        show_name=show_name,
        total_episodes=total_episodes,
        coverages=coverages,
    )


def display_outputs_dashboard(show_name: str | None = None) -> None:
    """Display the outputs dashboard for a show or all shows."""
    console.print(
        Panel(
            f"[bold {COLOR_PRIMARY}]📊 Outputs Dashboard[/bold {COLOR_PRIMARY}]",
            border_style="blue",
        )
    )

    if show_name:
        # display dashboard for specific show.
        health = scan_show_outputs(show_name)
        _display_show_dashboard(health)
    else:
        # display overview for all shows.
        shows = get_all_shows()
        if not shows:
            console.print(f"[{COLOR_WARNING}]No shows found[/{COLOR_WARNING}]")
            return

        _display_all_shows_overview(shows)


def _display_show_dashboard(health: ShowHealth) -> None:
    """Display detailed dashboard for a single show."""
    console.print(f"\n[bold]Show:[/bold] {health.show_name}")
    console.print(f"[bold]Episodes:[/bold] {health.total_episodes} total\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Output Type", style="cyan", width=25)
    table.add_column("Complete", justify="right", width=10)
    table.add_column("Missing", justify="right", width=10)
    table.add_column("Coverage", width=22)

    for coverage in health.coverages.values():
        # color code based on coverage.
        if coverage.coverage_percent >= 90:
            status_color = COLOR_SUCCESS
        elif coverage.coverage_percent >= 50:
            status_color = COLOR_WARNING
        else:
            status_color = COLOR_ERROR

        coverage_str = f"{coverage.coverage_bar()} {coverage.coverage_percent:.0f}%"

        table.add_row(
            coverage.output_type,
            str(coverage.complete),
            str(coverage.missing),
            f"[{status_color}]{coverage_str}[/{status_color}]",
        )

    console.print(table)

    # legend with formulas.
    console.print("\n[dim]📊 Legend:[/dim]")
    console.print("[dim]  ✓ ≥90% | ◐ ≥50% | ◔ <50% | ✗ 0%[/dim]")
    console.print("\n[dim]📐 Calculation Formulas:[/dim]")
    console.print("[dim]  • Transcripts: (episodes_with_transcripts / total_episodes) × 100[/dim]")
    console.print("[dim]  • Keyword Analysis: (episodes_with_keywords / total_episodes) × 100[/dim]")
    console.print("[dim]  • LLM Annotations: (episodes_with_llm / total_episodes) × 100[/dim]")
    console.print("[dim]  • Classifications: (utterances_with_8_models / total_non_empty_utterances) × 100[/dim]")
    console.print("[dim]  • Note: Empty segments (silence, pauses) are excluded from classification metrics[/dim]")

    # overall health.
    overall = health.overall_health
    if overall >= 90:
        health_color = COLOR_SUCCESS
        health_status = "Excellent"
    elif overall >= 70:
        health_color = COLOR_WARNING
        health_status = "Good"
    elif overall >= 50:
        health_color = COLOR_WARNING
        health_status = "Partial"
    else:
        health_color = COLOR_ERROR
        health_status = "Incomplete"

    console.print(
        f"\n[bold]Overall Health:[/bold] [{health_color}]{health_status} ({overall:.0f}%)[/{health_color}]"
    )


def _display_all_shows_overview(shows: list[str]) -> None:
    """Display overview dashboard for all shows."""
    table = Table(show_header=True, header_style="bold magenta", title="All Shows Overview")
    table.add_column("Show", style="cyan", width=35)
    table.add_column("Episodes", justify="right", width=10)
    table.add_column("Transcripts\n[dim](file-level)[/dim]", justify="center", width=15)
    table.add_column("Keywords\n[dim](file-level)[/dim]", justify="center", width=15)
    table.add_column("LLM\n[dim](file-level)[/dim]", justify="center", width=15)
    table.add_column("Labels\n[dim](utterance)[/dim]", justify="center", width=15)

    total_episodes = 0
    for show in shows:
        health = scan_show_outputs(show)
        total_episodes += health.total_episodes

        def _status_icon(coverage: OutputCoverage) -> str:
            if coverage.coverage_percent >= 90:
                return f"[{COLOR_SUCCESS}]✓ {coverage.coverage_percent:.0f}%[/{COLOR_SUCCESS}]"
            elif coverage.coverage_percent >= 50:
                return f"[{COLOR_WARNING}]◐ {coverage.coverage_percent:.0f}%[/{COLOR_WARNING}]"
            elif coverage.coverage_percent > 0:
                return f"[{COLOR_ERROR}]◔ {coverage.coverage_percent:.0f}%[/{COLOR_ERROR}]"
            else:
                return f"[{COLOR_ERROR}]✗ 0%[/{COLOR_ERROR}]"

        table.add_row(
            show[:33],
            str(health.total_episodes),
            _status_icon(health.coverages.get("Transcripts", OutputCoverage("", 0, 0, 0))),
            _status_icon(health.coverages.get("Keyword Analysis", OutputCoverage("", 0, 0, 0))),
            _status_icon(health.coverages.get("LLM Annotations", OutputCoverage("", 0, 0, 0))),
            _status_icon(health.coverages.get("Classifications", OutputCoverage("", 0, 0, 0))),
        )

    console.print(table)
    console.print(f"\n[bold]Total Episodes:[/bold] {total_episodes}")
    console.print("\n[dim]📊 Legend:[/dim]")
    console.print("[dim]  ✓ ≥90% | ◐ ≥50% | ◔ <50% | ✗ 0%[/dim]")
    console.print("\n[dim]📐 Calculation Formulas:[/dim]")
    console.print("[dim]  • Transcripts: (episodes_with_transcripts / total_episodes) × 100[/dim]")
    console.print("[dim]  • Keywords: (episodes_with_keywords / total_episodes) × 100[/dim]")
    console.print("[dim]  • LLM: (episodes_with_llm / total_episodes) × 100[/dim]")
    console.print("[dim]  • Labels: (utterances_with_8_models / total_utterances) × 100[/dim]")


def dashboard_menu() -> None:
    """Interactive dashboard menu."""
    while True:
        console.clear()
        console.print(
            Panel(
                f"[bold {COLOR_PRIMARY}]📊 Outputs Dashboard[/bold {COLOR_PRIMARY}]\n"
                "[dim]View output completeness and health status[/dim]",
                border_style="blue",
            )
        )

        table = Table(show_header=False, box=None)
        table.add_column("Option", style=COLOR_WARNING)
        table.add_column("Description")

        table.add_row("1", "📋 All shows overview")
        table.add_row("2", "📊 Single show detailed view")
        table.add_row("3", "🔍 Find missing outputs")
        table.add_row(KEY_BACK, "⬅️  Back")

        console.print(table)
        console.print()

        choice = Prompt.ask(f"[{COLOR_WARNING}]Select option[/{COLOR_WARNING}]", default=KEY_BACK)

        match choice.lower():
            case "1":
                console.clear()
                display_outputs_dashboard()
                prompt_and_save(console, screen_name="dashboard_all_shows")
            case "2":
                _select_show_dashboard()
            case "3":
                _find_missing_outputs()
            case _ if choice.lower() == KEY_BACK:
                break


def _select_show_dashboard() -> None:
    """Select a show and display its dashboard."""
    console.clear()
    shows = get_all_shows()
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
            display_outputs_dashboard(shows[idx])
            prompt_and_save(console, screen_name=f"dashboard_{shows[idx]}")
        else:
            console.print(f"[{COLOR_ERROR}]Invalid selection[/{COLOR_ERROR}]")
    except ValueError:
        console.print(f"[{COLOR_ERROR}]Invalid input[/{COLOR_ERROR}]")

    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def _find_missing_outputs() -> None:
    """Find and list all missing outputs across shows."""
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]🔍 Missing Outputs Report[/bold {COLOR_PRIMARY}]\n")

    shows = get_all_shows()
    missing_report: dict[str, dict[str, int]] = {}

    for show in shows:
        health = scan_show_outputs(show)
        show_missing: dict[str, int] = {}

        for output_type, coverage in health.coverages.items():
            if coverage.missing > 0:
                show_missing[output_type] = coverage.missing

        if show_missing:
            missing_report[show] = show_missing

    if not missing_report:
        console.print(f"[{COLOR_SUCCESS}]✓ All outputs complete for all shows![/{COLOR_SUCCESS}]")
    else:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Show", style="cyan", width=35)
        table.add_column("Missing Output Type", width=20)
        table.add_column("Count", justify="right", width=10)

        for show, missing in sorted(missing_report.items()):
            first_row = True
            for output_type, count in missing.items():
                table.add_row(
                    show if first_row else "",
                    output_type,
                    f"[{COLOR_ERROR}]{count}[/{COLOR_ERROR}]",
                )
                first_row = False

        console.print(table)

        # total missing.
        total_missing = sum(sum(m.values()) for m in missing_report.values())
        console.print(
            f"\n[bold]Total Missing:[/bold] [{COLOR_ERROR}]{total_missing}[/{COLOR_ERROR}] outputs"
        )

    prompt_and_save(console, screen_name="missing_outputs_report")
