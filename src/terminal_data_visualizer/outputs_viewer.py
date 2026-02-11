"""Interactive viewers for various output files."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterator

from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.table import Table

from terminal_data_visualizer.config import (
    CHUNK_SIZE,
    COLOR_ERROR,
    COLOR_PRIMARY,
    COLOR_WARNING,
    KEY_BACK,
    LLM_ANNOTATIONS_DIR,
    OUTPUTS_PATH,
)
from terminal_data_visualizer.features_explorer import features_explorer_menu
from terminal_data_visualizer.screen_capture import prompt_and_save

console = Console(record=True)


def view_document_labels_menu() -> None:
    """Interactive viewer for document labels."""
    console.print(f"[bold {COLOR_PRIMARY}]📋 Document Labels Viewer[/bold {COLOR_PRIMARY}]\n")

    # get available podcasts with document labels.
    doc_labels_path = OUTPUTS_PATH / "document_labels"
    if not doc_labels_path.exists():
        console.print(f"[{COLOR_WARNING}]No document labels found[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    podcasts = sorted([d for d in doc_labels_path.iterdir() if d.is_dir()])
    if not podcasts:
        console.print(f"[{COLOR_WARNING}]No podcast folders found[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    # display available podcasts.
    console.print(f"[{COLOR_PRIMARY}]Available podcasts:[/{COLOR_PRIMARY}]\n")
    for i, podcast in enumerate(podcasts, 1):
        console.print(f"  {i}. {podcast.name}")

    choice = Prompt.ask(
        f"\n[{COLOR_WARNING}]Select podcast number (or '{KEY_BACK}' to go back)[/{COLOR_WARNING}]",
        default=KEY_BACK,
    )

    if choice.lower() == KEY_BACK:
        return

    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(podcasts):
            console.print(f"[{COLOR_ERROR}]Invalid selection[/{COLOR_ERROR}]")
            Prompt.ask("\n[dim]Press Enter to continue[/dim]")
            return
        podcast_path = podcasts[idx]
    except ValueError:
        console.print(f"[{COLOR_ERROR}]Invalid input[/{COLOR_ERROR}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    # show available files.
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]📋 {podcast_path.name}[/bold {COLOR_PRIMARY}]\n")

    files = list(podcast_path.glob("*.md"))
    if not files:
        console.print(f"[{COLOR_WARNING}]No label files found[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    console.print(f"[{COLOR_PRIMARY}]Available files:[/{COLOR_PRIMARY}]\n")
    for i, file in enumerate(files, 1):
        console.print(f"  {i}. {file.name}")

    file_choice = Prompt.ask(
        f"\n[{COLOR_WARNING}]Select file number (or '{KEY_BACK}' to go back)[/{COLOR_WARNING}]",
        default=KEY_BACK,
    )

    if file_choice.lower() == KEY_BACK:
        return

    try:
        idx = int(file_choice) - 1
        if idx < 0 or idx >= len(files):
            console.print(f"[{COLOR_ERROR}]Invalid selection[/{COLOR_ERROR}]")
            Prompt.ask("\n[dim]Press Enter to continue[/dim]")
            return
        selected_file = files[idx]
    except ValueError:
        console.print(f"[{COLOR_ERROR}]Invalid input[/{COLOR_ERROR}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    # display file content.
    console.clear()
    display_markdown_file(selected_file)
    prompt_and_save(
        console, screen_name=f"document_labels_{podcast_path.name}_{selected_file.stem}"
    )


def display_markdown_file(file_path: Path, max_lines: int = 200) -> None:
    """Display markdown file with pagination."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        # count lines.
        lines = content.split("\n")
        total_lines = len(lines)

        if total_lines > max_lines:
            # show first max_lines lines.
            truncated_content = "\n".join(lines[:max_lines])
            md = Markdown(truncated_content)
            console.print(md)
            console.print(f"\n[yellow]Showing first {max_lines} of {total_lines} lines[/yellow]")
            console.print(f"[dim]Full file at: {file_path}[/dim]")
        else:
            md = Markdown(content)
            console.print(md)

    except Exception as e:
        console.print(f"[red]Error reading file: {e}[/red]")


def load_csv_chunked(
    file_path: Path, chunk_size: int = CHUNK_SIZE
) -> Iterator[list[dict[str, Any]]]:
    """Load CSV file in chunks for memory efficiency."""
    try:
        with open(file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            chunk = []
            for row in reader:
                chunk.append(row)
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
            if chunk:
                yield chunk
    except (FileNotFoundError, csv.Error) as e:
        console.print(f"[{COLOR_ERROR}]Error reading CSV: {e}[/{COLOR_ERROR}]")
        return


def view_features_csv_menu() -> None:
    """Interactive viewer for features CSV - redirect to comprehensive explorer."""
    features_explorer_menu()


def display_sample_rows(rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Display sample rows from features CSV."""
    console.clear()
    console.print("[bold cyan]Sample Rows[/bold cyan]\n")

    num_rows = min(10, len(rows))
    for i, row in enumerate(rows[:num_rows], 1):
        console.print(f"[yellow]Row {i}:[/yellow]")

        table = Table(show_header=True, header_style="bold magenta", box=None)
        table.add_column("Column", style="cyan")
        table.add_column("Value", style="green")

        # show key columns only.
        key_columns = ["file_name", "average_politeness_score", "average_switch_time_minutes"]
        for col in key_columns:
            if col in row:
                value = row[col]
                if len(str(value)) > 60:
                    value = str(value)[:57] + "..."
                table.add_row(col, str(value))

        console.print(table)
        console.print()


def search_features_by_name(rows: list[dict[str, Any]]) -> None:
    """Search features by podcast/show name."""
    console.clear()
    console.print("[bold cyan]Search Features[/bold cyan]\n")

    search_term = Prompt.ask("[yellow]Enter show/podcast name to search[/yellow]")

    if not search_term.strip():
        return

    # search in file_name or file_path columns.
    matches = []
    for row in rows:
        file_name = row.get("file_name", "").lower()
        file_path = row.get("file_path", "").lower()
        if search_term.lower() in file_name or search_term.lower() in file_path:
            matches.append(row)

    console.print(f"\n[green]Found {len(matches)} matches[/green]\n")

    if matches:
        for i, match in enumerate(matches[:20], 1):  # show first 20.
            console.print(f"[yellow]{i}. {match.get('file_name', 'Unknown')}[/yellow]")

            # show key metrics.
            metrics = {
                "Politeness": match.get("average_politeness_score", "N/A"),
                "Switch Time": match.get("average_switch_time_minutes", "N/A"),
            }

            for metric, value in metrics.items():
                console.print(f"   {metric}: {value}")
            console.print()

        if len(matches) > 20:
            console.print(f"[dim]... and {len(matches) - 20} more matches[/dim]\n")


def view_llm_annotations_menu() -> None:
    """Interactive viewer for LLM annotations."""
    console.print("[bold cyan]🤖 LLM Annotations Viewer[/bold cyan]\n")

    llm_path = OUTPUTS_PATH / LLM_ANNOTATIONS_DIR
    if not llm_path.exists():
        console.print(f"[yellow]LLM annotations not found at {llm_path}[/yellow]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    # get available podcasts.
    podcasts = sorted([d for d in llm_path.iterdir() if d.is_dir()])
    if not podcasts:
        console.print("[yellow]No podcast folders found[/yellow]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    console.print("[cyan]Available podcasts:[/cyan]\n")
    for i, podcast in enumerate(podcasts[:20], 1):
        console.print(f"  {i}. {podcast.name}")

    choice = Prompt.ask("\n[yellow]Select podcast number (or 'b' to go back)[/yellow]", default="b")

    if choice.lower() == "b":
        return

    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(podcasts):
            console.print("[red]Invalid selection[/red]")
            Prompt.ask("\n[dim]Press Enter to continue[/dim]")
            return
        podcast_path = podcasts[idx]
    except ValueError:
        console.print("[red]Invalid input[/red]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    # get JSON files.
    files = sorted(podcast_path.glob("*.json"))
    if not files:
        console.print("[yellow]No annotation files found[/yellow]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    console.clear()
    console.print(f"[bold cyan]🤖 {podcast_path.name}[/bold cyan]\n")
    console.print(f"[cyan]Found {len(files)} annotated episodes[/cyan]\n")

    # show summary statistics.
    display_llm_annotations_summary(files)

    # option to view individual file.
    view_individual = Prompt.ask("\n[yellow]View individual episode? (y/n)[/yellow]", default="n")

    if view_individual.lower() == "y":
        console.print("\n[cyan]Available episodes:[/cyan]\n")
        for i, file in enumerate(files[:20], 1):
            console.print(f"  {i}. {file.stem}")

        file_choice = Prompt.ask(
            "\n[yellow]Select episode number (or 'b' to go back)[/yellow]", default="b"
        )

        if file_choice.lower() != "b":
            try:
                idx = int(file_choice) - 1
                if 0 <= idx < len(files):
                    display_llm_annotation_file(files[idx])
            except ValueError:
                console.print("[red]Invalid input[/red]")

    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def display_llm_annotations_summary(files: list[Path]) -> None:
    """Display summary statistics for LLM annotations."""
    total_segments = 0
    hate_speech_count = 0
    ad_count = 0
    target_groups: dict[str, int] = {}

    # sample first 10 files for statistics.
    for file in files[:10]:
        try:
            with open(file, encoding="utf-8") as f:
                data = json.load(f)

            segments = data.get("segments", [])
            total_segments += len(segments)

            for seg in segments:
                llm_ann = seg.get("llm_annotation", {})
                if llm_ann.get("has_hate_speech"):
                    hate_speech_count += 1

                if llm_ann.get("has_advertisement"):
                    ad_count += 1

                target = llm_ann.get("target_group")
                if target:
                    target_groups[target] = target_groups.get(target, 0) + 1

        except Exception:
            continue

    # display summary.
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Files analyzed", str(len(files)))
    table.add_row("Segments (sample)", str(total_segments))
    table.add_row("Hate speech detected", str(hate_speech_count))
    table.add_row("Ads detected", str(ad_count))

    console.print(table)

    if target_groups:
        console.print("\n[cyan]Target groups mentioned:[/cyan]")
        for group, count in sorted(target_groups.items(), key=lambda x: x[1], reverse=True)[:10]:
            console.print(f"  {group}: {count}")


def display_llm_annotation_file(file_path: Path) -> None:
    """Display LLM annotations for a single file."""
    console.clear()
    console.print(f"[bold cyan]📄 {file_path.stem}[/bold cyan]\n")

    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        segments = data.get("segments", [])
        console.print(f"[cyan]Total segments:[/cyan] {len(segments)}\n")

        # display first few segments.
        for i, seg in enumerate(segments[:5], 1):
            console.print(f"[yellow]Segment {i}:[/yellow]")
            console.print(f"  Speaker: {seg.get('speaker', 'Unknown')}")
            console.print(f"  Time: {seg.get('start', 0):.1f}s - {seg.get('end', 0):.1f}s")

            text = seg.get("text", "")
            if len(text) > 150:
                text = text[:147] + "..."
            console.print(f"  Text: {text}")

            # llm annotation.
            llm_ann = seg.get("llm_annotation", {})
            if llm_ann:
                console.print(f"  Hate speech: {llm_ann.get('has_hate_speech', False)}")
                console.print(f"  Advertisement: {llm_ann.get('has_advertisement', False)}")
                if llm_ann.get("target_group"):
                    console.print(f"  Target group: {llm_ann.get('target_group')}")

            # classifications.
            classifications = seg.get("classifications", [])
            if classifications:
                console.print("  Classifications:")
                for cls in classifications[:3]:  # show first 3.
                    model = cls.get("model_name", "Unknown")
                    label = cls.get("label", "Unknown")
                    conf = cls.get("confidence", 0)
                    console.print(f"    {model}: {label} ({conf:.2%})")

            console.print()

        if len(segments) > 5:
            console.print(f"[dim]... and {len(segments) - 5} more segments[/dim]\n")

    except Exception as e:
        console.print(f"[red]Error reading file: {e}[/red]")
