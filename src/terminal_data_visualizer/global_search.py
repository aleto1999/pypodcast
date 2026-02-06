"""Global search across all output types."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table

from podcast_conversations.naming import sanitize_name
from terminal_data_visualizer.config import (
    ANALYSIS_DIR,
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


@dataclass
class SearchMatch:
    """A single search match with context."""

    output_type: str
    show_name: str
    episode_name: str
    file_path: Path
    context: str
    match_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResults:
    """Aggregated search results."""

    query: str
    total_matches: int = 0
    by_output_type: dict[str, int] = field(default_factory=dict)
    by_show: dict[str, int] = field(default_factory=dict)
    matches: list[SearchMatch] = field(default_factory=list)


@dataclass
class SearchFilters:
    """Filters for global search."""

    shows: list[str] | None = None  # None means all shows.
    output_types: list[str] | None = None  # None means all types.
    has_hate_speech: bool | None = None
    speaker: str | None = None
    category: str | None = None
    min_confidence: float | None = None


def get_available_shows() -> list[str]:
    """Get list of available shows."""
    transcripts_path = OUTPUTS_PATH / TRANSCRIPTS_DIR
    if not transcripts_path.exists():
        return []
    return sorted([d.name for d in transcripts_path.iterdir() if d.is_dir()])


def search_transcripts(
    query: str,
    shows: list[str] | None = None,
    max_results: int = 100,
) -> list[SearchMatch]:
    """Search across all transcripts."""
    matches: list[SearchMatch] = []
    pattern = re.compile(query, re.IGNORECASE)

    transcripts_path = OUTPUTS_PATH / TRANSCRIPTS_DIR
    if not transcripts_path.exists():
        return matches

    show_dirs = shows if shows else get_available_shows()

    for show in show_dirs:
        show_path = transcripts_path / show
        if not show_path.exists():
            continue

        for file in show_path.glob("*.json"):
            if len(matches) >= max_results:
                break

            try:
                with open(file, encoding="utf-8") as f:
                    data = json.load(f)

                for seg in data.get("segments", []):
                    text = seg.get("text", "")
                    if pattern.search(text):
                        # extract match context.
                        match_obj = pattern.search(text)
                        start = max(0, match_obj.start() - 50) if match_obj else 0
                        end = min(len(text), (match_obj.end() if match_obj else 0) + 50)
                        context = text[start:end]

                        matches.append(
                            SearchMatch(
                                output_type="Transcript",
                                show_name=show,
                                episode_name=file.stem,
                                file_path=file,
                                context=f"...{context}...",
                                match_text=match_obj.group() if match_obj else "",
                                metadata={
                                    "speaker": seg.get("speaker"),
                                    "start_time": seg.get("start"),
                                },
                            )
                        )

                        if len(matches) >= max_results:
                            break

            except (json.JSONDecodeError, OSError):
                pass

    return matches


def search_keywords(
    query: str,
    shows: list[str] | None = None,
    category: str | None = None,
    max_results: int = 100,
) -> list[SearchMatch]:
    """Search across keyword analysis files."""
    matches: list[SearchMatch] = []
    pattern = re.compile(query, re.IGNORECASE)

    analysis_path = OUTPUTS_PATH / ANALYSIS_DIR
    if not analysis_path.exists():
        return matches

    show_dirs = shows if shows else [d.name for d in analysis_path.iterdir() if d.is_dir()]

    for show in show_dirs:
        show_path = analysis_path / show
        if not show_path.exists():
            continue

        for file in show_path.glob("*.json"):
            if len(matches) >= max_results:
                break

            try:
                with open(file, encoding="utf-8") as f:
                    data = json.load(f)

                for match_data in data.get("matches", []):
                    keyword = match_data.get("keyword", "")
                    match_category = match_data.get("category", "")

                    # filter by category if specified.
                    if category and match_category.lower() != category.lower():
                        continue

                    if pattern.search(keyword) or pattern.search(match_data.get("context", "")):
                        context = match_data.get("context", keyword)
                        if len(context) > 100:
                            context = context[:97] + "..."

                        matches.append(
                            SearchMatch(
                                output_type="Keywords",
                                show_name=show,
                                episode_name=file.stem.replace("_keywords", ""),
                                file_path=file,
                                context=context,
                                match_text=keyword,
                                metadata={
                                    "category": match_category,
                                    "speaker": match_data.get("speaker"),
                                },
                            )
                        )

                        if len(matches) >= max_results:
                            break

            except (json.JSONDecodeError, OSError):
                pass

    return matches


def search_llm_annotations(
    query: str | None = None,
    shows: list[str] | None = None,
    has_hate_speech: bool | None = None,
    max_results: int = 100,
) -> list[SearchMatch]:
    """Search across LLM annotation files."""
    matches: list[SearchMatch] = []
    pattern = re.compile(query, re.IGNORECASE) if query else None

    llm_path = OUTPUTS_PATH / LLM_ANNOTATIONS_DIR
    if not llm_path.exists():
        return matches

    show_dirs = shows if shows else [d.name for d in llm_path.iterdir() if d.is_dir()]

    for show in show_dirs:
        show_path = llm_path / show
        if not show_path.exists():
            continue

        for file in show_path.glob("*.json"):
            if len(matches) >= max_results:
                break

            try:
                with open(file, encoding="utf-8") as f:
                    data = json.load(f)

                for seg in data.get("segments", []):
                    ann = seg.get("llm_annotation", {})

                    # filter by hate speech if specified.
                    if has_hate_speech is not None:
                        if ann.get("has_hate_speech") != has_hate_speech:
                            continue

                    text = seg.get("text", "")

                    # filter by query if specified.
                    if pattern and not pattern.search(text):
                        continue

                    context = text[:100] + "..." if len(text) > 100 else text

                    matches.append(
                        SearchMatch(
                            output_type="LLM Annotations",
                            show_name=show,
                            episode_name=file.stem,
                            file_path=file,
                            context=context,
                            match_text=pattern.search(text).group()
                            if pattern and pattern.search(text)
                            else "",
                            metadata={
                                "has_hate_speech": ann.get("has_hate_speech", False),
                                "has_advertisement": ann.get("has_advertisement", False),
                                "target_group": ann.get("target_group"),
                            },
                        )
                    )

                    if len(matches) >= max_results:
                        break

            except (json.JSONDecodeError, OSError):
                pass

    return matches


def global_search(
    query: str,
    filters: SearchFilters | None = None,
    max_results_per_type: int = 50,
) -> SearchResults:
    """Perform global search across all output types."""
    if filters is None:
        filters = SearchFilters()

    results = SearchResults(query=query)
    all_matches: list[SearchMatch] = []

    output_types = filters.output_types or ["Transcript", "Keywords", "LLM Annotations"]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        # search transcripts.
        if "Transcript" in output_types:
            progress.add_task("Searching transcripts...", total=None)
            transcript_matches = search_transcripts(query, filters.shows, max_results_per_type)
            all_matches.extend(transcript_matches)
            results.by_output_type["Transcripts"] = len(transcript_matches)

        # search keywords.
        if "Keywords" in output_types:
            progress.add_task("Searching keywords...", total=None)
            keyword_matches = search_keywords(
                query, filters.shows, filters.category, max_results_per_type
            )
            all_matches.extend(keyword_matches)
            results.by_output_type["Keywords"] = len(keyword_matches)

        # search llm annotations.
        if "LLM Annotations" in output_types:
            progress.add_task("Searching LLM annotations...", total=None)
            llm_matches = search_llm_annotations(
                query, filters.shows, filters.has_hate_speech, max_results_per_type
            )
            all_matches.extend(llm_matches)
            results.by_output_type["LLM Annotations"] = len(llm_matches)

    # aggregate results.
    results.matches = all_matches
    results.total_matches = len(all_matches)

    for match in all_matches:
        results.by_show[match.show_name] = results.by_show.get(match.show_name, 0) + 1

    return results


def display_search_results(results: SearchResults) -> None:
    """Display search results."""
    console.print(
        Panel(
            f"[bold {COLOR_PRIMARY}]🔍 Search Results[/bold {COLOR_PRIMARY}]",
            border_style="blue",
        )
    )

    console.print(f'\n[bold]Query:[/bold] "{results.query}"')
    console.print(f"[bold]Total Matches:[/bold] {results.total_matches}\n")

    # by output type.
    if results.by_output_type:
        console.print(f"[bold {COLOR_PRIMARY}]By Output Type:[/bold {COLOR_PRIMARY}]")
        for output_type, count in results.by_output_type.items():
            icon = (
                "📝"
                if output_type == "Transcripts"
                else "🔑"
                if output_type == "Keywords"
                else "🤖"
            )
            console.print(f"  {icon} {output_type}: {count}")

    # by show.
    if results.by_show:
        console.print(f"\n[bold {COLOR_PRIMARY}]By Show:[/bold {COLOR_PRIMARY}]")
        total = sum(results.by_show.values())
        for show, count in sorted(results.by_show.items(), key=lambda x: x[1], reverse=True)[:10]:
            pct = (count / total) * 100 if total > 0 else 0
            console.print(f"  {show}: {count} ({pct:.0f}%)")

    # sample matches.
    if results.matches:
        console.print(
            f"\n[bold {COLOR_PRIMARY}]Sample Matches (first 10):[/bold {COLOR_PRIMARY}]\n"
        )

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Type", style="cyan", width=12)
        table.add_column("Show", style="yellow", width=15)
        table.add_column("Episode", width=20)
        table.add_column("Context", width=40)

        for match in results.matches[:10]:
            context = match.context
            if len(context) > 38:
                context = context[:35] + "..."

            table.add_row(
                match.output_type[:10],
                match.show_name[:13],
                match.episode_name[:18],
                context,
            )

        console.print(table)


def global_search_menu() -> None:
    """Interactive global search menu."""
    while True:
        console.clear()
        console.print(
            Panel(
                f"[bold {COLOR_PRIMARY}]🔍 Global Search[/bold {COLOR_PRIMARY}]\n"
                "[dim]Search across all output types[/dim]",
                border_style="blue",
            )
        )

        table = Table(show_header=False, box=None)
        table.add_column("Option", style=COLOR_WARNING)
        table.add_column("Description")

        table.add_row("1", "🔍 Simple search (all outputs)")
        table.add_row("2", "📝 Search transcripts only")
        table.add_row("3", "🔑 Search keywords only")
        table.add_row("4", "🤖 Search LLM annotations only")
        table.add_row("5", "⚠️  Find all hate speech flags")
        table.add_row("6", "🎯 Advanced search with filters")
        table.add_row(KEY_BACK, "⬅️  Back")

        console.print(table)
        console.print()

        choice = Prompt.ask(f"[{COLOR_WARNING}]Select option[/{COLOR_WARNING}]", default=KEY_BACK)

        match choice.lower():
            case "1":
                _simple_search()
            case "2":
                _search_transcripts_only()
            case "3":
                _search_keywords_only()
            case "4":
                _search_llm_only()
            case "5":
                _find_hate_speech()
            case "6":
                _advanced_search()
            case _ if choice.lower() == KEY_BACK:
                break


def _simple_search() -> None:
    """Perform a simple search across all outputs."""
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]Simple Search[/bold {COLOR_PRIMARY}]\n")

    query = Prompt.ask(f"[{COLOR_WARNING}]Enter search query[/{COLOR_WARNING}]")

    if not query.strip():
        return

    console.print(f'\n[dim]Searching for "{query}"...[/dim]\n')

    results = global_search(query)
    display_search_results(results)

    prompt_and_save(console, screen_name=f"search_{sanitize_name(query[:20])}")


def _search_transcripts_only() -> None:
    """Search transcripts only."""
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]Search Transcripts[/bold {COLOR_PRIMARY}]\n")

    query = Prompt.ask(f"[{COLOR_WARNING}]Enter search query[/{COLOR_WARNING}]")

    if not query.strip():
        return

    filters = SearchFilters(output_types=["Transcript"])
    results = global_search(query, filters)

    console.clear()
    display_search_results(results)
    prompt_and_save(console, screen_name=f"search_transcripts_{query[:15]}")


def _search_keywords_only() -> None:
    """Search keywords only."""
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]Search Keywords[/bold {COLOR_PRIMARY}]\n")

    query = Prompt.ask(f"[{COLOR_WARNING}]Enter search query[/{COLOR_WARNING}]")

    if not query.strip():
        return

    category = Prompt.ask(
        f"[{COLOR_WARNING}]Filter by category (leave blank for all)[/{COLOR_WARNING}]",
        default="",
    )

    filters = SearchFilters(
        output_types=["Keywords"],
        category=category if category else None,
    )
    results = global_search(query, filters)

    console.clear()
    display_search_results(results)
    prompt_and_save(console, screen_name=f"search_keywords_{query[:15]}")


def _search_llm_only() -> None:
    """Search LLM annotations only."""
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]Search LLM Annotations[/bold {COLOR_PRIMARY}]\n")

    query = Prompt.ask(f"[{COLOR_WARNING}]Enter search query[/{COLOR_WARNING}]")

    if not query.strip():
        return

    filters = SearchFilters(output_types=["LLM Annotations"])
    results = global_search(query, filters)

    console.clear()
    display_search_results(results)
    prompt_and_save(console, screen_name=f"search_llm_{query[:15]}")


def _find_hate_speech() -> None:
    """Find all hate speech flags."""
    console.clear()
    console.print(f"[bold {COLOR_WARNING}]⚠️ Hate Speech Search[/bold {COLOR_WARNING}]\n")

    matches = search_llm_annotations(
        query=None,
        has_hate_speech=True,
        max_results=200,
    )

    if not matches:
        console.print(f"[{COLOR_SUCCESS}]✓ No hate speech flags found[/{COLOR_SUCCESS}]")
    else:
        console.print(
            f"[{COLOR_WARNING}]Found {len(matches)} segments with hate speech flags[/{COLOR_WARNING}]\n"
        )

        # group by show.
        by_show: dict[str, list[SearchMatch]] = {}
        for match in matches:
            if match.show_name not in by_show:
                by_show[match.show_name] = []
            by_show[match.show_name].append(match)

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Show", style="cyan", width=25)
        table.add_column("Count", justify="right", width=10)
        table.add_column("Episodes", justify="right", width=10)

        for show, show_matches in sorted(by_show.items(), key=lambda x: len(x[1]), reverse=True):
            episodes = len(set(m.episode_name for m in show_matches))
            table.add_row(show, str(len(show_matches)), str(episodes))

        console.print(table)

        # target groups.
        target_groups: dict[str, int] = {}
        for match in matches:
            target = match.metadata.get("target_group")
            if target:
                target_groups[target] = target_groups.get(target, 0) + 1

        if target_groups:
            console.print("\n[bold]Target Groups:[/bold]")
            for target, count in sorted(target_groups.items(), key=lambda x: x[1], reverse=True)[
                :10
            ]:
                console.print(f"  • {target}: {count}")

    prompt_and_save(console, screen_name="hate_speech_search")


def _advanced_search() -> None:
    """Advanced search with multiple filters."""
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]Advanced Search[/bold {COLOR_PRIMARY}]\n")

    query = Prompt.ask(f"[{COLOR_WARNING}]Enter search query[/{COLOR_WARNING}]")

    if not query.strip():
        return

    # select output types.
    console.print("\n[dim]Select output types (comma-separated, or 'all'):[/dim]")
    console.print("  1. Transcripts")
    console.print("  2. Keywords")
    console.print("  3. LLM Annotations")

    types_input = Prompt.ask(
        f"[{COLOR_WARNING}]Output types[/{COLOR_WARNING}]",
        default="all",
    )

    output_types: list[str] | None = None
    if types_input.lower() != "all":
        type_map = {"1": "Transcript", "2": "Keywords", "3": "LLM Annotations"}
        output_types = [
            type_map[t.strip()] for t in types_input.split(",") if t.strip() in type_map
        ]

    # select shows.
    shows = get_available_shows()
    console.print(f"\n[dim]Available shows: {', '.join(shows[:5])}...[/dim]")

    show_filter = Prompt.ask(
        f"[{COLOR_WARNING}]Filter by show (leave blank for all)[/{COLOR_WARNING}]",
        default="",
    )

    selected_shows: list[str] | None = None
    if show_filter.strip():
        selected_shows = [s for s in shows if show_filter.lower() in s.lower()]

    filters = SearchFilters(
        output_types=output_types,
        shows=selected_shows,
    )

    console.print("\n[dim]Searching...[/dim]\n")

    results = global_search(query, filters)

    console.clear()
    display_search_results(results)
    prompt_and_save(console, screen_name=f"advanced_search_{query[:15]}")
