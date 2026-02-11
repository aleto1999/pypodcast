"""Interactive menu system for terminal data visualizer."""

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table

from terminal_data_visualizer.bookmarks import bookmarks_menu
from terminal_data_visualizer.cache import get_episode_stats
from terminal_data_visualizer.classification_summary import classification_summary_menu
from terminal_data_visualizer.classification_viewer import classification_viewer_menu
from terminal_data_visualizer.config import (
    COLOR_ERROR,
    COLOR_PRIMARY,
    COLOR_WARNING,
    KEY_BACK,
    KEY_QUIT,
    OUTPUTS_PATH,
)
from terminal_data_visualizer.dashboard import dashboard_menu
from terminal_data_visualizer.data_export import (
    export_category_distribution,
    export_episode_stats_to_csv,
    export_episode_stats_to_json,
    get_export_path,
)
from terminal_data_visualizer.episode_view import episode_view_menu
from terminal_data_visualizer.exporter import (
    export_to_json,
    format_categories_for_export,
    format_folder_stats_for_export,
    format_keywords_for_export,
    format_matches_for_export,
    format_utterance_for_export,
)
from terminal_data_visualizer.features_explorer import features_explorer_menu
from terminal_data_visualizer.global_search import global_search_menu as advanced_search_menu
from terminal_data_visualizer.keyword_browser import (
    CategoryStats,
    display_categories_table,
    display_categories_with_stats,
    display_category_files,
    display_category_matches,
    display_keyword_matches,
    display_keywords_table,
    display_match_detail,
    display_utterance_detail,
    get_all_categories_parallel,
    get_all_keywords_parallel,
    get_category_stats_parallel,
    get_matches_for_category,
    get_subfolders,
    parse_analysis_file,
    search_keywords_parallel,
)
from terminal_data_visualizer.llm_annotation_viewer import llm_annotation_viewer_menu
from terminal_data_visualizer.outputs_viewer import (
    view_document_labels_menu,
    view_features_csv_menu,
    view_llm_annotations_menu,
)
from terminal_data_visualizer.report_generator import report_generator_menu
from terminal_data_visualizer.scanner import scan_directory_parallel
from terminal_data_visualizer.screen_capture import prompt_and_save
from terminal_data_visualizer.selectors import (
    confirm_action,
    select_podcast,
    select_podcast_and_episode,
)
from terminal_data_visualizer.statistics import (
    compare_episodes,
    display_category_distribution,
    display_keyword_timeline,
    display_speaker_distribution,
)
from terminal_data_visualizer.summary_generator import summary_generator_menu
from terminal_data_visualizer.visualizations import (
    heatmap_menu,
    interactions_menu,
    timeline_menu,
)
from terminal_data_visualizer.visualizer import (
    display_analysis_tree,
    display_file_contents,
    display_folder_stats,
    scan_with_progress,
)

console = Console(record=True)

# default paths.
ANALYSIS_PATH = OUTPUTS_PATH / "analysis"


# track current search state for export.
class SearchState:
    """Track current search state for export functionality."""

    keywords: list[str] = []
    podcast: str = ""
    context_n: int = 0
    matches: list[tuple[Path, Any]] = []


search_state = SearchState()


def clear_screen() -> None:
    """Clear the terminal screen."""
    console.clear()


def display_main_menu() -> None:
    """Display the main menu."""
    console.print(
        Panel(
            f"[bold {COLOR_PRIMARY}]Terminal Data Visualizer v2.2.3[/bold {COLOR_PRIMARY}]\n"
            "[dim]Explore outputs directory with parallel processing[/dim]",
            border_style="blue",
        )
    )

    table = Table(show_header=False, box=None)
    table.add_column("Option", style=COLOR_WARNING)
    table.add_column("Description")

    table.add_row("1", "📊 Outputs Dashboard (health & completeness)")
    table.add_row("2", "📄 Episode View (all outputs for one episode)")
    table.add_row("3", "🔍 Global Search (across all output types)")
    table.add_row("4", "📈 Summary Generator (show summaries)")
    table.add_row("5", "📝 Report Generator (research reports)")
    table.add_row("6", "📊 Visualizations (timelines, heatmaps)")
    table.add_row("7", "📌 Bookmarks (saved views)")
    table.add_row("8", "🔬 Features Explorer (CSV data analysis)")
    table.add_row("9", "🏷️  Classification Summary (pipeline statistics)")
    table.add_row("10", "📂 Browse & Search (legacy menus)")
    table.add_row(KEY_QUIT, "❌ Quit")

    console.print(table)
    console.print()


def scan_outputs_menu() -> None:
    """Scan and display outputs directory stats with drill-down navigation."""
    _browse_folder_stats(OUTPUTS_PATH, "📁 Outputs Directory Overview")


def _browse_folder_stats(current_path: Path, title: str) -> None:
    """Browse folder statistics with drill-down into subfolders.

    Args:
        current_path: The directory to scan and display.
        title: Title to show in the display.
    """
    while True:
        clear_screen()
        console.print(f"[bold]Scanning {current_path.name}/...[/bold]\n")

        if not current_path.exists():
            console.print(f"[red]{current_path} not found![/red]")
            Prompt.ask("\n[dim]Press Enter to return[/dim]")
            return

        stats = scan_with_progress(current_path)

        if not stats:
            # no subfolders, show files instead.
            console.print(f"[yellow]No subfolders in {current_path.name}/[/yellow]\n")
            _show_folder_files(current_path)
            Prompt.ask("\n[dim]Press Enter to return[/dim]")
            return

        # display folder stats with numbered rows.
        sorted_folders = display_folder_stats(stats, title=title, show_numbers=True)

        # show navigation options.
        console.print("\n[yellow]Options:[/yellow]")
        console.print("  [cyan]#[/cyan]   - Drill into folder by number")
        console.print("  [cyan]f #[/cyan] - Show files in folder #")
        console.print("  [cyan]e[/cyan]   - Export current stats")
        console.print("  [cyan]b[/cyan]   - Back to parent / main menu")

        choice = Prompt.ask(
            "\n[yellow]Enter choice[/yellow]",
            default="b",
        )

        if choice.lower() == "b":
            return

        elif choice.lower() == "e":
            export_data = format_folder_stats_for_export(stats, title=title)
            export_to_json(export_data, prefix="folder_stats")
            Prompt.ask("\n[dim]Press Enter to continue[/dim]")

        elif choice.lower().startswith("f "):
            # show files in specific folder.
            try:
                idx = int(choice[2:].strip()) - 1
                if 0 <= idx < len(sorted_folders):
                    folder_name = sorted_folders[idx]
                    folder_path = current_path / folder_name
                    clear_screen()
                    _show_folder_files(folder_path)
                    Prompt.ask("\n[dim]Press Enter to continue[/dim]")
                else:
                    console.print("[red]Invalid folder number.[/red]")
            except ValueError:
                console.print("[red]Please enter a valid number after 'f'.[/red]")

        elif choice.isdigit():
            # drill into subfolder.
            idx = int(choice) - 1
            if 0 <= idx < len(sorted_folders):
                folder_name = sorted_folders[idx]
                subfolder_path = current_path / folder_name
                _browse_folder_stats(subfolder_path, f"📁 {folder_name}/")
            else:
                console.print("[red]Invalid folder number.[/red]")

        else:
            console.print("[red]Invalid option.[/red]")


def _show_folder_files(folder_path: Path, max_files: int = 50) -> None:
    """Show files in a folder with sizes.

    Args:
        folder_path: Path to the folder.
        max_files: Maximum number of files to display.
    """
    console.print(f"[bold cyan]Files in {folder_path.name}/[/bold cyan]\n")

    # get all files (not directories).
    all_items = sorted(folder_path.iterdir(), key=lambda x: x.name)
    files = [f for f in all_items if f.is_file()]
    dirs = [d for d in all_items if d.is_dir()]

    # show subdirectories first.
    if dirs:
        console.print(f"[dim]Subdirectories: {len(dirs)}[/dim]")
        for d in dirs[:10]:
            console.print(f"  📁 {d.name}/")
        if len(dirs) > 10:
            console.print(f"  [dim]... and {len(dirs) - 10} more directories[/dim]")
        console.print()

    if not files:
        console.print("[yellow]No files in this folder.[/yellow]")
        return

    # create table for files.
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", style="yellow", justify="right", width=4)
    table.add_column("File", style="green")
    table.add_column("Size", justify="right", style="cyan")
    table.add_column("Modified", style="dim")

    from datetime import datetime

    display_files = files[:max_files]
    total_size = 0

    for idx, f in enumerate(display_files, 1):
        size = f.stat().st_size
        total_size += size
        modified = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")

        # format size.
        if size >= 1024 * 1024 * 1024:
            size_str = f"{size / (1024**3):.2f} GB"
        elif size >= 1024 * 1024:
            size_str = f"{size / (1024**2):.2f} MB"
        elif size >= 1024:
            size_str = f"{size / 1024:.2f} KB"
        else:
            size_str = f"{size} B"

        table.add_row(str(idx), f.name, size_str, modified)

    console.print(table)

    # show summary.
    if len(files) > max_files:
        console.print(f"\n[dim]Showing {max_files} of {len(files)} files[/dim]")

    # format total size.
    if total_size >= 1024 * 1024 * 1024:
        total_str = f"{total_size / (1024**3):.2f} GB"
    elif total_size >= 1024 * 1024:
        total_str = f"{total_size / (1024**2):.2f} MB"
    else:
        total_str = f"{total_size / 1024:.2f} KB"

    console.print(f"\n[dim]Total: {len(files)} files, {total_str}[/dim]")


def view_analysis_tree_menu() -> None:
    """View analysis folder as a tree."""
    clear_screen()
    console.print("[bold]Analysis Folder Structure[/bold]\n")

    display_analysis_tree(ANALYSIS_PATH)

    Prompt.ask("\n[dim]Press Enter to return to menu[/dim]")


def browse_analysis_files_menu() -> None:
    """Browse and view analysis files with keyword search."""
    while True:
        clear_screen()
        console.print(
            Panel(
                "[bold cyan]Analysis Files Browser[/bold cyan]\n"
                "[dim]Browse keyword analysis by podcast and keywords[/dim]",
                border_style="blue",
            )
        )

        # display subfolders.
        subfolders = get_subfolders()
        if not subfolders:
            console.print("[yellow]No keyword analysis folders found.[/yellow]")
            Prompt.ask("\n[dim]Press Enter to return to menu[/dim]")
            return

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", style="yellow", justify="right")
        table.add_column("Podcast", style="cyan")

        for idx, folder in enumerate(subfolders, 1):
            table.add_row(str(idx), folder.name)

        console.print(table)
        console.print(f"\n[dim]Total podcasts: {len(subfolders)}[/dim]\n")

        # display options.
        options_table = Table(show_header=False, box=None)
        options_table.add_column("Option", style="yellow")
        options_table.add_column("Description")
        options_table.add_row("k", "🔑 Show all keywords analyzed")
        options_table.add_row("c", "📂 Show all categories")
        options_table.add_row("s", "🔍 Search keywords in a podcast")
        options_table.add_row("b", "⬅️  Back to main menu")
        console.print(options_table)

        choice = Prompt.ask("\n[yellow]Enter podcast # or option[/yellow]", default="b")

        if choice.lower() == "b":
            return
        elif choice.lower() == "k":
            show_all_keywords_menu()
        elif choice.lower() == "c":
            show_all_categories_menu()
        elif choice.lower() == "s":
            search_keywords_in_podcast_menu(subfolders)
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(subfolders):
                    browse_podcast_folder_menu(subfolders[idx])
                else:
                    console.print("[red]Invalid selection.[/red]")
            except ValueError:
                console.print("[red]Please enter a valid number or option.[/red]")


def show_all_keywords_menu() -> None:
    """Show all unique keywords across all analysis files."""
    clear_screen()
    console.print("[bold]Collecting all keywords...[/bold]\n")

    all_keywords: set[str] = set()
    subfolders = get_subfolders()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Scanning folders...", total=len(subfolders))
        for folder in subfolders:
            keywords = get_all_keywords_parallel(folder)
            all_keywords.update(keywords)
            progress.advance(task)

    display_keywords_table(all_keywords, title="🔑 All Keywords Analyzed")

    # offer export.
    export_choice = Prompt.ask(
        "\n[yellow]'e' to export keywords, Enter to continue[/yellow]",
        default="",
    )
    if export_choice.lower() == "e":
        export_data = format_keywords_for_export(all_keywords, title="All Keywords Analyzed")
        export_to_json(export_data, prefix="all_keywords")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def show_all_categories_menu() -> None:
    """Show all unique categories across all analysis files with drill-down navigation."""
    clear_screen()
    console.print("[bold]Collecting category statistics...[/bold]\n")

    # collect category stats from all folders.
    all_stats: dict[str, CategoryStats] = {}
    subfolders = get_subfolders()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Scanning folders...", total=len(subfolders))
        for folder in subfolders:
            folder_stats = get_category_stats_parallel(folder)
            # merge stats.
            for cat, stats in folder_stats.items():
                if cat not in all_stats:
                    all_stats[cat] = CategoryStats(
                        category=cat, match_count=0, file_count=0, files=[]
                    )
                all_stats[cat].match_count += stats.match_count
                all_stats[cat].file_count += stats.file_count
                all_stats[cat].files.extend(stats.files)
            progress.advance(task)

    # main category browsing loop.
    while True:
        clear_screen()
        sorted_categories = display_categories_with_stats(all_stats, title="📂 All Categories")

        if not sorted_categories:
            Prompt.ask("\n[dim]Press Enter to continue[/dim]")
            return

        console.print("\n[yellow]Options:[/yellow]")
        console.print("  [cyan]#[/cyan] - Select category by number")
        console.print("  [cyan]e[/cyan] - Export categories")
        console.print("  [cyan]q[/cyan] - Return to menu")

        choice = Prompt.ask(
            "\n[yellow]Enter choice[/yellow]",
            default="q",
        )

        if choice.lower() == "q":
            return
        elif choice.lower() == "e":
            # export all categories.
            all_categories = set(all_stats.keys())
            export_data = format_categories_for_export(all_categories, title="All Categories")
            export_to_json(export_data, prefix="all_categories")
            Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(sorted_categories):
                category = sorted_categories[idx]
                _browse_category_files(category, all_stats[category])


def _browse_category_files(category: str, stats: CategoryStats) -> None:
    """Browse files containing matches for a specific category."""
    # collect match counts per file.
    file_counts: dict[Path, int] = {}
    for file_path in stats.files:
        matches = get_matches_for_category(file_path, category)
        if matches:
            file_counts[file_path] = len(matches)

    while True:
        clear_screen()
        files_with_counts = [(f, c) for f, c in file_counts.items()]
        sorted_files = display_category_files(category, files_with_counts)

        if not sorted_files:
            Prompt.ask("\n[dim]Press Enter to continue[/dim]")
            return

        console.print("\n[yellow]Options:[/yellow]")
        console.print("  [cyan]#[/cyan] - Select file by number")
        console.print("  [cyan]e[/cyan] - Export file list")
        console.print("  [cyan]b[/cyan] - Back to categories")

        choice = Prompt.ask(
            "\n[yellow]Enter choice[/yellow]",
            default="b",
        )

        if choice.lower() == "b":
            return
        elif choice.lower() == "e":
            # export files list.
            export_data = {
                "category": category,
                "total_files": len(sorted_files),
                "total_matches": sum(file_counts.values()),
                "files": [
                    {"file": f.name, "path": str(f), "matches": file_counts[f]}
                    for f in sorted_files
                ],
            }
            export_to_json(export_data, prefix=f"category_{category}_files")
            Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(sorted_files):
                file_path = sorted_files[idx]
                _browse_file_matches(category, file_path)


def _browse_file_matches(category: str, file_path: Path) -> None:
    """Browse matches for a category within a specific file."""
    matches = get_matches_for_category(file_path, category)

    while True:
        clear_screen()
        displayed_matches = display_category_matches(category, matches, file_path.name)

        if not displayed_matches:
            Prompt.ask("\n[dim]Press Enter to continue[/dim]")
            return

        console.print("\n[yellow]Options:[/yellow]")
        console.print("  [cyan]#[/cyan] - View match details")
        console.print("  [cyan]e[/cyan] - Export matches")
        console.print("  [cyan]b[/cyan] - Back to files")

        choice = Prompt.ask(
            "\n[yellow]Enter choice[/yellow]",
            default="b",
        )

        if choice.lower() == "b":
            return
        elif choice.lower() == "e":
            # export matches.
            export_data = {
                "category": category,
                "file": file_path.name,
                "path": str(file_path),
                "total_matches": len(matches),
                "matches": [
                    {
                        "keyword": m.keyword,
                        "matched_text": m.matched_text,
                        "speaker": m.speaker,
                        "start_time": m.start_time,
                        "end_time": m.end_time,
                        "confidence_tier": m.confidence_tier,
                        "context_before": m.context_before,
                        "context_after": m.context_after,
                    }
                    for m in matches
                ],
            }
            export_to_json(export_data, prefix=f"category_{category}_matches")
            Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(displayed_matches):
                match = displayed_matches[idx]
                _view_match_detail(match, file_path.name)


def _view_match_detail(match: Any, file_name: str) -> None:
    """View detailed information for a single match."""
    clear_screen()
    display_match_detail(match, file_name)

    console.print("\n[yellow]Options:[/yellow]")
    console.print("  [cyan]e[/cyan] - Export match")
    console.print("  [cyan]Enter[/cyan] - Back to matches")

    choice = Prompt.ask(
        "\n[yellow]Enter choice[/yellow]",
        default="",
    )

    if choice.lower() == "e":
        export_data = {
            "file": file_name,
            "keyword": match.keyword,
            "category": match.category,
            "matched_text": match.matched_text,
            "speaker": match.speaker,
            "start_time": match.start_time,
            "end_time": match.end_time,
            "confidence_tier": match.confidence_tier,
            "context_before": match.context_before,
            "context_after": match.context_after,
        }
        export_to_json(export_data, prefix="match_detail")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def search_keywords_in_podcast_menu(subfolders: list[Path]) -> None:
    """Search for specific keywords in a selected podcast."""
    clear_screen()
    console.print("[bold]Search Keywords in Podcast[/bold]\n")

    # display podcasts.
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", style="yellow", justify="right")
    table.add_column("Podcast", style="cyan")

    for idx, folder in enumerate(subfolders, 1):
        table.add_row(str(idx), folder.name)

    console.print(table)

    # select podcast.
    podcast_choice = Prompt.ask(
        "\n[yellow]Enter podcast number[/yellow]",
        default="1",
    )

    try:
        idx = int(podcast_choice) - 1
        if not (0 <= idx < len(subfolders)):
            console.print("[red]Invalid podcast selection.[/red]")
            Prompt.ask("\n[dim]Press Enter to continue[/dim]")
            return
    except ValueError:
        console.print("[red]Please enter a valid number.[/red]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    selected_folder = subfolders[idx]
    console.print(f"\n[cyan]Selected:[/cyan] {selected_folder.name}")

    # enter keywords.
    keywords_input = Prompt.ask(
        "\n[yellow]Enter keywords (comma-separated, e.g., 'gay, migrant')[/yellow]",
        default="",
    )

    if not keywords_input.strip():
        console.print("[yellow]No keywords entered.[/yellow]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]
    console.print(f"\n[cyan]Searching for:[/cyan] {', '.join(keywords)}\n")

    # search with progress.
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Searching...", total=None)
        matches = search_keywords_parallel(selected_folder, keywords)

    display_keyword_matches(
        matches, title=f"🔍 Matches for '{', '.join(keywords)}' in {selected_folder.name}"
    )

    # allow viewing individual matches with export option.
    if matches:
        browse_matches_detail(matches, keywords=keywords, podcast=selected_folder.name)
    else:
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def browse_matches_detail(
    matches: list[tuple[Path, Any]],
    transcript_file: str = "",
    keywords: list[str] | None = None,
    podcast: str = "",
) -> None:
    """Browse individual match details with context and export options."""
    # ask for context setting once.
    context_n = 0
    context_choice = Prompt.ask(
        "\n[yellow]Enter number of context segments (n) to show before/after match "
        "(0 for none)[/yellow]",
        default="2",
    )
    try:
        context_n = int(context_choice)
    except ValueError:
        context_n = 0

    # update search state for export.
    search_state.context_n = context_n
    search_state.matches = matches
    if keywords:
        search_state.keywords = keywords
    search_state.podcast = podcast

    while True:
        choice = Prompt.ask(
            "\n[yellow]Enter match # to view, 'n' to change context, "
            "'e' to export results, or 'b' to go back[/yellow]",
            default="b",
        )

        if choice.lower() == "b":
            return

        if choice.lower() == "e":
            # export current matches.
            export_data = format_matches_for_export(
                matches,
                keywords=search_state.keywords,
                podcast_name=podcast,
                context_n=context_n,
            )
            export_to_json(export_data, prefix="keyword_matches")
            Prompt.ask("\n[dim]Press Enter to continue[/dim]")
            continue

        if choice.lower() == "n":
            new_n = Prompt.ask(
                "[yellow]Enter new context value (n)[/yellow]",
                default=str(context_n),
            )
            try:
                context_n = int(new_n)
                search_state.context_n = context_n
                console.print(f"[green]Context set to {context_n} segments.[/green]")
            except ValueError:
                console.print("[red]Invalid number.[/red]")
            continue

        try:
            idx = int(choice) - 1
            if 0 <= idx < min(len(matches), 100):
                clear_screen()
                file_path, match = matches[idx]

                # get transcript file from analysis if not provided.
                tf = transcript_file
                if not tf:
                    analysis = parse_analysis_file(file_path)
                    if analysis:
                        tf = analysis.transcript_file

                segments, target_id = display_utterance_detail(
                    file_path, match, context_n=context_n, transcript_file=tf
                )

                # offer to export this utterance detail.
                export_choice = Prompt.ask(
                    "\n[yellow]'e' to export this utterance, Enter to continue[/yellow]",
                    default="",
                )
                if export_choice.lower() == "e":
                    export_data = format_utterance_for_export(
                        file_path, match, segments, target_id, context_n, tf
                    )
                    export_to_json(export_data, prefix="utterance_detail")

                # redisplay matches table.
                clear_screen()
                display_keyword_matches(matches, title="🔍 Search Results")
            else:
                console.print("[red]Invalid selection.[/red]")
        except ValueError:
            console.print("[red]Please enter a valid number.[/red]")


def browse_podcast_folder_menu(folder: Path) -> None:
    """Browse a specific podcast's keyword analysis files."""
    while True:
        clear_screen()
        console.print(
            Panel(
                f"[bold cyan]{folder.name}[/bold cyan]\n"
                "[dim]Keyword analysis for this podcast[/dim]",
                border_style="blue",
            )
        )

        # show folder stats.
        json_files = list(folder.glob("*.json"))
        console.print(f"[dim]Total files: {len(json_files)}[/dim]\n")

        # show keywords in this folder.
        keywords = get_all_keywords_parallel(folder)

        options_table = Table(show_header=False, box=None)
        options_table.add_column("Option", style="yellow")
        options_table.add_column("Description")
        options_table.add_row("k", f"🔑 Show keywords ({len(keywords)} unique)")
        options_table.add_row("c", "📂 Show categories")
        options_table.add_row("s", "🔍 Search specific keywords")
        options_table.add_row("f", "📄 List all files")
        options_table.add_row("b", "⬅️  Back")
        console.print(options_table)

        choice = Prompt.ask("\n[yellow]Select option[/yellow]", default="b")

        match choice.lower():
            case "b":
                return
            case "k":
                clear_screen()
                display_keywords_table(keywords, title=f"🔑 Keywords in {folder.name}")
                Prompt.ask("\n[dim]Press Enter to continue[/dim]")
            case "c":
                clear_screen()
                categories = get_all_categories_parallel(folder)
                display_categories_table(categories, title=f"📂 Categories in {folder.name}")
                Prompt.ask("\n[dim]Press Enter to continue[/dim]")
            case "s":
                search_in_single_folder(folder)
            case "f":
                list_folder_files(folder, json_files)
            case _:
                console.print("[red]Invalid option.[/red]")


def search_in_single_folder(folder: Path) -> None:
    """Search for keywords in a single folder."""
    clear_screen()
    console.print(f"[bold]Search in {folder.name}[/bold]\n")

    keywords_input = Prompt.ask(
        "[yellow]Enter keywords (comma-separated)[/yellow]",
        default="",
    )

    if not keywords_input.strip():
        return

    keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Searching...", total=None)
        matches = search_keywords_parallel(folder, keywords)

    display_keyword_matches(matches, title=f"🔍 Results for '{', '.join(keywords)}'")

    if matches:
        browse_matches_detail(matches, keywords=keywords, podcast=folder.name)
    else:
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def list_folder_files(folder: Path, json_files: list[Path]) -> None:
    """List and browse files in a folder."""
    while True:
        clear_screen()
        console.print(f"[bold]Files in {folder.name}[/bold]\n")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", style="yellow", justify="right")
        table.add_column("File", style="green")

        display_files = json_files[:50]  # limit display.
        for idx, f in enumerate(display_files, 1):
            table.add_row(str(idx), f.stem)

        console.print(table)
        if len(json_files) > 50:
            console.print(f"\n[dim]Showing 50 of {len(json_files)} files[/dim]")

        choice = Prompt.ask(
            "\n[yellow]Enter file # to view, or 'b' to go back[/yellow]",
            default="b",
        )

        if choice.lower() == "b":
            return

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(display_files):
                clear_screen()
                display_file_contents(display_files[idx])
                Prompt.ask("\n[dim]Press Enter to continue[/dim]")
            else:
                console.print("[red]Invalid selection.[/red]")
        except ValueError:
            console.print("[red]Please enter a valid number.[/red]")


def search_file_menu() -> None:
    """Search for a specific file."""
    clear_screen()
    console.print("[bold]Search and View File[/bold]\n")

    file_path_str = Prompt.ask(
        "[yellow]Enter file path (relative to outputs/)[/yellow]",
        default="",
    )

    if not file_path_str:
        return

    file_path = OUTPUTS_PATH / file_path_str

    if file_path.exists() and file_path.is_file():
        display_file_contents(file_path)
    elif file_path.exists() and file_path.is_dir():
        console.print(f"\n[cyan]Directory contents of {file_path}:[/cyan]\n")
        stats = scan_directory_parallel(file_path)
        if stats:
            display_folder_stats(stats, title=f"📁 {file_path.name}")
        else:
            # list files directly.
            files = list(file_path.iterdir())[:20]
            for f in files:
                icon = "📁" if f.is_dir() else "📄"
                console.print(f"  {icon} {f.name}")
            if len(list(file_path.iterdir())) > 20:
                console.print("  [dim]... and more[/dim]")
    else:
        console.print(f"[red]Path not found: {file_path}[/red]")

    Prompt.ask("\n[dim]Press Enter to return to menu[/dim]")


def global_keyword_search_menu() -> None:
    """Search keywords across all podcasts."""
    clear_screen()
    console.print(
        Panel(
            "[bold cyan]Global Keyword Search[/bold cyan]\n"
            "[dim]Search for keywords across all podcasts[/dim]",
            border_style="blue",
        )
    )

    keywords_input = Prompt.ask(
        "[yellow]Enter keywords (comma-separated, e.g., 'gay, migrant, immigrant')[/yellow]",
        default="",
    )

    if not keywords_input.strip():
        return

    keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]
    console.print(f"\n[cyan]Searching for:[/cyan] {', '.join(keywords)}\n")

    all_matches: list[tuple[Path, Any]] = []
    subfolders = get_subfolders()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Searching all podcasts...", total=len(subfolders))
        for folder in subfolders:
            matches = search_keywords_parallel(folder, keywords)
            all_matches.extend(matches)
            progress.advance(task)

    console.print(f"\n[green]Found {len(all_matches)} matches across all podcasts.[/green]\n")
    display_keyword_matches(all_matches, title=f"🔍 Global results for '{', '.join(keywords)}'")

    if all_matches:
        browse_matches_detail(all_matches, keywords=keywords, podcast="all_podcasts")
    else:
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def visualizations_menu() -> None:
    """Interactive menu for statistics and visualizations."""
    while True:
        clear_screen()
        console.print(
            Panel(
                f"[bold {COLOR_PRIMARY}]📈 Statistics & Visualizations[/bold {COLOR_PRIMARY}]\n"
                "[dim]Generate charts and analyze data[/dim]",
                border_style="blue",
            )
        )

        table = Table(show_header=False, box=None)
        table.add_column("Option", style=COLOR_WARNING)
        table.add_column("Description")

        table.add_row("1", "📊 Speaker distribution (by episode)")
        table.add_row("2", "📂 Category distribution (by keyword file)")
        table.add_row("3", "⏱️  Keyword timeline (by keyword file)")
        table.add_row("4", "📈 Compare multiple episodes")
        table.add_row(KEY_BACK, "⬅️  Back to main menu")

        console.print(table)
        console.print()

        choice = Prompt.ask(f"[{COLOR_WARNING}]Select option[/{COLOR_WARNING}]", default=KEY_BACK)

        match choice.lower():
            case "1":
                speaker_distribution_menu()
            case "2":
                category_distribution_menu()
            case "3":
                keyword_timeline_menu()
            case "4":
                compare_episodes_menu()
            case _ if choice.lower() == KEY_BACK:
                break
            case _:
                console.print(f"[{COLOR_ERROR}]Invalid option. Please try again.[/{COLOR_ERROR}]")
                Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def speaker_distribution_menu() -> None:
    """Display speaker distribution for a selected episode."""
    clear_screen()
    console.print(f"[bold {COLOR_PRIMARY}]Speaker Distribution[/bold {COLOR_PRIMARY}]\n")

    # use selector to get podcast and episode.
    result = select_podcast_and_episode()
    if not result:
        return

    podcast, episode = result

    # calculate and display statistics.
    clear_screen()
    stats = get_episode_stats(episode)
    if stats:
        display_speaker_distribution(stats)

        # offer export option.
        if confirm_action("Export statistics to file?", default=False):
            export_format = Prompt.ask(
                f"[{COLOR_WARNING}]Export format (csv/json)[/{COLOR_WARNING}]",
                choices=["csv", "json"],
                default="csv",
            )

            if export_format == "csv":
                output_file = get_export_path(f"speaker_stats_{stats.filename}", "csv")
                export_episode_stats_to_csv([stats], output_file)
            else:
                output_file = get_export_path(f"speaker_stats_{stats.filename}", "json")
                export_episode_stats_to_json([stats], output_file)
    else:
        msg = "Could not calculate statistics for this episode"
        console.print(f"[{COLOR_WARNING}]{msg}[/{COLOR_WARNING}]")

    prompt_and_save(
        console,
        screen_name=f"speaker_distribution_{stats.filename}" if stats else "speaker_distribution",
    )


def category_distribution_menu() -> None:
    """Display category distribution from keyword analysis."""
    clear_screen()
    console.print(f"[bold {COLOR_PRIMARY}]Category Distribution[/bold {COLOR_PRIMARY}]\n")

    # use selector to get podcast.
    podcast = select_podcast("Select Podcast for Category Distribution")
    if not podcast:
        return

    # get keyword analysis files.
    keyword_files = sorted(podcast.glob("*.json"))
    if not keyword_files:
        console.print(f"[{COLOR_WARNING}]No keyword analysis files found[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    # let user select file.
    console.print(f"\n[{COLOR_PRIMARY}]Files in {podcast.name}:[/{COLOR_PRIMARY}]\n")
    for i, file in enumerate(keyword_files[:20], 1):
        console.print(f"  {i}. {file.stem}")

    file_idx = Prompt.ask(
        f"\n[{COLOR_WARNING}]Select file number (or '{KEY_BACK}' to go back)[/{COLOR_WARNING}]",
        default=KEY_BACK,
    )

    if file_idx.lower() == KEY_BACK:
        return

    try:
        idx = int(file_idx) - 1
        if idx < 0 or idx >= len(keyword_files):
            console.print(f"[{COLOR_ERROR}]Invalid selection[/{COLOR_ERROR}]")
            Prompt.ask("\n[dim]Press Enter to continue[/dim]")
            return
        keyword_file = keyword_files[idx]
    except ValueError:
        console.print(f"[{COLOR_ERROR}]Invalid input[/{COLOR_ERROR}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    # display distribution.
    clear_screen()
    display_category_distribution(keyword_file)

    # offer export option.
    if confirm_action("Export category distribution?", default=False):
        # load data for export.
        import json
        from collections import Counter

        try:
            with open(keyword_file, encoding="utf-8") as f:
                data = json.load(f)

            matches = data.get("matches", [])
            if matches:
                category_counts: Counter[str] = Counter()
                for match in matches:
                    category = match.get("category", "unknown")
                    category_counts[category] += 1

                export_format = Prompt.ask(
                    f"[{COLOR_WARNING}]Export format (csv/json)[/{COLOR_WARNING}]",
                    choices=["csv", "json"],
                    default="csv",
                )

                output_file = get_export_path(f"category_dist_{keyword_file.stem}", export_format)
                export_category_distribution(
                    dict(category_counts), output_file, format=export_format
                )
        except Exception as e:
            console.print(f"[{COLOR_ERROR}]Could not export data: {e}[/{COLOR_ERROR}]")

    prompt_and_save(console, screen_name=f"category_distribution_{keyword_file.stem}")


def keyword_timeline_menu() -> None:
    """Display keyword timeline from keyword analysis."""
    clear_screen()
    console.print(f"[bold {COLOR_PRIMARY}]Keyword Timeline[/bold {COLOR_PRIMARY}]\n")

    # use selector to get podcast.
    podcast = select_podcast("Select Podcast for Keyword Timeline")
    if not podcast:
        return

    # get keyword analysis files.
    keyword_files = sorted(podcast.glob("*.json"))
    if not keyword_files:
        console.print(f"[{COLOR_WARNING}]No keyword analysis files found[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    # let user select file.
    console.print(f"\n[{COLOR_PRIMARY}]Files in {podcast.name}:[/{COLOR_PRIMARY}]\n")
    for i, file in enumerate(keyword_files[:20], 1):
        console.print(f"  {i}. {file.stem}")

    file_idx = Prompt.ask(
        f"\n[{COLOR_WARNING}]Select file number (or '{KEY_BACK}' to go back)[/{COLOR_WARNING}]",
        default=KEY_BACK,
    )

    if file_idx.lower() == KEY_BACK:
        return

    try:
        idx = int(file_idx) - 1
        if idx < 0 or idx >= len(keyword_files):
            console.print(f"[{COLOR_ERROR}]Invalid selection[/{COLOR_ERROR}]")
            Prompt.ask("\n[dim]Press Enter to continue[/dim]")
            return
        keyword_file = keyword_files[idx]
    except ValueError:
        console.print(f"[{COLOR_ERROR}]Invalid input[/{COLOR_ERROR}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    # display timeline.
    clear_screen()
    display_keyword_timeline(keyword_file)
    prompt_and_save(console, screen_name=f"keyword_timeline_{keyword_file.stem}")


def compare_episodes_menu() -> None:
    """Compare statistics across multiple episodes."""
    clear_screen()
    console.print(f"[bold {COLOR_PRIMARY}]Compare Episodes[/bold {COLOR_PRIMARY}]\n")

    # use selector to get podcast.
    podcast = select_podcast("Select Podcast to Compare Episodes")
    if not podcast:
        return

    # get transcript files for this podcast.
    from terminal_data_visualizer.selectors import get_transcript_files

    transcript_files = get_transcript_files(podcast.name)
    if not transcript_files:
        console.print(f"[{COLOR_WARNING}]No transcript files found[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    # compare all episodes.
    clear_screen()
    console.print(f"[{COLOR_PRIMARY}]Comparing episodes from {podcast.name}...[/{COLOR_PRIMARY}]\n")

    # use cached stats for better performance.
    stats_list = []
    for transcript_file in transcript_files:
        stats = get_episode_stats(transcript_file)
        if stats:
            stats_list.append(stats)

    if stats_list:
        compare_episodes(transcript_files)

        # offer export option.
        if confirm_action("Export comparison to file?", default=False):
            export_format = Prompt.ask(
                f"[{COLOR_WARNING}]Export format (csv/json)[/{COLOR_WARNING}]",
                choices=["csv", "json"],
                default="csv",
            )

            if export_format == "csv":
                output_file = get_export_path(f"episode_comparison_{podcast.name}", "csv")
                export_episode_stats_to_csv(stats_list, output_file)
            else:
                output_file = get_export_path(f"episode_comparison_{podcast.name}", "json")
                export_episode_stats_to_json(stats_list, output_file)
    else:
        msg = "Could not calculate statistics for episodes"
        console.print(f"[{COLOR_WARNING}]{msg}[/{COLOR_WARNING}]")

    prompt_and_save(console, screen_name=f"episode_comparison_{podcast.name}")


def outputs_viewer_menu() -> None:
    """Interactive menu for viewing various output files."""
    while True:
        clear_screen()
        console.print(
            Panel(
                f"[bold {COLOR_PRIMARY}]📂 Output Files Viewer[/bold {COLOR_PRIMARY}]\n"
                "[dim]View document labels, features, and LLM annotations[/dim]",
                border_style="blue",
            )
        )

        table = Table(show_header=False, box=None)
        table.add_column("Option", style=COLOR_WARNING)
        table.add_column("Description")

        table.add_row("1", "📋 Document labels (episode classifications)")
        table.add_row("2", "📊 Features CSV (episode metrics)")
        table.add_row("3", "🤖 LLM annotations (hate speech, ads)")
        table.add_row("4", "🏷️  Classification Viewer (ML model results)")
        table.add_row("5", "🎯 LLM Annotation Viewer (enhanced)")
        table.add_row(KEY_BACK, "⬅️  Back to main menu")

        console.print(table)
        console.print()

        choice = Prompt.ask(f"[{COLOR_WARNING}]Select option[/{COLOR_WARNING}]", default=KEY_BACK)

        match choice.lower():
            case "1":
                view_document_labels_menu()
            case "2":
                view_features_csv_menu()
            case "3":
                view_llm_annotations_menu()
            case "4":
                classification_viewer_menu()
            case "5":
                llm_annotation_viewer_menu()
            case _ if choice.lower() == KEY_BACK:
                break
            case _:
                console.print(f"[{COLOR_ERROR}]Invalid option. Please try again.[/{COLOR_ERROR}]")
                Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def legacy_menu() -> None:
    """Legacy menu with original navigation options."""
    while True:
        clear_screen()
        console.print(
            Panel(
                f"[bold {COLOR_PRIMARY}]📂 Browse & Search (Legacy)[/bold {COLOR_PRIMARY}]\n"
                "[dim]Original navigation and search features[/dim]",
                border_style="blue",
            )
        )

        table = Table(show_header=False, box=None)
        table.add_column("Option", style=COLOR_WARNING)
        table.add_column("Description")

        table.add_row("1", "📊 Scan outputs/ directory (folder stats)")
        table.add_row("2", "🌳 View analysis folder structure")
        table.add_row("3", "📄 Browse analysis files & keywords")
        table.add_row("4", "🔍 Search keywords across all podcasts")
        table.add_row("5", "📁 Search and view specific file")
        table.add_row("6", "📈 Statistics & visualizations")
        table.add_row("7", "📂 View output files")
        table.add_row(KEY_BACK, "⬅️  Back to main menu")

        console.print(table)
        console.print()

        choice = Prompt.ask(f"[{COLOR_WARNING}]Select option[/{COLOR_WARNING}]", default=KEY_BACK)

        match choice.lower():
            case "1":
                scan_outputs_menu()
            case "2":
                view_analysis_tree_menu()
            case "3":
                browse_analysis_files_menu()
            case "4":
                global_keyword_search_menu()
            case "5":
                search_file_menu()
            case "6":
                visualizations_menu()
            case "7":
                outputs_viewer_menu()
            case _ if choice.lower() == KEY_BACK:
                break
            case _:
                console.print(f"[{COLOR_ERROR}]Invalid option. Please try again.[/{COLOR_ERROR}]")
                Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def advanced_visualizations_menu() -> None:
    """Advanced visualizations submenu."""
    while True:
        clear_screen()
        console.print(
            Panel(
                f"[bold {COLOR_PRIMARY}]📊 Advanced Visualizations[/bold {COLOR_PRIMARY}]\n"
                "[dim]Timelines, heatmaps, and interaction graphs[/dim]",
                border_style="blue",
            )
        )

        table = Table(show_header=False, box=None)
        table.add_column("Option", style=COLOR_WARNING)
        table.add_column("Description")

        table.add_row("1", "📈 Timeline visualizations")
        table.add_row("2", "🎨 Category heatmaps")
        table.add_row("3", "👥 Speaker interactions")
        table.add_row("4", "📊 Basic statistics (legacy)")
        table.add_row(KEY_BACK, "⬅️  Back")

        console.print(table)
        console.print()

        choice = Prompt.ask(f"[{COLOR_WARNING}]Select option[/{COLOR_WARNING}]", default=KEY_BACK)

        match choice.lower():
            case "1":
                timeline_menu()
            case "2":
                heatmap_menu()
            case "3":
                interactions_menu()
            case "4":
                visualizations_menu()
            case _ if choice.lower() == KEY_BACK:
                break
            case _:
                console.print(f"[{COLOR_ERROR}]Invalid option. Please try again.[/{COLOR_ERROR}]")
                Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def main() -> None:
    """Main entry point with interactive menu."""
    while True:
        clear_screen()
        display_main_menu()

        choice = Prompt.ask(f"[{COLOR_WARNING}]Select option[/{COLOR_WARNING}]", default=KEY_QUIT)

        match choice.lower():
            case "1":
                dashboard_menu()
            case "2":
                episode_view_menu()
            case "3":
                advanced_search_menu()
            case "4":
                summary_generator_menu()
            case "5":
                report_generator_menu()
            case "6":
                advanced_visualizations_menu()
            case "7":
                bookmarks_menu()
            case "8":
                features_explorer_menu()
            case "9":
                classification_summary_menu()
            case "10":
                legacy_menu()
            case _ if choice.lower() == KEY_QUIT:
                console.print("[green]Goodbye![/green]")
                break
            case _:
                console.print(f"[{COLOR_ERROR}]Invalid option. Please try again.[/{COLOR_ERROR}]")


if __name__ == "__main__":
    main()
