"""Terminal visualization using Rich library."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.tree import Tree

from terminal_data_visualizer.config import (
    COLOR_DIM,
    COLOR_ERROR,
    COLOR_INFO,
    COLOR_PRIMARY,
    COLOR_SUCCESS,
)
from terminal_data_visualizer.scanner import (
    FolderStats,
    get_files_in_folder,
    scan_directory_parallel,
)

console = Console(record=True)


def display_folder_stats(
    stats: list[FolderStats],
    title: str = "Directory Overview",
    show_numbers: bool = False,
) -> list[str]:
    """Display folder statistics in a table.

    Args:
        stats: List of FolderStats to display.
        title: Title for the table.
        show_numbers: If True, show row numbers for selection.

    Returns:
        List of folder names in display order (for drill-down selection).
    """
    table = Table(title=title, show_header=True, header_style="bold magenta")

    if show_numbers:
        table.add_column("#", style="yellow", justify="right", width=4)

    table.add_column("Folder", style="cyan")
    table.add_column("Files", justify="right", style="green")
    table.add_column("Size", justify="right", style="yellow")

    total_files = 0
    total_size = 0
    folder_names: list[str] = []

    # sort by name for consistent ordering.
    sorted_stats = sorted(stats, key=lambda s: s.name.lower())

    for idx, stat in enumerate(sorted_stats, 1):
        folder_names.append(stat.name)
        if show_numbers:
            table.add_row(str(idx), stat.name, str(stat.file_count), stat.size_human)
        else:
            table.add_row(stat.name, str(stat.file_count), stat.size_human)
        total_files += stat.file_count
        total_size += stat.size_bytes

    # add total row.
    table.add_section()
    total_stat = FolderStats(
        name="TOTAL", path=Path(), file_count=total_files, size_bytes=total_size
    )
    if show_numbers:
        table.add_row(
            "",
            "[bold]TOTAL[/bold]",
            f"[bold]{total_files}[/bold]",
            f"[bold]{total_stat.size_human}[/bold]",
        )
    else:
        table.add_row(
            "[bold]TOTAL[/bold]",
            f"[bold]{total_files}[/bold]",
            f"[bold]{total_stat.size_human}[/bold]",
        )

    console.print(table)
    console.print(f"\n[{COLOR_DIM}]Total folders: {len(stats)}[/{COLOR_DIM}]")

    return folder_names


def display_analysis_tree(analysis_path: Path) -> None:
    """Display analysis folder structure as a tree."""
    if not analysis_path.exists():
        console.print(f"[{COLOR_ERROR}]Analysis folder not found.[/{COLOR_ERROR}]")
        return

    tree = Tree(f"📁 [bold {COLOR_PRIMARY}]{analysis_path.name}[/bold {COLOR_PRIMARY}]")

    def add_to_tree(parent: Tree, path: Path, depth: int = 0) -> None:
        if depth > 3:
            return

        try:
            items = sorted(path.iterdir())
        except PermissionError:
            return

        dirs = [p for p in items if p.is_dir()]
        files = [p for p in items if p.is_file()]

        for d in dirs:
            branch = parent.add(f"📁 [{COLOR_PRIMARY}]{d.name}[/{COLOR_PRIMARY}]")
            add_to_tree(branch, d, depth + 1)

        for f in files[:10]:  # limit files shown.
            size = f.stat().st_size if f.exists() else 0
            size_str = format_size(size)
            parent.add(f"📄 [{COLOR_SUCCESS}]{f.name}[/{COLOR_SUCCESS}] [{COLOR_DIM}]({size_str})[/{COLOR_DIM}]")

        if len(files) > 10:
            parent.add(f"[{COLOR_DIM}]... and {len(files) - 10} more files[/{COLOR_DIM}]")

    add_to_tree(tree, analysis_path)
    console.print(tree)


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def display_file_contents(file_path: Path, max_lines: int = 50) -> None:
    """Display file contents with syntax highlighting."""
    if not file_path.exists():
        console.print(f"[{COLOR_ERROR}]File not found: {file_path}[/{COLOR_ERROR}]")
        return

    try:
        content = file_path.read_text(encoding="utf-8")
        lines = content.split("\n")

        if len(lines) > max_lines:
            content = "\n".join(lines[:max_lines])
            content += f"\n\n[{COLOR_DIM}]... truncated ({len(lines) - max_lines} more lines)[/{COLOR_DIM}]"

        console.print(Panel(content, title=str(file_path.name), border_style=COLOR_INFO))
    except Exception as e:
        console.print(f"[{COLOR_ERROR}]Error reading file: {e}[/{COLOR_ERROR}]")


def scan_with_progress(base_path: Path) -> list[FolderStats]:
    """Scan directory with progress indicator."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(description="Scanning directories...", total=None)
        stats = scan_directory_parallel(base_path)
    return stats


def display_analysis_files_parallel(analysis_path: Path) -> list[tuple[str, Path]]:
    """Get all analysis files with parallel scanning."""
    if not analysis_path.exists():
        return []

    files: list[tuple[str, Path]] = []
    subfolders = [p for p in analysis_path.iterdir() if p.is_dir()]

    def get_folder_files(folder: Path) -> list[tuple[str, Path]]:
        result = []
        for f in get_files_in_folder(folder):
            result.append((folder.name, f))
        return result

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(get_folder_files, folder): folder for folder in subfolders}
        for future in as_completed(futures):
            try:
                files.extend(future.result())
            except Exception:
                pass

    # also get files directly in analysis folder.
    for f in analysis_path.iterdir():
        if f.is_file():
            files.append(("root", f))

    return sorted(files, key=lambda x: (x[0], x[1].name))
