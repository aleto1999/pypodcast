"""Centralized Rich library utilities for consistent styling across the application.

This module provides standardized Rich components to ensure visual consistency:
- Console factory with optional recording support
- Color constants for semantic styling
- Progress bar factories
- Table factories with consistent styling
- Print helpers for common message types

Usage:
    from terminal_data_visualizer.rich_utils import (
        get_console,
        create_progress,
        create_table,
        print_success,
        print_warning,
        print_error,
        COLOR_PRIMARY,
        COLOR_SUCCESS,
    )
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

if TYPE_CHECKING:
    from rich.progress import TaskID

# re-export color constants from config for convenience.
from terminal_data_visualizer.config import (
    COLOR_DIM,
    COLOR_ERROR,
    COLOR_INFO,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_SUCCESS,
    COLOR_WARNING,
)

__all__ = [
    # color constants.
    "COLOR_PRIMARY",
    "COLOR_SECONDARY",
    "COLOR_SUCCESS",
    "COLOR_WARNING",
    "COLOR_ERROR",
    "COLOR_INFO",
    "COLOR_DIM",
    # console.
    "get_console",
    "create_console",
    # progress.
    "create_progress",
    "create_spinner_progress",
    "create_bar_progress",
    # tables.
    "create_table",
    "create_menu_table",
    "create_stats_table",
    # print helpers.
    "print_success",
    "print_warning",
    "print_error",
    "print_info",
    "print_header",
    "print_panel",
]


# =============================================================================
# Console Factory
# =============================================================================

# module-level console instance for scripts (no recording).
_script_console: Console | None = None

# module-level console instance for interactive apps (with recording for screenshots).
_interactive_console: Console | None = None


def get_console(record: bool = False, force_new: bool = False) -> Console:
    """
    Get a console instance with optional recording support.

    Args:
        record: Enable recording for screenshot capture (default: False).
        force_new: Force creation of a new console instance.

    Returns:
        Console instance configured appropriately.
    """
    global _script_console, _interactive_console

    if force_new:
        return Console(record=record)

    if record:
        if _interactive_console is None:
            _interactive_console = Console(record=True)
        return _interactive_console
    else:
        if _script_console is None:
            _script_console = Console()
        return _script_console


def create_console(record: bool = False, width: int | None = None) -> Console:
    """
    Create a new console instance with specified options.

    Args:
        record: Enable recording for screenshot capture.
        width: Fixed width for the console output.

    Returns:
        New Console instance.
    """
    return Console(record=record, width=width)


# =============================================================================
# Progress Bar Factories
# =============================================================================


@contextmanager
def create_progress(
    transient: bool = True,
    console: Console | None = None,
) -> Iterator[Progress]:
    """
    Create a standard progress bar context manager.

    This is the default progress bar style with a spinner and description.

    Args:
        transient: Remove progress bar when complete (default: True).
        console: Optional console instance to use.

    Yields:
        Progress instance for adding tasks.

    Example:
        with create_progress() as progress:
            task = progress.add_task("Processing...", total=100)
            for i in range(100):
                progress.update(task, advance=1)
    """
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=transient,
        console=console,
    )
    with progress:
        yield progress


@contextmanager
def create_spinner_progress(
    transient: bool = True,
    console: Console | None = None,
) -> Iterator[Progress]:
    """
    Create a spinner-only progress bar (no percentage).

    Useful for indeterminate operations.

    Args:
        transient: Remove progress bar when complete.
        console: Optional console instance to use.

    Yields:
        Progress instance.
    """
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=transient,
        console=console,
    )
    with progress:
        yield progress


@contextmanager
def create_bar_progress(
    show_time: bool = True,
    show_percentage: bool = True,
    transient: bool = False,
    console: Console | None = None,
) -> Iterator[Progress]:
    """
    Create a progress bar with percentage and optional time display.

    Useful for determinate operations with known totals.

    Args:
        show_time: Show elapsed and remaining time.
        show_percentage: Show completion percentage.
        transient: Remove progress bar when complete.
        console: Optional console instance to use.

    Yields:
        Progress instance.
    """
    columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
    ]
    if show_percentage:
        columns.append(TaskProgressColumn())
    if show_time:
        columns.extend([TimeElapsedColumn(), TimeRemainingColumn()])

    progress = Progress(*columns, transient=transient, console=console)
    with progress:
        yield progress


# =============================================================================
# Table Factories
# =============================================================================


def create_table(
    title: str | None = None,
    show_header: bool = True,
    show_lines: bool = False,
    box: Any = None,
    expand: bool = False,
) -> Table:
    """
    Create a standard table with consistent styling.

    Args:
        title: Optional table title.
        show_header: Show column headers.
        show_lines: Show row separator lines.
        box: Box style (None for default).
        expand: Expand table to full width.

    Returns:
        Configured Table instance.
    """
    return Table(
        title=title,
        show_header=show_header,
        show_lines=show_lines,
        box=box,
        expand=expand,
    )


def create_menu_table(title: str | None = None) -> Table:
    """
    Create a table styled for menu options.

    Args:
        title: Optional table title.

    Returns:
        Table configured with option and description columns.
    """
    table = Table(title=title, show_header=True, box=None)
    table.add_column("Option", style=COLOR_WARNING, width=8)
    table.add_column("Description", style="white")
    return table


def create_stats_table(title: str | None = None) -> Table:
    """
    Create a table styled for statistics display.

    Args:
        title: Optional table title.

    Returns:
        Table configured for metric/value pairs.
    """
    table = Table(title=title, show_header=True, box=None)
    table.add_column("Metric", style=COLOR_PRIMARY)
    table.add_column("Value", style="white", justify="right")
    return table


# =============================================================================
# Print Helpers
# =============================================================================


def print_success(
    message: str,
    console: Console | None = None,
    prefix: str = "✓",
) -> None:
    """Print a success message in green."""
    c = console or get_console()
    c.print(f"[{COLOR_SUCCESS}]{prefix} {message}[/{COLOR_SUCCESS}]")


def print_warning(
    message: str,
    console: Console | None = None,
    prefix: str = "⚠",
) -> None:
    """Print a warning message in yellow."""
    c = console or get_console()
    c.print(f"[{COLOR_WARNING}]{prefix} {message}[/{COLOR_WARNING}]")


def print_error(
    message: str,
    console: Console | None = None,
    prefix: str = "✗",
) -> None:
    """Print an error message in red."""
    c = console or get_console()
    c.print(f"[{COLOR_ERROR}]{prefix} {message}[/{COLOR_ERROR}]")


def print_info(
    message: str,
    console: Console | None = None,
    prefix: str = "ℹ",
) -> None:
    """Print an info message in blue."""
    c = console or get_console()
    c.print(f"[{COLOR_INFO}]{prefix} {message}[/{COLOR_INFO}]")


def print_header(
    title: str,
    console: Console | None = None,
    style: str = COLOR_PRIMARY,
) -> None:
    """Print a bold header."""
    c = console or get_console()
    c.print(f"[bold {style}]{title}[/bold {style}]")


def print_panel(
    content: str,
    title: str | None = None,
    console: Console | None = None,
    style: str = COLOR_PRIMARY,
) -> None:
    """Print content in a styled panel."""
    c = console or get_console()
    c.print(Panel(content, title=title, border_style=style))
