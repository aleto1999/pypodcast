"""Rich library utilities for pipeline scripts.

This module provides standardized Rich components for CLI scripts:
- Console instance
- Color constants for semantic styling
- Progress bar helpers
- Logging setup
- Common print helpers

Usage:
    from utils.rich_utils import (
        console,
        setup_logging,
        create_progress,
        print_success,
        print_error,
        COLOR_PRIMARY,
    )
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from rich.console import Console
from rich.logging import RichHandler
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

# =============================================================================
# Color Constants
# =============================================================================

# semantic color names for consistent styling across scripts.
COLOR_PRIMARY = "cyan"
COLOR_SECONDARY = "magenta"
COLOR_SUCCESS = "green"
COLOR_WARNING = "yellow"
COLOR_ERROR = "red"
COLOR_INFO = "blue"
COLOR_DIM = "dim"

# =============================================================================
# Console Instance
# =============================================================================

# shared console instance for scripts.
console = Console()


# =============================================================================
# Logging Setup
# =============================================================================


def setup_logging(level: str = "INFO", console_instance: Console | None = None) -> None:
    """
    Configure logging with Rich handler for beautiful output.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        console_instance: Optional console instance (uses module console if not provided).
    """
    c = console_instance or console
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=c, rich_tracebacks=True, show_path=False)],
    )


# =============================================================================
# Progress Bar Helpers
# =============================================================================


@contextmanager
def create_progress(
    transient: bool = True,
    console_instance: Console | None = None,
) -> Iterator[Progress]:
    """
    Create a standard spinner progress bar.

    Args:
        transient: Remove progress bar when complete (default: True).
        console_instance: Optional console instance.

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
        console=console_instance or console,
    )
    with progress:
        yield progress


@contextmanager
def create_bar_progress(
    show_time: bool = True,
    transient: bool = False,
    console_instance: Console | None = None,
) -> Iterator[Progress]:
    """
    Create a progress bar with percentage and time display.

    Args:
        show_time: Show elapsed and remaining time.
        transient: Remove progress bar when complete.
        console_instance: Optional console instance.

    Yields:
        Progress instance.
    """
    columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ]
    if show_time:
        columns.extend([TimeElapsedColumn(), TimeRemainingColumn()])

    progress = Progress(*columns, transient=transient, console=console_instance or console)
    with progress:
        yield progress


# =============================================================================
# Table Helpers
# =============================================================================


def create_table(
    title: str | None = None,
    show_header: bool = True,
    box_style: str | None = None,
) -> Table:
    """
    Create a standard table.

    Args:
        title: Optional table title.
        show_header: Show column headers.
        box_style: Box style name (None for default).

    Returns:
        Configured Table instance.
    """
    return Table(title=title, show_header=show_header)


def create_stats_table(title: str | None = None) -> Table:
    """
    Create a table for displaying statistics.

    Args:
        title: Optional table title.

    Returns:
        Table configured for metric/value pairs.
    """
    table = Table(title=title, show_header=True)
    table.add_column("Metric", style=COLOR_PRIMARY)
    table.add_column("Value", justify="right")
    return table


# =============================================================================
# Print Helpers
# =============================================================================


def print_success(message: str, prefix: str = "✓") -> None:
    """Print a success message in green."""
    console.print(f"[{COLOR_SUCCESS}]{prefix} {message}[/{COLOR_SUCCESS}]")


def print_warning(message: str, prefix: str = "⚠") -> None:
    """Print a warning message in yellow."""
    console.print(f"[{COLOR_WARNING}]{prefix} {message}[/{COLOR_WARNING}]")


def print_error(message: str, prefix: str = "✗") -> None:
    """Print an error message in red."""
    console.print(f"[{COLOR_ERROR}]{prefix} {message}[/{COLOR_ERROR}]")


def print_info(message: str, prefix: str = "ℹ") -> None:
    """Print an info message in blue."""
    console.print(f"[{COLOR_INFO}]{prefix} {message}[/{COLOR_INFO}]")


def print_header(title: str, style: str = COLOR_PRIMARY) -> None:
    """Print a bold header."""
    console.print(f"[bold {style}]{title}[/bold {style}]")


def print_panel(content: str, title: str | None = None, style: str = COLOR_PRIMARY) -> None:
    """Print content in a styled panel."""
    console.print(Panel(content, title=title, border_style=style))


def print_separator(char: str = "=", width: int = 80) -> None:
    """Print a separator line."""
    console.print(f"[{COLOR_DIM}]{char * width}[/{COLOR_DIM}]")
