"""Shared utilities for pipeline scripts."""

from .cli_utils import get_default_workers, resolve_log_level
from .rich_utils import (
    COLOR_DIM,
    COLOR_ERROR,
    COLOR_INFO,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_SUCCESS,
    COLOR_WARNING,
    console,
    create_bar_progress,
    create_progress,
    create_stats_table,
    create_table,
    print_error,
    print_header,
    print_info,
    print_panel,
    print_separator,
    print_success,
    print_warning,
    setup_logging,
)

__all__ = [
    # cli_utils.
    "get_default_workers",
    "resolve_log_level",
    # rich_utils - colors.
    "COLOR_PRIMARY",
    "COLOR_SECONDARY",
    "COLOR_SUCCESS",
    "COLOR_WARNING",
    "COLOR_ERROR",
    "COLOR_INFO",
    "COLOR_DIM",
    # rich_utils - console.
    "console",
    "setup_logging",
    # rich_utils - progress.
    "create_progress",
    "create_bar_progress",
    # rich_utils - tables.
    "create_table",
    "create_stats_table",
    # rich_utils - print helpers.
    "print_success",
    "print_warning",
    "print_error",
    "print_info",
    "print_header",
    "print_panel",
    "print_separator",
]
