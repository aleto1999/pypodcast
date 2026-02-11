"""Terminal data visualizer package for exploring outputs directory."""

__version__ = "0.1.0"

from terminal_data_visualizer.classification_summary import classification_summary_menu
from terminal_data_visualizer.features_explorer import features_explorer_menu
from terminal_data_visualizer.rich_utils import (
    COLOR_DIM,
    COLOR_ERROR,
    COLOR_INFO,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_SUCCESS,
    COLOR_WARNING,
    create_bar_progress,
    create_menu_table,
    create_progress,
    create_stats_table,
    create_table,
    get_console,
    print_error,
    print_header,
    print_info,
    print_panel,
    print_success,
    print_warning,
)

__all__ = [
    "features_explorer_menu",
    "classification_summary_menu",
    # rich utilities - colors.
    "COLOR_PRIMARY",
    "COLOR_SECONDARY",
    "COLOR_SUCCESS",
    "COLOR_WARNING",
    "COLOR_ERROR",
    "COLOR_INFO",
    "COLOR_DIM",
    # rich utilities - console.
    "get_console",
    # rich utilities - progress.
    "create_progress",
    "create_bar_progress",
    # rich utilities - tables.
    "create_table",
    "create_menu_table",
    "create_stats_table",
    # rich utilities - print helpers.
    "print_success",
    "print_warning",
    "print_error",
    "print_info",
    "print_header",
    "print_panel",
]
