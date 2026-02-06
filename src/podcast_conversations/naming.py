"""Centralized file and folder naming conventions.

This module provides standardized naming utilities for all output files and folders
across the codebase. All modules should import from here to ensure consistency.

Naming Convention (from CLAUDE.md):
- lowercase all folder and file names
- replace whitespaces with _
- replace consecutive __ with single _
- replace special characters with _

Timestamp Convention:
- Filenames: YYYY_MM_DD_HH_MM_SS (e.g., 2026_02_06_12_30_45)
- Display/logs: YYYY-MM-DD HH:MM:SS (e.g., 2026-02-06 12:30:45)
"""

import re
from datetime import datetime


def sanitize_name(name: str, max_length: int = 100) -> str:
    """Sanitize a name following the project naming convention.

    Rules applied:
    - convert to lowercase
    - replace special characters with _
    - replace whitespaces with _
    - replace multiple consecutive __ with single _
    - strip leading/trailing underscores
    - truncate to max_length

    Args:
        name: the original name to sanitize
        max_length: maximum length (default 100)

    Returns:
        sanitized name following the naming convention
    """
    if not name:
        return "untitled"

    # convert to lowercase.
    result = name.lower()

    # replace special characters (keeping only alphanumeric and spaces).
    result = re.sub(r"[^a-z0-9\s]", "_", result)

    # replace whitespaces with underscore.
    result = re.sub(r"\s+", "_", result)

    # replace multiple consecutive underscores with single underscore.
    result = re.sub(r"_+", "_", result)

    # strip leading/trailing underscores.
    result = result.strip("_")

    # truncate to max length.
    if max_length and len(result) > max_length:
        result = result[:max_length].rstrip("_")

    # handle empty result.
    if not result:
        result = "untitled"

    return result


def sanitize_filename(name: str, max_length: int = 100, extension: str = "") -> str:
    """Sanitize a filename following the project naming convention.

    Args:
        name: the original filename or title
        max_length: maximum length for the base name (default 100)
        extension: optional file extension to append (e.g., ".mp3", ".json")

    Returns:
        sanitized filename with extension
    """
    base = sanitize_name(name, max_length=max_length)
    return f"{base}{extension}"


def sanitize_dirname(name: str, max_length: int = 50) -> str:
    """Sanitize a directory name following the project naming convention.

    Args:
        name: the original directory name or title
        max_length: maximum length for the directory name (default 50)

    Returns:
        sanitized directory name
    """
    return sanitize_name(name, max_length=max_length)


def generate_timestamp() -> str:
    """Generate a timestamp string for filenames.

    Format: YYYY_MM_DD_HH_MM_SS

    Returns:
        timestamp string suitable for filenames
    """
    return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


def generate_display_timestamp() -> str:
    """Generate a timestamp string for display/logs.

    Format: YYYY-MM-DD HH:MM:SS

    Returns:
        timestamp string suitable for display
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def generate_date_string() -> str:
    """Generate a date string for display.

    Format: YYYY-MM-DD

    Returns:
        date string suitable for display
    """
    return datetime.now().strftime("%Y-%m-%d")


def create_timestamped_filename(base_name: str, extension: str = "") -> str:
    """Create a filename with sanitized base name and timestamp.

    Args:
        base_name: the base name for the file
        extension: file extension (e.g., ".txt", ".json")

    Returns:
        filename in format: base_name_YYYY_MM_DD_HH_MM_SS.ext
    """
    safe_base = sanitize_name(base_name)
    timestamp = generate_timestamp()
    return f"{safe_base}_{timestamp}{extension}"
