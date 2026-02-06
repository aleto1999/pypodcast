"""filename sanitization utilities for consistent output file naming.

naming convention:
- lowercase only
- replace whitespaces with _
- replace multiple consecutive __ with one _
- replace special characters with _
"""

import re


def sanitize_filename(name: str, max_length: int = 100, extension: str = "") -> str:
    """
    sanitize a filename following the project naming convention.

    rules:
    - lowercase only
    - replace whitespaces with _
    - replace special characters with _
    - replace multiple consecutive __ with one _
    - truncate to max_length (not including extension)

    args:
        name: the original filename or title
        max_length: maximum length for the base name (default 100)
        extension: optional file extension to append (e.g., ".mp3")

    returns:
        sanitized filename following the naming convention
    """
    if not name:
        return f"untitled{extension}"

    # convert to lowercase
    result = name.lower()

    # replace special characters (keeping only alphanumeric and spaces)
    # this includes: < > : " / \ | ? * and any other non-alphanumeric
    result = re.sub(r"[^a-z0-9\s]", "_", result)

    # replace whitespaces with underscore
    result = re.sub(r"\s+", "_", result)

    # replace multiple consecutive underscores with single underscore
    result = re.sub(r"_+", "_", result)

    # strip leading/trailing underscores
    result = result.strip("_")

    # truncate to max length
    if max_length and len(result) > max_length:
        result = result[:max_length].rstrip("_")

    # handle empty result
    if not result:
        result = "untitled"

    return f"{result}{extension}"


def sanitize_dirname(name: str, max_length: int = 50) -> str:
    """
    sanitize a directory name following the project naming convention.

    args:
        name: the original directory name or title
        max_length: maximum length for the directory name (default 50)

    returns:
        sanitized directory name following the naming convention
    """
    return sanitize_filename(name, max_length=max_length, extension="")
