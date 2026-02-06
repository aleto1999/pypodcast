"""Directory scanner with parallel processing."""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Generator


@dataclass
class FolderStats:
    """Statistics for a folder."""

    name: str
    path: Path
    file_count: int
    size_bytes: int

    @property
    def size_human(self) -> str:
        """Return human-readable size."""
        size = self.size_bytes
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} TB"


def get_folder_stats(folder_path: Path) -> FolderStats:
    """Get statistics for a single folder."""
    file_count = 0
    total_size = 0

    for root, _, files in os.walk(folder_path):
        file_count += len(files)
        for file in files:
            try:
                total_size += os.path.getsize(os.path.join(root, file))
            except OSError:
                pass

    return FolderStats(
        name=folder_path.name,
        path=folder_path,
        file_count=file_count,
        size_bytes=total_size,
    )


def scan_directory_parallel(base_path: Path, max_workers: int = 8) -> list[FolderStats]:
    """Scan directory and get stats for all subfolders using parallel processing."""
    if not base_path.exists():
        return []

    # get immediate subfolders.
    subfolders = [p for p in base_path.iterdir() if p.is_dir()]

    if not subfolders:
        return []

    stats: list[FolderStats] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_folder_stats, folder): folder for folder in subfolders}

        for future in as_completed(futures):
            try:
                stats.append(future.result())
            except Exception:
                pass

    return sorted(stats, key=lambda x: x.name)


def get_files_in_folder(folder_path: Path) -> Generator[Path, None, None]:
    """Get all files in a folder recursively."""
    if not folder_path.exists():
        return

    for root, _, files in os.walk(folder_path):
        for file in files:
            yield Path(root) / file
