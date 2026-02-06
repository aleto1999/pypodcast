"""tests for terminal_data_visualizer.scanner module."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest

from terminal_data_visualizer.scanner import (
    FolderStats,
    get_folder_stats,
    scan_directory_parallel,
)


class TestFolderStats:
    """tests for FolderStats dataclass."""

    def test_create_folder_stats(self, tmp_path: Path):
        """should create folder stats with required fields."""
        stats = FolderStats(
            name="test_folder",
            path=tmp_path,
            file_count=10,
            size_bytes=1024,
        )
        assert stats.name == "test_folder"
        assert stats.path == tmp_path
        assert stats.file_count == 10
        assert stats.size_bytes == 1024

    def test_size_human_bytes(self):
        """should format bytes correctly."""
        stats = FolderStats(
            name="test", path=Path("/tmp"), file_count=0, size_bytes=512
        )
        assert "512.00 B" in stats.size_human

    def test_size_human_kilobytes(self):
        """should format kilobytes correctly."""
        stats = FolderStats(
            name="test", path=Path("/tmp"), file_count=0, size_bytes=1536
        )
        assert "KB" in stats.size_human

    def test_size_human_megabytes(self):
        """should format megabytes correctly."""
        stats = FolderStats(
            name="test", path=Path("/tmp"), file_count=0, size_bytes=2 * 1024 * 1024
        )
        assert "MB" in stats.size_human

    def test_size_human_gigabytes(self):
        """should format gigabytes correctly."""
        stats = FolderStats(
            name="test",
            path=Path("/tmp"),
            file_count=0,
            size_bytes=3 * 1024 * 1024 * 1024,
        )
        assert "GB" in stats.size_human


class TestGetFolderStats:
    """tests for get_folder_stats function."""

    def test_empty_folder(self, tmp_path: Path):
        """should handle empty folder."""
        stats = get_folder_stats(tmp_path)
        assert stats.name == tmp_path.name
        assert stats.file_count == 0
        assert stats.size_bytes == 0

    def test_folder_with_files(self, tmp_path: Path):
        """should count files correctly."""
        # create test files.
        (tmp_path / "file1.txt").write_text("hello")
        (tmp_path / "file2.txt").write_text("world")
        (tmp_path / "file3.txt").write_text("test")

        stats = get_folder_stats(tmp_path)
        assert stats.file_count == 3

    def test_folder_with_nested_files(self, tmp_path: Path):
        """should count nested files."""
        # create nested structure.
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "file1.txt").write_text("root")
        (subdir / "file2.txt").write_text("nested")

        stats = get_folder_stats(tmp_path)
        assert stats.file_count == 2

    def test_calculates_size(self, tmp_path: Path):
        """should calculate total size."""
        content = "x" * 100
        (tmp_path / "file.txt").write_text(content)

        stats = get_folder_stats(tmp_path)
        assert stats.size_bytes == 100


class TestScanDirectoryParallel:
    """tests for scan_directory_parallel function."""

    def test_nonexistent_directory(self, tmp_path: Path):
        """should return empty list for nonexistent directory."""
        result = scan_directory_parallel(tmp_path / "nonexistent")
        assert result == []

    def test_empty_directory(self, tmp_path: Path):
        """should return empty list for directory without subdirs."""
        result = scan_directory_parallel(tmp_path)
        assert result == []

    def test_with_subfolders(self, tmp_path: Path):
        """should scan subfolders."""
        # create subfolders.
        (tmp_path / "folder1").mkdir()
        (tmp_path / "folder2").mkdir()
        (tmp_path / "folder3").mkdir()

        result = scan_directory_parallel(tmp_path)
        assert len(result) == 3

    def test_returns_folder_stats(self, tmp_path: Path):
        """should return FolderStats objects."""
        (tmp_path / "test_folder").mkdir()

        result = scan_directory_parallel(tmp_path)
        assert len(result) == 1
        assert isinstance(result[0], FolderStats)
        assert result[0].name == "test_folder"

    def test_with_files_in_subfolders(self, tmp_path: Path):
        """should count files in subfolders."""
        folder = tmp_path / "folder"
        folder.mkdir()
        (folder / "file1.txt").write_text("test1")
        (folder / "file2.txt").write_text("test2")

        result = scan_directory_parallel(tmp_path)
        assert len(result) == 1
        assert result[0].file_count == 2

    def test_custom_max_workers(self, tmp_path: Path):
        """should respect max_workers parameter."""
        for i in range(5):
            (tmp_path / f"folder{i}").mkdir()

        result = scan_directory_parallel(tmp_path, max_workers=2)
        assert len(result) == 5

    def test_ignores_files_in_root(self, tmp_path: Path):
        """should only scan subfolders, not root files."""
        (tmp_path / "root_file.txt").write_text("root")
        folder = tmp_path / "subfolder"
        folder.mkdir()
        (folder / "nested_file.txt").write_text("nested")

        result = scan_directory_parallel(tmp_path)
        assert len(result) == 1
        assert result[0].name == "subfolder"


class TestScannerIntegration:
    """integration tests for scanner module."""

    def test_full_scan_pipeline(self, tmp_path: Path):
        """should perform complete directory scan."""
        # create complex directory structure.
        (tmp_path / "project1").mkdir()
        (tmp_path / "project1" / "src").mkdir()
        (tmp_path / "project1" / "src" / "main.py").write_text("print('hello')")
        (tmp_path / "project1" / "data.json").write_text('{"key": "value"}')

        (tmp_path / "project2").mkdir()
        (tmp_path / "project2" / "readme.md").write_text("# Project")

        result = scan_directory_parallel(tmp_path)
        assert len(result) == 2

        # find project1 stats.
        project1 = next((s for s in result if s.name == "project1"), None)
        assert project1 is not None
        assert project1.file_count == 2

        # find project2 stats.
        project2 = next((s for s in result if s.name == "project2"), None)
        assert project2 is not None
        assert project2.file_count == 1
