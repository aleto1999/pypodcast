"""tests for terminal_data_visualizer.data_export module."""

import csv
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest

from terminal_data_visualizer.models import EpisodeStats, SpeakerStats
from terminal_data_visualizer.data_export import (
    export_episode_stats_to_csv,
    export_episode_stats_to_json,
    export_category_distribution,
    get_export_path,
)


@pytest.fixture
def sample_episode_stats():
    """create sample episode statistics."""
    return [
        EpisodeStats(
            filename="episode_1.json",
            total_segments=100,
            total_duration=3600.0,
            speaker_count=2,
            total_words=5000,
            speakers={
                "SPEAKER_01": SpeakerStats(
                    speaker="SPEAKER_01",
                    segment_count=50,
                    total_words=2500,
                    total_duration=1800.0,
                    avg_segment_length=50.0,
                ),
                "SPEAKER_02": SpeakerStats(
                    speaker="SPEAKER_02",
                    segment_count=50,
                    total_words=2500,
                    total_duration=1800.0,
                    avg_segment_length=50.0,
                ),
            },
        ),
        EpisodeStats(
            filename="episode_2.json",
            total_segments=80,
            total_duration=2400.0,
            speaker_count=1,
            total_words=3500,
            speakers={
                "SPEAKER_01": SpeakerStats(
                    speaker="SPEAKER_01",
                    segment_count=80,
                    total_words=3500,
                    total_duration=2400.0,
                    avg_segment_length=43.75,
                ),
            },
        ),
    ]


class TestExportEpisodeStatsToCSV:
    """tests for export_episode_stats_to_csv function."""

    def test_creates_csv_file(self, tmp_path: Path, sample_episode_stats):
        """should create csv file."""
        output_file = tmp_path / "stats.csv"
        result = export_episode_stats_to_csv(sample_episode_stats, output_file)

        assert result is True
        assert output_file.exists()

    def test_csv_has_header(self, tmp_path: Path, sample_episode_stats):
        """should include header row."""
        output_file = tmp_path / "stats.csv"
        export_episode_stats_to_csv(sample_episode_stats, output_file)

        with open(output_file, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)

        assert "Episode" in header
        assert "Total Segments" in header
        assert "Total Duration (min)" in header

    def test_csv_has_data_rows(self, tmp_path: Path, sample_episode_stats):
        """should include data rows."""
        output_file = tmp_path / "stats.csv"
        export_episode_stats_to_csv(sample_episode_stats, output_file)

        with open(output_file, newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

        # header + 2 data rows.
        assert len(rows) == 3

    def test_creates_parent_directories(self, tmp_path: Path, sample_episode_stats):
        """should create parent directories if needed."""
        output_file = tmp_path / "nested" / "dir" / "stats.csv"
        result = export_episode_stats_to_csv(sample_episode_stats, output_file)

        assert result is True
        assert output_file.exists()

    def test_empty_stats_list(self, tmp_path: Path):
        """should handle empty list."""
        output_file = tmp_path / "empty.csv"
        result = export_episode_stats_to_csv([], output_file)

        assert result is True
        assert output_file.exists()


class TestExportEpisodeStatsToJSON:
    """tests for export_episode_stats_to_json function."""

    def test_creates_json_file(self, tmp_path: Path, sample_episode_stats):
        """should create json file."""
        output_file = tmp_path / "stats.json"
        result = export_episode_stats_to_json(sample_episode_stats, output_file)

        assert result is True
        assert output_file.exists()

    def test_json_is_valid(self, tmp_path: Path, sample_episode_stats):
        """should produce valid json."""
        output_file = tmp_path / "stats.json"
        export_episode_stats_to_json(sample_episode_stats, output_file)

        with open(output_file) as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) == 2

    def test_json_has_expected_fields(self, tmp_path: Path, sample_episode_stats):
        """should include expected fields."""
        output_file = tmp_path / "stats.json"
        export_episode_stats_to_json(sample_episode_stats, output_file)

        with open(output_file) as f:
            data = json.load(f)

        episode = data[0]
        assert "filename" in episode
        assert "total_segments" in episode
        assert "total_duration_seconds" in episode
        assert "total_duration_minutes" in episode
        assert "speakers" in episode

    def test_json_includes_speaker_data(self, tmp_path: Path, sample_episode_stats):
        """should include speaker statistics."""
        output_file = tmp_path / "stats.json"
        export_episode_stats_to_json(sample_episode_stats, output_file)

        with open(output_file) as f:
            data = json.load(f)

        speakers = data[0]["speakers"]
        assert "SPEAKER_01" in speakers
        assert speakers["SPEAKER_01"]["segment_count"] == 50

    def test_creates_parent_directories(self, tmp_path: Path, sample_episode_stats):
        """should create parent directories if needed."""
        output_file = tmp_path / "nested" / "dir" / "stats.json"
        result = export_episode_stats_to_json(sample_episode_stats, output_file)

        assert result is True
        assert output_file.exists()


class TestExportCategoryDistribution:
    """tests for export_category_distribution function."""

    @pytest.fixture
    def sample_category_data(self):
        """sample category distribution data."""
        return {
            "politics": 150,
            "sports": 75,
            "technology": 200,
            "entertainment": 100,
        }

    def test_export_csv(self, tmp_path: Path, sample_category_data):
        """should export to csv format."""
        output_file = tmp_path / "categories.csv"
        result = export_category_distribution(
            sample_category_data, output_file, format="csv"
        )

        assert result is True
        assert output_file.exists()

    def test_csv_sorted_by_count(self, tmp_path: Path, sample_category_data):
        """should sort by count descending."""
        output_file = tmp_path / "categories.csv"
        export_category_distribution(sample_category_data, output_file, format="csv")

        with open(output_file, newline="") as f:
            reader = csv.reader(f)
            next(reader)  # skip header.
            rows = list(reader)

        # technology has highest count (200).
        assert rows[0][0] == "technology"
        assert rows[0][1] == "200"

    def test_export_json(self, tmp_path: Path, sample_category_data):
        """should export to json format."""
        output_file = tmp_path / "categories.json"
        result = export_category_distribution(
            sample_category_data, output_file, format="json"
        )

        assert result is True
        with open(output_file) as f:
            data = json.load(f)
        assert data == sample_category_data

    def test_empty_data(self, tmp_path: Path):
        """should handle empty data."""
        output_file = tmp_path / "empty.csv"
        result = export_category_distribution({}, output_file, format="csv")

        assert result is True
        assert output_file.exists()


class TestGetExportPath:
    """tests for get_export_path function."""

    def test_returns_path(self):
        """should return a Path object."""
        result = get_export_path("test_export")
        assert isinstance(result, Path)

    def test_includes_timestamp(self):
        """should include timestamp in filename."""
        result = get_export_path("test_export")
        # filename should contain underscore-separated date parts.
        assert "_" in result.name
        assert "test_export" in result.name

    def test_includes_extension(self):
        """should include file extension."""
        result = get_export_path("test_export", "csv")
        assert result.suffix == ".csv"

    def test_custom_extension(self):
        """should support custom extension."""
        result = get_export_path("test_export", "json")
        assert result.suffix == ".json"

    def test_path_in_exports_directory(self):
        """should place file in exports subdirectory."""
        result = get_export_path("test_export")
        assert "exports" in str(result)


class TestDataExportIntegration:
    """integration tests for data export."""

    def test_full_export_pipeline(self, tmp_path: Path, sample_episode_stats):
        """should complete full export workflow."""
        # export to csv.
        csv_file = tmp_path / "stats.csv"
        assert export_episode_stats_to_csv(sample_episode_stats, csv_file)

        # export to json.
        json_file = tmp_path / "stats.json"
        assert export_episode_stats_to_json(sample_episode_stats, json_file)

        # verify both files exist and have content.
        assert csv_file.stat().st_size > 0
        assert json_file.stat().st_size > 0

    def test_export_overwrites_existing(self, tmp_path: Path, sample_episode_stats):
        """should overwrite existing files."""
        output_file = tmp_path / "stats.csv"

        # first export.
        export_episode_stats_to_csv(sample_episode_stats, output_file)
        first_size = output_file.stat().st_size

        # second export with same data.
        export_episode_stats_to_csv(sample_episode_stats, output_file)
        second_size = output_file.stat().st_size

        assert first_size == second_size
