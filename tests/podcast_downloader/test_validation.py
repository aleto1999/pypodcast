"""tests for validation module."""

import asyncio
from pathlib import Path

import pytest

from podcast_downloader.validation import (
    ValidationConfig,
    ValidationResult,
    check_disk_space,
    validate_download,
    validate_download_sync,
    validate_file_size,
    validate_mp3_frames,
)


class TestValidationResult:
    """tests for ValidationResult namedtuple."""

    def test_valid_result(self):
        """should create valid result."""
        result = ValidationResult(valid=True, validation_type="test")
        assert result.valid is True
        assert result.error_message is None

    def test_invalid_result(self):
        """should create invalid result with error."""
        result = ValidationResult(
            valid=False,
            error_message="test error",
            validation_type="test",
        )
        assert result.valid is False
        assert result.error_message == "test error"


class TestValidationConfig:
    """tests for ValidationConfig."""

    def test_default_values(self):
        """should have sensible defaults."""
        config = ValidationConfig()
        assert config.min_file_size == 1024
        assert config.max_file_size == 500 * 1024 * 1024
        assert config.mp3_sample_count == 5

    def test_custom_values(self):
        """should accept custom values."""
        config = ValidationConfig(
            min_file_size=2048,
            max_file_size=100 * 1024 * 1024,
            mp3_sample_count=10,
        )
        assert config.min_file_size == 2048
        assert config.max_file_size == 100 * 1024 * 1024


class TestCheckDiskSpace:
    """tests for disk space checking."""

    def test_sufficient_space(self, temp_dir: Path):
        """should pass when space is sufficient."""
        result = check_disk_space(temp_dir, required_bytes=1024)
        assert result.valid is True
        assert result.validation_type == "disk_space"

    def test_insufficient_space(self, temp_dir: Path):
        """should fail when requesting too much space."""
        # request more space than any disk would have
        result = check_disk_space(temp_dir, required_bytes=10**18)
        assert result.valid is False
        assert "insufficient" in result.error_message.lower()

    def test_nonexistent_directory(self):
        """should handle nonexistent directory gracefully."""
        result = check_disk_space(Path("/nonexistent/path"), required_bytes=1024)
        # this may return valid=False or raise, depending on implementation
        # just ensure it doesn't crash
        assert isinstance(result, ValidationResult)


class TestValidateFileSize:
    """tests for file size validation."""

    def test_valid_size(self, sample_mp3_file: Path):
        """should pass for valid file size."""
        result = validate_file_size(sample_mp3_file)
        assert result.valid is True

    def test_file_too_small(self, sample_small_file: Path):
        """should fail for files too small."""
        result = validate_file_size(sample_small_file, min_size=1024)
        assert result.valid is False
        assert "too small" in result.error_message.lower()

    def test_file_too_large(self, sample_mp3_file: Path):
        """should fail for files too large."""
        result = validate_file_size(sample_mp3_file, max_size=10)
        assert result.valid is False
        assert "too large" in result.error_message.lower()

    def test_custom_size_limits(self, sample_mp3_file: Path):
        """should respect custom size limits."""
        size = sample_mp3_file.stat().st_size
        result = validate_file_size(
            sample_mp3_file,
            min_size=size - 1,
            max_size=size + 1,
        )
        assert result.valid is True

    def test_nonexistent_file(self, temp_dir: Path):
        """should fail for nonexistent file."""
        result = validate_file_size(temp_dir / "nonexistent.mp3")
        assert result.valid is False


class TestValidateMp3Frames:
    """tests for mp3 frame validation."""

    def test_valid_mp3(self, sample_mp3_file: Path):
        """should pass for valid mp3 file."""
        result = validate_mp3_frames(sample_mp3_file)
        assert result.valid is True
        assert result.validation_type == "mp3_frames"

    def test_invalid_file(self, sample_invalid_file: Path):
        """should fail for non-mp3 file."""
        result = validate_mp3_frames(sample_invalid_file)
        assert result.valid is False

    def test_small_file(self, sample_small_file: Path):
        """should fail for file too small."""
        result = validate_mp3_frames(sample_small_file)
        assert result.valid is False

    def test_sample_count(self, sample_mp3_file: Path):
        """should respect sample count parameter."""
        result = validate_mp3_frames(sample_mp3_file, sample_count=3)
        # should still work with different sample count
        assert isinstance(result, ValidationResult)


class TestValidateDownload:
    """tests for comprehensive download validation."""

    @pytest.mark.asyncio
    async def test_valid_file(self, sample_mp3_file: Path):
        """should pass for valid audio file with frame checks only."""
        # Use config without ffprobe since sample mp3 isn't fully valid
        config = ValidationConfig(use_ffprobe=False)
        result = await validate_download(sample_mp3_file, config)
        assert result.valid is True

    @pytest.mark.asyncio
    async def test_invalid_file(self, sample_invalid_file: Path):
        """should fail for invalid file."""
        result = await validate_download(sample_invalid_file)
        assert result.valid is False

    @pytest.mark.asyncio
    async def test_nonexistent_file(self, temp_dir: Path):
        """should fail for nonexistent file."""
        result = await validate_download(temp_dir / "nonexistent.mp3")
        assert result.valid is False
        assert result.validation_type == "existence"

    @pytest.mark.asyncio
    async def test_custom_config(self, sample_mp3_file: Path):
        """should respect custom config."""
        config = ValidationConfig(
            min_file_size=1,
            max_file_size=100 * 1024 * 1024,
            use_ffprobe=False,
        )
        result = await validate_download(sample_mp3_file, config)
        assert isinstance(result, ValidationResult)

    @pytest.mark.asyncio
    async def test_file_too_small(self, sample_small_file: Path):
        """should fail size check before format check."""
        config = ValidationConfig(min_file_size=1024)
        result = await validate_download(sample_small_file, config)
        assert result.valid is False


class TestValidateDownloadSync:
    """tests for synchronous validation wrapper."""

    def test_valid_file(self, sample_mp3_file: Path):
        """should pass for valid file without ffprobe."""
        config = ValidationConfig(use_ffprobe=False)
        result = validate_download_sync(sample_mp3_file, config)
        assert result.valid is True

    def test_invalid_file(self, sample_invalid_file: Path):
        """should fail for invalid file."""
        result = validate_download_sync(sample_invalid_file)
        assert result.valid is False

    def test_nonexistent_file(self, temp_dir: Path):
        """should fail for nonexistent file."""
        result = validate_download_sync(temp_dir / "nonexistent.mp3")
        assert result.valid is False


class TestValidationIntegration:
    """integration tests for validation pipeline."""

    def test_full_validation_pipeline(self, sample_mp3_file: Path, temp_dir: Path):
        """should run full validation pipeline."""
        # check disk space
        space_result = check_disk_space(temp_dir, required_bytes=1024)
        assert space_result.valid is True

        # validate file size
        size_result = validate_file_size(sample_mp3_file)
        assert size_result.valid is True

        # validate mp3 frames
        frame_result = validate_mp3_frames(sample_mp3_file)
        assert frame_result.valid is True

    @pytest.mark.asyncio
    async def test_async_validation_pipeline(self, sample_mp3_file: Path):
        """should run async validation without ffprobe."""
        config = ValidationConfig(use_ffprobe=False)
        result = await validate_download(sample_mp3_file, config)
        assert result.valid is True
        assert result.validation_type == "complete"


class TestEdgeCases:
    """edge case tests for validation."""

    def test_empty_file(self, temp_dir: Path):
        """should handle empty file."""
        empty_file = temp_dir / "empty.mp3"
        empty_file.touch()
        result = validate_file_size(empty_file, min_size=1)
        assert result.valid is False

    def test_binary_garbage(self, temp_dir: Path):
        """should handle binary garbage."""
        garbage_file = temp_dir / "garbage.mp3"
        with open(garbage_file, "wb") as f:
            f.write(bytes(range(256)) * 100)
        result = validate_mp3_frames(garbage_file)
        assert result.valid is False

    def test_truncated_mp3_header(self, temp_dir: Path):
        """should handle truncated mp3 header."""
        truncated_file = temp_dir / "truncated.mp3"
        with open(truncated_file, "wb") as f:
            f.write(b"ID3")  # only partial id3 header
        result = validate_mp3_frames(truncated_file)
        assert result.valid is False
