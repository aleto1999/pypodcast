"""tests for audio module."""

from pathlib import Path

import pytest

from podcast_downloader.audio import (
    AUDIO_SIGNATURES,
    MIN_FILE_SIZE,
    SUPPORTED_AUDIO_TYPES,
    detect_audio_format,
    get_audio_info,
    validate_audio_file,
    validate_mp3,
)


class TestConstants:
    """tests for module constants."""

    def test_supported_audio_types(self):
        """should have common audio types."""
        assert "audio/mpeg" in SUPPORTED_AUDIO_TYPES
        assert "audio/mp3" in SUPPORTED_AUDIO_TYPES
        assert "audio/mp4" in SUPPORTED_AUDIO_TYPES
        assert "audio/ogg" in SUPPORTED_AUDIO_TYPES

    def test_audio_signatures(self):
        """should have common audio signatures."""
        assert b"ID3" in AUDIO_SIGNATURES
        assert b"ftyp" in AUDIO_SIGNATURES
        assert b"RIFF" in AUDIO_SIGNATURES
        assert b"OggS" in AUDIO_SIGNATURES

    def test_min_file_size(self):
        """should have reasonable minimum file size."""
        assert MIN_FILE_SIZE >= 1024


class TestDetectAudioFormat:
    """tests for audio format detection."""

    def test_detect_mp3_with_id3(self, sample_mp3_file: Path):
        """should detect mp3 with id3 tag."""
        result = detect_audio_format(sample_mp3_file)
        assert result == "mp3"

    def test_detect_invalid_file(self, sample_invalid_file: Path):
        """should return None for invalid file."""
        result = detect_audio_format(sample_invalid_file)
        assert result is None

    def test_detect_nonexistent_file(self, temp_dir: Path):
        """should return None for nonexistent file."""
        result = detect_audio_format(temp_dir / "nonexistent.mp3")
        assert result is None

    def test_detect_mp3_frame_sync(self, temp_dir: Path):
        """should detect mp3 from frame sync."""
        mp3_file = temp_dir / "frame_sync.mp3"
        # write mp3 frame sync without id3 tag
        with open(mp3_file, "wb") as f:
            f.write(b"\xff\xfb\x90\x00" + b"\x00" * 1000)
        result = detect_audio_format(mp3_file)
        assert result == "mp3"

    def test_detect_m4a(self, temp_dir: Path):
        """should detect m4a format."""
        m4a_file = temp_dir / "test.m4a"
        # m4a has ftyp at offset 4
        with open(m4a_file, "wb") as f:
            f.write(b"\x00\x00\x00\x20ftyp" + b"\x00" * 1000)
        result = detect_audio_format(m4a_file)
        assert result == "m4a"

    def test_detect_wav(self, temp_dir: Path):
        """should detect wav format."""
        wav_file = temp_dir / "test.wav"
        with open(wav_file, "wb") as f:
            f.write(b"RIFF" + b"\x00" * 1000)
        result = detect_audio_format(wav_file)
        assert result == "wav"

    def test_detect_ogg(self, temp_dir: Path):
        """should detect ogg format."""
        ogg_file = temp_dir / "test.ogg"
        with open(ogg_file, "wb") as f:
            f.write(b"OggS" + b"\x00" * 1000)
        result = detect_audio_format(ogg_file)
        assert result == "ogg"


class TestValidateMp3:
    """tests for mp3 validation."""

    def test_valid_mp3(self, sample_mp3_file: Path):
        """should validate valid mp3."""
        result = validate_mp3(sample_mp3_file)
        assert result is True

    def test_invalid_file(self, sample_invalid_file: Path):
        """should reject invalid file."""
        result = validate_mp3(sample_invalid_file)
        assert result is False

    def test_nonexistent_file(self, temp_dir: Path):
        """should reject nonexistent file."""
        result = validate_mp3(temp_dir / "nonexistent.mp3")
        assert result is False

    def test_truncated_file(self, temp_dir: Path):
        """should handle truncated file."""
        truncated = temp_dir / "truncated.mp3"
        with open(truncated, "wb") as f:
            f.write(b"ID3")
        result = validate_mp3(truncated)
        assert result is False

    def test_mp3_without_id3(self, temp_dir: Path):
        """should validate mp3 without id3 tag."""
        mp3_file = temp_dir / "no_id3.mp3"
        with open(mp3_file, "wb") as f:
            f.write(b"\xff\xfb\x90\x00" + b"\x00" * 1000)
        result = validate_mp3(mp3_file)
        assert result is True


class TestValidateAudioFile:
    """tests for comprehensive audio validation."""

    def test_valid_mp3(self, sample_mp3_file: Path):
        """should validate valid mp3 file."""
        result = validate_audio_file(sample_mp3_file)
        assert result is True

    def test_invalid_file(self, sample_invalid_file: Path):
        """should reject invalid file."""
        result = validate_audio_file(sample_invalid_file)
        assert result is False

    def test_small_file(self, sample_small_file: Path):
        """should reject file below minimum size."""
        result = validate_audio_file(sample_small_file)
        assert result is False

    def test_nonexistent_file(self, temp_dir: Path):
        """should reject nonexistent file."""
        result = validate_audio_file(temp_dir / "nonexistent.mp3")
        assert result is False

    def test_expected_format_match(self, sample_mp3_file: Path):
        """should pass when expected format matches."""
        result = validate_audio_file(sample_mp3_file, expected_format="mp3")
        assert result is True

    def test_expected_format_mismatch(self, sample_mp3_file: Path):
        """should fail when expected format doesn't match."""
        result = validate_audio_file(sample_mp3_file, expected_format="m4a")
        assert result is False

    def test_empty_file(self, temp_dir: Path):
        """should reject empty file."""
        empty = temp_dir / "empty.mp3"
        empty.touch()
        result = validate_audio_file(empty)
        assert result is False


class TestGetAudioInfo:
    """tests for audio info retrieval."""

    def test_valid_file_info(self, sample_mp3_file: Path):
        """should return info for valid file."""
        info = get_audio_info(sample_mp3_file)
        assert info is not None
        assert "path" in info
        assert "size_bytes" in info
        assert "size_mb" in info
        assert "format" in info
        assert "valid" in info

    def test_valid_mp3_info(self, sample_mp3_file: Path):
        """should return correct info for mp3."""
        info = get_audio_info(sample_mp3_file)
        assert info["format"] == "mp3"
        assert info["valid"] is True
        assert info["size_bytes"] > 0

    def test_invalid_file_info(self, sample_invalid_file: Path):
        """should return info for invalid file."""
        info = get_audio_info(sample_invalid_file)
        assert info is not None
        assert info["valid"] is False
        assert info["format"] is None

    def test_nonexistent_file_info(self, temp_dir: Path):
        """should return None for nonexistent file."""
        info = get_audio_info(temp_dir / "nonexistent.mp3")
        assert info is None


class TestEdgeCases:
    """edge case tests for audio module."""

    def test_file_with_wrong_extension(self, temp_dir: Path):
        """should detect format regardless of extension."""
        # mp3 content with .txt extension
        mp3_as_txt = temp_dir / "audio.txt"
        with open(mp3_as_txt, "wb") as f:
            f.write(b"ID3\x04\x00\x00\x00\x00\x00\x00")
            f.write(b"\xff\xfb\x90\x00" + b"\x00" * 1000)
        result = detect_audio_format(mp3_as_txt)
        assert result == "mp3"

    def test_binary_garbage(self, temp_dir: Path):
        """should handle binary garbage gracefully."""
        garbage = temp_dir / "garbage.mp3"
        with open(garbage, "wb") as f:
            f.write(bytes(range(256)) * 50)
        result = validate_audio_file(garbage)
        assert result is False

    def test_very_small_header(self, temp_dir: Path):
        """should handle files with very small headers."""
        tiny = temp_dir / "tiny.mp3"
        with open(tiny, "wb") as f:
            f.write(b"\xff\xfb")  # only 2 bytes
        result = detect_audio_format(tiny)
        # should not crash, may return mp3 or None
        assert result is None or result == "mp3"
