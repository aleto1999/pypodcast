"""audio file validation utilities."""

from pathlib import Path

# audio mime types we support.
SUPPORTED_AUDIO_TYPES = {
    "audio/mpeg",
    "audio/mp3",
    "audio/mp4",
    "audio/m4a",
    "audio/x-m4a",
    "audio/wav",
    "audio/x-wav",
    "audio/ogg",
    "audio/flac",
    "audio/aac",
}

# file signatures (magic bytes) for audio formats.
AUDIO_SIGNATURES = {
    b"\xff\xfb": "mp3",  # mp3 frame sync.
    b"\xff\xfa": "mp3",  # mp3 frame sync.
    b"\xff\xf3": "mp3",  # mp3 frame sync.
    b"\xff\xf2": "mp3",  # mp3 frame sync.
    b"ID3": "mp3",  # id3 tag.
    b"ftyp": "m4a",  # mp4/m4a.
    b"RIFF": "wav",  # wav.
    b"OggS": "ogg",  # ogg.
    b"fLaC": "flac",  # flac.
}

# minimum valid file sizes.
MIN_FILE_SIZE = 1024  # 1kb - anything smaller is likely corrupt.


def detect_audio_format(file_path: Path) -> str | None:
    """detect audio format from file signature."""
    try:
        with open(file_path, "rb") as f:
            header = f.read(12)

        if len(header) < 4:
            return None

        # check for id3 tag (mp3).
        if header[:3] == b"ID3":
            return "mp3"

        # check for mp3 frame sync.
        if header[0] == 0xFF and (header[1] & 0xE0) == 0xE0:
            return "mp3"

        # check for mp4/m4a (ftyp at offset 4).
        if header[4:8] == b"ftyp":
            return "m4a"

        # check other signatures.
        for sig, fmt in AUDIO_SIGNATURES.items():
            if header.startswith(sig):
                return fmt

        return None
    except OSError:
        return None


def validate_mp3(file_path: Path) -> bool:
    """
    basic mp3 validation - check for valid frame headers.

    this is a lightweight check, not a full file scan.
    """
    try:
        with open(file_path, "rb") as f:
            header = f.read(10)

            # skip id3v2 tag if present.
            if header[:3] == b"ID3":
                # id3v2 tag size is in bytes 6-9.
                if len(header) >= 10:
                    size = (
                        (header[6] & 0x7F) << 21
                        | (header[7] & 0x7F) << 14
                        | (header[8] & 0x7F) << 7
                        | (header[9] & 0x7F)
                    )
                    f.seek(10 + size)
                    header = f.read(4)
                else:
                    return False

            # check for mp3 frame sync.
            if len(header) >= 2:
                return header[0] == 0xFF and (header[1] & 0xE0) == 0xE0

        return False
    except OSError:
        return False


def validate_audio_file(file_path: Path, expected_format: str | None = None) -> bool:
    """
    validate an audio file.

    checks:
    - file exists and is readable.
    - file size is above minimum threshold.
    - file has valid audio signature.
    - for mp3, checks for valid frame header.
    """
    if not file_path.exists():
        return False

    # check file size.
    file_size = file_path.stat().st_size
    if file_size < MIN_FILE_SIZE:
        return False

    # detect format.
    detected_format = detect_audio_format(file_path)
    if detected_format is None:
        return False

    # check expected format if specified.
    if expected_format and detected_format != expected_format:
        return False

    # additional validation for mp3.
    if detected_format == "mp3":
        return validate_mp3(file_path)

    return True


def get_audio_info(file_path: Path) -> dict | None:
    """get basic audio file info."""
    if not file_path.exists():
        return None

    try:
        stat = file_path.stat()
        detected_format = detect_audio_format(file_path)

        return {
            "path": str(file_path),
            "size_bytes": stat.st_size,
            "size_mb": stat.st_size / (1024 * 1024),
            "format": detected_format,
            "valid": validate_audio_file(file_path),
        }
    except OSError:
        return None
