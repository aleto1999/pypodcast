"""comprehensive download validation utilities.

provides enhanced validation beyond basic audio format detection:
- mp3 frame integrity checking (samples multiple frames throughout file)
- ffprobe integration for deep validation
- disk space verification
- file size limits
"""

import asyncio
import shutil
from pathlib import Path
from typing import NamedTuple

from podcast_downloader.audio import detect_audio_format


class ValidationResult(NamedTuple):
    """result of file validation."""

    valid: bool
    error_message: str | None = None
    validation_type: str = "unknown"


class ValidationConfig(NamedTuple):
    """configuration for validation."""

    min_file_size: int = 1024  # 1kb minimum
    max_file_size: int = 500 * 1024 * 1024  # 500mb maximum
    min_disk_space: int = 100 * 1024 * 1024  # 100mb minimum free space
    mp3_sample_count: int = 5  # number of frames to sample for mp3
    use_ffprobe: bool = True  # use ffprobe for deep validation if available


# mp3 bitrate lookup table (mpeg audio layer 3).
MP3_BITRATES = {
    0b0001: 32,
    0b0010: 40,
    0b0011: 48,
    0b0100: 56,
    0b0101: 64,
    0b0110: 80,
    0b0111: 96,
    0b1000: 112,
    0b1001: 128,
    0b1010: 160,
    0b1011: 192,
    0b1100: 224,
    0b1101: 256,
    0b1110: 320,
}

# mp3 sample rate lookup table.
MP3_SAMPLE_RATES = {
    0b00: 44100,
    0b01: 48000,
    0b10: 32000,
}


def check_disk_space(output_dir: Path, required_bytes: int) -> ValidationResult:
    """check if sufficient disk space is available.

    args:
        output_dir: directory where file will be saved.
        required_bytes: minimum bytes of free space needed.

    returns:
        validation result indicating if space is available.
    """
    try:
        stat = shutil.disk_usage(output_dir)
        if stat.free < required_bytes:
            return ValidationResult(
                valid=False,
                error_message=f"insufficient disk space: {stat.free / (1024 * 1024):.1f}mb free, need {required_bytes / (1024 * 1024):.1f}mb",
                validation_type="disk_space",
            )
        return ValidationResult(valid=True, validation_type="disk_space")
    except OSError as e:
        return ValidationResult(
            valid=False,
            error_message=f"cannot check disk space: {e}",
            validation_type="disk_space",
        )


def validate_file_size(
    file_path: Path,
    min_size: int = 1024,
    max_size: int = 500 * 1024 * 1024,
) -> ValidationResult:
    """validate file size is within acceptable range.

    args:
        file_path: path to file to validate.
        min_size: minimum acceptable size in bytes.
        max_size: maximum acceptable size in bytes.

    returns:
        validation result.
    """
    try:
        size = file_path.stat().st_size
        if size < min_size:
            return ValidationResult(
                valid=False,
                error_message=f"file too small: {size} bytes (minimum {min_size})",
                validation_type="file_size",
            )
        if size > max_size:
            return ValidationResult(
                valid=False,
                error_message=f"file too large: {size / (1024 * 1024):.1f}mb (maximum {max_size / (1024 * 1024):.1f}mb)",
                validation_type="file_size",
            )
        return ValidationResult(valid=True, validation_type="file_size")
    except OSError as e:
        return ValidationResult(
            valid=False,
            error_message=f"cannot read file: {e}",
            validation_type="file_size",
        )


def _parse_mp3_frame_header(header: bytes) -> dict | None:
    """parse mp3 frame header bytes.

    args:
        header: 4 bytes of frame header.

    returns:
        dict with frame info or none if invalid.
    """
    if len(header) < 4:
        return None

    # check frame sync (11 bits set to 1).
    if header[0] != 0xFF or (header[1] & 0xE0) != 0xE0:
        return None

    # extract version, layer, bitrate, sample rate.
    version = (header[1] >> 3) & 0x03
    layer = (header[1] >> 1) & 0x03

    # we only validate mpeg audio layer 3.
    if layer != 0x01:  # layer 3
        return None

    bitrate_index = (header[2] >> 4) & 0x0F
    sample_rate_index = (header[2] >> 2) & 0x03
    padding = (header[2] >> 1) & 0x01

    if bitrate_index not in MP3_BITRATES or sample_rate_index not in MP3_SAMPLE_RATES:
        return None

    bitrate = MP3_BITRATES[bitrate_index]
    sample_rate = MP3_SAMPLE_RATES[sample_rate_index]

    # calculate frame size.
    frame_size = int((144 * bitrate * 1000 / sample_rate) + padding)

    return {
        "version": version,
        "layer": layer,
        "bitrate": bitrate,
        "sample_rate": sample_rate,
        "padding": padding,
        "frame_size": frame_size,
    }


def validate_mp3_frames(
    file_path: Path,
    sample_count: int = 5,
) -> ValidationResult:
    """validate mp3 by sampling frames throughout the file.

    samples multiple frames at different positions to detect
    corruption that might only appear in certain parts of the file.

    args:
        file_path: path to mp3 file.
        sample_count: number of frames to sample.

    returns:
        validation result.
    """
    try:
        file_size = file_path.stat().st_size
        if file_size < 1024:
            return ValidationResult(
                valid=False,
                error_message="file too small to be valid mp3",
                validation_type="mp3_frames",
            )

        with open(file_path, "rb") as f:
            # skip id3v2 tag if present.
            header = f.read(10)
            start_offset = 0

            if header[:3] == b"ID3":
                # parse id3v2 tag size.
                tag_size = (
                    (header[6] & 0x7F) << 21
                    | (header[7] & 0x7F) << 14
                    | (header[8] & 0x7F) << 7
                    | (header[9] & 0x7F)
                )
                start_offset = 10 + tag_size

            # calculate sample positions.
            usable_size = file_size - start_offset
            if usable_size < 1000:
                return ValidationResult(
                    valid=False,
                    error_message="audio data too small",
                    validation_type="mp3_frames",
                )

            # sample at evenly spaced positions.
            sample_positions = [
                start_offset + int(usable_size * i / (sample_count + 1))
                for i in range(1, sample_count + 1)
            ]

            valid_frames = 0
            for pos in sample_positions:
                f.seek(pos)

                # scan for frame sync in a small window.
                window = f.read(512)
                found_frame = False

                for i in range(len(window) - 4):
                    if window[i] == 0xFF and (window[i + 1] & 0xE0) == 0xE0:
                        frame_info = _parse_mp3_frame_header(window[i : i + 4])
                        if frame_info:
                            valid_frames += 1
                            found_frame = True
                            break

                if not found_frame:
                    # one missing frame is acceptable (might be at end).
                    pass

            # require at least half the samples to be valid.
            if valid_frames < sample_count // 2:
                return ValidationResult(
                    valid=False,
                    error_message=f"only {valid_frames}/{sample_count} sampled frames are valid",
                    validation_type="mp3_frames",
                )

            return ValidationResult(valid=True, validation_type="mp3_frames")

    except OSError as e:
        return ValidationResult(
            valid=False,
            error_message=f"cannot read file: {e}",
            validation_type="mp3_frames",
        )


async def validate_with_ffprobe(file_path: Path) -> ValidationResult:
    """validate audio file using ffprobe.

    uses ffprobe to perform deep validation of the audio file structure.
    this can detect issues that simple header checks miss.

    args:
        file_path: path to audio file.

    returns:
        validation result.
    """
    if not shutil.which("ffprobe"):
        return ValidationResult(
            valid=True,
            error_message="ffprobe not available, skipped",
            validation_type="ffprobe",
        )

    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration,size:stream=codec_type",
            "-of", "json",
            str(file_path),
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=30.0,
        )

        if process.returncode != 0:
            error_msg = stderr.decode().strip() if stderr else "unknown ffprobe error"
            return ValidationResult(
                valid=False,
                error_message=f"ffprobe error: {error_msg}",
                validation_type="ffprobe",
            )

        # parse json output.
        import json
        data = json.loads(stdout.decode())

        # check for audio stream.
        streams = data.get("streams", [])
        has_audio = any(s.get("codec_type") == "audio" for s in streams)

        if not has_audio:
            return ValidationResult(
                valid=False,
                error_message="no audio stream found in file",
                validation_type="ffprobe",
            )

        # check duration is reasonable.
        duration = float(data.get("format", {}).get("duration", 0))
        if duration < 1:
            return ValidationResult(
                valid=False,
                error_message="audio duration too short",
                validation_type="ffprobe",
            )

        return ValidationResult(valid=True, validation_type="ffprobe")

    except asyncio.TimeoutError:
        return ValidationResult(
            valid=False,
            error_message="ffprobe timed out",
            validation_type="ffprobe",
        )
    except Exception as e:
        return ValidationResult(
            valid=False,
            error_message=f"ffprobe validation failed: {e}",
            validation_type="ffprobe",
        )


async def validate_download(
    file_path: Path,
    config: ValidationConfig | None = None,
) -> ValidationResult:
    """perform comprehensive validation of a downloaded file.

    runs all validation checks:
    1. file size validation
    2. format detection
    3. mp3 frame sampling (for mp3 files)
    4. ffprobe validation (if available and enabled)

    args:
        file_path: path to downloaded file.
        config: validation configuration.

    returns:
        validation result (first failure encountered, or success).
    """
    config = config or ValidationConfig()

    # check file exists.
    if not file_path.exists():
        return ValidationResult(
            valid=False,
            error_message="file does not exist",
            validation_type="existence",
        )

    # validate file size.
    size_result = validate_file_size(
        file_path,
        min_size=config.min_file_size,
        max_size=config.max_file_size,
    )
    if not size_result.valid:
        return size_result

    # detect format.
    audio_format = detect_audio_format(file_path)
    if not audio_format:
        return ValidationResult(
            valid=False,
            error_message="could not detect audio format",
            validation_type="format",
        )

    # mp3-specific frame validation.
    if audio_format == "mp3":
        frame_result = validate_mp3_frames(file_path, config.mp3_sample_count)
        if not frame_result.valid:
            return frame_result

    # ffprobe validation if enabled.
    if config.use_ffprobe:
        ffprobe_result = await validate_with_ffprobe(file_path)
        if not ffprobe_result.valid and ffprobe_result.error_message != "ffprobe not available, skipped":
            return ffprobe_result

    return ValidationResult(valid=True, validation_type="complete")


def validate_download_sync(
    file_path: Path,
    config: ValidationConfig | None = None,
) -> ValidationResult:
    """synchronous wrapper for validate_download.

    use when async is not available or needed.

    args:
        file_path: path to downloaded file.
        config: validation configuration.

    returns:
        validation result.
    """
    config = config or ValidationConfig()

    # check file exists.
    if not file_path.exists():
        return ValidationResult(
            valid=False,
            error_message="file does not exist",
            validation_type="existence",
        )

    # validate file size.
    size_result = validate_file_size(
        file_path,
        min_size=config.min_file_size,
        max_size=config.max_file_size,
    )
    if not size_result.valid:
        return size_result

    # detect format.
    audio_format = detect_audio_format(file_path)
    if not audio_format:
        return ValidationResult(
            valid=False,
            error_message="could not detect audio format",
            validation_type="format",
        )

    # mp3-specific frame validation.
    if audio_format == "mp3":
        frame_result = validate_mp3_frames(file_path, config.mp3_sample_count)
        if not frame_result.valid:
            return frame_result

    return ValidationResult(valid=True, validation_type="complete")
