#!/usr/bin/env python3
"""
batch transcription script using whisperx.

recursively processes audio files and generates json transcripts.
"""

import os
import sys
from pathlib import Path

# setup cuDNN 8.x library path BEFORE any imports that trigger .so loading.
# ctranslate2 (used by faster-whisper/whisperx) needs libcudnn_ops_infer.so.8
# which comes from nvidia-cudnn-cu11. the dynamic linker only reads
# LD_LIBRARY_PATH at process startup, so we must set it and re-exec.
if not os.environ.get("_CUDNN_PATH_SET"):
    import site

    site_dirs = site.getsitepackages() + [site.getusersitepackages()]
    cudnn_dirs = []
    for site_dir in site_dirs:
        site_path = Path(site_dir)
        if not site_path.exists():
            continue
        for so_file in site_path.rglob("libcudnn_ops_infer.so.8*"):
            cudnn_dirs.append(str(so_file.parent))
            break

    if cudnn_dirs:
        current = os.environ.get("LD_LIBRARY_PATH", "")
        new_paths = ":".join(cudnn_dirs)
        os.environ["LD_LIBRARY_PATH"] = f"{new_paths}:{current}" if current else new_paths
        os.environ["_CUDNN_PATH_SET"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)

import json
import logging

# add src to path for imports FIRST.
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
src_dir = project_root / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# CRITICAL: apply PyTorch 2.6+ compatibility fix BEFORE importing torch/click/anything else.
# this registers omegaconf classes as safe globals for torch.load.
from src.podcast_conversations.transcription import torchaudio_compat  # noqa: F401, E402

import click

from src.podcast_conversations.monitoring import ResourceDisplay, print_system_info
from src.podcast_conversations.transcription import (
    TranscriptionConfig,
    create_transcription_pipeline,
    detect_transcription_device,
    get_mlx_transcription_status,
)
from utils.rich_utils import console, setup_logging

# supported audio formats.
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".wma", ".aac", ".mp4"}


def discover_audio_files(audio_dir: Path) -> list[Path]:
    """discover all audio files recursively."""
    files = []
    for ext in AUDIO_EXTENSIONS:
        files.extend(audio_dir.rglob(f"*{ext}"))
    return sorted(files)


def check_transcript_exists(output_path: Path) -> bool:
    """check if valid transcript already exists."""
    if not output_path.exists():
        return False

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # verify it has segments.
        segments = data.get("segments", [])
        return len(segments) > 0

    except (json.JSONDecodeError, KeyError):
        return False


@click.command()
@click.option(
    "--audio-dir",
    type=click.Path(exists=True, path_type=Path),
    default=project_root / "outputs" / "downloads",
    help="root directory containing audio files (default: outputs/downloads). transcripts will preserve subdirectory structure.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=project_root / "outputs" / "transcripts",
    help="output directory for transcript json files (default: outputs/transcripts)",
)
@click.option(
    "--model",
    type=click.Choice([
        "tiny", "base", "small", "medium", "large-v2", "large-v3",
        "turbo", "large-v3-turbo", "large-v3-8bit", "large-v3-4bit", "distil-large-v3"
    ]),
    default="large-v3",
    help="whisper model size (turbo/quantized variants available for mlx)",
)
@click.option(
    "--language",
    type=str,
    default=None,
    help="language code (en, es, etc.) or omit for auto-detect",
)
@click.option(
    "--compute-type",
    type=click.Choice(["float16", "int8", "float32"]),
    default="float16",
    help="compute precision type (ignored for mlx backend)",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="batch size (auto-calculated if not specified)",
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cuda", "mps", "mlx", "cpu"]),
    default="auto",
    help="Device: auto (recommended), cuda, mps, mlx (Apple Silicon), cpu",
)
@click.option(
    "--hf-token",
    type=str,
    envvar="HF_TOKEN",
    default=None,
    help="HuggingFace token (reserved for future use, not currently required)",
)
@click.option(
    "--force",
    is_flag=True,
    help="Force reprocessing of files with existing outputs",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be processed without running",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Logging level",
)
def transcribe_batch(
    audio_dir: Path,
    output_dir: Path,
    model: str,
    language: str | None,
    compute_type: str,
    batch_size: int | None,
    device: str,
    hf_token: str | None,  # noqa: ARG001  # reserved for future use
    force: bool,
    dry_run: bool,
    log_level: str,
) -> None:
    """
    batch transcribe audio files using whisperx.

    discovers audio files recursively and generates json transcripts
    with word-level timestamps.
    """
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    console.print("\n[bold cyan]🎙️  Podcast Transcription Pipeline[/bold cyan]\n")

    # discover audio files.
    console.print("[bold blue]discovering audio files...[/bold blue]")
    audio_files = discover_audio_files(audio_dir)

    if not audio_files:
        console.print(f"[yellow]no audio files found in {audio_dir}[/yellow]")
        sys.exit(0)

    console.print(f"[green]found {len(audio_files)} audio files[/green]")

    # filter out files that already have transcripts.
    files_to_process = []
    files_skipped = 0

    for audio_path in audio_files:
        # create output path preserving show subdirectory structure.
        # if audio_dir is a specific show (e.g., outputs/downloads/the_daily),
        # preserve that show name in the output path.
        rel_path = audio_path.relative_to(audio_dir)
        
        # if the audio file is directly in audio_dir (no subdirs),
        # add the show name from audio_dir's parent directory.
        if rel_path.parent == Path("."):
            show_name = audio_dir.name
            output_path = output_dir / show_name / rel_path.with_suffix(".json")
        else:
            # already has subdirectory structure, preserve it.
            output_path = output_dir / rel_path.with_suffix(".json")

        if not force and check_transcript_exists(output_path):
            files_skipped += 1
            logger.debug(f"skipping {audio_path.name} (transcript exists)")
        else:
            files_to_process.append((audio_path, output_path))

    if files_skipped > 0:
        console.print(f"[yellow]skipping {files_skipped} files with existing transcripts[/yellow]")

    console.print(f"[green]will process {len(files_to_process)} new files[/green]\n")

    if not files_to_process:
        console.print("[yellow]all files already have transcripts - nothing to process[/yellow]")
        console.print("  use --force to reprocess\n")
        sys.exit(0)

    # dry run mode.
    if dry_run:
        console.print("[bold yellow]dry run mode - files to process:[/bold yellow]\n")
        for i, (audio_path, output_path) in enumerate(files_to_process[:10], 1):
            rel = audio_path.relative_to(audio_dir)
            console.print(f"  {i}. {rel}")
        if len(files_to_process) > 10:
            console.print(f"  ... and {len(files_to_process) - 10} more files")
        return

    # resolve device.
    resolved_device = device
    if device == "auto":
        resolved_device = detect_transcription_device()

    # show mlx status on Apple Silicon.
    if resolved_device == "mlx":
        mlx_status = get_mlx_transcription_status()
        if mlx_status["mlx_whisper_available"]:
            console.print("[green]mlx-whisper available on Apple Silicon[/green]")
        else:
            console.print("[yellow]mlx-whisper not installed, falling back to cpu[/yellow]")
            if mlx_status["install_command"]:
                console.print(f"[dim]install with: {mlx_status['install_command']}[/dim]")
            resolved_device = "cpu"

    # create config and pipeline.
    config = TranscriptionConfig(
        model_size=model,
        language=language,
        compute_type=compute_type,
        batch_size=batch_size,
        device=resolved_device,
    )

    console.print(f"[dim]model: {model}, compute: {compute_type}, device: {resolved_device}[/dim]\n")

    # display system info before starting.
    print_system_info(console)

    pipeline = create_transcription_pipeline(config)

    # process files.
    stats = {"processed": 0, "successful": 0, "failed": 0, "skipped": files_skipped}

    with ResourceDisplay(console=console) as display:
        task = display.progress.add_task(
            "[cyan]transcribing | starting...",
            total=len(files_to_process),
        )

        for i, (audio_path, output_path) in enumerate(files_to_process):
            rel_path = audio_path.relative_to(audio_dir)

            try:
                pipeline.transcribe_to_json(audio_path, output_path)
                stats["successful"] += 1

                display.progress.update(
                    task,
                    description=(
                        f"[cyan]transcribing | "
                        f"{stats['successful']} ✓  {stats['failed']} ✗"
                    ),
                    advance=1,
                )

            except Exception as e:
                stats["failed"] += 1
                logger.error(f"failed {rel_path}: {e}")

                display.progress.update(
                    task,
                    description=(
                        f"[cyan]transcribing | "
                        f"{stats['successful']} ✓  {stats['failed']} ✗"
                    ),
                    advance=1,
                )

            stats["processed"] += 1

            # periodic cache clearing every 10 files.
            if (i + 1) % 10 == 0:
                pipeline.clear_cache()

        # print resource usage summary.
        display.print_summary()

    # cleanup.
    pipeline.unload()

    # summary.
    console.print("\n[bold green]✓ transcription complete![/bold green]\n")
    console.print(f"[yellow]skipped (already exists):[/yellow] {stats['skipped']}")
    console.print(f"[cyan]total processed:[/cyan] {stats['processed']}")
    console.print(f"[green]successful:[/green] {stats['successful']}")
    console.print(f"[red]failed:[/red] {stats['failed']}")
    console.print(f"\n[bold green]output saved to: {output_dir}[/bold green]\n")


if __name__ == "__main__":
    transcribe_batch()
