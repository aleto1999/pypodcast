#!/usr/bin/env python3
"""
batch speaker diarization script optimized for H100 GPU.

recursively processes podcast transcripts and generates RTTM diarization files.
supports multi-processing, dynamic batching, and I/O parallelization.
"""

import logging
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# critical: set these environment variables BEFORE importing torch or pyannote.
# this prevents PyTorch 2.8.0 multiprocessing bugs (PythonDispatcherTLS errors).
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_VERBOSE"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# additional PyTorch 2.8.0 multiprocessing workarounds.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# add src to path for imports.
# resolve symlinks and make absolute to ensure it works from any directory.
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
src_dir = project_root / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# add scripts to path for utils import.
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

import click

from podcast_conversations.config import DEFAULT_DIARIZATION_MODEL
from utils.cli_utils import get_default_workers
from utils.rich_utils import console, setup_logging

# import directly from submodules to avoid importing torch via __init__.py.
# the diarization/__init__.py imports DiarizationPipeline which imports torch.
from podcast_conversations.diarization.file_discovery import FileDiscovery, FileMapping
from podcast_conversations.diarization.pipeline import (
    detect_diarization_device,
    get_diarization_device_info,
)
from podcast_conversations.monitoring import ResourceDisplay, print_system_info

# configure PyTorch multiprocessing BEFORE creating any process pools.
# note: torch is imported by diarization/__init__.py even though we try to avoid it.
try:
    import torch
    import torch.multiprocessing

    # critical: use 'spawn' start method to avoid CUDA context issues.
    torch.multiprocessing.set_start_method('spawn', force=True)

    # set sharing strategy to avoid file descriptor issues.
    torch.multiprocessing.set_sharing_strategy('file_system')
except Exception:
    pass  # if torch not available or already configured, continue.


def worker_init(device: str) -> None:
    """
    initialize worker process with proper device settings.

    this runs before any other imports in the worker process,
    ensuring CUDA_VISIBLE_DEVICES is set correctly for CUDA devices.
    """
    import os
    import warnings

    if device.startswith("cuda:"):
        device_num = device.split(":")[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = device_num

    # critical: disable torch.compile to avoid PythonDispatcherTLS multiprocessing errors.
    os.environ["TORCH_COMPILE_DISABLE"] = "1"

    # set environment variables to avoid PyTorch multiprocessing issues (CUDA only).
    if device.startswith("cuda"):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torchinductor_cache"

    # suppress verbose torch inductor output.
    os.environ["TORCHDYNAMO_VERBOSE"] = "0"

    # suppress torch._dynamo warnings about untraced builtins.
    warnings.filterwarnings("ignore", category=UserWarning, module="torch._dynamo")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch._inductor")

    # suppress pyannote.audio pooling warnings (harmless statistical edge case).
    warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.models.blocks.pooling")
    warnings.filterwarnings("ignore", message=".*std.*degrees of freedom.*")

    # initialize torch early in worker to set up dispatcher state correctly.
    import torch

    # ensure PyTorch is properly initialized for the device.
    if device.startswith("cuda") and torch.cuda.is_available():
        try:
            torch.cuda.init()
        except Exception:
            pass  # may already be initialized
    elif device == "mps" and torch.backends.mps.is_available():
        # MPS is automatically initialized when used.
        pass


def get_audio_duration(audio_path: Path) -> float:
    """
    get audio file duration in seconds.

    Args:
        audio_path: path to audio file.

    Returns:
        duration in seconds, or 0.0 if unable to determine.
    """
    try:
        # use ffprobe instead of torchaudio to avoid importing torch in main process.
        import json
        import subprocess

        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(audio_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except Exception:
        # fallback: estimate based on file size (rough approximation).
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        return file_size_mb * 60  # rough estimate: 1MB ≈ 1 minute.


def batch_files_by_duration(
    file_mappings: list[FileMapping], batch_size: int = 4
) -> list[list[FileMapping]]:
    """
    group files into batches of similar durations for better GPU utilization.

    Args:
        file_mappings: list of file mappings to batch.
        batch_size: approximate number of files per batch.

    Returns:
        list of batches, each containing FileMapping objects.
    """
    if not file_mappings:
        return []

    # compute durations.
    files_with_duration = []
    for mapping in file_mappings:
        duration = get_audio_duration(mapping.audio_path)
        files_with_duration.append((duration, mapping))

    # sort by duration (process similar lengths together).
    files_with_duration.sort(key=lambda x: x[0])

    # create batches.
    batches = []
    for i in range(0, len(files_with_duration), batch_size):
        batch = [item[1] for item in files_with_duration[i : i + batch_size]]
        batches.append(batch)

    return batches


def process_single_file(
    mapping: FileMapping,
    model_name: str,
    hf_token: str,
    device: str,
    use_bf16: bool,
    use_compile: bool,
    compile_mode: str,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> tuple[FileMapping, bool, Optional[str]]:
    """
    worker function to process a single file (for multiprocessing).

    Args:
        mapping: file mapping with paths.
        model_name: HuggingFace model identifier.
        hf_token: HuggingFace API token.
        device: device to use.
        use_bf16: enable BFloat16.
        use_compile: enable torch.compile.
        compile_mode: compilation mode.
        min_speakers: minimum number of speakers (optional).
        max_speakers: maximum number of speakers (optional).

    Returns:
        tuple of (mapping, success, error_message).
    """
    import time

    # import torch-dependent modules inside worker process.
    from podcast_conversations.diarization import DiarizationPipeline, RTTMWriter

    logger = logging.getLogger(__name__)

    # each worker initializes its own pipeline.
    # CUDA_VISIBLE_DEVICES is set in worker_init, so we always use device 0 for CUDA.
    # torch.compile is disabled via TORCH_COMPILE_DISABLE env var in worker_init.
    if device.startswith("cuda"):
        worker_device = "cuda:0"
    elif device == "mps":
        worker_device = "mps"
    else:
        worker_device = "cpu"

    # retry logic for unexpected errors (torch.compile is already disabled in workers).
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            pipeline = DiarizationPipeline(
                model_name=model_name,
                hf_token=hf_token,
                device=worker_device,
                use_bf16=use_bf16,
                use_compile=False,  # always disabled in workers (TORCH_COMPILE_DISABLE=1)
                compile_mode=compile_mode,
            )

            # process file with speaker constraints.
            diarization = pipeline.process_file(
                mapping.audio_path,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )

            # write RTTM output.
            rttm_writer = RTTMWriter()
            rttm_writer.write(
                annotation=diarization,
                output_path=mapping.output_path,
                audio_filename=mapping.audio_path.name,
            )

            return (mapping, True, None)

        except Exception as e:
            error_msg = str(e)

            # check for PyTorch multiprocessing errors.
            is_pytorch_mp_error = (
                "PythonDispatcherTLS" in error_msg
                or "INTERNAL ASSERT FAILED" in error_msg
            )

            if is_pytorch_mp_error and attempt < max_retries:
                logger.warning(
                    f"PyTorch multiprocessing error on {mapping.transcript_path.name}, "
                    f"retrying without compile (attempt {attempt + 2}/{max_retries + 1})"
                )
                time.sleep(1)
                continue

            # final attempt failed or non-retryable error.
            logger.error(f"failed to process {mapping.transcript_path.name}: {e}")
            return (mapping, False, error_msg)


@click.command()
@click.option(
    "--transcripts-dir",
    type=click.Path(exists=True, path_type=Path),
    default=project_root / "outputs" / "transcripts",
    help="root directory containing JSON transcript files (default: outputs/transcripts)",
)
@click.option(
    "--audio-base-dir",
    type=click.Path(exists=True, path_type=Path),
    default=project_root / "outputs" / "downloads",
    help="root directory containing audio files (default: outputs/downloads)",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=project_root / "outputs" / "diarizations",
    help="output directory for RTTM diarization files (default: outputs/diarizations)",
)
@click.option(
    "--hf-token",
    envvar="HF_TOKEN",
    default=None,
    help="HuggingFace API token (or set HF_TOKEN env var, not required for --dry-run)",
)
@click.option(
    "--model",
    default=DEFAULT_DIARIZATION_MODEL,
    help="HuggingFace model identifier",
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cuda", "cpu", "mps"], case_sensitive=False),
    default="auto",
    help="Device for inference (auto-detects if not specified).",
)
@click.option(
    "--use-bf16/--no-bf16",
    default=False,
    help="enable BFloat16 mixed precision (CUDA only, H100/A100)",
)
@click.option(
    "--torch-compile/--no-torch-compile",
    default=False,
    help="Enable torch.compile optimization (experimental, may cause issues)",
)
@click.option(
    "--compile-mode",
    type=click.Choice(["default", "reduce-overhead", "max-autotune"]),
    default="reduce-overhead",
    help="torch.compile mode",
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
@click.option(
    "--workers",
    type=int,
    default=None,
    help="Number of parallel workers (default: auto-detect, recommended 3-5 for H100 80GB)",
)
@click.option(
    "--batch-size",
    type=int,
    default=4,
    help="number of files per batch for duration-based grouping",
)
@click.option(
    "--force",
    is_flag=True,
    help="Force reprocessing of files with existing outputs",
)
@click.option(
    "--min-speakers",
    type=int,
    default=None,
    help="minimum number of speakers (optional hint for pyannote)",
)
@click.option(
    "--max-speakers",
    type=int,
    default=None,
    help="maximum number of speakers (optional hint for pyannote)",
)
def diarize_batch(
    transcripts_dir: Path,
    audio_base_dir: Path,
    output_dir: Path,
    hf_token: str,
    model: str,
    device: str,
    use_bf16: bool,
    torch_compile: bool,
    compile_mode: str,
    dry_run: bool,
    log_level: str,
    workers: int | None,
    batch_size: int,
    force: bool,
    min_speakers: int | None,
    max_speakers: int | None,
) -> None:
    """
    batch process podcast transcripts for speaker diarization.

    discovers JSON transcript files recursively, processes corresponding audio files,
    and generates RTTM diarization outputs with mirrored directory structure.
    """
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    # auto-detect workers if not specified (use 1GB per worker for diarization).
    if workers is None:
        workers = get_default_workers(memory_per_worker_gb=1.0, cpu_fraction=0.5)

    console.print("\n[bold cyan]🎙️  Podcast Diarization Pipeline[/bold cyan]\n")

    # resolve device and show info.
    resolved_device = device
    if device == "auto":
        resolved_device = detect_diarization_device()

    device_info = get_diarization_device_info()
    if resolved_device == "mps":
        console.print("[green]using MPS (Apple Silicon GPU)[/green]")
        # recommend single worker for MPS to avoid memory issues.
        if workers > 2:
            console.print(
                f"[yellow]note: reducing workers from {workers} to 2 for MPS "
                "(Apple Silicon works better with fewer parallel processes)[/yellow]"
            )
            workers = min(workers, 2)
    elif resolved_device.startswith("cuda"):
        cuda_name = device_info.get("cuda_device_name", "Unknown GPU")
        cuda_mem = device_info.get("cuda_memory_gb", 0)
        console.print(f"[green]using CUDA: {cuda_name} ({cuda_mem:.1f}GB)[/green]")
    else:
        console.print("[yellow]using CPU (no GPU acceleration)[/yellow]")

    # warn if torch.compile is enabled with multiprocessing.
    if torch_compile and workers > 1:
        console.print(
            "[bold yellow]warning: torch.compile is automatically disabled in worker processes "
            "to prevent PythonDispatcherTLS errors[/bold yellow]\n"
        )

    console.print("[bold blue]discovering files...[/bold blue]")
    try:
        discoverer = FileDiscovery(transcripts_dir, audio_base_dir, output_dir)
        file_mappings = discoverer.discover_files()
    except Exception as e:
        console.print(f"error during file discovery: {e}", style="bold red", markup=False)
        sys.exit(1)

    if not file_mappings:
        console.print("[yellow]no transcript-audio pairs found[/yellow]")
        sys.exit(0)

    console.print(f"[green]found {len(file_mappings)} transcript-audio pairs[/green]")

    # ensure output directory exists.
    output_dir.mkdir(parents=True, exist_ok=True)

    # filter out files that already have RTTM outputs (unless --force).
    console.print("[bold blue]checking for existing RTTM files...[/bold blue]")
    files_to_process = []
    files_skipped = []

    for mapping in file_mappings:
        if not force and mapping.output_path.exists():
            files_skipped.append(mapping)
            logger.debug(f"skipping {mapping.transcript_path.name} (RTTM already exists)")
        else:
            files_to_process.append(mapping)

    if files_skipped:
        console.print(f"[yellow]skipping {len(files_skipped)} files with existing RTTM files[/yellow]")
    console.print(f"[green]will process {len(files_to_process)} new files[/green]\n")

    if not files_to_process:
        console.print("[yellow]all files already have RTTM outputs - nothing to process[/yellow]")
        console.print("  use --force to reprocess\n")
        sys.exit(0)

    if dry_run:
        console.print("[bold yellow]dry run mode - showing files to process:[/bold yellow]\n")
        for i, mapping in enumerate(files_to_process[:10], 1):
            console.print(f"  {i}. {mapping.transcript_path.name}")
            console.print(f"     → audio: {mapping.audio_path.relative_to(audio_base_dir)}")
            console.print(
                f"     → output: {mapping.output_path.relative_to(output_dir)}\n"
            )
        if len(files_to_process) > 10:
            console.print(f"  ... and {len(files_to_process) - 10} more files")
        return

    if not hf_token:
        console.print(
            "[bold red]error: --hf-token is required when not in dry-run mode[/bold red]"
        )
        console.print("set HF_TOKEN environment variable or pass --hf-token option")
        sys.exit(1)

    # batch files by duration for better GPU utilization.
    console.print("[bold blue]grouping files by duration...[/bold blue]")
    batches = batch_files_by_duration(files_to_process, batch_size=batch_size)
    console.print(f"[green]created {len(batches)} batches of similar-duration files[/green]\n")

    # flatten batches back to list for parallel processing.
    # (batching is primarily for better ordering, not for true batch inference).
    files_to_process_ordered = [f for batch in batches for f in batch]

    # display system info before starting.
    print_system_info(console)

    console.print(
        f"[bold blue]processing files with {workers} parallel workers...[/bold blue]\n"
    )

    stats = {"processed": 0, "successful": 0, "failed": 0, "skipped": len(files_skipped)}

    # use spawn method to avoid CUDA context issues in forked processes.
    mp_context = mp.get_context("spawn")

    with ResourceDisplay(console=console) as display:
        task = display.progress.add_task(
            "[cyan]diarization | starting...",
            total=len(files_to_process_ordered)
        )

        # use ProcessPoolExecutor for parallel processing.
        # initializer sets CUDA_VISIBLE_DEVICES before imports in worker.
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp_context,
            initializer=worker_init,
            initargs=(resolved_device,),
        ) as executor:
            # submit all tasks.
            futures = {
                executor.submit(
                    process_single_file,
                    mapping,
                    model,
                    hf_token,
                    resolved_device,
                    use_bf16,
                    torch_compile,
                    compile_mode,
                    min_speakers,
                    max_speakers,
                ): mapping
                for mapping in files_to_process_ordered
            }

            # process results as they complete.
            for future in as_completed(futures):
                try:
                    mapping, success, error = future.result()

                    # update stats.
                    file_name = mapping.transcript_path.stem
                    show_name = mapping.transcript_path.parent.name

                    if success:
                        stats["successful"] += 1
                        console.print(
                            f"[green]✓[/green] {show_name}/{file_name}"
                        )
                    else:
                        stats["failed"] += 1
                        console.print(
                            f"[red]✗[/red] {show_name}/{file_name}"
                        )
                        if error:
                            logger.error(
                                f"failed {mapping.transcript_path.name}: {error}"
                            )

                except Exception as e:
                    mapping = futures[future]
                    file_name = mapping.transcript_path.stem
                    show_name = mapping.transcript_path.parent.name

                    console.print(
                        f"[red]✗[/red] {show_name}/{file_name} - exception"
                    )
                    logger.error(
                        f"worker exception for {mapping.transcript_path.name}: {e}"
                    )
                    stats["failed"] += 1

                stats["processed"] += 1

                # update progress with running statistics.
                display.progress.update(
                    task,
                    description=(
                        f"[cyan]diarization | "
                        f"{stats['successful']} ✓  {stats['failed']} ✗"
                    ),
                    advance=1,
                )

        # print resource usage summary.
        display.print_summary()

    console.print("\n[bold green]✓ processing complete![/bold green]\n")
    console.print(f"[yellow]skipped (already exists):[/yellow] {stats['skipped']}")
    console.print(f"[cyan]total processed:[/cyan] {stats['processed']}")
    console.print(f"[green]successful:[/green] {stats['successful']}")
    console.print(f"[red]failed:[/red] {stats['failed']}\n")


if __name__ == "__main__":
    diarize_batch()
