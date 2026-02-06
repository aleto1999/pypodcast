#!/usr/bin/env python3
"""
full pipeline script for podcast conversations analysis with metadata tracking.

runs all 7 pipeline stages in sequence:
1. download rss metadata - fetch podcast metadata for speaker identification
2. transcription - generate transcripts from audio files using whisperx
3. diarization - identify speakers in audio files
4. combine transcripts with diarizations - merge speaker labels with transcripts
5. combine consecutive speakers - merge fragmented same-speaker segments
6. speaker labeling - label speakers using rss metadata and heuristics/llm
7. utterance classification - classify segments with trained models

each stage writes a metadata file documenting execution details to:
outputs/metadata_pipelines/SHOW/PIPELINE_NAME_TIMESTAMP.txt

usage:
    # run full pipeline with default settings.
    uv run python scripts/run_full_pipeline.py --audio-dir outputs/downloads

    # run with speaker labeling using LLM.
    uv run python scripts/run_full_pipeline.py --audio-dir outputs/downloads --use-llm

    # skip transcription and diarization (use existing files).
    uv run python scripts/run_full_pipeline.py --audio-dir outputs/downloads \\
        --skip-transcription --skip-diarization

    # dry run to see what would be executed.
    uv run python scripts/run_full_pipeline.py --audio-dir outputs/downloads --dry-run
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# add src to path for imports.
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
src_dir = project_root / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

# CRITICAL: apply PyTorch 2.6+ compatibility fix BEFORE importing torch.
# this registers omegaconf classes as safe globals for torch.load.
from podcast_conversations.transcription import torchaudio_compat  # noqa: F401, E402

import click
from rich.panel import Panel
from rich.table import Table

from podcast_conversations.monitoring import print_system_info
from utils.rich_utils import console


# =============================================================================
# pipeline metadata tracking
# =============================================================================


@dataclass
class PipelineMetadata:
    """metadata for a pipeline execution."""

    pipeline_name: str
    inputs: list[str] = field(default_factory=list)
    outputs_generated: list[str] = field(default_factory=list)
    output_paths: list[str] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    duration_seconds: float = 0.0
    num_outputs_generated: int = 0
    success: bool = False
    error_message: str | None = None
    command_executed: str = ""
    extra_data: dict[str, Any] = field(default_factory=dict)


def get_system_info() -> dict[str, Any]:
    """collect system information for reproducibility."""
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": platform.node(),
    }

    # cpu info.
    try:
        info["cpu_count"] = os.cpu_count()
    except Exception:
        pass

    # memory info.
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["total_memory_gb"] = round(mem.total / (1024**3), 2)
        info["available_memory_gb"] = round(mem.available / (1024**3), 2)
    except ImportError:
        pass

    # gpu info.
    try:
        import torch
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
            )
        elif torch.backends.mps.is_available():
            info["mps_available"] = True
        else:
            info["gpu_available"] = False
    except ImportError:
        pass

    # package versions.
    try:
        import importlib.metadata as metadata
        packages = [
            "torch", "transformers", "whisperx", "pyannote.audio",
            "rich", "click", "httpx",
        ]
        info["package_versions"] = {}
        for pkg in packages:
            try:
                info["package_versions"][pkg] = metadata.version(pkg)
            except metadata.PackageNotFoundError:
                pass
    except ImportError:
        pass

    return info


def get_file_count(directory: Path, pattern: str = "*.json") -> int:
    """count files matching pattern in directory."""
    if not directory.exists():
        return 0
    return len(list(directory.rglob(pattern)))


def get_dir_count(directory: Path) -> int:
    """count subdirectories in directory."""
    if not directory.exists():
        return 0
    return len([d for d in directory.iterdir() if d.is_dir() and not d.name.startswith(".")])


def format_duration(seconds: float) -> str:
    """format duration in human-readable form."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def write_pipeline_metadata(
    metadata: PipelineMetadata,
    metadata_dir: Path,
    show_name: str | None = None,
) -> Path:
    """
    write pipeline metadata to a text file.

    Args:
        metadata: the pipeline metadata to write.
        metadata_dir: base directory for metadata files.
        show_name: optional show name for subdirectory organization.

    Returns:
        path to the written metadata file.
    """
    # create subdirectory structure.
    if show_name:
        output_dir = metadata_dir / show_name
    else:
        output_dir = metadata_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # generate filename with timestamp.
    timestamp = metadata.start_time.strftime("%Y_%m_%d_%H_%M_%S")
    safe_name = metadata.pipeline_name.lower().replace(" ", "_").replace("-", "_")
    filename = f"{safe_name}_{timestamp}.txt"
    metadata_path = output_dir / filename

    # collect system info.
    system_info = get_system_info()

    # format the metadata file content.
    content = f"""================================================================================
                     PIPELINE EXECUTION METADATA
================================================================================

Pipeline Name:          {metadata.pipeline_name}
Execution Date:         {metadata.start_time.strftime('%Y-%m-%d')}
Execution Start Time:   {metadata.start_time.strftime('%H:%M:%S')}
Execution End Time:     {metadata.end_time.strftime('%H:%M:%S') if metadata.end_time else 'N/A'}
Duration:               {format_duration(metadata.duration_seconds)}
Status:                 {'SUCCESS' if metadata.success else 'FAILED'}

--------------------------------------------------------------------------------
                              INPUTS
--------------------------------------------------------------------------------
"""

    for i, input_path in enumerate(metadata.inputs, 1):
        content += f"  {i}. {input_path}\n"

    content += f"""
--------------------------------------------------------------------------------
                           OUTPUTS GENERATED
--------------------------------------------------------------------------------
Number of Outputs:      {metadata.num_outputs_generated}

Output Paths:
"""

    for i, output_path in enumerate(metadata.output_paths, 1):
        content += f"  {i}. {output_path}\n"

    if metadata.outputs_generated:
        content += "\nOutput Files/Directories:\n"
        for i, output in enumerate(metadata.outputs_generated[:20], 1):
            content += f"  {i}. {output}\n"
        if len(metadata.outputs_generated) > 20:
            content += f"  ... and {len(metadata.outputs_generated) - 20} more\n"

    content += f"""
--------------------------------------------------------------------------------
                          COMMAND EXECUTED
--------------------------------------------------------------------------------
{metadata.command_executed}

--------------------------------------------------------------------------------
                         SYSTEM INFORMATION
--------------------------------------------------------------------------------
Platform:               {system_info.get('platform', 'N/A')}
Python Version:         {system_info.get('python_version', 'N/A')}
Machine:                {system_info.get('machine', 'N/A')}
Hostname:               {system_info.get('hostname', 'N/A')}
CPU Count:              {system_info.get('cpu_count', 'N/A')}
Total Memory:           {system_info.get('total_memory_gb', 'N/A')} GB
Available Memory:       {system_info.get('available_memory_gb', 'N/A')} GB
"""

    if system_info.get('cuda_available'):
        content += f"""
GPU:                    {system_info.get('cuda_device_name', 'N/A')}
GPU Memory:             {system_info.get('cuda_memory_gb', 'N/A')} GB
"""
    elif system_info.get('mps_available'):
        content += "\nGPU:                    Apple Silicon (MPS)\n"

    if system_info.get('package_versions'):
        content += "\nPackage Versions:\n"
        for pkg, version in system_info['package_versions'].items():
            content += f"  {pkg}: {version}\n"

    if metadata.extra_data:
        content += """
--------------------------------------------------------------------------------
                         ADDITIONAL INFORMATION
--------------------------------------------------------------------------------
"""
        for key, value in metadata.extra_data.items():
            content += f"  {key}: {value}\n"

    if metadata.error_message:
        content += f"""
--------------------------------------------------------------------------------
                              ERROR
--------------------------------------------------------------------------------
{metadata.error_message}
"""

    content += """
================================================================================
                              END METADATA
================================================================================
"""

    # write to file.
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write(content)

    return metadata_path


# =============================================================================
# pipeline execution helpers
# =============================================================================


def run_command(
    cmd: list[str],
    description: str,
    dry_run: bool = False,
) -> tuple[bool, str | None]:
    """
    run a command and return success status and error message.

    Returns:
        tuple of (success, error_message).
    """
    prefix = "[DRY RUN] " if dry_run else ""
    console.print(f"\n[bold blue]{prefix}Running: {description}[/bold blue]")
    console.print(f"[dim]{' '.join(cmd)}[/dim]\n")

    if dry_run:
        return True, None

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result.returncode == 0, None
    except subprocess.CalledProcessError as e:
        error_msg = f"Command failed with exit code {e.returncode}"
        console.print(f"[bold red]Error running {description}: {error_msg}[/bold red]")
        return False, error_msg
    except FileNotFoundError as e:
        error_msg = f"Command not found: {e}"
        console.print(f"[bold red]{error_msg}[/bold red]")
        return False, error_msg


def get_show_names_from_dir(directory: Path) -> list[str]:
    """get list of show names (subdirectories) from a directory."""
    if not directory.exists():
        return []
    return [d.name for d in directory.iterdir() if d.is_dir() and not d.name.startswith(".")]


def run_pipeline_stage(
    cmd: list[str],
    pipeline_name: str,
    inputs: list[str],
    output_paths: list[str],
    metadata_base_dir: Path,
    show_name: str | None,
    dry_run: bool = False,
    extra_data: dict[str, Any] | None = None,
) -> tuple[bool, PipelineMetadata]:
    """
    run a pipeline stage and write metadata.

    Returns:
        tuple of (success, metadata).
    """
    metadata = PipelineMetadata(
        pipeline_name=pipeline_name,
        inputs=inputs,
        output_paths=output_paths,
        command_executed=" ".join(cmd),
        extra_data=extra_data or {},
    )

    success, error = run_command(cmd, pipeline_name, dry_run)

    metadata.end_time = datetime.now()
    metadata.duration_seconds = (metadata.end_time - metadata.start_time).total_seconds()
    metadata.success = success
    metadata.error_message = error

    # count outputs.
    for output_path in output_paths:
        path = Path(output_path)
        if path.exists():
            if path.is_dir():
                # count files in directory.
                json_count = get_file_count(path, "*.json")
                rttm_count = get_file_count(path, "*.rttm")
                metadata.num_outputs_generated += json_count + rttm_count
                # collect some output filenames.
                for f in list(path.rglob("*.json"))[:10]:
                    metadata.outputs_generated.append(str(f.relative_to(path.parent)))
                for f in list(path.rglob("*.rttm"))[:10]:
                    metadata.outputs_generated.append(str(f.relative_to(path.parent)))
            else:
                metadata.num_outputs_generated += 1
                metadata.outputs_generated.append(path.name)

    # write metadata file.
    if not dry_run:
        metadata_path = write_pipeline_metadata(metadata, metadata_base_dir, show_name)
        console.print(f"[dim]Metadata written to: {metadata_path}[/dim]")
    else:
        console.print(
            f"[dim][DRY RUN] Would write metadata to: "
            f"{metadata_base_dir / (show_name or '')}/{pipeline_name.lower().replace(' ', '_')}_*.txt[/dim]"
        )

    return success, metadata


# =============================================================================
# main pipeline
# =============================================================================


def generate_final_summary(
    outputs_dir: Path,
    audio_dir: Path,
    start_time: datetime,
    end_time: datetime,
    stages_completed: list[str],
    stages_failed: list[str],
    stages_skipped: list[str],
) -> str:
    """generate a summary report of the full pipeline run."""
    duration = end_time - start_time
    hours, remainder = divmod(int(duration.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)

    # count output files.
    rss_metadata_count = get_dir_count(outputs_dir / "rss_metadata")
    transcripts_count = get_file_count(outputs_dir / "transcripts")
    diarizations_count = get_file_count(outputs_dir / "diarizations", "*.rttm")
    transcripts_with_diarization_count = get_file_count(
        outputs_dir / "transcripts_with_diarization_labels"
    )
    transcripts_postprocessed_count = get_file_count(
        outputs_dir / "transcripts_with_diarization_labels_postprocessed"
    )
    speaker_labeled_count = get_file_count(
        outputs_dir / "transcripts_with_speaker_labels_postprocessed"
    )
    classified_count = get_file_count(
        outputs_dir / "transcripts_with_speaker_labels_postprocessed_with_classification_labels"
    )

    summary = f"""
================================================================================
                    FULL PIPELINE EXECUTION REPORT
================================================================================

Run Date: {start_time.strftime('%Y-%m-%d %H:%M:%S')}
Duration: {hours}h {minutes}m {seconds}s

--------------------------------------------------------------------------------
                                 INPUT
--------------------------------------------------------------------------------
Audio Directory: {audio_dir}

--------------------------------------------------------------------------------
                              PIPELINE STAGES
--------------------------------------------------------------------------------
"""

    all_stages = [
        "1. Download RSS Metadata",
        "2. Transcription",
        "3. Diarization",
        "4. Combine Transcripts with Diarizations",
        "5. Combine Consecutive Speakers",
        "6. Speaker Labeling",
        "7. Utterance Classification",
    ]

    for stage in all_stages:
        if stage in stages_completed:
            summary += f"  [OK] {stage}\n"
        elif stage in stages_failed:
            summary += f"  [X]  {stage} (FAILED)\n"
        elif stage in stages_skipped:
            summary += f"  [-]  {stage} (skipped)\n"
        else:
            summary += f"  [?]  {stage}\n"

    summary += f"""
--------------------------------------------------------------------------------
                              OUTPUT SUMMARY
--------------------------------------------------------------------------------
RSS Metadata:                                   {rss_metadata_count} podcasts
Transcripts:                                    {transcripts_count} files
Diarizations (RTTM):                            {diarizations_count} files
Transcripts with Diarization:                   {transcripts_with_diarization_count} files
Transcripts Postprocessed:                      {transcripts_postprocessed_count} files
Speaker Labeled Transcripts:                    {speaker_labeled_count} files
Classified Transcripts:                         {classified_count} files

--------------------------------------------------------------------------------
                              OUTPUT LOCATIONS
--------------------------------------------------------------------------------
RSS Metadata:           {outputs_dir / 'rss_metadata'}
Transcripts:            {outputs_dir / 'transcripts'}
Diarizations:           {outputs_dir / 'diarizations'}
With Diarization:       {outputs_dir / 'transcripts_with_diarization_labels'}
Postprocessed:          {outputs_dir / 'transcripts_with_diarization_labels_postprocessed'}
Speaker Labeled:        {outputs_dir / 'transcripts_with_speaker_labels_postprocessed'}
Classified:             {outputs_dir / 'transcripts_with_speaker_labels_postprocessed_with_classification_labels'}
Pipeline Metadata:      {outputs_dir / 'metadata_pipelines'}

================================================================================
                                  END REPORT
================================================================================
"""
    return summary


@click.command()
@click.option(
    "--audio-dir",
    type=click.Path(exists=True, path_type=Path),
    default=Path("outputs/downloads"),
    help="root directory containing audio files (default: outputs/downloads)",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("outputs"),
    help="base output directory for all pipeline outputs (default: outputs)",
)
# rss metadata options.
@click.option(
    "--skip-rss-download",
    is_flag=True,
    help="skip rss metadata download (use existing metadata)",
)
# transcription options.
@click.option(
    "--skip-transcription",
    is_flag=True,
    help="skip transcription stage (use existing transcripts)",
)
@click.option(
    "--whisper-model",
    type=click.Choice([
        "tiny", "base", "small", "medium", "large-v2", "large-v3",
        "turbo", "large-v3-turbo", "large-v3-8bit", "large-v3-4bit", "distil-large-v3"
    ]),
    default="large-v3",
    help="whisperx model size for transcription",
)
@click.option(
    "--language",
    type=str,
    default=None,
    help="language code for transcription (e.g., en, es). auto-detect if not specified.",
)
@click.option(
    "--transcription-device",
    type=click.Choice(["auto", "cuda", "mps", "mlx", "cpu"]),
    default="auto",
    help="device for transcription: auto, cuda, mps, mlx, cpu",
)
# diarization options.
@click.option(
    "--skip-diarization",
    is_flag=True,
    help="skip diarization stage (use existing RTTM files)",
)
@click.option(
    "--hf-token",
    type=str,
    envvar="HF_TOKEN",
    default=None,
    help="HuggingFace API token for diarization (or set HF_TOKEN env var)",
)
@click.option(
    "--diarization-device",
    type=click.Choice(["auto", "cuda", "cpu", "mps"]),
    default="auto",
    help="device for diarization: auto, cuda, cpu, mps",
)
@click.option(
    "--diarization-workers",
    type=int,
    default=None,
    help="number of parallel workers for diarization (auto-detect if not specified)",
)
@click.option(
    "--torch-compile/--no-torch-compile",
    default=False,
    help="enable torch.compile for diarization (experimental)",
)
@click.option(
    "--min-speakers",
    type=int,
    default=None,
    help="minimum number of speakers hint for diarization",
)
@click.option(
    "--max-speakers",
    type=int,
    default=None,
    help="maximum number of speakers hint for diarization",
)
# speaker labeling options.
@click.option(
    "--skip-speaker-labeling",
    is_flag=True,
    help="skip speaker labeling stage",
)
@click.option(
    "--use-llm",
    is_flag=True,
    help="use OpenAI API for speaker role classification (requires OPENAI_API_KEY)",
)
@click.option(
    "--llm-model",
    type=str,
    default="gpt-4o-mini",
    help="OpenAI model to use for speaker classification",
)
@click.option(
    "--use-local",
    is_flag=True,
    help="use local transformers model with CUDA for speaker classification",
)
@click.option(
    "--local-model",
    type=str,
    default=None,
    help="HuggingFace model name for local inference (auto-selects if not specified)",
)
@click.option(
    "--4bit",
    "load_4bit",
    is_flag=True,
    help="use 4-bit quantization for local model (reduces memory ~75%%)",
)
@click.option(
    "--8bit",
    "load_8bit",
    is_flag=True,
    help="use 8-bit quantization for local model (reduces memory ~50%%)",
)
# classification options.
@click.option(
    "--skip-classification",
    is_flag=True,
    help="skip utterance classification stage",
)
@click.option(
    "--classifiers-config",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="utterance classification configuration file",
)
@click.option(
    "--keywords-config",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="keyword analysis configuration file",
)
@click.option(
    "--min-matches",
    type=int,
    default=1,
    help="minimum keyword matches for classification filtering",
)
# general options.
@click.option(
    "--force",
    is_flag=True,
    help="force reprocessing of all files (passed to all stages)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="show what would be run without executing",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="logging level (passed to all stages)",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="enable verbose logging (same as --log-level DEBUG)",
)
def run_full_pipeline(
    audio_dir: Path,
    output_dir: Path,
    # rss options.
    skip_rss_download: bool,
    # transcription options.
    skip_transcription: bool,
    whisper_model: str,
    language: str | None,
    transcription_device: str,
    # diarization options.
    skip_diarization: bool,
    hf_token: str | None,
    diarization_device: str,
    diarization_workers: int | None,
    torch_compile: bool,
    min_speakers: int | None,
    max_speakers: int | None,
    # speaker labeling options.
    skip_speaker_labeling: bool,
    use_llm: bool,
    llm_model: str,
    use_local: bool,
    local_model: str | None,
    load_4bit: bool,
    load_8bit: bool,
    # classification options.
    skip_classification: bool,
    classifiers_config: Path | None,
    keywords_config: Path | None,
    min_matches: int,
    # general options.
    force: bool,
    dry_run: bool,
    log_level: str,
    verbose: bool,
) -> None:
    """
    run the complete podcast conversations pipeline with metadata tracking.

    executes all 7 stages in sequence:
    1. download rss metadata
    2. transcription (whisperx)
    3. diarization (pyannote)
    4. combine transcripts with diarization labels
    5. combine consecutive speakers
    6. speaker labeling
    7. utterance classification

    each stage writes a metadata file to outputs/metadata_pipelines/ documenting:
    - inputs and outputs
    - execution time and duration
    - system configuration
    - package versions

    \b
    examples:
        # run full pipeline with default settings.
        uv run python scripts/run_full_pipeline.py --audio-dir outputs/downloads

        # run with LLM-based speaker classification.
        uv run python scripts/run_full_pipeline.py --audio-dir outputs/downloads --use-llm

        # skip transcription and diarization (use existing files).
        uv run python scripts/run_full_pipeline.py --audio-dir outputs/downloads \\
            --skip-transcription --skip-diarization

        # dry run to see what would be executed.
        uv run python scripts/run_full_pipeline.py --audio-dir outputs/downloads --dry-run
    """
    start_time = datetime.now()

    # handle verbose flag.
    if verbose:
        log_level = "DEBUG"

    console.print(Panel.fit(
        "[bold cyan]Full Podcast Pipeline - All Stages[/bold cyan]\n\n"
        f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}",
        border_style="cyan",
    ))

    # display system info.
    print_system_info(console)

    # resolve paths.
    audio_dir = audio_dir.resolve()
    output_dir = output_dir.resolve()

    # define output directories.
    rss_metadata_dir = output_dir / "rss_metadata"
    transcripts_dir = output_dir / "transcripts"
    diarizations_dir = output_dir / "diarizations"
    transcripts_with_diarization_dir = output_dir / "transcripts_with_diarization_labels"
    transcripts_postprocessed_dir = output_dir / "transcripts_with_diarization_labels_postprocessed"
    speaker_labeled_dir = output_dir / "transcripts_with_speaker_labels_postprocessed"
    classified_dir = output_dir / "transcripts_with_speaker_labels_postprocessed_with_classification_labels"
    metadata_base_dir = output_dir / "metadata_pipelines"

    # ensure directories exist.
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_base_dir.mkdir(parents=True, exist_ok=True)

    stages_completed = []
    stages_failed = []
    stages_skipped = []
    all_metadata: list[PipelineMetadata] = []

    # get show names for metadata organization.
    show_names = get_show_names_from_dir(audio_dir)
    primary_show = show_names[0] if show_names else "unknown_show"

    # =========================================================================
    # stage 1: download rss metadata
    # =========================================================================
    if not skip_rss_download:
        console.print("\n" + "=" * 80)
        console.print("[bold magenta]STAGE 1/7: Download RSS Metadata[/bold magenta]")
        console.print("=" * 80)

        cmd = [
            "uv", "run", "python", str(script_dir / "download_rss_metadata.py"),
            "--output-dir", str(rss_metadata_dir),
            "--downloads-dir", str(audio_dir),
            "--update-existing",
        ]
        if force:
            cmd.append("--force")

        success, metadata = run_pipeline_stage(
            cmd=cmd,
            pipeline_name="Download RSS Metadata",
            inputs=[str(audio_dir)],
            output_paths=[str(rss_metadata_dir)],
            metadata_base_dir=metadata_base_dir,
            show_name=primary_show,
            dry_run=dry_run,
            extra_data={"downloads_dir": str(audio_dir)},
        )
        all_metadata.append(metadata)

        if success:
            stages_completed.append("1. Download RSS Metadata")
        else:
            stages_failed.append("1. Download RSS Metadata")
            console.print(
                "[bold yellow]RSS metadata download failed. "
                "Continuing without metadata...[/bold yellow]"
            )
    else:
        console.print("\n[yellow]Skipping Stage 1: Download RSS Metadata[/yellow]")
        stages_skipped.append("1. Download RSS Metadata")

    # =========================================================================
    # stage 2: transcription
    # =========================================================================
    if not skip_transcription:
        console.print("\n" + "=" * 80)
        console.print("[bold magenta]STAGE 2/7: Transcription (WhisperX)[/bold magenta]")
        console.print("=" * 80)

        cmd = [
            "uv", "run", "python", str(script_dir / "transcribe_batch.py"),
            "--audio-dir", str(audio_dir),
            "--output-dir", str(transcripts_dir),
            "--model", whisper_model,
            "--device", transcription_device,
            "--log-level", log_level,
        ]
        if language:
            cmd.extend(["--language", language])
        if force:
            cmd.append("--force")

        success, metadata = run_pipeline_stage(
            cmd=cmd,
            pipeline_name="Transcription",
            inputs=[str(audio_dir)],
            output_paths=[str(transcripts_dir)],
            metadata_base_dir=metadata_base_dir,
            show_name=primary_show,
            dry_run=dry_run,
            extra_data={
                "whisper_model": whisper_model,
                "language": language or "auto-detect",
                "device": transcription_device,
            },
        )
        all_metadata.append(metadata)

        if success:
            stages_completed.append("2. Transcription")
        else:
            stages_failed.append("2. Transcription")
            console.print("[bold red]Transcription failed. Stopping pipeline.[/bold red]")
            sys.exit(1)
    else:
        console.print("\n[yellow]Skipping Stage 2: Transcription[/yellow]")
        stages_skipped.append("2. Transcription")

    # =========================================================================
    # stage 3: diarization
    # =========================================================================
    if not skip_diarization:
        console.print("\n" + "=" * 80)
        console.print("[bold magenta]STAGE 3/7: Diarization (Pyannote)[/bold magenta]")
        console.print("=" * 80)

        # check for hf_token.
        if not hf_token and not dry_run:
            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                console.print(
                    "[bold red]Error: HF_TOKEN required for diarization. "
                    "Set HF_TOKEN env var or pass --hf-token.[/bold red]"
                )
                stages_failed.append("3. Diarization")
                sys.exit(1)

        cmd = [
            "uv", "run", "python", str(script_dir / "diarize_batch.py"),
            "--transcripts-dir", str(transcripts_dir),
            "--audio-base-dir", str(audio_dir),
            "--output-dir", str(diarizations_dir),
            "--device", diarization_device,
            "--log-level", log_level,
        ]
        if hf_token:
            cmd.extend(["--hf-token", hf_token])
        if diarization_workers:
            cmd.extend(["--workers", str(diarization_workers)])
        if torch_compile:
            cmd.append("--torch-compile")
        if min_speakers is not None:
            cmd.extend(["--min-speakers", str(min_speakers)])
        if max_speakers is not None:
            cmd.extend(["--max-speakers", str(max_speakers)])
        if force:
            cmd.append("--force")

        success, metadata = run_pipeline_stage(
            cmd=cmd,
            pipeline_name="Diarization",
            inputs=[str(audio_dir), str(transcripts_dir)],
            output_paths=[str(diarizations_dir)],
            metadata_base_dir=metadata_base_dir,
            show_name=primary_show,
            dry_run=dry_run,
            extra_data={
                "device": diarization_device,
                "torch_compile": torch_compile,
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
            },
        )
        all_metadata.append(metadata)

        if success:
            stages_completed.append("3. Diarization")
        else:
            stages_failed.append("3. Diarization")
            console.print("[bold red]Diarization failed. Stopping pipeline.[/bold red]")
            sys.exit(1)
    else:
        console.print("\n[yellow]Skipping Stage 3: Diarization[/yellow]")
        stages_skipped.append("3. Diarization")

    # =========================================================================
    # stage 4: combine transcripts with diarizations
    # =========================================================================
    console.print("\n" + "=" * 80)
    console.print("[bold magenta]STAGE 4/7: Combine Transcripts with Diarizations[/bold magenta]")
    console.print("=" * 80)

    cmd = [
        "uv", "run", "python", str(script_dir / "combine_transcripts_with_speakers.py"),
        "--transcripts-dir", str(transcripts_dir),
        "--diarizations-dir", str(diarizations_dir),
        "--output-dir", str(transcripts_with_diarization_dir),
        "--log-level", log_level,
    ]
    if force:
        cmd.append("--force")

    success, metadata = run_pipeline_stage(
        cmd=cmd,
        pipeline_name="Combine Transcripts with Diarizations",
        inputs=[str(transcripts_dir), str(diarizations_dir)],
        output_paths=[str(transcripts_with_diarization_dir)],
        metadata_base_dir=metadata_base_dir,
        show_name=primary_show,
        dry_run=dry_run,
    )
    all_metadata.append(metadata)

    if success:
        stages_completed.append("4. Combine Transcripts with Diarizations")
    else:
        stages_failed.append("4. Combine Transcripts with Diarizations")
        console.print(
            "[bold red]Combine transcripts with diarizations failed. "
            "Stopping pipeline.[/bold red]"
        )
        sys.exit(1)

    # =========================================================================
    # stage 5: combine consecutive speakers
    # =========================================================================
    console.print("\n" + "=" * 80)
    console.print("[bold magenta]STAGE 5/7: Combine Consecutive Speakers[/bold magenta]")
    console.print("=" * 80)

    cmd = [
        "uv", "run", "python", str(script_dir / "combine_consecutive_speakers.py"),
        "--transcripts-dir", str(transcripts_with_diarization_dir),
        "--output-dir", str(transcripts_postprocessed_dir),
        "--log-level", log_level,
    ]
    if force:
        cmd.append("--force")

    success, metadata = run_pipeline_stage(
        cmd=cmd,
        pipeline_name="Combine Consecutive Speakers",
        inputs=[str(transcripts_with_diarization_dir)],
        output_paths=[str(transcripts_postprocessed_dir)],
        metadata_base_dir=metadata_base_dir,
        show_name=primary_show,
        dry_run=dry_run,
    )
    all_metadata.append(metadata)

    if success:
        stages_completed.append("5. Combine Consecutive Speakers")
    else:
        stages_failed.append("5. Combine Consecutive Speakers")
        console.print(
            "[bold red]Combine consecutive speakers failed. "
            "Stopping pipeline.[/bold red]"
        )
        sys.exit(1)

    # =========================================================================
    # stage 6: speaker labeling
    # =========================================================================
    if not skip_speaker_labeling:
        console.print("\n" + "=" * 80)
        console.print("[bold magenta]STAGE 6/7: Speaker Labeling[/bold magenta]")
        console.print("=" * 80)

        # build command for speaker labeling pipeline.
        cmd = [
            "uv", "run", "python", str(script_dir / "run_speaker_labeling_pipeline.py"),
            "--audio-dir", str(audio_dir),
            "--output-dir", str(output_dir),
            "--skip-rss-download",  # already done in stage 1.
            "--skip-transcription",  # already done in stage 2.
            "--skip-diarization",  # already done in stage 3.
            "--log-level", log_level,
        ]
        if use_llm:
            cmd.append("--use-llm")
            cmd.extend(["--llm-model", llm_model])
        if use_local:
            cmd.append("--use-local")
            if local_model:
                cmd.extend(["--local-model", local_model])
        if load_4bit:
            cmd.append("--4bit")
        if load_8bit:
            cmd.append("--8bit")
        if force:
            cmd.append("--force")

        success, metadata = run_pipeline_stage(
            cmd=cmd,
            pipeline_name="Speaker Labeling",
            inputs=[str(transcripts_postprocessed_dir), str(rss_metadata_dir)],
            output_paths=[str(speaker_labeled_dir)],
            metadata_base_dir=metadata_base_dir,
            show_name=primary_show,
            dry_run=dry_run,
            extra_data={
                "use_llm": use_llm,
                "llm_model": llm_model if use_llm else None,
                "use_local": use_local,
                "local_model": local_model,
                "4bit": load_4bit,
                "8bit": load_8bit,
            },
        )
        all_metadata.append(metadata)

        if success:
            stages_completed.append("6. Speaker Labeling")
        else:
            stages_failed.append("6. Speaker Labeling")
            console.print("[bold yellow]Speaker labeling failed. Continuing...[/bold yellow]")
    else:
        console.print("\n[yellow]Skipping Stage 6: Speaker Labeling[/yellow]")
        stages_skipped.append("6. Speaker Labeling")

    # =========================================================================
    # stage 7: utterance classification
    # =========================================================================
    if not skip_classification:
        console.print("\n" + "=" * 80)
        console.print("[bold magenta]STAGE 7/7: Utterance Classification[/bold magenta]")
        console.print("=" * 80)

        # determine input directory (prefer speaker labeled if available).
        input_dir = speaker_labeled_dir if speaker_labeled_dir.exists() else transcripts_postprocessed_dir

        cmd = [
            "uv", "run", "python", str(script_dir / "classify_utterances.py"),
            "--transcripts-dir", str(input_dir),
            "--output-dir", str(classified_dir),
            "--min-matches", str(min_matches),
            "--log-level", log_level,
        ]
        if classifiers_config:
            cmd.extend(["--config", str(classifiers_config)])
        if keywords_config:
            cmd.extend(["--keywords-config", str(keywords_config)])
        if force:
            cmd.append("--force")

        success, metadata = run_pipeline_stage(
            cmd=cmd,
            pipeline_name="Utterance Classification",
            inputs=[str(input_dir)],
            output_paths=[str(classified_dir)],
            metadata_base_dir=metadata_base_dir,
            show_name=primary_show,
            dry_run=dry_run,
            extra_data={
                "classifiers_config": str(classifiers_config) if classifiers_config else "default",
                "keywords_config": str(keywords_config) if keywords_config else "default",
                "min_matches": min_matches,
            },
        )
        all_metadata.append(metadata)

        if success:
            stages_completed.append("7. Utterance Classification")
        else:
            stages_failed.append("7. Utterance Classification")
            console.print("[bold yellow]Utterance classification failed.[/bold yellow]")
    else:
        console.print("\n[yellow]Skipping Stage 7: Utterance Classification[/yellow]")
        stages_skipped.append("7. Utterance Classification")

    # =========================================================================
    # generate final summary
    # =========================================================================
    end_time = datetime.now()

    summary_text = generate_final_summary(
        outputs_dir=output_dir,
        audio_dir=audio_dir,
        start_time=start_time,
        end_time=end_time,
        stages_completed=stages_completed,
        stages_failed=stages_failed,
        stages_skipped=stages_skipped,
    )

    # save summary to file.
    summary_filename = f"full_pipeline_{end_time.strftime('%Y_%m_%d_%H_%M_%S')}.txt"
    summary_path = output_dir / summary_filename

    if not dry_run:
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_text)
        console.print(f"\n[bold green]Summary saved to: {summary_path}[/bold green]")

        # also save to metadata directory.
        metadata_summary_path = metadata_base_dir / primary_show / summary_filename
        metadata_summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_summary_path, "w", encoding="utf-8") as f:
            f.write(summary_text)
        console.print(f"[bold green]Metadata summary saved to: {metadata_summary_path}[/bold green]")
    else:
        console.print(
            f"\n[bold yellow][DRY RUN] Would save summary to: {summary_path}[/bold yellow]"
        )

    # print summary to console.
    console.print(summary_text)

    # print metadata files table.
    if all_metadata and not dry_run:
        console.print("\n")
        table = Table(title="Pipeline Metadata Files")
        table.add_column("Stage", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Duration", justify="right")
        table.add_column("Outputs", justify="right")

        for m in all_metadata:
            status = "[green]OK[/green]" if m.success else "[red]FAILED[/red]"
            table.add_row(
                m.pipeline_name,
                status,
                format_duration(m.duration_seconds),
                str(m.num_outputs_generated),
            )

        console.print(table)
        console.print(f"\n[dim]Metadata files written to: {metadata_base_dir / primary_show}[/dim]")

    # final status.
    if stages_failed:
        fail_count = len(stages_failed)
        msg = f"[bold yellow]Pipeline completed with {fail_count} failed stage(s)[/bold yellow]"
        console.print(Panel.fit(msg, border_style="yellow"))
        sys.exit(1)
    else:
        console.print(Panel.fit(
            "[bold green]Full Pipeline completed successfully![/bold green]",
            border_style="green",
        ))


if __name__ == "__main__":
    run_full_pipeline()
