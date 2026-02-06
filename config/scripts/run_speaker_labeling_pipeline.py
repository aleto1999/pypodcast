#!/usr/bin/env python3
"""
full speaker labeling pipeline for podcast conversations.

runs all 6 pipeline stages in sequence:
1. rss metadata download - fetch podcast metadata for speaker identification
2. transcription - generate transcripts from audio files using whisperx
3. diarization - identify speakers in audio files
4. combine transcripts with diarization - merge speaker labels with transcripts
5. combine consecutive speakers - merge fragmented same-speaker segments
6. speaker labeling - label speakers using rss metadata and heuristics/llm

generates a summary report upon completion.

usage:
    # run full pipeline with default settings.
    uv run python scripts/run_speaker_labeling_pipeline.py --audio-dir outputs/downloads

    # run with LLM-based speaker classification.
    uv run python scripts/run_speaker_labeling_pipeline.py --audio-dir outputs/downloads --use-llm

    # run with local CUDA model for speaker classification.
    uv run python scripts/run_speaker_labeling_pipeline.py --audio-dir outputs/downloads --use-local

    # skip transcription and diarization (use existing files).
    uv run python scripts/run_speaker_labeling_pipeline.py --audio-dir outputs/downloads \\
        --skip-transcription --skip-diarization
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# add src to path for imports.
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
src_dir = project_root / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

import click
from rich.panel import Panel

from podcast_conversations.monitoring import print_system_info
from utils.rich_utils import console


def run_command(cmd: list[str], description: str, dry_run: bool = False) -> bool:
    """run a command and return success status."""
    prefix = "[DRY RUN] " if dry_run else ""
    console.print(f"\n[bold blue]{prefix}Running: {description}[/bold blue]")
    console.print(f"[dim]{' '.join(cmd)}[/dim]\n")

    if dry_run:
        return True

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]Error running {description}: {e}[/bold red]")
        return False
    except FileNotFoundError as e:
        console.print(f"[bold red]Command not found: {e}[/bold red]")
        return False


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


def generate_summary(
    outputs_dir: Path,
    audio_dir: Path,
    start_time: datetime,
    end_time: datetime,
    stages_completed: list[str],
    stages_failed: list[str],
    stages_skipped: list[str],
) -> str:
    """generate a summary report of the pipeline run."""
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

    summary = f"""
================================================================================
               SPEAKER LABELING PIPELINE - FULL PIPELINE REPORT
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
        "1. RSS Metadata Download",
        "2. Transcription",
        "3. Diarization",
        "4. Combine Transcripts with Diarization",
        "5. Combine Consecutive Speakers",
        "6. Speaker Labeling",
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
RSS Metadata:                           {rss_metadata_count} podcasts
Transcripts:                            {transcripts_count} files
Diarizations (RTTM):                    {diarizations_count} files
Transcripts with Diarization:           {transcripts_with_diarization_count} files
Transcripts Postprocessed:              {transcripts_postprocessed_count} files
Speaker Labeled Transcripts:            {speaker_labeled_count} files

--------------------------------------------------------------------------------
                              OUTPUT LOCATIONS
--------------------------------------------------------------------------------
RSS Metadata:           {outputs_dir / 'rss_metadata'}
Transcripts:            {outputs_dir / 'transcripts'}
Diarizations:           {outputs_dir / 'diarizations'}
With Diarization:       {outputs_dir / 'transcripts_with_diarization_labels'}
Postprocessed:          {outputs_dir / 'transcripts_with_diarization_labels_postprocessed'}
Speaker Labeled:        {outputs_dir / 'transcripts_with_speaker_labels_postprocessed'}

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
@click.option(
    "--rss-force",
    is_flag=True,
    help="force re-download of existing rss metadata",
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
@click.option(
    "--transcription-batch-size",
    type=int,
    default=None,
    help="batch size for transcription (auto-calculated if not specified)",
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
    "--use-bf16/--no-bf16",
    default=False,
    help="enable BFloat16 mixed precision for diarization (CUDA H100/A100)",
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
def run_speaker_labeling_pipeline(
    audio_dir: Path,
    output_dir: Path,
    # rss options.
    skip_rss_download: bool,
    rss_force: bool,
    # transcription options.
    skip_transcription: bool,
    whisper_model: str,
    language: str | None,
    transcription_device: str,
    transcription_batch_size: int | None,
    # diarization options.
    skip_diarization: bool,
    hf_token: str | None,
    diarization_device: str,
    diarization_workers: int | None,
    use_bf16: bool,
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
    # general options.
    force: bool,
    dry_run: bool,
    log_level: str,
    verbose: bool,
) -> None:
    """
    run the complete speaker labeling pipeline for podcast conversations.

    executes all 6 stages in sequence:
    1. rss metadata download
    2. transcription (whisperx)
    3. diarization (pyannote)
    4. combine transcripts with diarization labels
    5. combine consecutive speakers
    6. speaker labeling

    \b
    examples:
        # run full pipeline with default settings.
        uv run python scripts/run_speaker_labeling_pipeline.py --audio-dir outputs/downloads

        # run with LLM-based speaker classification.
        uv run python scripts/run_speaker_labeling_pipeline.py --audio-dir outputs/downloads --use-llm

        # skip transcription and diarization (use existing files).
        uv run python scripts/run_speaker_labeling_pipeline.py --audio-dir outputs/downloads \\
            --skip-transcription --skip-diarization

        # run on specific language with local GPU model.
        uv run python scripts/run_speaker_labeling_pipeline.py --audio-dir outputs/downloads \\
            --language es --use-local --4bit
    """
    start_time = datetime.now()

    # handle verbose flag.
    if verbose:
        log_level = "DEBUG"

    console.print(Panel.fit(
        "[bold cyan]Speaker Labeling Pipeline - Full Pipeline[/bold cyan]\n\n"
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

    # ensure base output directory exists.
    output_dir.mkdir(parents=True, exist_ok=True)

    stages_completed = []
    stages_failed = []
    stages_skipped = []

    # =========================================================================
    # stage 1: rss metadata download
    # =========================================================================
    if not skip_rss_download:
        console.print("\n" + "=" * 80)
        console.print("[bold magenta]STAGE 1/6: RSS Metadata Download[/bold magenta]")
        console.print("=" * 80)

        cmd = [
            "uv", "run", "python", str(script_dir / "download_rss_metadata.py"),
            "--output-dir", str(rss_metadata_dir),
            "--downloads-dir", str(audio_dir),
            "--update-existing",
        ]
        if rss_force or force:
            cmd.append("--force")

        if run_command(cmd, "RSS Metadata Download", dry_run):
            stages_completed.append("1. RSS Metadata Download")
        else:
            stages_failed.append("1. RSS Metadata Download")
            console.print(
                "[bold yellow]RSS metadata download failed. "
                "Continuing without metadata...[/bold yellow]"
            )
    else:
        console.print("\n[yellow]Skipping Stage 1: RSS Metadata Download[/yellow]")
        stages_skipped.append("1. RSS Metadata Download")

    # =========================================================================
    # stage 2: transcription
    # =========================================================================
    if not skip_transcription:
        console.print("\n" + "=" * 80)
        console.print("[bold magenta]STAGE 2/6: Transcription (WhisperX)[/bold magenta]")
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
        if transcription_batch_size:
            cmd.extend(["--batch-size", str(transcription_batch_size)])
        if force:
            cmd.append("--force")

        if run_command(cmd, "Transcription (WhisperX)", dry_run):
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
        console.print("[bold magenta]STAGE 3/6: Diarization (Pyannote)[/bold magenta]")
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
                console.print("[bold red]Diarization failed. Stopping pipeline.[/bold red]")
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
        if use_bf16:
            cmd.append("--use-bf16")
        if torch_compile:
            cmd.append("--torch-compile")
        if min_speakers is not None:
            cmd.extend(["--min-speakers", str(min_speakers)])
        if max_speakers is not None:
            cmd.extend(["--max-speakers", str(max_speakers)])
        if force:
            cmd.append("--force")

        if run_command(cmd, "Speaker Diarization (Pyannote)", dry_run):
            stages_completed.append("3. Diarization")
        else:
            stages_failed.append("3. Diarization")
            console.print("[bold red]Diarization failed. Stopping pipeline.[/bold red]")
            sys.exit(1)
    else:
        console.print("\n[yellow]Skipping Stage 3: Diarization[/yellow]")
        stages_skipped.append("3. Diarization")

    # =========================================================================
    # stage 4: combine transcripts with diarization labels
    # =========================================================================
    console.print("\n" + "=" * 80)
    console.print("[bold magenta]STAGE 4/6: Combine Transcripts with Diarization[/bold magenta]")
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

    if run_command(cmd, "Combine Transcripts with Diarization", dry_run):
        stages_completed.append("4. Combine Transcripts with Diarization")
    else:
        stages_failed.append("4. Combine Transcripts with Diarization")
        console.print(
            "[bold red]Combine transcripts with diarization failed. "
            "Stopping pipeline.[/bold red]"
        )
        sys.exit(1)

    # =========================================================================
    # stage 5: combine consecutive speakers
    # =========================================================================
    console.print("\n" + "=" * 80)
    console.print("[bold magenta]STAGE 5/6: Combine Consecutive Speakers[/bold magenta]")
    console.print("=" * 80)

    cmd = [
        "uv", "run", "python", str(script_dir / "combine_consecutive_speakers.py"),
        "--transcripts-dir", str(transcripts_with_diarization_dir),
        "--output-dir", str(transcripts_postprocessed_dir),
        "--log-level", log_level,
    ]
    if force:
        cmd.append("--force")

    if run_command(cmd, "Combine Consecutive Speakers", dry_run):
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
        console.print("[bold magenta]STAGE 6/6: Speaker Labeling[/bold magenta]")
        console.print("=" * 80)

        # import and run the speaker labeling pipeline directly.
        # this avoids subprocess overhead and allows better error handling.
        try:
            from podcast_conversations.speaker_labeling.pipeline import create_pipeline
            from podcast_conversations.speaker_labeling.summary_generator import (
                create_summary_generator
            )

            if not dry_run:
                # check rss metadata availability.
                rss_dir = rss_metadata_dir if rss_metadata_dir.exists() else None

                console.print("[bold blue]Creating speaker labeling pipeline...[/bold blue]")

                pipeline = create_pipeline(
                    input_dir=transcripts_postprocessed_dir,
                    output_dir=speaker_labeled_dir,
                    rss_dir=rss_dir,
                    use_llm=use_llm,
                    llm_model=llm_model,
                    use_local=use_local,
                    local_model=local_model,
                    load_in_4bit=load_4bit,
                    load_in_8bit=load_8bit,
                )

                console.print("[bold blue]Running speaker labeling pipeline...[/bold blue]")
                summary = pipeline.run()

                # generate summaries.
                console.print("[bold blue]Generating summary reports...[/bold blue]")
                summary_generator = create_summary_generator(speaker_labeled_dir)
                general_path, _ = summary_generator.generate_all(summary)

                # print speaker labeling summary.
                console.print(f"\n[green]Podcasts processed: {summary.total_podcasts}[/green]")
                console.print(f"[green]Episodes processed: {summary.total_episodes_processed}[/green]")
                console.print(f"[yellow]Episodes failed: {summary.total_episodes_failed}[/yellow]")
                console.print(f"[green]Speakers labeled: {summary.total_speakers_labeled}[/green]")
                console.print(f"[green]Average confidence: {summary.average_confidence:.2f}[/green]")
                console.print(f"[dim]General summary: {general_path}[/dim]")

                stages_completed.append("6. Speaker Labeling")
            else:
                console.print(
                    "[bold yellow][DRY RUN] Would run speaker labeling pipeline[/bold yellow]"
                )
                console.print(f"  Input: {transcripts_postprocessed_dir}")
                console.print(f"  Output: {speaker_labeled_dir}")
                console.print(f"  RSS metadata: {rss_metadata_dir}")
                console.print(f"  Use LLM: {use_llm}")
                console.print(f"  Use Local: {use_local}")
                stages_completed.append("6. Speaker Labeling")

        except Exception as e:
            console.print(f"[bold red]Speaker labeling failed: {e}[/bold red]")
            stages_failed.append("6. Speaker Labeling")
    else:
        console.print("\n[yellow]Skipping Stage 6: Speaker Labeling[/yellow]")
        stages_skipped.append("6. Speaker Labeling")

    # =========================================================================
    # generate final summary
    # =========================================================================
    end_time = datetime.now()

    summary_text = generate_summary(
        outputs_dir=output_dir,
        audio_dir=audio_dir,
        start_time=start_time,
        end_time=end_time,
        stages_completed=stages_completed,
        stages_failed=stages_failed,
        stages_skipped=stages_skipped,
    )

    # save summary to file.
    summary_filename = f"speaker_labeling_pipeline_{end_time.strftime('%Y_%m_%d_%H_%M_%S')}.txt"
    summary_path = output_dir / summary_filename

    if not dry_run:
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_text)
        console.print(f"\n[bold green]Summary saved to: {summary_path}[/bold green]")
    else:
        console.print(
            f"\n[bold yellow][DRY RUN] Would save summary to: {summary_path}[/bold yellow]"
        )

    # print summary to console.
    console.print(summary_text)

    # final status.
    if stages_failed:
        fail_count = len(stages_failed)
        msg = f"[bold yellow]Pipeline completed with {fail_count} failed stage(s)[/bold yellow]"
        console.print(Panel.fit(msg, border_style="yellow"))
        sys.exit(1)
    else:
        console.print(Panel.fit(
            "[bold green]Speaker Labeling Pipeline completed successfully![/bold green]",
            border_style="green",
        ))


if __name__ == "__main__":
    run_speaker_labeling_pipeline()
