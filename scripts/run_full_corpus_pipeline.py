#!/usr/bin/env python3
"""
full pipeline script for podcast conversations analysis.

runs all 7 pipeline stages in sequence:
1. transcription - generate transcripts from audio files using whisperx
2. diarization - identify speakers in audio files
3. combine transcripts with speakers - merge speaker labels with transcripts
4. combine consecutive speakers - merge fragmented same-speaker segments
5. embeddings analysis - semantic similarity analysis
6. keyword analysis - extract keyword mentions with context
7. utterance classification - classify transcripts with keyword matches

generates a summary report upon completion.
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

# add src to path for imports.
script_dir_setup = Path(__file__).resolve().parent
project_root = script_dir_setup.parent
src_dir = project_root / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(script_dir_setup) not in sys.path:
    sys.path.insert(0, str(script_dir_setup))

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
    transcripts_count = get_file_count(outputs_dir / "transcripts")
    diarizations_count = get_file_count(outputs_dir / "diarizations", "*.rttm")
    transcripts_with_diarization_count = get_file_count(outputs_dir / "transcripts_with_diarization_labels")
    transcripts_postprocessed_count = get_file_count(outputs_dir / "transcripts_with_diarization_labels_postprocessed")
    embeddings_count = get_file_count(outputs_dir / "analysis" / "embeddings_method")
    keyword_analysis_count = get_file_count(outputs_dir / "analysis" / "keyword_analysis")
    classified_count = get_file_count(
        outputs_dir / "transcripts_with_speaker_labels_postprocessed_with_classification_labels"
    )

    summary = f"""
================================================================================
                    PODCAST CONVERSATIONS ANALYSIS - FULL PIPELINE
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
        "1. Transcription",
        "2. Diarization",
        "3. Combine Transcripts with Speakers",
        "4. Combine Consecutive Speakers",
        "5. Embeddings Analysis",
        "6. Keyword Analysis",
        "7. Utterance Classification",
    ]

    for stage in all_stages:
        if stage in stages_completed:
            summary += f"  ✓ {stage}\n"
        elif stage in stages_failed:
            summary += f"  ✗ {stage} (FAILED)\n"
        elif stage in stages_skipped:
            summary += f"  - {stage} (skipped)\n"
        else:
            summary += f"  ? {stage}\n"

    summary += f"""
--------------------------------------------------------------------------------
                              OUTPUT SUMMARY
--------------------------------------------------------------------------------
Transcripts:                            {transcripts_count} files
Diarizations (RTTM):                    {diarizations_count} files
Transcripts with Diarization:           {transcripts_with_diarization_count} files
Transcripts Postprocessed:              {transcripts_postprocessed_count} files
Embeddings Analysis Results:            {embeddings_count} files
Keyword Analysis Results:               {keyword_analysis_count} files
Classified Transcripts:                 {classified_count} files

--------------------------------------------------------------------------------
                              OUTPUT LOCATIONS
--------------------------------------------------------------------------------
Transcripts:            {outputs_dir / 'transcripts'}
Diarizations:           {outputs_dir / 'diarizations'}
With Diarization:       {outputs_dir / 'transcripts_with_diarization_labels'}
Postprocessed:          {outputs_dir / 'transcripts_with_diarization_labels_postprocessed'}
Embeddings:             {outputs_dir / 'analysis' / 'embeddings_method'}
Keyword Analysis:       {outputs_dir / 'analysis' / 'keyword_analysis'}
Classified:             {outputs_dir / 'transcripts_with_speaker_labels_postprocessed_with_classification_labels'}

================================================================================
                                  END REPORT
================================================================================
"""
    return summary


@click.command()
@click.option(
    "--audio-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="root directory containing audio files",
)
@click.option(
    "--output-dir",
    "--outputs-dir",  # alias for backwards compatibility.
    type=click.Path(path_type=Path),
    default=Path("outputs"),
    help="base output directory for all pipeline outputs",
)
@click.option(
    "--hf-token",
    type=str,
    default=None,
    help="HuggingFace API token (or set HF_TOKEN env var)",
)
# transcription options.
@click.option(
    "--whisper-model",
    type=click.Choice(["tiny", "base", "small", "medium", "large-v2", "large-v3"]),
    default="large-v3",
    help="whisperx model size for transcription",
)
@click.option(
    "--language",
    type=str,
    default=None,
    help="Language code for transcription (e.g., en, es, de). Auto-detect if not specified.",
)
# diarization options.
@click.option(
    "--torch-compile/--no-torch-compile",
    default=False,
    help="enable torch.compile for diarization (slower first run, faster after)",
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
@click.option(
    "--diarization-workers",
    type=int,
    default=4,
    help="number of parallel workers for diarization",
)
# analysis options.
@click.option(
    "--keywords-config",
    "--config",  # alias for backwards compatibility.
    type=click.Path(exists=True, path_type=Path),
    default=Path("config/keyword_analysis_config.yaml"),
    help="keyword analysis configuration file",
)
@click.option(
    "--classifiers-config",
    type=click.Path(exists=True, path_type=Path),
    default=Path("config/classifiers.yaml"),
    help="utterance classification configuration file",
)
@click.option(
    "--min-matches",
    type=int,
    default=1,
    help="minimum keyword matches for classification filtering",
)
@click.option(
    "--threshold",
    type=float,
    default=0.35,
    help="similarity threshold for embeddings analysis",
)
# skip options.
@click.option(
    "--skip-transcription",
    is_flag=True,
    help="skip transcription stage (use existing transcripts)",
)
@click.option(
    "--skip-diarization",
    is_flag=True,
    help="skip diarization stage (use existing RTTM files)",
)
@click.option(
    "--skip-classification",
    is_flag=True,
    help="skip classification stage",
)
@click.option(
    "--skip-embeddings",
    is_flag=True,
    help="skip embeddings analysis stage",
)
# general options.
@click.option(
    "--dry-run",
    is_flag=True,
    help="show what would be run without executing",
)
@click.option(
    "--force",
    is_flag=True,
    help="force reprocessing of all files (passed to all stages)",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="logging level (passed to all stages)",
)
def run_full_pipeline(
    audio_dir: Path,
    output_dir: Path,
    hf_token: str | None,
    whisper_model: str,
    language: str,
    torch_compile: bool,
    min_speakers: int | None,
    max_speakers: int | None,
    diarization_workers: int,
    keywords_config: Path,
    classifiers_config: Path,
    min_matches: int,
    threshold: float,
    skip_transcription: bool,
    skip_diarization: bool,
    skip_classification: bool,
    skip_embeddings: bool,
    dry_run: bool,
    force: bool,
    log_level: str,
) -> None:
    """
    run the complete podcast conversations analysis pipeline.

    executes all 7 stages in sequence:
    1. transcription (whisperx)
    2. diarization (pyannote)
    3. combine transcripts with speakers
    4. combine consecutive speakers
    5. embeddings analysis
    6. keyword analysis
    7. utterance classification

    Example:
        python scripts/run_full_pipeline.py \\
            --audio-dir /path/to/audio \\
            --hf-token YOUR_HF_TOKEN \\
            --whisper-model large-v3 \\
            --torch-compile
    """
    start_time = datetime.now()

    console.print(Panel.fit(
        "[bold cyan]Podcast Conversations Analysis - Full Pipeline[/bold cyan]\n\n"
        f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}",
        border_style="cyan",
    ))

    # display system info.
    print_system_info(console)

    # resolve paths.
    audio_dir = audio_dir.resolve()
    output_dir = output_dir.resolve()
    keywords_config = keywords_config.resolve()
    classifiers_config = classifiers_config.resolve()

    # define output directories.
    transcripts_dir = output_dir / "transcripts"
    diarizations_dir = output_dir / "diarizations"
    transcripts_with_diarization_dir = output_dir / "transcripts_with_diarization_labels"
    transcripts_postprocessed_dir = output_dir / "transcripts_with_diarization_labels_postprocessed"
    embeddings_dir = output_dir / "analysis" / "embeddings_method"
    keyword_analysis_dir = output_dir / "analysis" / "keyword_analysis"
    classified_dir = output_dir / "transcripts_with_speaker_labels_postprocessed_with_classification_labels"

    # ensure base output directory exists.
    output_dir.mkdir(parents=True, exist_ok=True)

    stages_completed = []
    stages_failed = []
    stages_skipped = []

    # get script directory.
    script_dir = Path(__file__).resolve().parent

    # stage 1: transcription.
    if not skip_transcription:
        console.print("\n" + "=" * 80)
        console.print("[bold magenta]STAGE 1/7: Transcription (WhisperX)[/bold magenta]")
        console.print("=" * 80)

        cmd = [
            "uv", "run", "python", str(script_dir / "transcribe_batch.py"),
            "--audio-dir", str(audio_dir),
            "--output-dir", str(transcripts_dir),
            "--model", whisper_model,
            "--language", language,
            "--log-level", log_level,
        ]
        if force:
            cmd.append("--force")

        if run_command(cmd, "Transcription (WhisperX)", dry_run):
            stages_completed.append("1. Transcription")
        else:
            stages_failed.append("1. Transcription")
            console.print("[bold red]Transcription failed. Stopping pipeline.[/bold red]")
            sys.exit(1)
    else:
        console.print("\n[yellow]Skipping Stage 1: Transcription[/yellow]")
        stages_skipped.append("1. Transcription")

    # stage 2: diarization.
    if not skip_diarization:
        console.print("\n" + "=" * 80)
        console.print("[bold magenta]STAGE 2/7: Diarization (Pyannote)[/bold magenta]")
        console.print("=" * 80)

        cmd = [
            "uv", "run", "python", str(script_dir / "diarize_batch.py"),
            "--transcripts-dir", str(transcripts_dir),
            "--audio-base-dir", str(audio_dir),
            "--output-dir", str(diarizations_dir),
            "--workers", str(diarization_workers),
            "--log-level", log_level,
        ]
        if hf_token:
            cmd.extend(["--hf-token", hf_token])
        if torch_compile:
            cmd.append("--torch-compile")
        if min_speakers is not None:
            cmd.extend(["--min-speakers", str(min_speakers)])
        if max_speakers is not None:
            cmd.extend(["--max-speakers", str(max_speakers)])
        if force:
            cmd.append("--force")

        if run_command(cmd, "Speaker Diarization", dry_run):
            stages_completed.append("2. Diarization")
        else:
            stages_failed.append("2. Diarization")
            console.print("[bold red]Diarization failed. Stopping pipeline.[/bold red]")
            sys.exit(1)
    else:
        console.print("\n[yellow]Skipping Stage 2: Diarization[/yellow]")
        stages_skipped.append("2. Diarization")

    # stage 3: combine transcripts with speakers.
    console.print("\n" + "=" * 80)
    console.print("[bold magenta]STAGE 3/7: Combine Transcripts with Speakers[/bold magenta]")
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

    if run_command(cmd, "Combine Transcripts with Speakers", dry_run):
        stages_completed.append("3. Combine Transcripts with Speakers")
    else:
        stages_failed.append("3. Combine Transcripts with Speakers")
        console.print("[bold red]Combine transcripts failed. Stopping pipeline.[/bold red]")
        sys.exit(1)

    # stage 4: combine consecutive speakers.
    console.print("\n" + "=" * 80)
    console.print("[bold magenta]STAGE 4/7: Combine Consecutive Speakers[/bold magenta]")
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
        stages_completed.append("4. Combine Consecutive Speakers")
    else:
        stages_failed.append("4. Combine Consecutive Speakers")
        console.print(
            "[bold red]Combine consecutive speakers failed. Stopping pipeline.[/bold red]"
        )
        sys.exit(1)

    # stage 5: embeddings analysis.
    if not skip_embeddings:
        console.print("\n" + "=" * 80)
        console.print("[bold magenta]STAGE 5/7: Embeddings Analysis[/bold magenta]")
        console.print("=" * 80)

        cmd = [
            "uv", "run", "python", str(script_dir / "analyze_embeddings.py"),
            "--process-all",
            "--threshold", str(threshold),
            "--transcripts-dir", str(transcripts_postprocessed_dir),
            "--output-dir", str(embeddings_dir),
            "--log-level", log_level,
        ]
        if force:
            cmd.append("--force")

        if run_command(cmd, "Embeddings Analysis", dry_run):
            stages_completed.append("5. Embeddings Analysis")
        else:
            stages_failed.append("5. Embeddings Analysis")
            console.print("[bold yellow]Embeddings analysis failed. Continuing...[/bold yellow]")
    else:
        console.print("\n[yellow]Skipping Stage 5: Embeddings Analysis[/yellow]")
        stages_skipped.append("5. Embeddings Analysis")

    # stage 6: keyword analysis.
    console.print("\n" + "=" * 80)
    console.print("[bold magenta]STAGE 6/7: Keyword Analysis[/bold magenta]")
    console.print("=" * 80)

    cmd = [
        "uv", "run", "python", str(script_dir / "analyze_keywords.py"),
        "--transcripts-dir", str(transcripts_postprocessed_dir),
        "--config", str(keywords_config),
        "--output-dir", str(keyword_analysis_dir),
        "--log-level", log_level,
    ]
    if force:
        cmd.append("--force")

    if run_command(cmd, "Keyword Analysis", dry_run):
        stages_completed.append("6. Keyword Analysis")
    else:
        stages_failed.append("6. Keyword Analysis")
        console.print("[bold red]Keyword analysis failed. Stopping pipeline.[/bold red]")
        sys.exit(1)

    # stage 7: utterance classification.
    if not skip_classification:
        console.print("\n" + "=" * 80)
        console.print("[bold magenta]STAGE 7/7: Utterance Classification[/bold magenta]")
        console.print("=" * 80)

        cmd = [
            "uv", "run", "python", str(script_dir / "classify_utterances.py"),
            "--transcripts-dir", str(transcripts_postprocessed_dir),
            "--output-dir", str(classified_dir),
            "--config", str(classifiers_config),
            "--keywords-dir", str(keyword_analysis_dir),
            "--min-matches", str(min_matches),
            "--log-level", log_level,
        ]
        if force:
            cmd.append("--force")

        if run_command(cmd, "Utterance Classification", dry_run):
            stages_completed.append("7. Utterance Classification")
        else:
            stages_failed.append("7. Utterance Classification")
            console.print(
                "[bold yellow]Classification failed.[/bold yellow]"
            )
    else:
        console.print("\n[yellow]Skipping Stage 7: Utterance Classification[/yellow]")
        stages_skipped.append("7. Utterance Classification")

    # generate summary.
    end_time = datetime.now()

    summary = generate_summary(
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
            f.write(summary)
        console.print(f"\n[bold green]Summary saved to: {summary_path}[/bold green]")
    else:
        console.print(
            f"\n[bold yellow][DRY RUN] Would save summary to: {summary_path}[/bold yellow]"
        )

    # print summary to console.
    console.print(summary)

    # final status.
    if stages_failed:
        fail_count = len(stages_failed)
        msg = f"[bold yellow]Pipeline completed with {fail_count} failed stage(s)[/bold yellow]"
        console.print(Panel.fit(msg, border_style="yellow"))
        sys.exit(1)
    else:
        console.print(Panel.fit(
            "[bold green]Pipeline completed successfully![/bold green]",
            border_style="green",
        ))


if __name__ == "__main__":
    run_full_pipeline()
