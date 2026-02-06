#!/usr/bin/env python3
"""
automated pipeline for downloading and processing podcast episodes.

performs the following stages:
1. download - fetch episodes from rss feeds or by podcast name
2. transcribe - generate word-level transcripts using whisperx
3. diarize - identify speakers in audio files
4. combine - merge speaker labels with transcripts

generates a summary report upon completion.
"""

import os
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

import click
from rich.panel import Panel
from rich.table import Table

# add scripts to path for utils import.
if str(script_dir_setup) not in sys.path:
    sys.path.insert(0, str(script_dir_setup))

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


def get_audio_file_count(directory: Path) -> int:
    """count audio files in directory."""
    if not directory.exists():
        return 0
    audio_extensions = ["*.mp3", "*.wav", "*.m4a", "*.flac", "*.ogg", "*.aac"]
    count = 0
    for ext in audio_extensions:
        count += len(list(directory.rglob(ext)))
    return count


def generate_summary(
    downloads_dir: Path,
    outputs_dir: Path,
    start_time: datetime,
    end_time: datetime,
    stages_completed: list[str],
    stages_failed: list[str],
    stages_skipped: list[str],
    podcast_names: list[str],
) -> str:
    """generate a summary report of the pipeline run."""
    duration = end_time - start_time
    hours, remainder = divmod(int(duration.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)

    # count files.
    audio_count = get_audio_file_count(downloads_dir)
    transcripts_count = get_file_count(outputs_dir / "transcripts")
    diarizations_count = get_file_count(outputs_dir / "diarizations", "*.rttm")
    transcripts_with_diarization_count = get_file_count(outputs_dir / "transcripts_with_diarization_labels")

    summary = f"""
================================================================================
              PODCAST DOWNLOAD AND PROCESSING PIPELINE - SUMMARY
================================================================================

Run Date: {start_time.strftime('%Y-%m-%d %H:%M:%S')}
Duration: {hours}h {minutes}m {seconds}s

--------------------------------------------------------------------------------
                                 INPUT
--------------------------------------------------------------------------------
Podcasts Requested: {', '.join(podcast_names) if podcast_names else 'N/A'}
Downloads Directory: {downloads_dir}

--------------------------------------------------------------------------------
                              PIPELINE STAGES
--------------------------------------------------------------------------------
"""

    all_stages = [
        "1. Download Episodes",
        "2. Transcription",
        "3. Diarization",
        "4. Combine Transcripts with Speakers",
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
Audio Files Downloaded:             {audio_count} files
Transcripts Generated:              {transcripts_count} files
Diarization Files (RTTM):           {diarizations_count} files
Transcripts with Diarization:       {transcripts_with_diarization_count} files

--------------------------------------------------------------------------------
                              OUTPUT LOCATIONS
--------------------------------------------------------------------------------
Downloads:          {downloads_dir}
Transcripts:        {outputs_dir / 'transcripts'}
Diarizations:       {outputs_dir / 'diarizations'}
With Diarization:   {outputs_dir / 'transcripts_with_diarization_labels'}

================================================================================
                                  END REPORT
================================================================================
"""
    return summary


@click.command()
@click.option(
    "--podcasts",
    "-p",
    multiple=True,
    help="podcast names to download (can specify multiple times)",
)
@click.option(
    "--podcasts-file",
    "-f",
    type=click.Path(exists=True, path_type=Path),
    help="file with podcast names (one per line)",
)
@click.option(
    "--feed-url",
    "-u",
    type=str,
    help="direct rss feed url to download from",
)
@click.option(
    "--downloads-dir",
    type=click.Path(path_type=Path),
    default=Path("downloads"),
    help="directory to download audio files to",
)
@click.option(
    "--output-dir",
    "--outputs-dir",  # alias for backwards compatibility.
    type=click.Path(path_type=Path),
    default=Path("outputs"),
    help="base output directory for processing outputs",
)
# download options.
@click.option(
    "--max-episodes",
    "-n",
    type=int,
    default=None,
    help="maximum episodes to download per podcast",
)
@click.option(
    "--start-date",
    type=str,
    help="earliest episode date (YYYY-MM-DD)",
)
@click.option(
    "--end-date",
    type=str,
    help="latest episode date (YYYY-MM-DD)",
)
@click.option(
    "--random-sample",
    type=int,
    help="randomly sample n episodes per podcast",
)
@click.option(
    "--concurrent-downloads",
    type=int,
    default=4,
    help="number of concurrent downloads",
)
# transcription options.
@click.option(
    "--whisper-model",
    type=click.Choice([
        "tiny", "base", "small", "medium", "large-v2", "large-v3",
        "large-v3-turbo", "large-v3-8bit", "large-v3-4bit", "distil-large-v3",
    ]),
    default="large-v3",
    help="whisperx model size for transcription",
)
@click.option(
    "--language",
    type=str,
    default=None,
    help="language code for transcription (e.g., en, es). auto-detect if not specified",
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cuda", "mlx", "mps", "cpu"]),
    default="auto",
    help="device for transcription and diarization",
)
# diarization options.
@click.option(
    "--hf-token",
    type=str,
    default=None,
    help="huggingface api token (or set HF_TOKEN env var)",
)
@click.option(
    "--torch-compile/--no-torch-compile",
    default=False,
    help="enable torch.compile for diarization (slower first run, faster after)",
)
@click.option(
    "--diarization-workers",
    type=int,
    default=4,
    help="number of parallel workers for diarization",
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
# skip options.
@click.option(
    "--skip-download",
    is_flag=True,
    help="skip download stage (use existing audio files)",
)
@click.option(
    "--skip-transcription",
    is_flag=True,
    help="skip transcription stage (use existing transcripts)",
)
@click.option(
    "--skip-diarization",
    is_flag=True,
    help="skip diarization stage (use existing rttm files)",
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
    help="force reprocessing of all files",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="logging level",
)
def download_and_process(
    podcasts: tuple[str, ...],
    podcasts_file: Path | None,
    feed_url: str | None,
    downloads_dir: Path,
    output_dir: Path,
    max_episodes: int | None,
    start_date: str | None,
    end_date: str | None,
    random_sample: int | None,
    concurrent_downloads: int,
    whisper_model: str,
    language: str | None,
    device: str,
    hf_token: str | None,
    torch_compile: bool,
    diarization_workers: int,
    min_speakers: int | None,
    max_speakers: int | None,
    skip_download: bool,
    skip_transcription: bool,
    skip_diarization: bool,
    dry_run: bool,
    force: bool,
    log_level: str,
) -> None:
    """
    download and process podcast episodes through the full pipeline.

    this script automates:
    1. downloading episodes from rss feeds
    2. transcribing audio to text with word-level timestamps
    3. identifying speakers (diarization)
    4. combining transcripts with speaker labels

    \b
    examples:
        # download and process a single podcast
        uv run python scripts/download_and_process.py -p "The Daily" --max-episodes 5

        # download multiple podcasts
        uv run python scripts/download_and_process.py -p "The Daily" -p "The Ezra Klein Show"

        # download from a file list
        uv run python scripts/download_and_process.py -f podcasts.txt --max-episodes 10

        # download from direct rss feed
        uv run python scripts/download_and_process.py -u https://example.com/feed.xml

        # process existing downloads (skip download stage)
        uv run python scripts/download_and_process.py --skip-download --downloads-dir ./my_audio

        # full pipeline with date filtering
        uv run python scripts/download_and_process.py \\
            -p "The Daily" \\
            --start-date 2025-01-01 \\
            --end-date 2025-01-31 \\
            --hf-token $HF_TOKEN
    """
    start_time = datetime.now()

    console.print(Panel.fit(
        "[bold cyan]Podcast Download and Processing Pipeline[/bold cyan]\n\n"
        f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}",
        border_style="cyan",
    ))

    # display system info.
    print_system_info(console)

    # collect podcast names.
    podcast_names = list(podcasts)
    if podcasts_file:
        with open(podcasts_file) as f:
            podcast_names.extend(line.strip() for line in f if line.strip())

    # validate inputs.
    if not skip_download and not podcast_names and not feed_url:
        console.print(
            "[bold red]Error: Provide podcast names (-p), a file (-f), "
            "or a feed URL (-u), or use --skip-download[/bold red]"
        )
        sys.exit(1)

    # get hf token from environment if not provided.
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN")

    if not skip_diarization and not hf_token:
        console.print(
            "[bold yellow]Warning: No HuggingFace token provided. "
            "Diarization requires a token.[/bold yellow]"
        )
        console.print(
            "[yellow]Set HF_TOKEN env var or use --hf-token flag.[/yellow]"
        )

    # resolve paths.
    downloads_dir = downloads_dir.resolve()
    output_dir = output_dir.resolve()

    # define output directories.
    transcripts_dir = output_dir / "transcripts"
    diarizations_dir = output_dir / "diarizations"
    transcripts_with_diarization_dir = output_dir / "transcripts_with_diarization_labels"

    # ensure directories exist.
    downloads_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    stages_completed = []
    stages_failed = []
    stages_skipped = []

    # get script directory.
    script_dir = Path(__file__).resolve().parent

    # display configuration.
    console.print("\n[bold]Configuration:[/bold]")
    config_table = Table(show_header=False, box=None)
    config_table.add_column("Key", style="cyan")
    config_table.add_column("Value")

    if podcast_names:
        config_table.add_row("Podcasts", ", ".join(podcast_names[:3]) + ("..." if len(podcast_names) > 3 else ""))
    if feed_url:
        config_table.add_row("Feed URL", feed_url[:60] + "..." if len(feed_url) > 60 else feed_url)
    config_table.add_row("Downloads", str(downloads_dir))
    config_table.add_row("Outputs", str(output_dir))
    config_table.add_row("Whisper Model", whisper_model)
    config_table.add_row("Device", device)
    if max_episodes:
        config_table.add_row("Max Episodes", str(max_episodes))

    console.print(config_table)

    # stage 1: download.
    if not skip_download:
        console.print("\n" + "=" * 80)
        console.print("[bold magenta]STAGE 1/4: Download Episodes[/bold magenta]")
        console.print("=" * 80)

        cmd = [
            "uv", "run", "python", "-m", "podcast_downloader", "download",
            "--output-dir", str(downloads_dir),
            "--transcripts-dir", str(transcripts_dir),
            "--concurrent", str(concurrent_downloads),
        ]

        # add podcast names or feed url.
        if feed_url:
            cmd.extend(["--feed-url", feed_url])
            if podcast_names:
                cmd.append(podcast_names[0])  # use first name as identifier.
        else:
            cmd.extend(podcast_names)

        # add optional filters.
        if max_episodes:
            cmd.extend(["--max-episodes", str(max_episodes)])
        if start_date:
            cmd.extend(["--start-date", start_date])
        if end_date:
            cmd.extend(["--end-date", end_date])
        if random_sample:
            cmd.extend(["--random-sample", str(random_sample)])

        # skip existing unless force is set.
        if not force:
            cmd.append("--skip-existing")
        else:
            cmd.append("--overwrite")

        if run_command(cmd, "Download Episodes", dry_run):
            stages_completed.append("1. Download Episodes")
        else:
            stages_failed.append("1. Download Episodes")
            console.print("[bold red]Download failed. Stopping pipeline.[/bold red]")
            sys.exit(1)
    else:
        console.print("\n[yellow]Skipping Stage 1: Download Episodes[/yellow]")
        stages_skipped.append("1. Download Episodes")

    # stage 2: transcription.
    if not skip_transcription:
        console.print("\n" + "=" * 80)
        console.print("[bold magenta]STAGE 2/4: Transcription (WhisperX)[/bold magenta]")
        console.print("=" * 80)

        cmd = [
            "uv", "run", "python", str(script_dir / "transcribe_batch.py"),
            "--audio-dir", str(downloads_dir),
            "--output-dir", str(transcripts_dir),
            "--model", whisper_model,
            "--device", device,
        ]
        if language:
            cmd.extend(["--language", language])
        if force:
            cmd.append("--force")
        cmd.extend(["--log-level", log_level])

        if run_command(cmd, "Transcription (WhisperX)", dry_run):
            stages_completed.append("2. Transcription")
        else:
            stages_failed.append("2. Transcription")
            console.print("[bold red]Transcription failed. Stopping pipeline.[/bold red]")
            sys.exit(1)
    else:
        console.print("\n[yellow]Skipping Stage 2: Transcription[/yellow]")
        stages_skipped.append("2. Transcription")

    # stage 3: diarization.
    if not skip_diarization:
        console.print("\n" + "=" * 80)
        console.print("[bold magenta]STAGE 3/4: Speaker Diarization[/bold magenta]")
        console.print("=" * 80)

        if not hf_token:
            console.print(
                "[bold red]Error: Diarization requires a HuggingFace token.[/bold red]"
            )
            console.print("[red]Use --hf-token or set HF_TOKEN env var.[/red]")
            stages_failed.append("3. Diarization")
        else:
            cmd = [
                "uv", "run", "python", str(script_dir / "diarize_batch.py"),
                "--transcripts-dir", str(transcripts_dir),
                "--audio-base-dir", str(downloads_dir),
                "--output-dir", str(diarizations_dir),
                "--hf-token", hf_token,
                "--workers", str(diarization_workers),
                "--device", device,
            ]
            if torch_compile:
                cmd.append("--torch-compile")
            if min_speakers is not None:
                cmd.extend(["--min-speakers", str(min_speakers)])
            if max_speakers is not None:
                cmd.extend(["--max-speakers", str(max_speakers)])
            if force:
                cmd.append("--force")
            cmd.extend(["--log-level", log_level])

            if run_command(cmd, "Speaker Diarization", dry_run):
                stages_completed.append("3. Diarization")
            else:
                stages_failed.append("3. Diarization")
                console.print("[bold red]Diarization failed. Stopping pipeline.[/bold red]")
                sys.exit(1)
    else:
        console.print("\n[yellow]Skipping Stage 3: Diarization[/yellow]")
        stages_skipped.append("3. Diarization")

    # stage 4: combine transcripts with speakers.
    console.print("\n" + "=" * 80)
    console.print("[bold magenta]STAGE 4/4: Combine Transcripts with Speakers[/bold magenta]")
    console.print("=" * 80)

    cmd = [
        "uv", "run", "python", str(script_dir / "combine_transcripts_with_speakers.py"),
        "--transcripts-dir", str(transcripts_dir),
        "--diarizations-dir", str(diarizations_dir),
        "--output-dir", str(transcripts_with_diarization_dir),
    ]
    if force:
        cmd.append("--force")
    cmd.extend(["--log-level", log_level])

    if run_command(cmd, "Combine Transcripts with Speakers", dry_run):
        stages_completed.append("4. Combine Transcripts with Speakers")
    else:
        stages_failed.append("4. Combine Transcripts with Speakers")
        console.print(
            "[bold yellow]Combine transcripts failed. Check that transcripts "
            "and diarizations exist.[/bold yellow]"
        )

    # generate summary.
    end_time = datetime.now()

    summary = generate_summary(
        downloads_dir=downloads_dir,
        outputs_dir=output_dir,
        start_time=start_time,
        end_time=end_time,
        stages_completed=stages_completed,
        stages_failed=stages_failed,
        stages_skipped=stages_skipped,
        podcast_names=podcast_names,
    )

    # save summary to file.
    summary_filename = f"download_and_process_{end_time.strftime('%Y_%m_%d_%H_%M_%S')}.txt"
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
            "[bold green]Pipeline completed successfully![/bold green]\n\n"
            f"Transcripts with diarization labels available at:\n{transcripts_with_diarization_dir}",
            border_style="green",
        ))


if __name__ == "__main__":
    download_and_process()
