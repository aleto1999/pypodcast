#!/usr/bin/env python3
"""
analysis pipeline for podcast conversations.

runs all 4 analysis stages in sequence:
1. feature extraction - extract conversational features (questions, turn-taking, etc.)
2. keyword analysis - search for keywords defined in configuration
3. utterance classification - classify utterances using transformer models
4. llm annotation - annotate with LLM for detailed analysis

generates a summary report upon completion.

usage:
    # run full pipeline with default settings.
    uv run python scripts/run_analysis_pipeline.py

    # run with specific input directory.
    uv run python scripts/run_analysis_pipeline.py \\
        --input-dir outputs/transcripts_with_speaker_labels_postprocessed

    # skip feature extraction and keyword analysis.
    uv run python scripts/run_analysis_pipeline.py --skip-features --skip-keywords

    # run classification only on specific shows.
    uv run python scripts/run_analysis_pipeline.py --shows the_daily --shows pod_save_america

    # run with LLM annotation using MLX (Apple Silicon).
    uv run python scripts/run_analysis_pipeline.py --use-mlx
"""

from __future__ import annotations

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
    input_dir: Path,
    output_dir: Path,
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

    # count input files.
    input_count = get_file_count(input_dir)

    # count output files.
    features_count = get_file_count(output_dir / "features", "*.csv")
    keyword_analysis_count = get_file_count(output_dir / "keyword_analysis")
    classification_dir = (
        output_dir
        / "utterance_and_document_classification"
        / "transcripts_with_speaker_labels_postprocessed_with_classification_labels"
    )
    classification_count = get_file_count(classification_dir)
    llm_annotation_dir = (
        output_dir
        / "llm_annotation_method"
        / "transcripts_with_speaker_labels_postprocessed_with_classification_labels_and_llm_annotation"
    )
    llm_annotation_count = get_file_count(llm_annotation_dir)

    summary = f"""
================================================================================
                    ANALYSIS PIPELINE - FULL PIPELINE REPORT
================================================================================

Run Date: {start_time.strftime('%Y-%m-%d %H:%M:%S')}
Duration: {hours}h {minutes}m {seconds}s

--------------------------------------------------------------------------------
                                 INPUT
--------------------------------------------------------------------------------
Input Directory:                {input_dir}
Input Files:                    {input_count}

--------------------------------------------------------------------------------
                              PIPELINE STAGES
--------------------------------------------------------------------------------
"""

    all_stages = [
        "1. Feature Extraction",
        "2. Keyword Analysis",
        "3. Utterance Classification",
        "4. LLM Annotation",
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
Feature Extraction:             {features_count} CSV files
Keyword Analysis:               {keyword_analysis_count} files
Utterance Classification:       {classification_count} files
LLM Annotation:                 {llm_annotation_count} files

--------------------------------------------------------------------------------
                              OUTPUT LOCATIONS
--------------------------------------------------------------------------------
Features:           {output_dir / 'features'}
Keyword Analysis:   {output_dir / 'keyword_analysis'}
Classification:     {classification_dir}
LLM Annotation:     {llm_annotation_dir}

================================================================================
                                  END REPORT
================================================================================
"""
    return summary


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, path_type=Path),
    default=Path("outputs/transcripts_with_speaker_labels_postprocessed"),
    help="input directory containing transcript JSON files with speaker labels",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("outputs/analysis"),
    help="base output directory for all analysis outputs",
)
# feature extraction options.
@click.option(
    "--skip-features",
    is_flag=True,
    help="skip feature extraction stage",
)
@click.option(
    "--use-convokit",
    is_flag=True,
    help="use convokit for advanced politeness analysis (requires convokit + spacy)",
)
@click.option(
    "--features-workers",
    type=int,
    default=None,
    help="number of parallel workers for feature extraction (auto-detect if not set)",
)
# keyword analysis options.
@click.option(
    "--skip-keywords",
    is_flag=True,
    help="skip keyword analysis stage",
)
@click.option(
    "--keywords-config",
    type=click.Path(exists=True, path_type=Path),
    default=Path("config/keyword_analysis_config.yaml"),
    help="keyword analysis configuration file",
)
@click.option(
    "--keywords-format",
    type=click.Choice(["json", "csv", "txt", "all"]),
    default="json",
    help="output format for keyword analysis results",
)
@click.option(
    "--keywords-workers",
    type=int,
    default=None,
    help="number of parallel workers for keyword analysis (auto-detect if not set)",
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
    default=Path("config/classifiers.yaml"),
    help="classifiers configuration file",
)
@click.option(
    "--classification-device",
    type=click.Choice(["auto", "cuda", "mps", "mlx", "cpu"]),
    default="auto",
    help="device for classification inference",
)
@click.option(
    "--classification-batch-size",
    type=int,
    default=None,
    help="batch size for classification (auto-detect if not set)",
)
@click.option(
    "--min-keyword-matches",
    type=int,
    default=None,
    help="only classify transcripts with at least this many keyword matches",
)
@click.option(
    "--shows",
    multiple=True,
    help="process only specific shows (can be specified multiple times)",
)
# llm annotation options.
@click.option(
    "--skip-llm-annotation",
    is_flag=True,
    help="skip LLM annotation stage",
)
@click.option(
    "--llm-model",
    type=str,
    default="meta-llama/Llama-3.3-70B-Instruct",
    help="HuggingFace model for LLM annotation",
)
@click.option(
    "--use-mlx",
    is_flag=True,
    help="use MLX for LLM annotation (Apple Silicon only)",
)
@click.option(
    "--mlx-model",
    type=str,
    default="mlx-community/Llama-3.3-70B-Instruct-4bit",
    help="MLX model identifier for llm CLI (only used with --use-mlx)",
)
@click.option(
    "--llm-device",
    type=click.Choice(["auto", "cuda", "mps", "cpu"]),
    default="auto",
    help="device for LLM annotation",
)
@click.option(
    "--llm-batch-size",
    type=int,
    default=None,
    help="batch size for LLM annotation (auto-detect if not set)",
)
@click.option(
    "--load-in-4bit/--no-4bit",
    default=True,
    help="load LLM in 4-bit precision (recommended for 70B models)",
)
@click.option(
    "--load-in-8bit",
    is_flag=True,
    help="load LLM in 8-bit precision instead of 4-bit",
)
@click.option(
    "--hf-token",
    type=str,
    envvar="HF_TOKEN",
    default=None,
    help="HuggingFace API token for gated models (or set HF_TOKEN env var)",
)
@click.option(
    "--filter-segments/--no-filter-segments",
    default=True,
    help="only annotate segments with negative classification labels",
)
@click.option(
    "--min-negative-labels",
    type=int,
    default=2,
    help="minimum negative labels required for LLM annotation",
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
def run_analysis_pipeline(
    input_dir: Path,
    output_dir: Path,
    # feature extraction options.
    skip_features: bool,
    use_convokit: bool,
    features_workers: int | None,
    # keyword analysis options.
    skip_keywords: bool,
    keywords_config: Path,
    keywords_format: str,
    keywords_workers: int | None,
    # classification options.
    skip_classification: bool,
    classifiers_config: Path,
    classification_device: str,
    classification_batch_size: int | None,
    min_keyword_matches: int | None,
    shows: tuple[str, ...],
    # llm annotation options.
    skip_llm_annotation: bool,
    llm_model: str,
    use_mlx: bool,
    mlx_model: str,
    llm_device: str,
    llm_batch_size: int | None,
    load_in_4bit: bool,
    load_in_8bit: bool,
    hf_token: str | None,
    filter_segments: bool,
    min_negative_labels: int,
    # general options.
    force: bool,
    dry_run: bool,
    log_level: str,
    verbose: bool,
) -> None:
    """
    run the complete analysis pipeline for podcast conversations.

    executes all 4 analysis stages in sequence:
    1. feature extraction (questions, turn-taking, vocabulary)
    2. keyword analysis (search for configured keywords)
    3. utterance classification (transformer-based classifiers)
    4. llm annotation (detailed LLM-based analysis)

    \b
    examples:
        # run full pipeline with default settings.
        uv run python scripts/run_analysis_pipeline.py

        # run with custom input directory.
        uv run python scripts/run_analysis_pipeline.py \\
            --input-dir outputs/transcripts_with_speaker_labels_postprocessed

        # skip feature extraction (already done).
        uv run python scripts/run_analysis_pipeline.py --skip-features

        # run classification only on transcripts with keyword matches.
        uv run python scripts/run_analysis_pipeline.py --min-keyword-matches 5

        # run LLM annotation with MLX on Apple Silicon.
        uv run python scripts/run_analysis_pipeline.py --use-mlx

        # process specific shows only.
        uv run python scripts/run_analysis_pipeline.py --shows the_daily --shows latinus
    """
    start_time = datetime.now()

    # handle verbose flag.
    if verbose:
        log_level = "DEBUG"

    console.print(Panel.fit(
        "[bold cyan]Analysis Pipeline - Full Pipeline[/bold cyan]\n\n"
        f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}",
        border_style="cyan",
    ))

    # display system info.
    print_system_info(console)

    # resolve paths.
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    # define output directories.
    features_dir = output_dir / "features"
    keyword_analysis_dir = output_dir / "keyword_analysis"
    classification_dir = (
        output_dir
        / "utterance_and_document_classification"
        / "transcripts_with_speaker_labels_postprocessed_with_classification_labels"
    )
    llm_annotation_dir = (
        output_dir
        / "llm_annotation_method"
        / "transcripts_with_speaker_labels_postprocessed_with_classification_labels_and_llm_annotation"
    )

    # ensure base output directory exists.
    output_dir.mkdir(parents=True, exist_ok=True)

    stages_completed = []
    stages_failed = []
    stages_skipped = []

    # =========================================================================
    # stage 1: feature extraction
    # =========================================================================
    if not skip_features:
        console.print("\n" + "=" * 80)
        console.print("[bold magenta]STAGE 1/4: Feature Extraction[/bold magenta]")
        console.print("=" * 80)

        cmd = [
            "uv", "run", "python", str(script_dir / "extract_features.py"),
            "--transcripts-dir", str(input_dir),
            "--output-dir", str(features_dir),
            "--log-level", log_level,
        ]
        if use_convokit:
            cmd.append("--use-convokit")
        if features_workers:
            cmd.extend(["--workers", str(features_workers)])
        if force:
            cmd.append("--force")

        if run_command(cmd, "Feature Extraction", dry_run):
            stages_completed.append("1. Feature Extraction")
        else:
            stages_failed.append("1. Feature Extraction")
            console.print(
                "[bold yellow]Feature extraction failed. Continuing...[/bold yellow]"
            )
    else:
        console.print("\n[yellow]Skipping Stage 1: Feature Extraction[/yellow]")
        stages_skipped.append("1. Feature Extraction")

    # =========================================================================
    # stage 2: keyword analysis
    # =========================================================================
    if not skip_keywords:
        console.print("\n" + "=" * 80)
        console.print("[bold magenta]STAGE 2/4: Keyword Analysis[/bold magenta]")
        console.print("=" * 80)

        cmd = [
            "uv", "run", "python", str(script_dir / "analyze_keywords.py"),
            "--transcripts-dir", str(input_dir),
            "--config", str(keywords_config),
            "--output-dir", str(keyword_analysis_dir),
            "--format", keywords_format,
            "--log-level", log_level,
        ]
        if keywords_workers:
            cmd.extend(["--workers", str(keywords_workers)])
        if force:
            cmd.append("--force")

        if run_command(cmd, "Keyword Analysis", dry_run):
            stages_completed.append("2. Keyword Analysis")
        else:
            stages_failed.append("2. Keyword Analysis")
            console.print(
                "[bold yellow]Keyword analysis failed. Continuing...[/bold yellow]"
            )
    else:
        console.print("\n[yellow]Skipping Stage 2: Keyword Analysis[/yellow]")
        stages_skipped.append("2. Keyword Analysis")

    # =========================================================================
    # stage 3: utterance classification
    # =========================================================================
    if not skip_classification:
        console.print("\n" + "=" * 80)
        console.print("[bold magenta]STAGE 3/4: Utterance Classification[/bold magenta]")
        console.print("=" * 80)

        cmd = [
            "uv", "run", "python", str(script_dir / "classify_utterances.py"),
            "--transcripts-dir", str(input_dir),
            "--output-dir", str(classification_dir),
            "--config", str(classifiers_config),
            "--device", classification_device,
            "--log-level", log_level,
        ]
        if classification_batch_size:
            cmd.extend(["--batch-size", str(classification_batch_size)])
        if hf_token:
            cmd.extend(["--hf-token", hf_token])
        if min_keyword_matches is not None:
            cmd.extend([
                "--keywords-dir", str(keyword_analysis_dir),
                "--min-matches", str(min_keyword_matches),
            ])
        for show in shows:
            cmd.extend(["--shows", show])
        if force:
            cmd.append("--force")

        if run_command(cmd, "Utterance Classification", dry_run):
            stages_completed.append("3. Utterance Classification")
        else:
            stages_failed.append("3. Utterance Classification")
            console.print(
                "[bold yellow]Utterance classification failed. Continuing...[/bold yellow]"
            )
    else:
        console.print("\n[yellow]Skipping Stage 3: Utterance Classification[/yellow]")
        stages_skipped.append("3. Utterance Classification")

    # =========================================================================
    # stage 4: llm annotation
    # =========================================================================
    if not skip_llm_annotation:
        console.print("\n" + "=" * 80)
        console.print("[bold magenta]STAGE 4/4: LLM Annotation[/bold magenta]")
        console.print("=" * 80)

        # use classified output as input if classification was run.
        llm_input_dir = classification_dir if classification_dir.exists() else input_dir

        cmd = [
            "uv", "run", "python", str(script_dir / "annotate_with_llm.py"),
            "--transcripts-dir", str(llm_input_dir),
            "--output-dir", str(llm_annotation_dir),
            "--device", llm_device,
            "--log-level", log_level,
        ]

        if use_mlx:
            cmd.append("--use-mlx")
            cmd.extend(["--mlx-model", mlx_model])
        else:
            cmd.extend(["--model", llm_model])
            if load_in_8bit:
                cmd.append("--load-in-8bit")
            elif load_in_4bit:
                cmd.append("--load-in-4bit")
            else:
                cmd.append("--no-4bit")

        if llm_batch_size:
            cmd.extend(["--batch-size", str(llm_batch_size)])
        if hf_token:
            cmd.extend(["--hf-token", hf_token])
        if filter_segments:
            cmd.append("--filter-segments")
            cmd.extend(["--min-negative-labels", str(min_negative_labels)])
        else:
            cmd.append("--no-filter-segments")
        if force:
            cmd.append("--force")
        # auto-select batch size to avoid interactive prompt.
        cmd.append("--auto-batch")

        if run_command(cmd, "LLM Annotation", dry_run):
            stages_completed.append("4. LLM Annotation")
        else:
            stages_failed.append("4. LLM Annotation")
            console.print(
                "[bold yellow]LLM annotation failed.[/bold yellow]"
            )
    else:
        console.print("\n[yellow]Skipping Stage 4: LLM Annotation[/yellow]")
        stages_skipped.append("4. LLM Annotation")

    # =========================================================================
    # generate final summary
    # =========================================================================
    end_time = datetime.now()

    summary_text = generate_summary(
        input_dir=input_dir,
        output_dir=output_dir,
        start_time=start_time,
        end_time=end_time,
        stages_completed=stages_completed,
        stages_failed=stages_failed,
        stages_skipped=stages_skipped,
    )

    # save summary to file.
    summary_filename = f"analysis_pipeline_{end_time.strftime('%Y_%m_%d_%H_%M_%S')}.txt"
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
            "[bold green]Analysis Pipeline completed successfully![/bold green]",
            border_style="green",
        ))


if __name__ == "__main__":
    run_analysis_pipeline()
