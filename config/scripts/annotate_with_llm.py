#!/usr/bin/env python3
"""LLM annotation CLI for podcast transcripts using local GPU inference."""

import logging
import sys
from datetime import datetime
from pathlib import Path

# add src to path for imports.
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
src_dir = project_root / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import click

# add scripts to path for utils import.
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from podcast_conversations.llm_annotation import (
    AnnotationConfig,
    BatchConfig,
    DeviceType,
    FileTask,
    LLMAnnotator,
    ParallelAnnotationPipeline,
    calculate_optimal_batch_config,
    check_annotations_exist,
    detect_device,
    detect_optimal_io_workers,
    detect_optimal_prefetch,
    is_mlx_available,
)
from podcast_conversations.monitoring import ResourceDisplay, print_system_info
from utils.rich_utils import console, setup_logging


def display_batch_options_and_select(
    batch_config: BatchConfig,
    auto_select: bool = False,
) -> int:
    """Display batch configuration options and let user select.

    Shows the system-recommended batch size along with alternatives,
    explains the trade-offs, and allows the user to choose.

    Args:
        batch_config: Calculated batch configuration with recommendations.
        auto_select: If True, automatically use recommended without prompting.

    Returns:
        Selected batch size.
    """
    from rich.table import Table

    # display resource summary.
    console.print("\n[bold cyan]Batch Configuration[/bold cyan]")
    console.print(f"[dim]Total segments to process: {batch_config.total_segments:,}[/dim]")
    console.print(f"[dim]GPU memory: {batch_config.gpu_memory_gb:.1f}GB total, "
                  f"{batch_config.available_memory_gb:.1f}GB available[/dim]")

    # get batch options.
    options = batch_config.get_batch_options()

    # create options table.
    table = Table(title="Batch Size Options", show_header=True, header_style="bold")
    table.add_column("#", style="dim", width=3)
    table.add_column("Mode", style="cyan")
    table.add_column("Batch Size", justify="right")
    table.add_column("Total Batches", justify="right")
    table.add_column("Description")

    for i, opt in enumerate(options, 1):
        mode_label = opt["label"]
        if opt["size"] == batch_config.recommended_batch_size:
            mode_label = f"[green]{mode_label} (recommended)[/green]"

        table.add_row(
            str(i),
            mode_label,
            str(opt["size"]),
            f"{opt['batches']:,}",
            opt["description"],
        )

    # add custom option.
    table.add_row(
        str(len(options) + 1),
        "[yellow]Custom[/yellow]",
        "?",
        "?",
        "Enter your own batch size",
    )

    console.print(table)

    # auto-select if requested.
    if auto_select:
        console.print(f"\n[green]Auto-selected recommended batch size: "
                      f"{batch_config.recommended_batch_size}[/green]")
        return batch_config.recommended_batch_size

    # prompt user for selection.
    console.print(f"\n[dim]Press Enter to use recommended ({batch_config.recommended_batch_size}), "
                  f"or enter option number:[/dim]")

    try:
        user_input = input().strip()

        if not user_input:
            # use recommended.
            console.print(f"[green]Using recommended batch size: "
                          f"{batch_config.recommended_batch_size}[/green]")
            return batch_config.recommended_batch_size

        choice = int(user_input)

        if 1 <= choice <= len(options):
            selected = options[choice - 1]
            console.print(f"[green]Selected {selected['label']}: "
                          f"batch size {selected['size']}[/green]")
            return selected["size"]

        elif choice == len(options) + 1:
            # custom input - no upper limit, scales with available resources.
            console.print("[dim]Enter custom batch size (minimum 1):[/dim]")
            custom_size = int(input().strip())
            custom_size = max(1, custom_size)
            console.print(f"[green]Using custom batch size: {custom_size}[/green]")
            return custom_size

        else:
            console.print("[yellow]Invalid choice, using recommended[/yellow]")
            return batch_config.recommended_batch_size

    except (ValueError, EOFError, KeyboardInterrupt):
        console.print(f"[yellow]Using recommended batch size: "
                      f"{batch_config.recommended_batch_size}[/yellow]")
        return batch_config.recommended_batch_size




def generate_summary(
    transcripts_dir: Path,
    output_dir: Path,
    model: str,
    start_time: datetime,
    end_time: datetime,
    files_processed: int,
    files_skipped: int,
    files_failed: int,
    total_segments: int,
    batch_size: int,
    precision: str,
    device: str,
    platform: str = "CUDA",
    throughput_metrics: dict | None = None,
) -> str:
    """Generate a summary report of the LLM annotation run."""
    duration = end_time - start_time
    hours, remainder = divmod(int(duration.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)

    # count output files.
    output_count = len(list(output_dir.rglob("*.json"))) if output_dir.exists() else 0

    # throughput section.
    throughput_section = ""
    if throughput_metrics:
        throughput_section = f"""
--------------------------------------------------------------------------------
                              THROUGHPUT METRICS
--------------------------------------------------------------------------------
Segments/Second:        {throughput_metrics.get('segments_per_second', 0):.2f}
Files/Minute:           {throughput_metrics.get('files_per_minute', 0):.2f}
Avg Batch Time:         {throughput_metrics.get('avg_batch_time_ms', 0):.1f}ms
"""

    summary = f"""
================================================================================
                    LLM ANNOTATION PIPELINE - SUMMARY REPORT
================================================================================

Run Date: {start_time.strftime('%Y-%m-%d %H:%M:%S')}
Duration: {hours}h {minutes}m {seconds}s

--------------------------------------------------------------------------------
                                 CONFIGURATION
--------------------------------------------------------------------------------
Model:              {model}
Device:             {device}
Platform:           {platform}
Precision:          {precision}
Batch Size:         {batch_size}

--------------------------------------------------------------------------------
                                    INPUT
--------------------------------------------------------------------------------
Transcripts Directory:  {transcripts_dir}
Input Files Found:      {files_processed + files_skipped + files_failed}

--------------------------------------------------------------------------------
                              PROCESSING RESULTS
--------------------------------------------------------------------------------
Files Processed:        {files_processed}
Files Skipped:          {files_skipped} (already annotated)
Files Failed:           {files_failed}
Total Segments:         {total_segments:,}
{throughput_section}
--------------------------------------------------------------------------------
                                   OUTPUT
--------------------------------------------------------------------------------
Output Directory:       {output_dir}
Output Files:           {output_count}

================================================================================
                                  END REPORT
================================================================================
"""
    return summary


@click.command()
@click.option(
    "--transcripts-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing transcript JSON files",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("outputs/analysis/llm_annotation_method"),
    help="Output directory for annotated transcripts",
)
@click.option(
    "--model",
    type=str,
    default="meta-llama/Llama-3.3-70B-Instruct",
    help="Model name/identifier from HuggingFace (or MLX model if --use-mlx is set)",
)
@click.option(
    "--use-mlx",
    is_flag=True,
    default=False,
    help="Use MLX for local inference via llm CLI (Apple Silicon only)",
)
@click.option(
    "--mlx-model",
    type=str,
    default="mlx-community/Llama-3.3-70B-Instruct-4bit",
    help="MLX model identifier for llm CLI (only used with --use-mlx)",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Batch size for inference (auto-detected based on GPU memory if not specified)",
)
@click.option(
    "--temperature",
    type=float,
    default=0.1,
    help="LLM temperature (lower = more deterministic)",
)
@click.option(
    "--load-in-4bit/--no-4bit",
    default=True,
    help="Load model in 4-bit precision (recommended for 70B on 80GB GPU)",
)
@click.option(
    "--load-in-8bit",
    is_flag=True,
    default=False,
    help="Load model in 8-bit precision instead of 4-bit",
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cuda", "cpu", "mps"], case_sensitive=False),
    default="auto",
    help="Device for inference (auto-detects if not specified).",
)
@click.option(
    "--flash-attention/--no-flash-attention",
    default=True,
    help="Use Flash Attention 2 for faster inference",
)
@click.option(
    "--hf-token",
    type=str,
    envvar="HF_TOKEN",
    help="HuggingFace token for gated models (or set HF_TOKEN env var)",
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
@click.option(
    "--prefetch",
    type=int,
    default=None,
    help="Number of files to prefetch (auto-detected if not set)",
)
@click.option(
    "--workers",
    type=int,
    default=None,
    help="Number of parallel workers (auto-detected if not set)",
)
@click.option(
    "--cross-file-batching/--no-cross-file-batching",
    default=True,
    help="Batch segments across files for better GPU utilization",
)
@click.option(
    "--auto-batch/--interactive-batch",
    default=False,
    help="Auto-select batch size (default) or interactively choose from options",
)
@click.option(
    "--torch-compile/--no-torch-compile",
    default=False,
    help="Use torch.compile() for faster inference (experimental, may cause issues)",
)
@click.option(
    "--filter-segments/--no-filter-segments",
    default=True,
    help="Only annotate segments with 2+ negative labels from classifiers (excludes ads)",
)
@click.option(
    "--min-negative-labels",
    type=int,
    default=2,
    help="Minimum negative labels required for annotation (default: 2)",
)
def annotate_with_llm(
    transcripts_dir: Path,
    output_dir: Path,
    model: str,
    use_mlx: bool,
    mlx_model: str,
    batch_size: int | None,
    temperature: float,
    load_in_4bit: bool,
    load_in_8bit: bool,
    device: str,
    flash_attention: bool,
    hf_token: str | None,
    force: bool,
    dry_run: bool,
    log_level: str,
    prefetch: int,
    workers: int,
    cross_file_batching: bool,
    auto_batch: bool,
    torch_compile: bool,
    filter_segments: bool,
    min_negative_labels: int,
) -> None:
    """
    Annotate podcast transcripts using LLM (Llama 3.3 70B Instruct).

    Supports two inference backends:
    1. HuggingFace Transformers (default): GPU inference with CUDA/MPS
    2. MLX (--use-mlx): Apple Silicon optimized via llm CLI

    By default, only annotates segments that:
    - Have 2+ negative labels from classification models (hate_speech_detection,
      hostile_content, fine_grained_hate_speech_detection, hate_against_minorities)
    - Are NOT classified as advertisements

    Use --no-filter-segments to annotate all segments.

    Analyzes each utterance for:
    - Hate speech detection
    - Target group identification
    - Hate speech type classification
    - Advertisement detection
    - Main topic extraction

    Example (HuggingFace):
        python scripts/annotate_with_llm.py \\
            --transcripts-dir outputs/transcripts_with_speaker_labels_postprocessed_with_classification_labels

    Example (MLX on Apple Silicon):
        python scripts/annotate_with_llm.py \\
            --transcripts-dir outputs/transcripts_with_diarization_labels_postprocessed \\
            --use-mlx

    Example (MLX with custom model):
        python scripts/annotate_with_llm.py \\
            --transcripts-dir outputs/transcripts_with_diarization_labels_postprocessed \\
            --use-mlx \\
            --mlx-model mlx-community/Llama-3.3-70B-Instruct-4bit

    Example (annotate all segments):
        python scripts/annotate_with_llm.py \\
            --transcripts-dir outputs/transcripts_with_diarization_labels_postprocessed \\
            --no-filter-segments
    """
    setup_logging(log_level)
    start_time = datetime.now()

    console.print("\n[bold cyan]🤖 LLM Annotation Pipeline[/bold cyan]\n")

    # check if MLX mode is requested.
    if use_mlx:
        console.print("[bold cyan]Using MLX backend for local inference[/bold cyan]\n")
        
        # import MLX annotator.
        from podcast_conversations.llm_annotation import MLXAnnotator
        
        # verify llm CLI is available.
        try:
            import subprocess
            result = subprocess.run(
                ["llm", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                console.print("[bold red]Error: llm CLI not found[/bold red]")
                console.print("Install with: pip install llm llm-mlx")
                sys.exit(1)
            console.print(f"[dim]llm CLI version: {result.stdout.strip()}[/dim]")
        except (subprocess.SubprocessError, FileNotFoundError):
            console.print("[bold red]Error: llm CLI not found[/bold red]")
            console.print("Install with: pip install llm llm-mlx")
            sys.exit(1)
        
        # show MLX model info.
        console.print(f"[dim]MLX Model: {mlx_model}[/dim]")
        console.print("[dim]Backend: llm CLI with MLX[/dim]")
        console.print(f"[dim]Temperature: {temperature}[/dim]")
        
        # MLX doesn't support batching via llm CLI, so we process sequentially.
        if batch_size and batch_size > 1:
            console.print(
                "[yellow]Note: MLX via llm CLI processes segments sequentially. "
                "Batch size setting ignored.[/yellow]"
            )
        
        # configure annotation for MLX.
        config = AnnotationConfig(
            model_name=mlx_model,  # use MLX model identifier.
            temperature=temperature,
        )
        
        # skip device detection for MLX - it will use Apple Silicon by default.
        use_mlx_backend = True
        
    else:
        # original HuggingFace transformers path.
        use_mlx_backend = False

    # detect and validate device (skip for MLX).
    if not use_mlx_backend:
        try:
            import torch

            # auto-detect device if not specified.
            if device is None:
                device_info = detect_device()
                device = device_info.device_str
                console.print(f"[green]Auto-detected device: {device}[/green]")
            else:
                device_info = detect_device()

            # handle device-specific setup.
            if device.startswith("cuda"):
                if not torch.cuda.is_available():
                    console.print("[bold red]Error: CUDA not available[/bold red]")
                    console.print("This script requires an NVIDIA GPU with CUDA support")
                    console.print("Use --device mps for Apple Silicon or --device cpu for CPU")
                    sys.exit(1)

                # get number of visible GPUs.
                num_gpus = torch.cuda.device_count()
                console.print(f"[dim]Visible GPUs: {num_gpus}[/dim]")

                # get device index.
                device_idx = 0
                if ":" in device:
                    requested_idx = int(device.split(":")[1])
                    if requested_idx >= num_gpus:
                        # device index out of range - use device 0 instead.
                        console.print(
                            f"[yellow]Warning: Requested device cuda:{requested_idx} but only "
                            f"{num_gpus} GPU(s) visible. Using cuda:0 instead.[/yellow]"
                        )
                        console.print(
                            "[dim]Hint: CUDA_VISIBLE_DEVICES may be set, making GPU appear as device 0[/dim]"
                        )
                        device_idx = 0
                        device = "cuda:0"  # update device for model loading.
                    else:
                        device_idx = requested_idx

                gpu_name = torch.cuda.get_device_name(device_idx)
                gpu_mem = torch.cuda.get_device_properties(device_idx).total_memory / (1024**3)
                console.print(f"[dim]Device: {device} - {gpu_name} ({gpu_mem:.1f}GB)[/dim]")

            elif device == "mps":
                if not torch.backends.mps.is_available():
                    console.print("[bold red]Error: MPS (Apple Silicon) not available[/bold red]")
                    console.print("MPS requires macOS 12.3+ and an Apple Silicon Mac")
                    sys.exit(1)

                console.print(f"[dim]Device: {device} - {device_info.name}[/dim]")
                console.print(
                    f"[dim]Unified memory: {device_info.total_memory_gb:.1f}GB total, "
                    f"{device_info.available_memory_gb:.1f}GB available[/dim]"
                )

                # warn about bitsandbytes not being available, but MLX has quantization.
                if load_in_4bit or load_in_8bit:
                    console.print(
                        "[yellow]Note: bitsandbytes quantization not supported on Apple Silicon. "
                        "Using float16 instead.[/yellow]"
                    )

                # check for MLX availability and suggest it for quantization.
                if is_mlx_available():
                    console.print(
                        "[green]MLX detected![/green] For better Apple Silicon performance with quantization:"
                    )
                    console.print(
                        "[dim]  • mlx-lm supports native 4-bit and 8-bit quantization[/dim]"
                    )
                    console.print(
                        "[dim]  • Pre-quantize models: mlx_lm.convert --hf-path <model> -q --q-bits 4[/dim]"
                    )
                else:
                    console.print(
                        "[dim]Tip: Install MLX for Apple Silicon optimized inference with quantization:[/dim]"
                    )
                    console.print(
                        "[dim]  pip install mlx mlx-lm[/dim]"
                    )

            else:
                console.print(f"[dim]Device: {device}[/dim]")
                console.print(
                    "[yellow]Warning: Running on CPU will be very slow for large models[/yellow]"
                )
        except ImportError:
            console.print("[bold red]Error: PyTorch not installed[/bold red]")
            sys.exit(1)

        # configure annotation for HuggingFace.
        config = AnnotationConfig(
            model_name=model,
            temperature=temperature,
        )

        console.print(f"[dim]Model: {model}[/dim]")
        if load_in_8bit:
            console.print("[dim]Precision: 8-bit quantization[/dim]")
        elif load_in_4bit:
            console.print("[dim]Precision: 4-bit quantization (NF4)[/dim]")
        else:
            console.print("[dim]Precision: bfloat16[/dim]")
        if flash_attention:
            console.print("[dim]Flash Attention 2: enabled[/dim]")

    # discover files.
    console.print("\n[bold blue]Discovering transcript files...[/bold blue]")
    json_files = sorted(transcripts_dir.rglob("*.json"))

    if not json_files:
        console.print(f"[yellow]No JSON files found in {transcripts_dir}[/yellow]")
        sys.exit(0)

    console.print(f"[green]Found {len(json_files)} transcript files[/green]")

    # ensure output directory exists.
    output_dir.mkdir(parents=True, exist_ok=True)

    # filter files that need processing.
    files_to_process: list[tuple[Path, Path]] = []
    files_skipped = 0

    for input_path in json_files:
        relative_path = input_path.relative_to(transcripts_dir)
        output_path = output_dir / relative_path

        if not force and check_annotations_exist(output_path):
            files_skipped += 1
        else:
            files_to_process.append((input_path, output_path))

    if files_skipped > 0:
        console.print(
            f"[yellow]Skipping {files_skipped} files with existing annotations[/yellow]"
        )

    console.print(f"[green]Will process {len(files_to_process)} files[/green]\n")

    if not files_to_process:
        console.print(
            "[yellow]All files already have annotations - nothing to process[/yellow]"
        )
        console.print("  Use --force to reprocess\n")
        sys.exit(0)

    # dry-run mode.
    if dry_run:
        console.print(
            "\n[bold yellow]Dry run mode - files that would be processed:[/bold yellow]\n"
        )
        for i, (input_path, _) in enumerate(files_to_process[:10], 1):
            relative_path = input_path.relative_to(transcripts_dir)
            console.print(f"  {i}. {relative_path}")
        if len(files_to_process) > 10:
            console.print(f"  ... and {len(files_to_process) - 10} more files")
        console.print()
        return

    # display system info.
    print_system_info(console)

    # auto-detect workers and prefetch if not specified.
    actual_workers = workers if workers is not None else detect_optimal_io_workers()
    actual_prefetch = prefetch if prefetch is not None else detect_optimal_prefetch()

    if workers is None or prefetch is None:
        console.print(
            f"[dim]Auto-detected: workers={actual_workers}, prefetch={actual_prefetch}[/dim]"
        )

    # count total segments before loading model (for batch size selection).
    console.print("[dim]Counting segments...[/dim]")
    import json
    total_segment_count = 0
    for input_path, _ in files_to_process:
        try:
            with open(input_path, encoding="utf-8") as f:
                data = json.load(f)
            total_segment_count += len(data.get("segments", []))
        except Exception:
            pass
    console.print(f"[dim]Total segments to process: {total_segment_count:,}[/dim]")

    # initialize annotator.
    console.print("\n[bold blue]Loading model...[/bold blue]")
    # initialize annotator based on backend.
    if use_mlx_backend:
        # use MLX annotator.
        from podcast_conversations.llm_annotation import MLXAnnotator
        
        console.print("\n[bold blue]Initializing MLX annotator...[/bold blue]")
        mlx_annotator = MLXAnnotator(
            model_name=mlx_model,
            max_tokens=2048,
            temperature=temperature,
        )
        console.print("[green]MLX annotator ready![/green]")
        
        # for MLX, we don't do batch optimization - it processes sequentially.
        actual_batch_size = 1
        console.print("[dim]MLX processes segments sequentially via llm CLI[/dim]\n")
        
        # use regular annotator workflow but will need to adapt for MLX.
        # for now, create a minimal wrapper that uses MLX.
        annotator = None  # MLX doesn't use the same interface.
        use_parallel_pipeline = False  # MLX processes sequentially.
        
    else:
        # original HuggingFace path.
        console.print("\n[bold blue]Initializing model...[/bold blue]")
        annotator = LLMAnnotator(
            config=config,
            device=device,
            batch_size=batch_size,
            use_flash_attention=flash_attention,
            load_in_4bit=load_in_4bit and not load_in_8bit,
            load_in_8bit=load_in_8bit,
            hf_token=hf_token,
            auto_optimize=False,  # disable auto-optimize, we'll do it with user input.
            use_torch_compile=torch_compile,
        )
        annotator.load_model()
        console.print("[green]Model loaded![/green]")

        # calculate optimal batch configuration and let user select.
        if batch_size is None:
            batch_config = calculate_optimal_batch_config(
                total_segments=total_segment_count,
                current_batch_size=annotator.batch_size,
                device_info=device_info,
            )

            # display options and get user selection.
            selected_batch_size = display_batch_options_and_select(
                batch_config=batch_config,
                auto_select=auto_batch,
            )

            # update annotator batch size.
            annotator.batch_size = selected_batch_size
        else:
            console.print(f"[dim]Using specified batch size: {batch_size}[/dim]")

        console.print(f"\n[green]Final batch size: {annotator.batch_size}[/green]")
        console.print(
            f"[dim]Parallel I/O: prefetch={actual_prefetch}, workers={actual_workers}[/dim]"
        )
        console.print(f"[dim]Cross-file batching: {cross_file_batching}[/dim]")
        if filter_segments:
            console.print(
                f"[dim]Segment filtering: enabled (min {min_negative_labels} negative labels, excluding ads)[/dim]\n"
            )
        else:
            console.print("[dim]Segment filtering: disabled (annotating all segments)[/dim]\n")

    # create file tasks.
    file_tasks = [
        FileTask(input_path=inp, output_path=out)
        for inp, out in files_to_process
    ]

    # process based on backend.
    if use_mlx_backend:
        # MLX processing - sequential via llm CLI.
        console.print("\n[bold]Processing with MLX (sequential)...[/bold]\n")
        
        from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
        
        processed = 0
        failed = 0
        total_segments = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "[cyan]Annotating...",
                total=len(file_tasks),
            )
            
            for file_task in file_tasks:
                try:
                    # load transcript.
                    with open(file_task.input_path, encoding="utf-8") as f:
                        transcript_data = json.load(f)
                    
                    segments = transcript_data.get("segments", [])
                    
                    # filter segments if requested.
                    if filter_segments:
                        filtered_segments = []
                        for segment in segments:
                            classifications = segment.get("classifications", [])
                            # count negative labels.
                            negative_count = sum(
                                1 for c in classifications
                                if c.get("label") in ["HATE", "OFFENSIVE", "toxic", "LABEL_0"]
                                and c.get("model_name") != "ad_content_detection"
                            )
                            if negative_count >= min_negative_labels:
                                filtered_segments.append(segment)
                        segments_to_annotate = filtered_segments
                    else:
                        segments_to_annotate = segments
                    
                    # annotate each segment.
                    for segment in segments_to_annotate:
                        text = segment.get("text", "").strip()
                        if not text:
                            continue
                        
                        # build prompt from config.
                        from podcast_conversations.llm_annotation import load_questions_config
                        questions = load_questions_config()
                        
                        # create simple prompt.
                        prompt_template = f"""Analyze this podcast utterance and provide JSON output:

{questions}

Utterance: {{text}}

Respond with valid JSON only."""
                        
                        try:
                            # annotate with MLX.
                            annotation = mlx_annotator.annotate(text, prompt_template)
                            segment["llm_annotation"] = annotation
                            total_segments += 1
                        except Exception as e:
                            logger.error(f"Failed to annotate segment: {e}")
                            segment["llm_annotation"] = {"error": str(e)}
                    
                    # save annotated transcript.
                    file_task.output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(file_task.output_path, "w", encoding="utf-8") as f:
                        json.dump(transcript_data, f, indent=2, ensure_ascii=False)
                    
                    processed += 1
                    
                except Exception as e:
                    logger.error(f"Failed to process {file_task.input_path}: {e}")
                    failed += 1
                
                progress.update(task, advance=1)
        
        # set metrics.
        throughput_metrics = {
            "total_segments": total_segments,
            "avg_segments_per_second": None,  # MLX doesn't track this.
        }
        precision = "4-bit (MLX)"
        platform = "MLX via llm CLI"
        
    else:
        # original HuggingFace parallel pipeline.
        pipeline = ParallelAnnotationPipeline(
            annotator=annotator,
            prefetch_count=actual_prefetch,
            io_workers=actual_workers,
            save_workers=actual_workers,
            cross_file_batching=cross_file_batching,
            max_files_per_batch=8,
            filter_segments=filter_segments,
            min_negative_labels=min_negative_labels,
        )

        # process files with progress bar and resource monitoring.
        with ResourceDisplay(console=console) as display:
            task = display.progress.add_task(
                "[cyan]annotating | starting...",
                total=len(files_to_process),
            )

            def progress_callback(completed: int, _total: int) -> None:
                current = display.progress.tasks[task].completed
                if completed > current:
                    display.progress.update(
                        task,
                        advance=completed - current,
                        description=(
                            f"[cyan]annotating | "
                            f"{pipeline.metrics.total_segments:,} segments in {completed} files"
                        ),
                    )

            processed, failed, total_segments = pipeline.process_files(
                file_tasks, progress_callback=progress_callback
            )

            # print resource usage summary.
            display.print_summary()

        # get throughput metrics.
        throughput_metrics = pipeline.get_metrics_summary()

        # unload model.
        annotator.unload_model()

        # determine precision string based on device.
        if device_info.device_type == DeviceType.MPS:
            precision = "float16 (Apple Silicon)"
        elif load_in_8bit:
            precision = "8-bit quantization"
        elif load_in_4bit:
            precision = "4-bit quantization (NF4)"
        else:
            precision = "bfloat16"
        
        platform = f"{device_info.device_type.value} - {device_info.name}"

    # determine platform string.
    if use_mlx_backend:
        platform_str = platform  # already set for MLX.
        actual_batch_size = 1
        model_name = mlx_model
    else:
        if device_info.device_type == DeviceType.CUDA:
            platform_str = f"NVIDIA CUDA ({device_info.name})"
        elif device_info.device_type == DeviceType.MPS:
            platform_str = f"Apple Silicon MPS ({device_info.name})"
        else:
            platform_str = "CPU"
        actual_batch_size = annotator.batch_size
        model_name = model

    # generate and save summary.
    end_time = datetime.now()
    summary = generate_summary(
        transcripts_dir=transcripts_dir,
        output_dir=output_dir,
        model=model_name,
        start_time=start_time,
        end_time=end_time,
        files_processed=processed,
        files_skipped=files_skipped,
        files_failed=failed,
        total_segments=total_segments,
        batch_size=actual_batch_size,
        precision=precision,
        device=device if not use_mlx_backend else "mlx",
        platform=platform_str,
        throughput_metrics=throughput_metrics,
    )

    # save summary to file.
    summary_filename = f"llm_annotation_{end_time.strftime('%Y_%m_%d_%H_%M_%S')}.txt"
    summary_path = output_dir / summary_filename

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    # display summary.
    console.print("\n[bold green]✓ Annotation complete![/bold green]\n")
    console.print("[bold]Statistics:[/bold]")
    console.print(f"  • Files processed: {processed}")
    if files_skipped > 0:
        console.print(f"  • Files skipped: {files_skipped}")
    if failed > 0:
        console.print(f"  • Files failed: {failed}")
    console.print(f"  • Total segments annotated: {total_segments:,}")
    if filter_segments and throughput_metrics.get("segments_skipped", 0) > 0:
        console.print(f"  • Segments skipped (filtered): {throughput_metrics['segments_skipped']:,}")
    console.print(f"  • Model: {model_name}")
    console.print(f"  • Backend: {'MLX via llm CLI' if use_mlx_backend else 'HuggingFace Transformers'}")
    console.print(f"  • Batch size: {actual_batch_size}")

    # throughput stats.
    console.print("\n[bold]Throughput:[/bold]")
    console.print(f"  • Segments/second: {throughput_metrics['segments_per_second']:.2f}")
    console.print(f"  • Files/minute: {throughput_metrics['files_per_minute']:.2f}")
    console.print(f"  • Avg batch time: {throughput_metrics['avg_batch_time_ms']:.1f}ms")

    console.print(f"\n[bold green]Output saved to: {output_dir}[/bold green]")
    console.print(f"[bold green]Summary saved to: {summary_path}[/bold green]\n")


if __name__ == "__main__":
    annotate_with_llm()
