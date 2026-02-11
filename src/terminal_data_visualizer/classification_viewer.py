"""Interactive viewers for utterance classification results."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from terminal_data_visualizer.classification_stats import (
    calculate_confidence_distribution,
    calculate_label_distribution,
    calculate_label_statistics,
    calculate_model_agreement,
    export_classification_matrix,
    load_classifications_from_file,
)
from terminal_data_visualizer.config import (
    COLOR_ERROR,
    COLOR_PRIMARY,
    COLOR_SUCCESS,
    COLOR_WARNING,
    KEY_BACK,
    OUTPUTS_PATH,
)
from terminal_data_visualizer.models import Classification
from terminal_data_visualizer.screen_capture import prompt_and_save
from terminal_data_visualizer.selectors import select_podcast, select_podcast_and_episode

console = Console(record=True)

# path to classified transcripts (use same config as main).
from terminal_data_visualizer.config import TRANSCRIPTS_DIR
CLASSIFIED_DIR = OUTPUTS_PATH / TRANSCRIPTS_DIR

# model names from config/classifiers.yaml.
AVAILABLE_MODELS = [
    "hate_speech_detection",
    "fine_grained_hate_speech_detection",
    "hate_against_minorities",
    "hostile_content",
    "ad_content_detection",
    "multilingual_hate_speech",
    "hate_speech_multilabel_bert",
    "beto_contextualized_hate_speech",
]


def view_per_model_results() -> None:
    """view classification results for a specific model."""
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]📊 Per-Model Classification Results[/bold {COLOR_PRIMARY}]\n")
    
    # check if classified directory exists.
    if not CLASSIFIED_DIR.exists():
        console.print(f"[{COLOR_WARNING}]No classified transcripts found at {CLASSIFIED_DIR}[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return
    
    # display available models.
    console.print(f"[{COLOR_PRIMARY}]Available Models:[/{COLOR_PRIMARY}]\n")
    for i, model in enumerate(AVAILABLE_MODELS, 1):
        console.print(f"  {i}. {model}")
    
    choice = Prompt.ask(
        f"\n[{COLOR_WARNING}]Select model number (or '{KEY_BACK}' to go back)[/{COLOR_WARNING}]",
        default=KEY_BACK,
    )
    
    if choice.lower() == KEY_BACK:
        return
    
    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(AVAILABLE_MODELS):
            console.print(f"[{COLOR_ERROR}]Invalid selection[/{COLOR_ERROR}]")
            Prompt.ask("\n[dim]Press Enter to continue[/dim]")
            return
        model_name = AVAILABLE_MODELS[idx]
    except ValueError:
        console.print(f"[{COLOR_ERROR}]Invalid input[/{COLOR_ERROR}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return
    
    # select podcast and episode.
    result = select_podcast_and_episode()
    if not result:
        return
    
    podcast_path, episode_path = result
    
    # find classified file.
    relative_path = episode_path.relative_to(OUTPUTS_PATH / "transcripts_with_diarization_labels_postprocessed")
    classified_file = CLASSIFIED_DIR / relative_path
    
    if not classified_file.exists():
        console.print(f"[{COLOR_WARNING}]No classified version found for this episode[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return
    
    # load classifications.
    all_classifications = load_classifications_from_file(classified_file)
    
    # filter by selected model.
    model_results = []
    for segment_classifications in all_classifications:
        for cls in segment_classifications:
            if cls.model_name == model_name:
                model_results.append(cls)
    
    if not model_results:
        console.print(f"[{COLOR_WARNING}]No classifications found for model: {model_name}[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return
    
    # display results.
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]{model_name} - {episode_path.stem}[/bold {COLOR_PRIMARY}]\n")
    
    # count labels.
    from collections import Counter
    label_counts = Counter(cls.label for cls in model_results)
    
    table = Table(title=f"Classification Results ({len(model_results)} segments)")
    table.add_column("Label", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Percentage", justify="right")
    
    for label, count in label_counts.most_common():
        percentage = (count / len(model_results)) * 100
        table.add_row(label, str(count), f"{percentage:.1f}%")
    
    console.print(table)
    
    # offer to save.
    prompt_and_save(console)
    
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def view_label_statistics() -> None:
    """view label statistics across all models."""
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]📊 Label Statistics[/bold {COLOR_PRIMARY}]\n")
    
    # select episode.
    result = select_podcast_and_episode()
    if not result:
        return
    
    podcast_path, episode_path = result
    
    # find classified file.
    relative_path = episode_path.relative_to(OUTPUTS_PATH / "transcripts_with_diarization_labels_postprocessed")
    classified_file = CLASSIFIED_DIR / relative_path
    
    if not classified_file.exists():
        console.print(f"[{COLOR_WARNING}]No classified version found for this episode[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return
    
    # load and calculate statistics.
    all_classifications = load_classifications_from_file(classified_file)
    stats = calculate_label_statistics(all_classifications)
    
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]Label Statistics - {episode_path.stem}[/bold {COLOR_PRIMARY}]\n")
    
    for model, model_stats in stats.items():
        table = Table(title=f"{model} ({model_stats['total_classifications']} classifications)")
        table.add_column("Label", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("Percentage", justify="right")
        
        for label, count in model_stats["label_counts"].items():
            percentage = model_stats["label_percentages"][label]
            table.add_row(label, str(count), f"{percentage:.1f}%")
        
        console.print(table)
        console.print()
    
    # offer to save.
    prompt_and_save(console)
    
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def view_cross_model_analysis() -> None:
    """view cross-model agreement analysis."""
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]📊 Cross-Model Analysis[/bold {COLOR_PRIMARY}]\n")
    
    # select episode.
    result = select_podcast_and_episode()
    if not result:
        return
    
    podcast_path, episode_path = result
    
    # find classified file.
    relative_path = episode_path.relative_to(OUTPUTS_PATH / "transcripts_with_diarization_labels_postprocessed")
    classified_file = CLASSIFIED_DIR / relative_path
    
    if not classified_file.exists():
        console.print(f"[{COLOR_WARNING}]No classified version found for this episode[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return
    
    # load classifications.
    all_classifications = load_classifications_from_file(classified_file)
    
    # calculate agreement.
    agreement_stats = calculate_model_agreement(all_classifications)
    
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]Cross-Model Analysis - {episode_path.stem}[/bold {COLOR_PRIMARY}]\n")
    
    if "error" in agreement_stats:
        console.print(f"[{COLOR_WARNING}]{agreement_stats['error']}[/{COLOR_WARNING}]")
    else:
        table = Table(title="Model Agreement Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        
        table.add_row("Models Compared", str(len(agreement_stats["models_compared"])))
        table.add_row("Total Segments", str(agreement_stats["total_segments"]))
        table.add_row("Agreements", str(agreement_stats["agreements"]))
        table.add_row("Disagreements", str(agreement_stats["disagreements"]))
        table.add_row("Agreement Rate", f"{agreement_stats['agreement_rate']*100:.1f}%")
        
        console.print(table)
        console.print()
        
        console.print(f"[{COLOR_PRIMARY}]Models Compared:[/{COLOR_PRIMARY}]")
        for model in agreement_stats["models_compared"]:
            console.print(f"  • {model}")
    
    # offer to save.
    prompt_and_save(console)
    
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def view_show_classification_summary() -> None:
    """view classification summary for entire show."""
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]📊 Show Classification Summary[/bold {COLOR_PRIMARY}]\n")
    
    # select podcast.
    podcast_path = select_podcast("Select podcast for summary")
    if not podcast_path:
        return
    
    # find all classified episodes for this show.
    show_classified_dir = CLASSIFIED_DIR / podcast_path.name
    if not show_classified_dir.exists():
        console.print(f"[{COLOR_WARNING}]No classified episodes found for {podcast_path.name}[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return
    
    classified_files = list(show_classified_dir.glob("*.json"))
    
    if not classified_files:
        console.print(f"[{COLOR_WARNING}]No classified episodes found[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return
    
    console.print(f"[dim]Loading {len(classified_files)} episodes...[/dim]")
    
    # aggregate classifications from all episodes.
    all_show_classifications = []
    for file in classified_files:
        episode_classifications = load_classifications_from_file(file)
        all_show_classifications.extend(episode_classifications)
    
    # calculate statistics.
    stats = calculate_label_statistics(all_show_classifications)
    
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]{podcast_path.name} - Classification Summary[/bold {COLOR_PRIMARY}]\n")
    console.print(f"[dim]Episodes: {len(classified_files)} | Total Segments: {len(all_show_classifications)}[/dim]\n")
    
    for model, model_stats in stats.items():
        table = Table(title=f"{model}")
        table.add_column("Label", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("Percentage", justify="right")
        
        for label, count in sorted(model_stats["label_counts"].items(), key=lambda x: -x[1]):
            percentage = model_stats["label_percentages"][label]
            table.add_row(label, f"{count:,}", f"{percentage:.1f}%")
        
        console.print(table)
        console.print()
    
    # offer to save.
    prompt_and_save(console)
    
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def view_episode_classification_summary() -> None:
    """view classification summary for specific episode."""
    view_label_statistics()  # same functionality.


def classification_viewer_menu() -> None:
    """main menu for classification viewing."""
    while True:
        console.clear()
        console.print(f"[bold {COLOR_PRIMARY}]📊 Classification Viewer[/bold {COLOR_PRIMARY}]\n")
        
        options = [
            "1. View Per-Model Results",
            "2. View Label Statistics",
            "3. Cross-Model Analysis",
            "4. Show Classification Summary",
            "5. Episode Classification Summary",
            f"{KEY_BACK}. Back to Main Menu",
        ]
        
        for option in options:
            console.print(f"  {option}")
        
        choice = Prompt.ask(
            f"\n[{COLOR_WARNING}]Select option[/{COLOR_WARNING}]",
            default=KEY_BACK,
        )
        
        if choice == "1":
            view_per_model_results()
        elif choice == "2":
            view_label_statistics()
        elif choice == "3":
            view_cross_model_analysis()
        elif choice == "4":
            view_show_classification_summary()
        elif choice == "5":
            view_episode_classification_summary()
        elif choice.lower() == KEY_BACK:
            break
        else:
            console.print(f"[{COLOR_ERROR}]Invalid option[/{COLOR_ERROR}]")
            Prompt.ask("\n[dim]Press Enter to continue[/dim]")
