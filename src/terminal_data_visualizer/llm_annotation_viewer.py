"""Interactive viewers for LLM annotation results."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from terminal_data_visualizer.config import (
    COLOR_ERROR,
    COLOR_PRIMARY,
    COLOR_SUCCESS,
    COLOR_WARNING,
    KEY_BACK,
    LLM_ANNOTATIONS_DIR,
    OUTPUTS_PATH,
)
from terminal_data_visualizer.llm_annotation_stats import (
    calculate_advertisement_distribution,
    calculate_comprehensive_stats,
    calculate_hate_speech_distribution,
    calculate_hate_speech_type_stats,
    calculate_target_group_stats,
    calculate_topic_frequency,
    export_annotation_csv,
    export_annotation_dataset,
    load_llm_annotations_from_file,
)
from terminal_data_visualizer.screen_capture import prompt_and_save
from terminal_data_visualizer.selectors import select_podcast, select_podcast_and_episode

console = Console(record=True)


def find_llm_annotation_dir() -> Path | None:
    """find the LLM annotation directory (handles wildcards)."""
    analysis_dir = OUTPUTS_PATH / "analysis"
    if not analysis_dir.exists():
        return None
    
    # look for llm_annotation_* directories.
    llm_dirs = list(analysis_dir.glob("llm_annotation_*"))
    if llm_dirs:
        # use the most recent one.
        return sorted(llm_dirs)[-1]
    
    return None


def view_by_annotation_type() -> None:
    """view annotations filtered by type (hate speech or advertisements)."""
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]🤖 Annotations by Type[/bold {COLOR_PRIMARY}]\n")
    
    llm_dir = find_llm_annotation_dir()
    if not llm_dir:
        console.print(f"[{COLOR_WARNING}]No LLM annotation directory found[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return
    
    # select annotation type.
    console.print(f"[{COLOR_PRIMARY}]Filter by:[/{COLOR_PRIMARY}]\n")
    console.print("  1. Hate Speech (has_hate_speech=true)")
    console.print("  2. Advertisements (has_advertisement=true)")
    console.print("  3. All Annotations")
    
    choice = Prompt.ask(
        f"\n[{COLOR_WARNING}]Select type (or '{KEY_BACK}' to go back)[/{COLOR_WARNING}]",
        default=KEY_BACK,
    )
    
    if choice.lower() == KEY_BACK:
        return
    
    if choice not in ["1", "2", "3"]:
        console.print(f"[{COLOR_ERROR}]Invalid selection[/{COLOR_ERROR}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return
    
    filter_type = {"1": "hate_speech", "2": "advertisement", "3": "all"}[choice]
    
    # select podcast and episode.
    result = select_podcast_and_episode()
    if not result:
        return
    
    podcast_path, episode_path = result
    
    # find LLM annotation file.
    annotation_file = llm_dir / podcast_path.name / episode_path.name
    
    if not annotation_file.exists():
        console.print(f"[{COLOR_WARNING}]No LLM annotations found for this episode[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return
    
    # load annotations.
    annotations = load_llm_annotations_from_file(annotation_file)
    
    if not annotations:
        console.print(f"[{COLOR_WARNING}]No annotations found in file[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return
    
    # filter annotations.
    if filter_type == "hate_speech":
        filtered = [ann for ann in annotations if ann.has_hate_speech]
        title = "Hate Speech Annotations"
    elif filter_type == "advertisement":
        filtered = [ann for ann in annotations if ann.has_advertisement]
        title = "Advertisement Annotations"
    else:
        filtered = annotations
        title = "All Annotations"
    
    # display results.
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]{title} - {episode_path.stem}[/bold {COLOR_PRIMARY}]\n")
    console.print(f"[dim]Total: {len(filtered)} / {len(annotations)} segments[/dim]\n")
    
    if not filtered:
        console.print(f"[{COLOR_WARNING}]No annotations match the filter[/{COLOR_WARNING}]")
    else:
        # sample first 20.
        for i, ann in enumerate(filtered[:20], 1):
            console.print(f"{i}. [cyan]{ann.main_topic}[/cyan]")
            if ann.target_group:
                console.print(f"   Target: {ann.target_group}")
            if ann.hate_speech_type:
                console.print(f"   Type: {ann.hate_speech_type}")
            console.print()
        
        if len(filtered) > 20:
            console.print(f"[dim]... and {len(filtered) - 20} more[/dim]")
    
    # offer to save.
    prompt_and_save(console)
    
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def view_by_target_group() -> None:
    """view annotations filtered by target group."""
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]🎯 Annotations by Target Group[/bold {COLOR_PRIMARY}]\n")
    
    llm_dir = find_llm_annotation_dir()
    if not llm_dir:
        console.print(f"[{COLOR_WARNING}]No LLM annotation directory found[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return
    
    # target group options.
    target_groups = [
        "racial and ethnic minorities",
        "religious minorities",
        "women",
        "lgbtq+ population",
        "disabled population",
        "immigrant population",
    ]
    
    console.print(f"[{COLOR_PRIMARY}]Select Target Group:[/{COLOR_PRIMARY}]\n")
    for i, group in enumerate(target_groups, 1):
        console.print(f"  {i}. {group}")
    
    choice = Prompt.ask(
        f"\n[{COLOR_WARNING}]Select group (or '{KEY_BACK}' to go back)[/{COLOR_WARNING}]",
        default=KEY_BACK,
    )
    
    if choice.lower() == KEY_BACK:
        return
    
    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(target_groups):
            console.print(f"[{COLOR_ERROR}]Invalid selection[/{COLOR_ERROR}]")
            Prompt.ask("\n[dim]Press Enter to continue[/dim]")
            return
        selected_group = target_groups[idx]
    except ValueError:
        console.print(f"[{COLOR_ERROR}]Invalid input[/{COLOR_ERROR}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return
    
    # select podcast and episode.
    result = select_podcast_and_episode()
    if not result:
        return
    
    podcast_path, episode_path = result
    
    # find LLM annotation file.
    annotation_file = llm_dir / podcast_path.name / episode_path.name
    
    if not annotation_file.exists():
        console.print(f"[{COLOR_WARNING}]No LLM annotations found for this episode[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return
    
    # load and filter annotations.
    annotations = load_llm_annotations_from_file(annotation_file)
    filtered = [ann for ann in annotations if ann.target_group == selected_group]
    
    # display results.
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]Target Group: {selected_group}[/bold {COLOR_PRIMARY}]\n")
    console.print(f"[dim]{episode_path.stem}[/dim]\n")
    console.print(f"[dim]Found: {len(filtered)} segments[/dim]\n")
    
    if not filtered:
        console.print(f"[{COLOR_WARNING}]No annotations targeting this group[/{COLOR_WARNING}]")
    else:
        for i, ann in enumerate(filtered[:20], 1):
            console.print(f"{i}. [cyan]{ann.main_topic}[/cyan]")
            if ann.hate_speech_type:
                console.print(f"   Type: {ann.hate_speech_type}")
            console.print()
        
        if len(filtered) > 20:
            console.print(f"[dim]... and {len(filtered) - 20} more[/dim]")
    
    # offer to save.
    prompt_and_save(console)
    
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def view_by_hate_speech_type() -> None:
    """view annotations filtered by hate speech type."""
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]📑 Annotations by Hate Speech Type[/bold {COLOR_PRIMARY}]\n")
    
    llm_dir = find_llm_annotation_dir()
    if not llm_dir:
        console.print(f"[{COLOR_WARNING}]No LLM annotation directory found[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return
    
    # hate speech type options.
    hate_types = [
        "Threat to culture or identity",
        "Threat to survival or physical security",
        "Vilification",
        "Explicit dehumanization",
    ]
    
    console.print(f"[{COLOR_PRIMARY}]Select Hate Speech Type:[/{COLOR_PRIMARY}]\n")
    for i, ht in enumerate(hate_types, 1):
        console.print(f"  {i}. {ht}")
    
    choice = Prompt.ask(
        f"\n[{COLOR_WARNING}]Select type (or '{KEY_BACK}' to go back)[/{COLOR_WARNING}]",
        default=KEY_BACK,
    )
    
    if choice.lower() == KEY_BACK:
        return
    
    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(hate_types):
            console.print(f"[{COLOR_ERROR}]Invalid selection[/{COLOR_ERROR}]")
            Prompt.ask("\n[dim]Press Enter to continue[/dim]")
            return
        selected_type = hate_types[idx]
    except ValueError:
        console.print(f"[{COLOR_ERROR}]Invalid input[/{COLOR_ERROR}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return
    
    # select podcast and episode.
    result = select_podcast_and_episode()
    if not result:
        return
    
    podcast_path, episode_path = result
    
    # find LLM annotation file.
    annotation_file = llm_dir / podcast_path.name / episode_path.name
    
    if not annotation_file.exists():
        console.print(f"[{COLOR_WARNING}]No LLM annotations found for this episode[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return
    
    # load and filter annotations.
    annotations = load_llm_annotations_from_file(annotation_file)
    filtered = [ann for ann in annotations if ann.hate_speech_type == selected_type]
    
    # display results.
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]{selected_type}[/bold {COLOR_PRIMARY}]\n")
    console.print(f"[dim]{episode_path.stem}[/dim]\n")
    console.print(f"[dim]Found: {len(filtered)} segments[/dim]\n")
    
    if not filtered:
        console.print(f"[{COLOR_WARNING}]No annotations of this type[/{COLOR_WARNING}]")
    else:
        for i, ann in enumerate(filtered[:20], 1):
            console.print(f"{i}. [cyan]{ann.main_topic}[/cyan]")
            if ann.target_group:
                console.print(f"   Target: {ann.target_group}")
            console.print()
        
        if len(filtered) > 20:
            console.print(f"[dim]... and {len(filtered) - 20} more[/dim]")
    
    # offer to save.
    prompt_and_save(console)
    
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def view_topic_analysis() -> None:
    """view topic frequency analysis."""
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]📚 Topic Analysis[/bold {COLOR_PRIMARY}]\n")
    
    llm_dir = find_llm_annotation_dir()
    if not llm_dir:
        console.print(f"[{COLOR_WARNING}]No LLM annotation directory found[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return
    
    # select podcast and episode.
    result = select_podcast_and_episode()
    if not result:
        return
    
    podcast_path, episode_path = result
    
    # find LLM annotation file.
    annotation_file = llm_dir / podcast_path.name / episode_path.name
    
    if not annotation_file.exists():
        console.print(f"[{COLOR_WARNING}]No LLM annotations found for this episode[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return
    
    # load annotations and calculate topic frequency.
    annotations = load_llm_annotations_from_file(annotation_file)
    topics = calculate_topic_frequency(annotations)
    
    # display results.
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]Topic Analysis - {episode_path.stem}[/bold {COLOR_PRIMARY}]\n")
    console.print(f"[dim]Total annotations: {len(annotations)} | Unique topics: {len(topics)}[/dim]\n")
    
    if not topics:
        console.print(f"[{COLOR_WARNING}]No topics found[/{COLOR_WARNING}]")
    else:
        table = Table(title="Most Common Topics")
        table.add_column("Topic", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("Percentage", justify="right")
        
        for topic, count in sorted(topics.items(), key=lambda x: -x[1])[:30]:
            percentage = (count / len(annotations)) * 100
            table.add_row(topic, str(count), f"{percentage:.1f}%")
        
        console.print(table)
    
    # offer to save.
    prompt_and_save(console)
    
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def view_show_annotation_summary() -> None:
    """view annotation summary for entire show."""
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]🤖 Show Annotation Summary[/bold {COLOR_PRIMARY}]\n")
    
    llm_dir = find_llm_annotation_dir()
    if not llm_dir:
        console.print(f"[{COLOR_WARNING}]No LLM annotation directory found[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return
    
    # select podcast.
    podcast_path = select_podcast("Select podcast for summary")
    if not podcast_path:
        return
    
    # find all annotated episodes for this show.
    show_annotation_dir = llm_dir / podcast_path.name
    if not show_annotation_dir.exists():
        console.print(f"[{COLOR_WARNING}]No annotations found for {podcast_path.name}[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return
    
    annotation_files = list(show_annotation_dir.glob("*.json"))
    
    if not annotation_files:
        console.print(f"[{COLOR_WARNING}]No annotated episodes found[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return
    
    console.print(f"[dim]Loading {len(annotation_files)} episodes...[/dim]")
    
    # aggregate annotations from all episodes.
    all_annotations = []
    for file in annotation_files:
        episode_annotations = load_llm_annotations_from_file(file)
        all_annotations.extend(episode_annotations)
    
    # calculate comprehensive statistics.
    stats = calculate_comprehensive_stats(all_annotations)
    
    # display results.
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]{podcast_path.name} - Annotation Summary[/bold {COLOR_PRIMARY}]\n")
    console.print(f"[dim]Episodes: {len(annotation_files)} | Total Annotations: {stats['total_annotations']}[/dim]\n")
    
    # hate speech stats.
    hs_stats = stats["hate_speech"]
    table1 = Table(title="Hate Speech Detection")
    table1.add_column("Category", style="cyan")
    table1.add_column("Count", justify="right")
    table1.add_column("Rate", justify="right")
    table1.add_row("Has Hate Speech", str(hs_stats["has_hate_speech"]), f"{hs_stats['hate_speech_rate']:.1f}%")
    table1.add_row("No Hate Speech", str(hs_stats["no_hate_speech"]), "")
    console.print(table1)
    console.print()
    
    # target groups.
    if stats["target_groups"]:
        table2 = Table(title="Target Groups")
        table2.add_column("Group", style="cyan")
        table2.add_column("Count", justify="right")
        for group, count in sorted(stats["target_groups"].items(), key=lambda x: -x[1]):
            table2.add_row(group, str(count))
        console.print(table2)
        console.print()
    
    # hate speech types.
    if stats["hate_speech_types"]:
        table3 = Table(title="Hate Speech Types")
        table3.add_column("Type", style="cyan")
        table3.add_column("Count", justify="right")
        for ht, count in sorted(stats["hate_speech_types"].items(), key=lambda x: -x[1]):
            table3.add_row(ht, str(count))
        console.print(table3)
        console.print()
    
    # offer to save.
    prompt_and_save(console)
    
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def view_episode_annotation_summary() -> None:
    """view annotation summary for specific episode."""
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]🤖 Episode Annotation Summary[/bold {COLOR_PRIMARY}]\n")
    
    llm_dir = find_llm_annotation_dir()
    if not llm_dir:
        console.print(f"[{COLOR_WARNING}]No LLM annotation directory found[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return
    
    # select podcast and episode.
    result = select_podcast_and_episode()
    if not result:
        return
    
    podcast_path, episode_path = result
    
    # find LLM annotation file.
    annotation_file = llm_dir / podcast_path.name / episode_path.name
    
    if not annotation_file.exists():
        console.print(f"[{COLOR_WARNING}]No LLM annotations found for this episode[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return
    
    # load annotations and calculate statistics.
    annotations = load_llm_annotations_from_file(annotation_file)
    stats = calculate_comprehensive_stats(annotations)
    
    # display results.
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]{episode_path.stem} - Annotation Summary[/bold {COLOR_PRIMARY}]\n")
    console.print(f"[dim]Total Annotations: {stats['total_annotations']}[/dim]\n")
    
    # hate speech stats.
    hs_stats = stats["hate_speech"]
    table1 = Table(title="Hate Speech Detection")
    table1.add_column("Category", style="cyan")
    table1.add_column("Count", justify="right")
    table1.add_column("Rate", justify="right")
    table1.add_row("Has Hate Speech", str(hs_stats["has_hate_speech"]), f"{hs_stats['hate_speech_rate']:.1f}%")
    table1.add_row("No Hate Speech", str(hs_stats["no_hate_speech"]), "")
    console.print(table1)
    console.print()
    
    # advertisement stats.
    ad_stats = stats["advertisement"]
    table2 = Table(title="Advertisement Detection")
    table2.add_column("Category", style="cyan")
    table2.add_column("Count", justify="right")
    table2.add_column("Rate", justify="right")
    table2.add_row("Has Advertisement", str(ad_stats["has_advertisement"]), f"{ad_stats['advertisement_rate']:.1f}%")
    table2.add_row("No Advertisement", str(ad_stats["no_advertisement"]), "")
    console.print(table2)
    console.print()
    
    # target groups.
    if stats["target_groups"]:
        table3 = Table(title="Target Groups")
        table3.add_column("Group", style="cyan")
        table3.add_column("Count", justify="right")
        for group, count in sorted(stats["target_groups"].items(), key=lambda x: -x[1]):
            table3.add_row(group, str(count))
        console.print(table3)
        console.print()
    
    # offer to save.
    prompt_and_save(console)
    
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def llm_annotation_viewer_menu() -> None:
    """main menu for LLM annotation viewing."""
    while True:
        console.clear()
        console.print(f"[bold {COLOR_PRIMARY}]🤖 LLM Annotation Viewer[/bold {COLOR_PRIMARY}]\n")
        
        options = [
            "1. View by Annotation Type",
            "2. View by Target Group",
            "3. View by Hate Speech Type",
            "4. Topic Analysis",
            "5. Show Annotation Summary",
            "6. Episode Annotation Summary",
            f"{KEY_BACK}. Back to Main Menu",
        ]
        
        for option in options:
            console.print(f"  {option}")
        
        choice = Prompt.ask(
            f"\n[{COLOR_WARNING}]Select option[/{COLOR_WARNING}]",
            default=KEY_BACK,
        )
        
        if choice == "1":
            view_by_annotation_type()
        elif choice == "2":
            view_by_target_group()
        elif choice == "3":
            view_by_hate_speech_type()
        elif choice == "4":
            view_topic_analysis()
        elif choice == "5":
            view_show_annotation_summary()
        elif choice == "6":
            view_episode_annotation_summary()
        elif choice.lower() == KEY_BACK:
            break
        else:
            console.print(f"[{COLOR_ERROR}]Invalid option[/{COLOR_ERROR}]")
            Prompt.ask("\n[dim]Press Enter to continue[/dim]")
