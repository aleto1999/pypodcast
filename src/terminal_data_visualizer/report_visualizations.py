"""Generate and export visualizations for reports."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from terminal_data_visualizer.config import BAR_WIDTH, COLOR_PRIMARY


def create_speaker_distribution_chart(
    speaker_data: dict[str, float],
    title: str = "Speaker Distribution",
    max_speakers: int = 10,
) -> Console:
    """Create a speaker distribution bar chart.
    
    Args:
        speaker_data: Dict mapping speaker name to percentage
        title: Chart title
        max_speakers: Maximum number of speakers to show
        
    Returns:
        Console with rendered chart
    """
    console = Console(record=True, width=100)
    
    # sort and limit speakers.
    top_speakers = sorted(speaker_data.items(), key=lambda x: x[1], reverse=True)[:max_speakers]
    
    console.print(f"\n[bold {COLOR_PRIMARY}]{title}[/bold {COLOR_PRIMARY}]\n")
    
    for speaker, pct in top_speakers:
        bar_len = int((pct / 100) * BAR_WIDTH)
        bar = "█" * bar_len + "░" * (BAR_WIDTH - bar_len)
        display_name = speaker[:30] if len(speaker) > 30 else speaker
        console.print(f"{display_name:<30} {bar} {pct:>5.1f}%")
    
    console.print()
    return console


def create_category_distribution_chart(
    category_data: dict[str, int],
    title: str = "Category Distribution",
    max_categories: int = 10,
) -> Console:
    """Create a category distribution bar chart.
    
    Args:
        category_data: Dict mapping category to count
        title: Chart title
        max_categories: Maximum categories to show
        
    Returns:
        Console with rendered chart
    """
    console = Console(record=True, width=100)
    
    # sort and limit categories.
    top_categories = sorted(category_data.items(), key=lambda x: x[1], reverse=True)[:max_categories]
    
    if not top_categories:
        return console
    
    # calculate max for scaling.
    max_count = max(count for _, count in top_categories)
    
    console.print(f"\n[bold {COLOR_PRIMARY}]{title}[/bold {COLOR_PRIMARY}]\n")
    
    for category, count in top_categories:
        bar_len = int((count / max_count) * BAR_WIDTH) if max_count > 0 else 0
        bar = "█" * bar_len + "░" * (BAR_WIDTH - bar_len)
        display_name = category[:30] if len(category) > 30 else category
        console.print(f"{display_name:<30} {bar} {count:>10,}")
    
    console.print()
    return console


def create_label_distribution_chart(
    label_data: dict[str, int],
    total_segments: int,
    title: str = "Classification Label Distribution",
    max_labels: int = 15,
) -> Console:
    """Create a classification label distribution chart with percentages.
    
    Args:
        label_data: Dict mapping label to count
        total_segments: Total segments analyzed
        title: Chart title
        max_labels: Maximum labels to show
        
    Returns:
        Console with rendered chart
    """
    console = Console(record=True, width=110)
    
    # sort and limit labels.
    top_labels = sorted(label_data.items(), key=lambda x: x[1], reverse=True)[:max_labels]
    
    if not top_labels:
        return console
    
    # calculate max for scaling.
    max_count = max(count for _, count in top_labels)
    
    console.print(f"\n[bold {COLOR_PRIMARY}]{title}[/bold {COLOR_PRIMARY}]\n")
    
    for label, count in top_labels:
        pct = (count / total_segments * 100) if total_segments > 0 else 0
        bar_len = int((count / max_count) * BAR_WIDTH) if max_count > 0 else 0
        bar = "█" * bar_len + "░" * (BAR_WIDTH - bar_len)
        display_name = label[:25] if len(label) > 25 else label
        console.print(f"{display_name:<25} {bar} {count:>10,} ({pct:>5.1f}%)")
    
    console.print()
    return console


def create_comparison_table(
    summaries: list,
    metrics: list[tuple[str, list]],
) -> Console:
    """Create a comparison table for multiple shows.
    
    Args:
        summaries: List of ShowSummary objects
        metrics: List of (metric_name, values) tuples
        
    Returns:
        Console with rendered table
    """
    console = Console(record=True, width=140)
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=18)
    
    for summary in summaries:
        table.add_column(summary.show_name[:12], justify="right", width=14)
    
    if len(summaries) > 1:
        table.add_column("Average", justify="right", width=14, style="yellow")
    
    for metric_name, values in metrics:
        row = [metric_name]
        for val in values:
            row.append(str(val))
        
        if len(summaries) > 1:
            try:
                if "%" in metric_name:
                    avg = sum(float(v.rstrip('%')) for v in values) / len(values)
                    row.append(f"{avg:.0f}%")
                elif metric_name in ("Episodes", "Utterances", "Keywords"):
                    avg = sum(int(v) for v in values) / len(values)
                    row.append(f"{avg:.0f}")
                else:
                    row.append("-")
            except (ValueError, TypeError):
                row.append("-")
        
        table.add_row(*row)
    
    console.print(table)
    return console


def export_console_to_svg(console: Console, output_path: Path) -> None:
    """Export console output to SVG file.
    
    Args:
        console: Console with recorded content
        output_path: Path to save SVG file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    svg_content = console.export_svg(title="Podcast Analysis Visualization")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_content)
