"""Interactive explorer for features CSV data with aggregation and filtering."""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import plotext as plt
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from terminal_data_visualizer.config import (
    COLOR_ERROR,
    COLOR_PRIMARY,
    COLOR_SUCCESS,
    COLOR_WARNING,
    FEATURES_FILE,
    KEY_BACK,
    OUTPUTS_PATH,
)
from terminal_data_visualizer.screen_capture import prompt_and_save

console = Console(record=True)

# create plots output directory.
PLOTS_DIR = OUTPUTS_PATH / "terminal_viewer" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_features_csv() -> tuple[list[dict[str, Any]], list[str]] | None:
    """Load features CSV file and return rows and column names."""
    features_file = OUTPUTS_PATH / FEATURES_FILE
    if not features_file.exists():
        console.print(
            f"[{COLOR_WARNING}]Features file not found at {features_file}[/{COLOR_WARNING}]"
        )
        return None

    try:
        with open(features_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            console.print(f"[{COLOR_WARNING}]No data found in features file[/{COLOR_WARNING}]")
            return None

        columns = list(rows[0].keys())
        return rows, columns

    except (FileNotFoundError, OSError, csv.Error) as e:
        console.print(f"[{COLOR_ERROR}]Error reading features file: {e}[/{COLOR_ERROR}]")
        return None


def extract_show_name(file_path: str) -> str:
    """Extract show name from file path."""
    # path format: outputs/transcripts_with_diarization_labels/show_name/episode.json
    parts = Path(file_path).parts
    if len(parts) >= 3:
        return parts[-2]  # show name is second to last.
    return "unknown"


def get_numeric_value(value: str | None) -> float | None:
    """Convert string value to float, return None if not numeric."""
    if not value or value.strip() == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def prompt_save_plot(plot_name: str) -> None:
    """Prompt user to save the current plotext plot."""
    console.print()
    save = Confirm.ask(
        f"[{COLOR_WARNING}]Save this plot?[/{COLOR_WARNING}]",
        default=False,
    )

    if save:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        # save as HTML (interactive, can be opened in browser).
        html_file = PLOTS_DIR / f"{plot_name}_{timestamp}.html"
        plt.save_fig(str(html_file), append=False)
        console.print(f"[{COLOR_SUCCESS}]✓ Plot saved to: {html_file}[/{COLOR_SUCCESS}]")

        # also save as text file.
        txt_file = PLOTS_DIR / f"{plot_name}_{timestamp}.txt"
        with open(txt_file, "w") as f:
            f.write(plt.build())
        console.print(f"[{COLOR_SUCCESS}]✓ Text version saved to: {txt_file}[/{COLOR_SUCCESS}]")


def features_explorer_menu() -> None:
    """Main menu for features explorer."""
    while True:
        console.clear()
        console.print(
            Panel(
                f"[bold {COLOR_PRIMARY}]📊 Features Explorer[/bold {COLOR_PRIMARY}]\n"
                "[dim]Explore conversation features with aggregations and filtering[/dim]",
                border_style="blue",
            )
        )

        table = Table(show_header=False, box=None)
        table.add_column("Option", style=COLOR_WARNING)
        table.add_column("Description")

        table.add_row("1", "📈 Overview Statistics")
        table.add_row("2", "📁 Per-Show Analysis")
        table.add_row("3", "📄 Per-Episode Details")
        table.add_row("4", "📊 Feature Distributions")
        table.add_row("5", "🔍 Custom Filtering")
        table.add_row("6", "🔗 Feature Correlations")
        table.add_row("7", "🎯 Top/Bottom Rankings")
        table.add_row(KEY_BACK, "⬅️  Back to Main Menu")

        console.print(table)
        console.print()

        choice = Prompt.ask(
            f"[{COLOR_WARNING}]Select option[/{COLOR_WARNING}]",
            default=KEY_BACK,
        )

        if choice.lower() == KEY_BACK:
            return
        elif choice == "1":
            show_overview_statistics()
        elif choice == "2":
            show_per_show_analysis()
        elif choice == "3":
            show_per_episode_details()
        elif choice == "4":
            show_feature_distributions()
        elif choice == "5":
            custom_filtering_menu()
        elif choice == "6":
            show_feature_correlations()
        elif choice == "7":
            show_rankings_menu()
        else:
            console.print(f"[{COLOR_ERROR}]Invalid option[/{COLOR_ERROR}]")
            Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def show_overview_statistics() -> None:
    """Display overall statistics across all features."""
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]📈 Overview Statistics[/bold {COLOR_PRIMARY}]\n")

    result = load_features_csv()
    if not result:
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    rows, columns = result

    # extract show names.
    for row in rows:
        row["show_name"] = extract_show_name(row.get("file_path", ""))

    total_episodes = len(rows)
    total_shows = len(set(row["show_name"] for row in rows))

    console.print(f"[{COLOR_SUCCESS}]Total Episodes:[/{COLOR_SUCCESS}] {total_episodes}")
    console.print(f"[{COLOR_SUCCESS}]Total Shows:[/{COLOR_SUCCESS}] {total_shows}\n")

    # key metrics summary.
    metrics = [
        ("average_politeness_score", "Average Politeness"),
        ("average_switch_time_minutes", "Avg Switch Time (min)"),
        ("question_ratio", "Question Ratio"),
        ("dominance_index", "Dominance Index"),
        ("type_token_ratio", "Type-Token Ratio"),
        ("total_duration_minutes", "Total Duration (min)"),
        ("total_questions", "Total Questions"),
        ("total_turns", "Total Turns"),
    ]

    table = Table(title="Key Metrics Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Mean", justify="right", style="green")
    table.add_column("Median", justify="right", style="yellow")
    table.add_column("Min", justify="right", style="blue")
    table.add_column("Max", justify="right", style="red")

    for col, label in metrics:
        values = [get_numeric_value(row.get(col)) for row in rows]
        values = [v for v in values if v is not None]

        if values:
            mean_val = statistics.mean(values)
            median_val = statistics.median(values)
            min_val = min(values)
            max_val = max(values)

            table.add_row(
                label,
                f"{mean_val:.2f}",
                f"{median_val:.2f}",
                f"{min_val:.2f}",
                f"{max_val:.2f}",
            )
        else:
            table.add_row(label, "N/A", "N/A", "N/A", "N/A")

    console.print(table)

    # offer screen capture.
    console.print()
    prompt_and_save(console)
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def show_per_show_analysis() -> None:
    """Display aggregated statistics per show."""
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]📁 Per-Show Analysis[/bold {COLOR_PRIMARY}]\n")

    result = load_features_csv()
    if not result:
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    rows, columns = result

    # group by show.
    for row in rows:
        row["show_name"] = extract_show_name(row.get("file_path", ""))

    show_groups = defaultdict(list)
    for row in rows:
        show_groups[row["show_name"]].append(row)

    # select metric to aggregate.
    console.print("[bold]Select metric to analyze:[/bold]\n")
    metrics = [
        ("average_politeness_score", "Politeness Score"),
        ("average_switch_time_minutes", "Switch Time (min)"),
        ("question_ratio", "Question Ratio"),
        ("dominance_index", "Dominance Index"),
        ("type_token_ratio", "Type-Token Ratio"),
        ("total_duration_minutes", "Episode Duration (min)"),
        ("total_questions", "Questions per Episode"),
        ("total_turns", "Turns per Episode"),
    ]

    for i, (col, label) in enumerate(metrics, 1):
        console.print(f"  {i}. {label}")

    choice = Prompt.ask(
        f"\n[{COLOR_WARNING}]Select metric (1-{len(metrics)})[/{COLOR_WARNING}]",
        default="1",
    )

    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(metrics):
            console.print(f"[{COLOR_ERROR}]Invalid selection[/{COLOR_ERROR}]")
            Prompt.ask("\n[dim]Press Enter to continue[/dim]")
            return
        metric_col, metric_label = metrics[idx]
    except ValueError:
        console.print(f"[{COLOR_ERROR}]Invalid input[/{COLOR_ERROR}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    # aggregate by show.
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]📁 {metric_label} by Show[/bold {COLOR_PRIMARY}]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Show", style="cyan")
    table.add_column("Episodes", justify="right", style="yellow")
    table.add_column("Mean", justify="right", style="green")
    table.add_column("Median", justify="right", style="blue")
    table.add_column("Min", justify="right", style="white")
    table.add_column("Max", justify="right", style="red")

    show_stats = []
    for show_name, show_rows in show_groups.items():
        values = [get_numeric_value(row.get(metric_col)) for row in show_rows]
        values = [v for v in values if v is not None]

        if values:
            mean_val = statistics.mean(values)
            median_val = statistics.median(values)
            min_val = min(values)
            max_val = max(values)

            show_stats.append((show_name, len(show_rows), mean_val, median_val, min_val, max_val))

    # sort by mean descending.
    show_stats.sort(key=lambda x: x[2], reverse=True)

    for show_name, ep_count, mean_val, median_val, min_val, max_val in show_stats:
        table.add_row(
            show_name,
            str(ep_count),
            f"{mean_val:.2f}",
            f"{median_val:.2f}",
            f"{min_val:.2f}",
            f"{max_val:.2f}",
        )

    console.print(table)
    console.print()

    # visualize with bar chart.
    display_show_comparison_chart(show_stats, metric_label)

    console.print()
    prompt_and_save(console)
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def display_show_comparison_chart(
    show_stats: list[tuple[str, int, float, float, float, float]], metric_label: str
) -> None:
    """Display bar chart comparing shows by metric mean."""
    if not show_stats:
        return

    # limit to top 15 for readability.
    show_stats = show_stats[:15]

    shows = [stat[0][:25] for stat in show_stats]  # truncate long names.
    means = [stat[2] for stat in show_stats]

    plt.clf()
    plt.bar(shows, means, orientation="horizontal", width=0.3)
    plt.title(f"Mean {metric_label} by Show (Top 15)")
    plt.xlabel(metric_label)
    plt.ylabel("Show")
    plt.theme("pro")
    plt.plotsize(100, max(15, len(shows) + 5))
    plt.show()

    # offer to save plot.
    plot_name = metric_label.lower().replace(" ", "_").replace("(", "").replace(")", "")
    prompt_save_plot(f"show_comparison_{plot_name}")
    console.print()  # add spacing after plot


def show_per_episode_details() -> None:
    """Display detailed view of specific episodes."""
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]📄 Per-Episode Details[/bold {COLOR_PRIMARY}]\n")

    result = load_features_csv()
    if not result:
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    rows, columns = result

    for row in rows:
        row["show_name"] = extract_show_name(row.get("file_path", ""))

    # search by show or episode name.
    search_term = Prompt.ask(
        f"\n[{COLOR_WARNING}]Search for show/episode (or '{KEY_BACK}' to cancel)[/{COLOR_WARNING}]",
        default=KEY_BACK,
    )

    if search_term.lower() == KEY_BACK:
        return

    # filter matching episodes.
    matching = [
        row
        for row in rows
        if search_term.lower() in row.get("file_name", "").lower()
        or search_term.lower() in row["show_name"].lower()
    ]

    if not matching:
        console.print(f"\n[{COLOR_WARNING}]No matching episodes found[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]Found {len(matching)} episodes[/bold {COLOR_PRIMARY}]\n")

    # display list.
    for i, row in enumerate(matching[:20], 1):  # limit to 20.
        show = row["show_name"]
        episode = row.get("file_name", "unknown")
        console.print(f"  {i}. [{COLOR_PRIMARY}]{show}[/{COLOR_PRIMARY}] - {episode}")

    if len(matching) > 20:
        console.print(f"\n[dim]... and {len(matching) - 20} more[/dim]")

    # select episode to view.
    max_choice = min(len(matching), 20)
    choice = Prompt.ask(
        f"\n[{COLOR_WARNING}]Select episode number (1-{max_choice}) "
        f"or '{KEY_BACK}'[/{COLOR_WARNING}]",
        default=KEY_BACK,
    )

    if choice.lower() == KEY_BACK:
        return

    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= min(len(matching), 20):
            console.print(f"[{COLOR_ERROR}]Invalid selection[/{COLOR_ERROR}]")
            Prompt.ask("\n[dim]Press Enter to continue[/dim]")
            return
        selected_row = matching[idx]
    except ValueError:
        console.print(f"[{COLOR_ERROR}]Invalid input[/{COLOR_ERROR}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    # display all features for selected episode.
    display_episode_features(selected_row, columns)


def display_episode_features(row: dict[str, Any], columns: list[str]) -> None:
    """Display all features for a specific episode."""
    console.clear()
    episode_name = row.get("file_name", "unknown")
    console.print(f"[bold {COLOR_PRIMARY}]📄 {episode_name}[/bold {COLOR_PRIMARY}]\n")

    # group features by category.
    core_features = [
        "file_name",
        "file_path",
        "total_duration_minutes",
        "total_turns",
        "total_questions",
        "total_words",
    ]

    aggregate_features = [
        "average_politeness_score",
        "average_switch_time_minutes",
        "question_ratio",
        "dominance_index",
        "type_token_ratio",
        "root_type_token_ratio",
    ]

    # display core info.
    table = Table(title="Core Information", show_header=True, header_style="bold magenta", box=None)
    table.add_column("Feature", style="cyan")
    table.add_column("Value", style="green")

    for col in core_features:
        if col in row:
            table.add_row(col, str(row[col]))

    console.print(table)
    console.print()

    # display aggregate features.
    table = Table(
        title="Aggregate Features", show_header=True, header_style="bold magenta", box=None
    )
    table.add_column("Feature", style="cyan")
    table.add_column("Value", style="green")

    for col in aggregate_features:
        if col in row:
            val = row[col]
            if val and val.strip():
                table.add_row(col, f"{float(val):.4f}" if val else "N/A")

    console.print(table)
    console.print()

    # display per-speaker metrics (up to 5 speakers).
    console.print("[bold]Per-Speaker Metrics (first 5 speakers):[/bold]\n")
    speaker_metrics = [
        ("speaking_time", "Speaking Time (min)"),
        ("turns", "Turns"),
        ("questions", "Questions"),
        ("politeness", "Politeness"),
        ("ttr", "Type-Token Ratio"),
    ]

    for metric_prefix, metric_label in speaker_metrics:
        console.print(f"[{COLOR_PRIMARY}]{metric_label}:[/{COLOR_PRIMARY}]")

        table = Table(show_header=False, box=None)
        table.add_column("Speaker", style="yellow")
        table.add_column("Value", style="green")

        for i in range(5):
            col_name = f"{metric_prefix}_SPEAKER_{i:02d}"
            if col_name in row and row[col_name] and row[col_name].strip():
                val = row[col_name]
                try:
                    val_float = float(val)
                    table.add_row(f"SPEAKER_{i:02d}", f"{val_float:.2f}")
                except ValueError:
                    table.add_row(f"SPEAKER_{i:02d}", val)

        console.print(table)
        console.print()

    prompt_and_save(console)
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def show_feature_distributions() -> None:
    """Display distribution of values for a selected feature."""
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]📊 Feature Distributions[/bold {COLOR_PRIMARY}]\n")

    result = load_features_csv()
    if not result:
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    rows, columns = result

    # select feature.
    console.print("[bold]Select feature to analyze:[/bold]\n")
    features = [
        ("average_politeness_score", "Politeness Score"),
        ("average_switch_time_minutes", "Switch Time (min)"),
        ("question_ratio", "Question Ratio"),
        ("dominance_index", "Dominance Index"),
        ("type_token_ratio", "Type-Token Ratio"),
        ("total_duration_minutes", "Episode Duration (min)"),
        ("total_questions", "Questions"),
        ("total_turns", "Turns"),
        ("total_words", "Words"),
    ]

    for i, (col, label) in enumerate(features, 1):
        console.print(f"  {i}. {label}")

    choice = Prompt.ask(
        f"\n[{COLOR_WARNING}]Select feature (1-{len(features)})[/{COLOR_WARNING}]",
        default="1",
    )

    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(features):
            console.print(f"[{COLOR_ERROR}]Invalid selection[/{COLOR_ERROR}]")
            Prompt.ask("\n[dim]Press Enter to continue[/dim]")
            return
        feature_col, feature_label = features[idx]
    except ValueError:
        console.print(f"[{COLOR_ERROR}]Invalid input[/{COLOR_ERROR}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    # get values.
    values = [get_numeric_value(row.get(feature_col)) for row in rows]
    values = [v for v in values if v is not None]

    if not values:
        console.print(
            f"\n[{COLOR_WARNING}]No numeric values found for this feature[/{COLOR_WARNING}]"
        )
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]📊 {feature_label} Distribution[/bold {COLOR_PRIMARY}]\n")

    # display statistics.
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Statistic", style="cyan")
    table.add_column("Value", justify="right", style="green")

    table.add_row("Count", str(len(values)))
    table.add_row("Mean", f"{statistics.mean(values):.4f}")
    table.add_row("Median", f"{statistics.median(values):.4f}")
    table.add_row("Std Dev", f"{statistics.stdev(values):.4f}" if len(values) > 1 else "N/A")
    table.add_row("Min", f"{min(values):.4f}")
    table.add_row("Max", f"{max(values):.4f}")
    table.add_row("Range", f"{max(values) - min(values):.4f}")

    console.print(table)
    console.print()

    # display histogram with plotext.
    display_plotext_histogram(values, feature_label)

    prompt_and_save(console)
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def display_plotext_histogram(values: list[float], label: str) -> None:
    """Display a terminal-based histogram using plotext."""
    plt.clf()
    plt.hist(values, bins=20, label=label)
    plt.title(f"{label} Distribution")
    plt.xlabel(label)
    plt.ylabel("Frequency")
    plt.theme("pro")
    plt.plotsize(100, 25)
    plt.show()

    # offer to save plot.
    plot_name = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
    prompt_save_plot(f"histogram_{plot_name}")
    console.print()  # add spacing after plot


def custom_filtering_menu() -> None:
    """Allow custom filtering of episodes based on feature criteria."""
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]🔍 Custom Filtering[/bold {COLOR_PRIMARY}]\n")

    result = load_features_csv()
    if not result:
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    rows, columns = result

    # select feature to filter by.
    console.print("[bold]Filter episodes by feature:[/bold]\n")
    features = [
        ("average_politeness_score", "Politeness Score"),
        ("average_switch_time_minutes", "Switch Time (min)"),
        ("question_ratio", "Question Ratio"),
        ("dominance_index", "Dominance Index"),
        ("total_duration_minutes", "Episode Duration (min)"),
        ("total_questions", "Questions"),
    ]

    for i, (col, label) in enumerate(features, 1):
        console.print(f"  {i}. {label}")

    choice = Prompt.ask(
        f"\n[{COLOR_WARNING}]Select feature (1-{len(features)})[/{COLOR_WARNING}]",
        default="1",
    )

    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(features):
            console.print(f"[{COLOR_ERROR}]Invalid selection[/{COLOR_ERROR}]")
            Prompt.ask("\n[dim]Press Enter to continue[/dim]")
            return
        feature_col, feature_label = features[idx]
    except ValueError:
        console.print(f"[{COLOR_ERROR}]Invalid input[/{COLOR_ERROR}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    # get threshold.
    threshold = Prompt.ask(
        f"\n[{COLOR_WARNING}]Enter minimum value for {feature_label}[/{COLOR_WARNING}]",
        default="0",
    )

    try:
        threshold_val = float(threshold)
    except ValueError:
        console.print(f"[{COLOR_ERROR}]Invalid number[/{COLOR_ERROR}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    # filter rows.
    for row in rows:
        row["show_name"] = extract_show_name(row.get("file_path", ""))

    filtered = []
    for row in rows:
        val = get_numeric_value(row.get(feature_col))
        if val is not None and val >= threshold_val:
            filtered.append(row)

    console.clear()
    console.print(
        f"[bold {COLOR_PRIMARY}]Found {len(filtered)} episodes where "
        f"{feature_label} >= {threshold_val}[/bold {COLOR_PRIMARY}]\n"
    )

    if not filtered:
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    # display results.
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Show", style="cyan")
    table.add_column("Episode", style="yellow")
    table.add_column(feature_label, justify="right", style="green")

    for row in filtered[:50]:  # limit to 50.
        show = row["show_name"]
        episode = row.get("file_name", "unknown")
        val = get_numeric_value(row.get(feature_col))
        table.add_row(show, episode, f"{val:.2f}" if val else "N/A")

    console.print(table)

    if len(filtered) > 50:
        console.print(f"\n[dim]... and {len(filtered) - 50} more episodes[/dim]")

    prompt_and_save(console)
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def show_feature_correlations() -> None:
    """Show correlations between two features."""
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]🔗 Feature Correlations[/bold {COLOR_PRIMARY}]\n")

    result = load_features_csv()
    if not result:
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    rows, columns = result

    features = [
        ("average_politeness_score", "Politeness"),
        ("average_switch_time_minutes", "Switch Time"),
        ("question_ratio", "Question Ratio"),
        ("dominance_index", "Dominance Index"),
        ("type_token_ratio", "Type-Token Ratio"),
        ("total_questions", "Total Questions"),
        ("total_turns", "Total Turns"),
    ]

    console.print("[bold]Select first feature:[/bold]\n")
    for i, (col, label) in enumerate(features, 1):
        console.print(f"  {i}. {label}")

    choice1 = Prompt.ask(
        f"\n[{COLOR_WARNING}]Select feature (1-{len(features)})[/{COLOR_WARNING}]",
        default="1",
    )

    console.print("\n[bold]Select second feature:[/bold]\n")
    for i, (col, label) in enumerate(features, 1):
        console.print(f"  {i}. {label}")

    choice2 = Prompt.ask(
        f"\n[{COLOR_WARNING}]Select feature (1-{len(features)})[/{COLOR_WARNING}]",
        default="2",
    )

    try:
        idx1 = int(choice1) - 1
        idx2 = int(choice2) - 1
        if idx1 < 0 or idx1 >= len(features) or idx2 < 0 or idx2 >= len(features):
            console.print(f"[{COLOR_ERROR}]Invalid selection[/{COLOR_ERROR}]")
            Prompt.ask("\n[dim]Press Enter to continue[/dim]")
            return
        col1, label1 = features[idx1]
        col2, label2 = features[idx2]
    except ValueError:
        console.print(f"[{COLOR_ERROR}]Invalid input[/{COLOR_ERROR}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    # calculate correlation.
    pairs = []
    for row in rows:
        val1 = get_numeric_value(row.get(col1))
        val2 = get_numeric_value(row.get(col2))
        if val1 is not None and val2 is not None:
            pairs.append((val1, val2))

    if len(pairs) < 2:
        console.print(
            f"\n[{COLOR_WARNING}]Not enough data points for correlation[/{COLOR_WARNING}]"
        )
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    # compute pearson correlation.
    x_values = [p[0] for p in pairs]
    y_values = [p[1] for p in pairs]

    mean_x = statistics.mean(x_values)
    mean_y = statistics.mean(y_values)

    numerator = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    denominator = (
        (sum((x - mean_x) ** 2 for x in x_values) ** 0.5)
        * (sum((y - mean_y) ** 2 for y in y_values) ** 0.5)
    )

    correlation = numerator / denominator if denominator != 0 else 0

    console.clear()
    console.print(
        f"[bold {COLOR_PRIMARY}]🔗 Correlation: {label1} vs {label2}"
        f"[/bold {COLOR_PRIMARY}]\n"
    )

    console.print(f"[{COLOR_SUCCESS}]Pearson Correlation:[/{COLOR_SUCCESS}] {correlation:.4f}")
    console.print(f"[{COLOR_SUCCESS}]Sample Size:[/{COLOR_SUCCESS}] {len(pairs)} episodes\n")

    # interpretation.
    if abs(correlation) >= 0.7:
        strength = "Strong"
    elif abs(correlation) >= 0.4:
        strength = "Moderate"
    elif abs(correlation) >= 0.2:
        strength = "Weak"
    else:
        strength = "Very weak"

    direction = "positive" if correlation > 0 else "negative"
    console.print(f"[dim]{strength} {direction} correlation[/dim]\n")

    # display scatter plot with plotext.
    display_plotext_scatter(x_values, y_values, label1, label2, correlation)

    prompt_and_save(console)
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def display_plotext_scatter(
    x_values: list[float],
    y_values: list[float],
    xlabel: str,
    ylabel: str,
    correlation: float,
) -> None:
    """Display a terminal-based scatter plot using plotext."""
    plt.clf()
    plt.scatter(x_values, y_values, marker="•")
    plt.title(f"{xlabel} vs {ylabel} (r={correlation:.3f})")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.theme("pro")
    plt.plotsize(100, 25)
    plt.show()

    # offer to save plot.
    x_name = xlabel.lower().replace(" ", "_").replace("(", "").replace(")", "")
    y_name = ylabel.lower().replace(" ", "_").replace("(", "").replace(")", "")
    prompt_save_plot(f"scatter_{x_name}_vs_{y_name}")
    console.print()  # add spacing after plot


def show_rankings_menu() -> None:
    """Show top/bottom ranked episodes for a selected feature."""
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]🎯 Top/Bottom Rankings[/bold {COLOR_PRIMARY}]\n")

    result = load_features_csv()
    if not result:
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    rows, columns = result

    # select feature.
    console.print("[bold]Select feature to rank by:[/bold]\n")
    features = [
        ("average_politeness_score", "Politeness Score"),
        ("average_switch_time_minutes", "Switch Time (min)"),
        ("question_ratio", "Question Ratio"),
        ("dominance_index", "Dominance Index"),
        ("type_token_ratio", "Type-Token Ratio"),
        ("total_duration_minutes", "Episode Duration (min)"),
        ("total_questions", "Questions"),
        ("total_turns", "Turns"),
    ]

    for i, (col, label) in enumerate(features, 1):
        console.print(f"  {i}. {label}")

    choice = Prompt.ask(
        f"\n[{COLOR_WARNING}]Select feature (1-{len(features)})[/{COLOR_WARNING}]",
        default="1",
    )

    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(features):
            console.print(f"[{COLOR_ERROR}]Invalid selection[/{COLOR_ERROR}]")
            Prompt.ask("\n[dim]Press Enter to continue[/dim]")
            return
        feature_col, feature_label = features[idx]
    except ValueError:
        console.print(f"[{COLOR_ERROR}]Invalid input[/{COLOR_ERROR}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    # add show names.
    for row in rows:
        row["show_name"] = extract_show_name(row.get("file_path", ""))

    # sort by feature.
    rows_with_values = [
        (row, get_numeric_value(row.get(feature_col)))
        for row in rows
        if get_numeric_value(row.get(feature_col)) is not None
    ]
    rows_with_values.sort(key=lambda x: x[1], reverse=True)

    console.clear()
    console.print(
        f"[bold {COLOR_PRIMARY}]🎯 Top/Bottom Rankings: {feature_label}"
        f"[/bold {COLOR_PRIMARY}]\n"
    )

    # top 10.
    console.print("[bold green]Top 10 Episodes:[/bold green]\n")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Rank", justify="right", style="cyan")
    table.add_column("Show", style="yellow")
    table.add_column("Episode", style="white")
    table.add_column(feature_label, justify="right", style="green")

    for i, (row, val) in enumerate(rows_with_values[:10], 1):
        table.add_row(
            str(i),
            row["show_name"],
            row.get("file_name", "unknown"),
            f"{val:.2f}" if val else "N/A",
        )

    console.print(table)
    console.print()

    # bottom 10.
    console.print("[bold red]Bottom 10 Episodes:[/bold red]\n")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Rank", justify="right", style="cyan")
    table.add_column("Show", style="yellow")
    table.add_column("Episode", style="white")
    table.add_column(feature_label, justify="right", style="red")

    bottom_10 = rows_with_values[-10:][::-1]
    for i, (row, val) in enumerate(bottom_10, 1):
        table.add_row(
            str(i),
            row["show_name"],
            row.get("file_name", "unknown"),
            f"{val:.2f}" if val else "N/A",
        )

    console.print(table)
    console.print()

    # visualize top 10 with bar chart.
    display_plotext_bar_chart(rows_with_values[:10], feature_label, "Top 10")

    prompt_and_save(console)
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def display_plotext_bar_chart(
    ranked_data: list[tuple[dict, float]], feature_label: str, title: str
) -> None:
    """Display a horizontal bar chart for rankings."""
    if not ranked_data:
        return

    labels = []
    values = []
    for row, val in ranked_data:
        show = row.get("show_name", "unknown")[:20]  # truncate long names.
        episode = row.get("file_name", "unknown")[:30]
        labels.append(f"{show}: {episode}"[:40])  # limit total length.
        values.append(val)

    plt.clf()
    plt.bar(labels, values, orientation="horizontal", width=0.3)
    plt.title(f"{title}: {feature_label}")
    plt.xlabel(feature_label)
    plt.theme("pro")
    plt.plotsize(100, max(15, len(labels) + 5))
    plt.show()

    # offer to save plot.
    plot_name = feature_label.lower().replace(" ", "_").replace("(", "").replace(")", "")
    prompt_save_plot(f"rankings_{plot_name}")
    console.print()  # add spacing after plot
