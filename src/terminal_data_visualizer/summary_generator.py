"""Show-level and cross-show summary generation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table

from terminal_data_visualizer.config import (
    ANALYSIS_DIR,
    BAR_WIDTH,
    COLOR_ERROR,
    COLOR_PRIMARY,
    COLOR_SUCCESS,
    COLOR_WARNING,
    DOCUMENT_LABELS_DIR,
    KEY_BACK,
    LLM_ANNOTATIONS_DIR,
    OUTPUTS_PATH,
    TRANSCRIPTS_DIR,
)
from terminal_data_visualizer.document_label_parser import parse_show_document_labels
from terminal_data_visualizer.models import Segment
from terminal_data_visualizer.screen_capture import prompt_and_save

console = Console(record=True)


@dataclass
class ShowSummary:
    """Comprehensive summary for a podcast show."""

    show_name: str
    total_episodes: int = 0
    total_duration_hours: float = 0.0
    total_words: int = 0
    total_segments: int = 0
    avg_duration_minutes: float = 0.0
    avg_words_per_episode: float = 0.0
    speaker_distribution: dict[str, float] = field(default_factory=dict)
    keyword_matches: int = 0
    episodes_with_keywords: int = 0  # file-level keyword coverage.
    top_categories: dict[str, int] = field(default_factory=dict)
    hate_speech_count: int = 0
    ad_count: int = 0
    target_groups: dict[str, int] = field(default_factory=dict)
    episodes_analyzed: int = 0
    classification_coverage: int = 0  # utterances with all 8 models.
    total_utterances: int = 0  # total utterances across all episodes.
    
    # document-level labels (from document_labels summary files).
    overall_document_label: str = "UNKNOWN"  # POSITIVE, NEGATIVE, or UNKNOWN.
    positive_episodes: int = 0
    negative_episodes: int = 0
    unknown_episodes: int = 0
    total_positive_labels: int = 0
    total_negative_labels: int = 0
    segments_analyzed_for_labels: int = 0
    ad_segments_skipped: int = 0
    top_classification_labels: dict[str, int] = field(default_factory=dict)  # top labels across all models.


@dataclass
class ProblematicContent:
    """Aggregated problematic content across shows."""

    total_hate_speech: int = 0
    by_show: dict[str, int] = field(default_factory=dict)
    by_target_group: dict[str, int] = field(default_factory=dict)
    classifier_flags: dict[str, int] = field(default_factory=dict)


def generate_show_summary(show_name: str) -> ShowSummary:
    """Generate comprehensive summary for a show."""
    summary = ShowSummary(show_name=show_name)

    # process transcripts.
    transcripts_path = OUTPUTS_PATH / TRANSCRIPTS_DIR / show_name
    if transcripts_path.exists():
        transcript_files = list(transcripts_path.glob("*.json"))
        summary.total_episodes = len(transcript_files)

        speaker_words: dict[str, int] = {}
        total_words_all = 0
        
        # track classification coverage.
        utterances_with_all_8 = 0
        total_utterances = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(f"Analyzing {show_name}...", total=len(transcript_files))

            for file in transcript_files:
                try:
                    with open(file, encoding="utf-8") as f:
                        data = json.load(f)

                    segments = data.get("segments", [])

                    for seg_data in segments:
                        # create Segment for basic stats.
                        seg = Segment.from_dict(seg_data)
                        
                        summary.total_segments += 1
                        summary.total_duration_hours += seg.duration / 3600
                        summary.total_words += seg.word_count
                        total_words_all += seg.word_count

                        if seg.speaker not in speaker_words:
                            speaker_words[seg.speaker] = 0
                        speaker_words[seg.speaker] += seg.word_count
                        
                        # check classification coverage (skip empty segments).
                        text = seg_data.get("text", "").strip()
                        if text:
                            # only count non-empty segments for classification coverage.
                            total_utterances += 1
                            classifications = seg_data.get("classifications", [])
                            if len(classifications) == 8:
                                utterances_with_all_8 += 1

                except (json.JSONDecodeError, OSError):
                    pass

                progress.advance(task)

        summary.classification_coverage = utterances_with_all_8
        summary.total_utterances = total_utterances

        # calculate averages.
        if summary.total_episodes > 0:
            summary.avg_duration_minutes = (
                summary.total_duration_hours * 60 / summary.total_episodes
            )
            summary.avg_words_per_episode = summary.total_words / summary.total_episodes

        # calculate speaker distribution.
        if total_words_all > 0:
            summary.speaker_distribution = {
                speaker: (words / total_words_all) * 100
                for speaker, words in sorted(
                    speaker_words.items(), key=lambda x: x[1], reverse=True
                )[:10]
            }

    # process keyword analysis.
    analysis_path = OUTPUTS_PATH / ANALYSIS_DIR / show_name
    if analysis_path.exists():
        category_counts: dict[str, int] = {}
        episodes_with_keywords = 0
        
        for file in analysis_path.glob("*_keywords.json"):
            try:
                with open(file, encoding="utf-8") as f:
                    data = json.load(f)

                matches = data.get("matches", [])
                summary.keyword_matches += len(matches)
                
                # file-level coverage: count files with at least one match.
                if len(matches) > 0:
                    episodes_with_keywords += 1

                for match in matches:
                    cat = match.get("category", "Unknown")
                    category_counts[cat] = category_counts.get(cat, 0) + 1

            except (json.JSONDecodeError, OSError):
                pass

        summary.episodes_with_keywords = episodes_with_keywords
        
        # top categories.
        summary.top_categories = dict(
            sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        )

    # process document-level labels.
    doc_labels_path = OUTPUTS_PATH / DOCUMENT_LABELS_DIR / show_name / "show_summary.md"
    doc_stats = parse_show_document_labels(doc_labels_path)
    if doc_stats:
        summary.overall_document_label = doc_stats.overall_label
        summary.positive_episodes = doc_stats.positive_episodes
        summary.negative_episodes = doc_stats.negative_episodes
        summary.unknown_episodes = doc_stats.unknown_episodes
        summary.total_positive_labels = doc_stats.total_positive_labels
        summary.total_negative_labels = doc_stats.total_negative_labels
        summary.segments_analyzed_for_labels = doc_stats.total_segments_analyzed
        summary.ad_segments_skipped = doc_stats.total_ad_segments_skipped
        
        # use ad detection from classification model (LABEL_1 from ad_content_detection).
        if "ad_content_detection" in doc_stats.label_distribution:
            ad_labels = doc_stats.label_distribution["ad_content_detection"]
            summary.ad_count = ad_labels.get("LABEL_1", 0)
        
        # aggregate top labels across all models.
        all_labels: dict[str, int] = {}
        for model_name, label_dist in doc_stats.label_distribution.items():
            for label, count in label_dist.items():
                # skip "safe" labels like LABEL_0, NOT-HATE, SAFE, Appropriate.
                if label in ("LABEL_0", "NOT-HATE", "SAFE", "Appropriate"):
                    continue
                all_labels[label] = all_labels.get(label, 0) + count
        
        summary.top_classification_labels = dict(
            sorted(all_labels.items(), key=lambda x: x[1], reverse=True)[:10]
        )

    # process llm annotations (optional - may not exist).
    llm_path = OUTPUTS_PATH / LLM_ANNOTATIONS_DIR / show_name
    if llm_path.exists():
        target_counts: dict[str, int] = {}

        for file in llm_path.glob("*.json"):
            summary.episodes_analyzed += 1
            try:
                with open(file, encoding="utf-8") as f:
                    data = json.load(f)

                for seg in data.get("segments", []):
                    ann = seg.get("llm_annotation", {})
                    if ann.get("has_hate_speech"):
                        summary.hate_speech_count += 1
                        target = ann.get("target_group")
                        if target:
                            target_counts[target] = target_counts.get(target, 0) + 1

                    # only count LLM ad detections if we don't have classification model data.
                    if ann.get("has_advertisement") and summary.ad_count == 0:
                        summary.ad_count += 1

            except (json.JSONDecodeError, OSError):
                pass

        summary.target_groups = dict(
            sorted(target_counts.items(), key=lambda x: x[1], reverse=True)
        )

    return summary


def display_show_summary(summary: ShowSummary) -> None:
    """Display comprehensive show summary."""
    console.print(
        Panel(
            f"[bold {COLOR_PRIMARY}]📈 Show Summary: {summary.show_name}[/bold {COLOR_PRIMARY}]",
            border_style="blue",
        )
    )

    # overview section.
    console.print(f"\n[bold {COLOR_PRIMARY}]OVERVIEW[/bold {COLOR_PRIMARY}]")
    console.print("─" * 40)

    overview_table = Table(show_header=False, box=None)
    overview_table.add_column("Metric", style="cyan", width=25)
    overview_table.add_column("Value", style="green", width=30)

    overview_table.add_row("Episodes Analyzed", f"{summary.total_episodes:,}")
    overview_table.add_row("Total Duration", f"{summary.total_duration_hours:.1f}h")
    overview_table.add_row("Total Words", f"{summary.total_words:,}")
    overview_table.add_row(
        "Average Episode",
        f"{summary.avg_duration_minutes:.0f}m, {summary.avg_words_per_episode:.0f} words",
    )

    console.print(overview_table)

    # speaker distribution.
    if summary.speaker_distribution:
        console.print(f"\n[bold {COLOR_PRIMARY}]SPEAKER DISTRIBUTION[/bold {COLOR_PRIMARY}]")
        console.print("─" * 40)

        for speaker, pct in list(summary.speaker_distribution.items())[:5]:
            bar_len = int((pct / 100) * BAR_WIDTH)
            bar = "█" * bar_len + "░" * (BAR_WIDTH - bar_len)
            # truncate long speaker names.
            display_name = speaker[:20] if len(speaker) > 20 else speaker
            console.print(f"{display_name:<20} {bar} {pct:.1f}%")

    # content analysis.
    console.print(f"\n[bold {COLOR_PRIMARY}]CONTENT ANALYSIS[/bold {COLOR_PRIMARY}]")
    console.print("─" * 40)

    console.print(f"Keyword Matches: {summary.keyword_matches:,}")
    
    # keyword coverage (file-level).
    if summary.total_episodes > 0:
        keyword_pct = (summary.episodes_with_keywords / summary.total_episodes) * 100
        console.print(f"Episode Coverage: {summary.episodes_with_keywords}/{summary.total_episodes} ({keyword_pct:.0f}%)")

    if summary.top_categories:
        console.print("\nTop Categories:")
        for cat, count in list(summary.top_categories.items())[:5]:
            console.print(f"  • {cat}: {count:,} matches")
    
    # classification coverage (utterance-level).
    if summary.total_utterances > 0:
        classification_pct = (summary.classification_coverage / summary.total_utterances) * 100
        console.print(f"\n[bold cyan]Classification Coverage[/bold cyan]: {summary.classification_coverage:,}/{summary.total_utterances:,} non-empty utterances ({classification_pct:.0f}%)")
        console.print(f"[dim]  (utterances with all 8 classification models)[/dim]")
    
    # document-level labels.
    if summary.overall_document_label != "UNKNOWN":
        console.print(f"\n[bold {COLOR_PRIMARY}]DOCUMENT LABELS[/bold {COLOR_PRIMARY}]")
        console.print("─" * 40)
        
        # overall label with color coding.
        if summary.overall_document_label == "POSITIVE":
            label_color = COLOR_SUCCESS
        elif summary.overall_document_label == "NEGATIVE":
            label_color = COLOR_ERROR
        else:
            label_color = COLOR_WARNING
        
        console.print(f"Overall Label: [{label_color}]{summary.overall_document_label}[/{label_color}]")
        
        # episode breakdown.
        if summary.total_episodes > 0:
            pos_pct = (summary.positive_episodes / summary.total_episodes) * 100
            neg_pct = (summary.negative_episodes / summary.total_episodes) * 100
            console.print(f"\nEpisode Breakdown:")
            console.print(f"  • Positive: {summary.positive_episodes} ({pos_pct:.0f}%)")
            console.print(f"  • Negative: {summary.negative_episodes} ({neg_pct:.0f}%)")
            if summary.unknown_episodes > 0:
                console.print(f"  • Unknown: {summary.unknown_episodes}")
        
        # label statistics.
        console.print(f"\nLabel Statistics:")
        console.print(f"  • Positive labels: {summary.total_positive_labels:,}")
        console.print(f"  • Negative labels: {summary.total_negative_labels:,}")
        console.print(f"  • Segments analyzed: {summary.segments_analyzed_for_labels:,}")
        console.print(f"  • Ad segments skipped: {summary.ad_segments_skipped:,}")
        
        # top classification labels.
        if summary.top_classification_labels:
            console.print(f"\nTop Classification Labels:")
            for label, count in list(summary.top_classification_labels.items())[:5]:
                console.print(f"  • {label}: {count:,}")

    # llm annotations.
    if summary.episodes_analyzed > 0:
        console.print(
            f"\n[bold {COLOR_PRIMARY}]LLM ANNOTATIONS[/bold {COLOR_PRIMARY}] ({summary.episodes_analyzed} episodes)"
        )
        console.print("─" * 40)

        if summary.hate_speech_count > 0:
            console.print(
                f"[{COLOR_WARNING}]Hate Speech Detected:[/{COLOR_WARNING}] {summary.hate_speech_count} segments"
            )
        else:
            console.print(f"[{COLOR_SUCCESS}]Hate Speech Detected:[/{COLOR_SUCCESS}] 0 segments")

        console.print(f"Advertisements: {summary.ad_count} segments")

        if summary.target_groups:
            console.print("\nTop Target Groups:")
            for target, count in list(summary.target_groups.items())[:5]:
                console.print(f"  • {target}: {count}")


def compare_shows(show_names: list[str]) -> None:
    """Display comparison table for multiple shows."""
    console.print(
        Panel(
            f"[bold {COLOR_PRIMARY}]📊 Show Comparison[/bold {COLOR_PRIMARY}]",
            border_style="blue",
        )
    )

    summaries: list[ShowSummary] = []
    for show in show_names:
        summaries.append(generate_show_summary(show))

    # create comparison table.
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=18)

    for summary in summaries:
        table.add_column(summary.show_name[:12], justify="right", width=14)

    if len(summaries) > 1:
        table.add_column("Average", justify="right", width=14, style="yellow")

    # add rows.
    metrics = [
        ("Episodes", [s.total_episodes for s in summaries]),
        ("Utterances", [s.total_utterances for s in summaries]),
        ("Avg Duration", [f"{s.avg_duration_minutes:.0f}m" for s in summaries]),
        ("Words/Ep", [f"{s.avg_words_per_episode:.0f}" for s in summaries]),
        ("Keywords", [s.keyword_matches for s in summaries]),
        ("Keyword Cov%", [f"{(s.episodes_with_keywords / s.total_episodes * 100) if s.total_episodes > 0 else 0:.0f}%" for s in summaries]),
        ("Class Cov%", [f"{(s.classification_coverage / s.total_utterances * 100) if s.total_utterances > 0 else 0:.0f}%" for s in summaries]),
        ("Hate Speech", [s.hate_speech_count for s in summaries]),
        ("Ads", [s.ad_count for s in summaries]),
    ]

    for metric_name, values in metrics:
        row = [metric_name]
        for val in values:
            row.append(str(val))

        if len(summaries) > 1:
            # calculate average for numeric values.
            try:
                if metric_name in ("Episodes", "Utterances", "Keywords", "Hate Speech", "Ads"):
                    avg = sum(int(v) for v in values) / len(values)
                    row.append(f"{avg:.0f}")
                elif metric_name in ("Keyword Cov%", "Class Cov%"):
                    # extract numbers from percentage strings.
                    avg = sum(float(v.rstrip('%')) for v in values) / len(values)
                    row.append(f"{avg:.0f}%")
                else:
                    row.append("-")
            except (ValueError, TypeError):
                row.append("-")

        table.add_row(*row)

    console.print(table)


def get_problematic_content_summary() -> ProblematicContent:
    """Aggregate all problematic content across shows."""
    result = ProblematicContent()

    llm_base = OUTPUTS_PATH / LLM_ANNOTATIONS_DIR
    if not llm_base.exists():
        return result

    for show_dir in llm_base.iterdir():
        if not show_dir.is_dir():
            continue

        show_count = 0
        for file in show_dir.glob("*.json"):
            try:
                with open(file, encoding="utf-8") as f:
                    data = json.load(f)

                for seg in data.get("segments", []):
                    ann = seg.get("llm_annotation", {})
                    if ann.get("has_hate_speech"):
                        show_count += 1
                        result.total_hate_speech += 1

                        target = ann.get("target_group")
                        if target:
                            result.by_target_group[target] = (
                                result.by_target_group.get(target, 0) + 1
                            )

                    # check classifications.
                    for cls in seg.get("classifications", []):
                        label = cls.get("label", "")
                        if label and any(
                            flag in label.lower()
                            for flag in ["hate", "toxic", "hostile", "offensive"]
                        ):
                            result.classifier_flags[label] = (
                                result.classifier_flags.get(label, 0) + 1
                            )

            except (json.JSONDecodeError, OSError):
                pass

        if show_count > 0:
            result.by_show[show_dir.name] = show_count

    return result


def display_problematic_content_summary(content: ProblematicContent) -> None:
    """Display aggregated problematic content summary."""
    console.print(
        Panel(
            f"[bold {COLOR_WARNING}]⚠️ Problematic Content Summary[/bold {COLOR_WARNING}]",
            border_style="yellow",
        )
    )

    console.print("\n[bold]HATE SPEECH DETECTIONS (LLM Annotated)[/bold]")
    console.print("─" * 40)
    console.print(
        f"Total: {content.total_hate_speech} segments across {len(content.by_show)} shows\n"
    )

    if content.by_show:
        console.print("[bold]By Show:[/bold]")
        total = sum(content.by_show.values())
        for show, count in sorted(content.by_show.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total) * 100 if total > 0 else 0
            bar_len = int((pct / 100) * 30)
            bar = "█" * bar_len
            console.print(f"  {show[:20]:<20}: {count:>5} ({pct:.0f}%) {bar}")

    if content.by_target_group:
        console.print("\n[bold]By Target Group:[/bold]")
        for target, count in sorted(
            content.by_target_group.items(), key=lambda x: x[1], reverse=True
        )[:10]:
            pct = (count / content.total_hate_speech) * 100 if content.total_hate_speech > 0 else 0
            console.print(f"  {target}: {count} ({pct:.0f}%)")

    if content.classifier_flags:
        console.print("\n[bold]CLASSIFIER FLAGS[/bold]")
        console.print("─" * 40)
        for label, count in sorted(
            content.classifier_flags.items(), key=lambda x: x[1], reverse=True
        )[:10]:
            console.print(f"  {label}: {count} segments")


def export_show_summary_markdown(summary: ShowSummary, output_dir: Path) -> Path:
    """Export show summary to markdown file."""
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_file = output_dir / f"{summary.show_name}_summary_{timestamp}.md"

    classification_pct = (summary.classification_coverage / summary.total_utterances * 100) if summary.total_utterances > 0 else 0
    keyword_pct = (summary.episodes_with_keywords / summary.total_episodes * 100) if summary.total_episodes > 0 else 0

    lines = [
        f"# Show Summary: {summary.show_name}",
        f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## Overview\n",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Episodes | {summary.total_episodes:,} |",
        f"| Total Duration | {summary.total_duration_hours:.1f} hours |",
        f"| Total Words | {summary.total_words:,} |",
        f"| Total Utterances | {summary.total_utterances:,} |",
        f"| Avg Episode Duration | {summary.avg_duration_minutes:.0f} minutes |",
        f"| Avg Words/Episode | {summary.avg_words_per_episode:.0f} |",
        "\n## Content Analysis\n",
        f"- **Keyword Matches**: {summary.keyword_matches:,}",
        f"- **Episodes with Keywords**: {summary.episodes_with_keywords}/{summary.total_episodes} ({keyword_pct:.0f}%)",
        f"- **Classification Coverage**: {summary.classification_coverage:,}/{summary.total_utterances:,} ({classification_pct:.0f}%)",
        f"  - *Utterances with all 8 classification models*",
    ]

    if summary.top_categories:
        lines.append("\n### Top Categories\n")
        for cat, count in summary.top_categories.items():
            lines.append(f"- {cat}: {count:,}")
    
    # document labels section.
    if summary.overall_document_label != "UNKNOWN":
        pos_pct = (summary.positive_episodes / summary.total_episodes * 100) if summary.total_episodes > 0 else 0
        neg_pct = (summary.negative_episodes / summary.total_episodes * 100) if summary.total_episodes > 0 else 0
        
        lines.extend(
            [
                "\n## Document Labels\n",
                f"- **Overall Label**: {summary.overall_document_label}",
                f"- **Positive Episodes**: {summary.positive_episodes} ({pos_pct:.0f}%)",
                f"- **Negative Episodes**: {summary.negative_episodes} ({neg_pct:.0f}%)",
                f"- **Positive Labels**: {summary.total_positive_labels:,}",
                f"- **Negative Labels**: {summary.total_negative_labels:,}",
                f"- **Segments Analyzed**: {summary.segments_analyzed_for_labels:,}",
                f"- **Ad Segments Skipped**: {summary.ad_segments_skipped:,}",
            ]
        )
        
        if summary.top_classification_labels:
            lines.append("\n### Top Classification Labels\n")
            for label, count in list(summary.top_classification_labels.items())[:10]:
                lines.append(f"- {label}: {count:,}")

    if summary.episodes_analyzed > 0:
        lines.extend(
            [
                f"\n## LLM Annotations ({summary.episodes_analyzed} episodes)\n",
                f"- **Hate Speech**: {summary.hate_speech_count} segments",
                f"- **Advertisements**: {summary.ad_count} segments",
            ]
        )

        if summary.target_groups:
            lines.append("\n### Target Groups\n")
            for target, count in summary.target_groups.items():
                lines.append(f"- {target}: {count}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return output_file


def summary_generator_menu() -> None:
    """Interactive menu for summary generation."""
    while True:
        console.clear()
        console.print(
            Panel(
                f"[bold {COLOR_PRIMARY}]📈 Summary Generator[/bold {COLOR_PRIMARY}]\n"
                "[dim]Generate comprehensive summaries and reports[/dim]",
                border_style="blue",
            )
        )

        table = Table(show_header=False, box=None)
        table.add_column("Option", style=COLOR_WARNING)
        table.add_column("Description")

        table.add_row("1", "📊 Single show summary")
        table.add_row("2", "📈 Compare multiple shows")
        table.add_row("3", "⚠️  Problematic content summary")
        table.add_row("4", "📤 Export summary to markdown")
        table.add_row(KEY_BACK, "⬅️  Back")

        console.print(table)
        console.print()

        choice = Prompt.ask(f"[{COLOR_WARNING}]Select option[/{COLOR_WARNING}]", default=KEY_BACK)

        match choice.lower():
            case "1":
                _single_show_summary()
            case "2":
                _compare_shows_menu()
            case "3":
                _problematic_content_menu()
            case "4":
                _export_summary_menu()
            case _ if choice.lower() == KEY_BACK:
                break


def _get_available_shows() -> list[str]:
    """Get list of available shows."""
    transcripts_path = OUTPUTS_PATH / TRANSCRIPTS_DIR
    if not transcripts_path.exists():
        return []
    return sorted([d.name for d in transcripts_path.iterdir() if d.is_dir()])


def _single_show_summary() -> None:
    """Display summary for a single show."""
    console.clear()
    shows = _get_available_shows()

    if not shows:
        console.print(f"[{COLOR_WARNING}]No shows found[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    console.print(f"[bold {COLOR_PRIMARY}]Select Show[/bold {COLOR_PRIMARY}]\n")
    for i, show in enumerate(shows, 1):
        console.print(f"  {i}. {show}")

    choice = Prompt.ask(
        f"\n[{COLOR_WARNING}]Select show number (or '{KEY_BACK}' to go back)[/{COLOR_WARNING}]",
        default=KEY_BACK,
    )

    if choice.lower() == KEY_BACK:
        return

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(shows):
            console.clear()
            summary = generate_show_summary(shows[idx])
            display_show_summary(summary)
            prompt_and_save(console, screen_name=f"summary_{shows[idx]}")
        else:
            console.print(f"[{COLOR_ERROR}]Invalid selection[/{COLOR_ERROR}]")
    except ValueError:
        console.print(f"[{COLOR_ERROR}]Invalid input[/{COLOR_ERROR}]")

    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def _compare_shows_menu() -> None:
    """Compare multiple shows."""
    console.clear()
    shows = _get_available_shows()

    if len(shows) < 2:
        console.print(f"[{COLOR_WARNING}]Need at least 2 shows to compare[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    console.print(f"[bold {COLOR_PRIMARY}]Select Shows to Compare[/bold {COLOR_PRIMARY}]\n")
    for i, show in enumerate(shows, 1):
        console.print(f"  {i}. {show}")

    console.print("\n[dim]Enter numbers separated by commas (e.g., 1,2,3)[/dim]")

    choice = Prompt.ask(
        f"\n[{COLOR_WARNING}]Select shows (or '{KEY_BACK}' to go back)[/{COLOR_WARNING}]",
        default=KEY_BACK,
    )

    if choice.lower() == KEY_BACK:
        return

    try:
        indices = [int(x.strip()) - 1 for x in choice.split(",")]
        selected_shows = [shows[i] for i in indices if 0 <= i < len(shows)]

        if len(selected_shows) >= 2:
            console.clear()
            compare_shows(selected_shows)
            prompt_and_save(console, screen_name="show_comparison")
        else:
            console.print(f"[{COLOR_ERROR}]Please select at least 2 shows[/{COLOR_ERROR}]")

    except (ValueError, IndexError):
        console.print(f"[{COLOR_ERROR}]Invalid input[/{COLOR_ERROR}]")

    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def _problematic_content_menu() -> None:
    """Display problematic content summary."""
    console.clear()
    content = get_problematic_content_summary()
    display_problematic_content_summary(content)
    prompt_and_save(console, screen_name="problematic_content_summary")


def _export_summary_menu() -> None:
    """Export summary to markdown."""
    console.clear()
    shows = _get_available_shows()

    if not shows:
        console.print(f"[{COLOR_WARNING}]No shows found[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    console.print(f"[bold {COLOR_PRIMARY}]Export Show Summary[/bold {COLOR_PRIMARY}]\n")
    for i, show in enumerate(shows, 1):
        console.print(f"  {i}. {show}")

    choice = Prompt.ask(
        f"\n[{COLOR_WARNING}]Select show number (or '{KEY_BACK}' to go back)[/{COLOR_WARNING}]",
        default=KEY_BACK,
    )

    if choice.lower() == KEY_BACK:
        return

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(shows):
            summary = generate_show_summary(shows[idx])
            export_dir = OUTPUTS_PATH / "terminal_viewer" / "exports"
            output_file = export_show_summary_markdown(summary, export_dir)
            console.print(
                f"\n[{COLOR_SUCCESS}]✓ Summary exported to: {output_file}[/{COLOR_SUCCESS}]"
            )
        else:
            console.print(f"[{COLOR_ERROR}]Invalid selection[/{COLOR_ERROR}]")
    except ValueError:
        console.print(f"[{COLOR_ERROR}]Invalid input[/{COLOR_ERROR}]")

    Prompt.ask("\n[dim]Press Enter to continue[/dim]")
