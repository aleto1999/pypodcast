"""Research report generation module."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table

from podcast_conversations.naming import sanitize_name
from terminal_data_visualizer.config import (
    COLOR_ERROR,
    COLOR_PRIMARY,
    COLOR_SUCCESS,
    COLOR_WARNING,
    DOCUMENT_LABELS_DIR,
    KEY_BACK,
    OUTPUTS_PATH,
)
from terminal_data_visualizer.document_label_parser import parse_global_document_labels
from terminal_data_visualizer.report_visualizations import (
    create_category_distribution_chart,
    create_label_distribution_chart,
    create_speaker_distribution_chart,
    export_console_to_svg,
)
from terminal_data_visualizer.summary_generator import (
    ShowSummary,
    generate_show_summary,
    get_problematic_content_summary,
)

console = Console(record=True)

# report output directory.
REPORTS_DIR = OUTPUTS_PATH / "terminal_viewer" / "reports"


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    title: str
    shows: list[str]
    report_type: str  # "comprehensive", "summary", "hate_speech", "custom"
    format: str  # "markdown", "html", "json"
    include_sections: list[str]


def generate_comprehensive_report(
    shows: list[str],
    title: str = "Podcast Analysis Report",
    format: str = "markdown",
) -> Path:
    """Generate a comprehensive research report."""
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    # ensure output directory exists.
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if format == "markdown":
        output_file = REPORTS_DIR / f"report_{timestamp}.md"
        _generate_markdown_report(shows, title, output_file)
    elif format == "json":
        output_file = REPORTS_DIR / f"report_{timestamp}.json"
        _generate_json_report(shows, title, output_file)
    else:
        output_file = REPORTS_DIR / f"report_{timestamp}.md"
        _generate_markdown_report(shows, title, output_file)

    return output_file


def _generate_markdown_report(shows: list[str], title: str, output_file: Path) -> None:
    """Generate markdown report with embedded visualizations."""
    lines: list[str] = []

    # create visualizations directory next to report.
    viz_dir = output_file.parent / f"{output_file.stem}_visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # header.
    lines.extend(
        [
            f"# {title}",
            f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n**Shows Analyzed**: {len(shows)}",
            "\n---\n",
        ]
    )

    total_episodes = 0
    total_duration = 0.0
    total_words = 0
    total_hate_speech = 0
    total_keywords_matched = 0
    total_utterances = 0
    total_classified_utterances = 0
    total_ads = 0
    total_episodes_with_keywords = 0

    # aggregate speaker data across all shows.
    global_speaker_words: dict[str, int] = {}

    summaries: list[ShowSummary] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"Analyzing {len(shows)} shows...", total=len(shows))

        for show in shows:
            summary = generate_show_summary(show)
            summaries.append(summary)

            total_episodes += summary.total_episodes
            total_duration += summary.total_duration_hours
            total_words += summary.total_words
            total_hate_speech += summary.hate_speech_count
            total_keywords_matched += summary.keyword_matches
            total_utterances += summary.total_utterances
            total_classified_utterances += summary.classification_coverage
            total_ads += summary.ad_count
            total_episodes_with_keywords += summary.episodes_with_keywords

            # aggregate speaker data (weight by total words in show).
            for speaker, pct in summary.speaker_distribution.items():
                words_for_speaker = int((pct / 100) * summary.total_words)
                global_speaker_words[speaker] = global_speaker_words.get(speaker, 0) + words_for_speaker

            progress.advance(task)

    # corpus overview section.
    lines.extend(
        [
            "## Corpus Overview\n",
            f"This report analyzes **{len(shows)} podcast shows** comprising **{total_episodes:,} episodes** covering political discourse, news commentary, and social issues.",
            f"The corpus contains **{total_duration:.1f} hours** ({total_duration / 24:.1f} days) of audio content with over **{total_words:,} words** across **{total_utterances:,} utterances**.\n",
            "### Pipeline Coverage\n",
            "| Pipeline | Status | Coverage |",
            "|----------|--------|----------|",
            f"| Transcription | ✓ Complete | {total_episodes:,}/{total_episodes:,} episodes (100%) |",
            f"| Keyword Analysis | ✓ Complete | {total_episodes_with_keywords:,}/{total_episodes:,} episodes ({(total_episodes_with_keywords / total_episodes * 100) if total_episodes > 0 else 0:.1f}%) |",
            f"| Classification (8 models) | {'✓ Complete' if total_classified_utterances == total_utterances else '⚠ Partial'} | {total_classified_utterances:,}/{total_utterances:,} utterances ({(total_classified_utterances / total_utterances * 100) if total_utterances > 0 else 0:.0f}%) |",
            "| Document Labels | ✓ Complete | See below |",
            f"| LLM Annotations | {'✓ Complete' if total_hate_speech > 0 else '✗ Not Run'} | {total_hate_speech:,} episodes |",
            "\n---\n",
            "## Executive Summary\n",
            "### Volume Metrics\n",
            f"- **Total Episodes**: {total_episodes:,}",
            f"- **Total Duration**: {total_duration:.1f} hours ({total_duration / 24:.1f} days)",
            f"- **Total Words**: {total_words:,}",
            f"- **Total Utterances**: {total_utterances:,}",
            f"- **Avg Words per Episode**: {total_words / total_episodes if total_episodes > 0 else 0:,.0f}",
            "\n### Pipeline Results\n",
            f"- **Keyword Matches**: {total_keywords_matched:,}",
            f"- **Episodes with Keywords**: {total_episodes_with_keywords:,}/{total_episodes:,} ({(total_episodes_with_keywords / total_episodes * 100) if total_episodes > 0 else 0:.1f}%)",
            f"- **Classification Coverage**: {total_classified_utterances:,}/{total_utterances:,} non-empty utterances ({(total_classified_utterances / total_utterances * 100) if total_utterances > 0 else 0:.0f}%)",
            f"- **Ad Segments Detected (ML Model)**: {total_ads:,} ({(total_ads / total_utterances * 100) if total_utterances > 0 else 0:.1f}% of utterances)",
            f"- **Hate Speech Flags (LLM)**: {total_hate_speech:,}",
            "\n",
        ]
    )

    # try to load global document labels.
    global_labels_path = OUTPUTS_PATH / DOCUMENT_LABELS_DIR / "global_summary.md"
    global_doc_labels = parse_global_document_labels(global_labels_path)

    if global_doc_labels:
        lines.extend(
            [
                "### Document-Level Labels\n",
                f"- **Total Shows**: {global_doc_labels.total_shows} ({global_doc_labels.positive_shows} positive, {global_doc_labels.negative_shows} negative)",
                f"- **Positive Episodes**: {global_doc_labels.positive_episodes:,} ({(global_doc_labels.positive_episodes / global_doc_labels.total_episodes * 100) if global_doc_labels.total_episodes > 0 else 0:.1f}%)",
                f"- **Negative Episodes**: {global_doc_labels.negative_episodes:,} ({(global_doc_labels.negative_episodes / global_doc_labels.total_episodes * 100) if global_doc_labels.total_episodes > 0 else 0:.1f}%)",
                f"- **Total Positive Labels**: {global_doc_labels.total_positive_labels:,}",
                f"- **Total Negative Labels**: {global_doc_labels.total_negative_labels:,}",
                f"- **Segments Analyzed**: {global_doc_labels.total_segments_analyzed:,}",
                f"- **Ad Segments Skipped**: {global_doc_labels.total_ad_segments_skipped:,}",
                "\n",
            ]
        )

    # key findings section with show rankings.
    lines.extend(
        [
            "### Key Findings\n",
            "#### Top Shows by Volume\n",
        ]
    )

    # create show rankings.
    shows_by_episodes = sorted(summaries, key=lambda s: s.total_episodes, reverse=True)[:5]
    shows_by_duration = sorted(summaries, key=lambda s: s.total_duration_hours, reverse=True)[:5]
    shows_by_keywords = sorted(summaries, key=lambda s: s.keyword_matches, reverse=True)[:5]

    lines.append("\n**Most Episodes:**\n")
    for i, show in enumerate(shows_by_episodes, 1):
        lines.append(f"{i}. {show.show_name}: {show.total_episodes:,} episodes")

    lines.append("\n**Longest Duration:**\n")
    for i, show in enumerate(shows_by_duration, 1):
        lines.append(f"{i}. {show.show_name}: {show.total_duration_hours:.1f} hours ({show.total_duration_hours / 24:.1f} days)")

    lines.append("\n**Most Keyword Matches:**\n")
    for i, show in enumerate(shows_by_keywords, 1):
        lines.append(f"{i}. {show.show_name}: {show.keyword_matches:,} matches")

    # add dominant classification label if available.
    if global_doc_labels and global_doc_labels.label_distribution:
        lines.append("\n#### Content Patterns\n")
        all_labels: dict[str, int] = {}
        for model_name, label_dist in global_doc_labels.label_distribution.items():
            for label, count in label_dist.items():
                if label not in ("LABEL_0", "NOT-HATE", "SAFE", "Appropriate"):
                    all_labels[label] = all_labels.get(label, 0) + count

        if all_labels:
            top_label, top_count = max(all_labels.items(), key=lambda x: x[1])
            top_pct = (top_count / global_doc_labels.total_segments_analyzed * 100) if global_doc_labels.total_segments_analyzed > 0 else 0
            lines.append(f"- **Most Common Classification Label**: {top_label} ({top_count:,} occurrences, {top_pct:.1f}% of analyzed segments)")

        lines.append(f"- **Overall Document Sentiment**: {global_doc_labels.positive_shows} shows positive, {global_doc_labels.negative_shows} shows negative")
        lines.append(f"- **Ad Content Detection**: {total_ads:,} ad segments detected ({(total_ads / total_utterances * 100) if total_utterances > 0 else 0:.1f}% of corpus)")

    lines.append("\n---\n")
    lines.append("## Global Analysis\n")

    # global speaker distribution.
    if global_speaker_words:
        total_global_words = sum(global_speaker_words.values())
        global_speaker_pct = {
            speaker: (words / total_global_words) * 100
            for speaker, words in global_speaker_words.items()
        }
        top_speakers = sorted(global_speaker_pct.items(), key=lambda x: x[1], reverse=True)[:10]

        lines.extend(
            [
                "### Global Speaker Distribution\n",
            ]
        )
        for speaker, pct in top_speakers:
            lines.append(f"- {speaker}: {pct:.1f}% ({global_speaker_words[speaker]:,} words)")
        lines.append("\n")

        # create and save speaker distribution visualization.
        speaker_chart = create_speaker_distribution_chart(
            dict(top_speakers),
            title="Figure 1: Global Speaker Distribution (Top 10)",
            max_speakers=10,
        )
        speaker_viz_path = viz_dir / "global_speaker_distribution.svg"
        export_console_to_svg(speaker_chart, speaker_viz_path)

        lines.extend(
            [
                f"![Figure 1: Global Speaker Distribution - Bar chart showing top 10 speakers by word count percentage across the entire corpus]({viz_dir.name}/global_speaker_distribution.svg)\n",
                "*Figure 1: Distribution of speaking time across the top 10 most frequent speakers in the corpus. Percentages represent proportion of total words spoken.*\n",
                "\n",
            ]
        )

    # global top classification labels (if document labels exist).
    if global_doc_labels and global_doc_labels.label_distribution:
        lines.extend(
            [
                "### Global Classification Label Distribution\n",
            ]
        )

        # aggregate all labels excluding safe ones.
        all_global_labels: dict[str, int] = {}
        for model_name, label_dist in global_doc_labels.label_distribution.items():
            for label, count in label_dist.items():
                if label not in ("LABEL_0", "NOT-HATE", "SAFE", "Appropriate"):
                    all_global_labels[label] = all_global_labels.get(label, 0) + count

        top_global_labels = sorted(all_global_labels.items(), key=lambda x: x[1], reverse=True)[:15]
        for label, count in top_global_labels:
            pct = (count / global_doc_labels.total_segments_analyzed * 100) if global_doc_labels.total_segments_analyzed > 0 else 0
            lines.append(f"- {label}: {count:,} ({pct:.1f}% of analyzed segments)")
        lines.append("\n")

        # create and save label distribution visualization.
        label_chart = create_label_distribution_chart(
            dict(all_global_labels),
            total_segments=global_doc_labels.total_segments_analyzed,
            title="Figure 2: Global Classification Label Distribution (Top 15)",
            max_labels=15,
        )
        label_viz_path = viz_dir / "global_label_distribution.svg"
        export_console_to_svg(label_chart, label_viz_path)

        lines.extend(
            [
                f"![Figure 2: Global Classification Label Distribution - Bar chart showing top 15 classification labels by occurrence count across all analyzed segments]({viz_dir.name}/global_label_distribution.svg)\n",
                "*Figure 2: Most common classification labels detected by the 8 ML models. Safe labels (LABEL_0, NOT-HATE, SAFE, Appropriate) are excluded for clarity. Percentages calculated against total segments analyzed.*\n",
                "\n",
            ]
        )

    lines.append("---\n")

    # per-show summaries.
    lines.extend(
        [
            "## Show Summaries\n",
        ]
    )

    for summary in summaries:
        classification_pct = (summary.classification_coverage / summary.total_utterances * 100) if summary.total_utterances > 0 else 0
        keyword_pct = (summary.episodes_with_keywords / summary.total_episodes * 100) if summary.total_episodes > 0 else 0

        lines.extend(
            [
                f"### {summary.show_name}\n",
                "#### Overview\n",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Episodes | {summary.total_episodes:,} |",
                f"| Duration | {summary.total_duration_hours:.1f} hours |",
                f"| Total Words | {summary.total_words:,} |",
                f"| Total Utterances | {summary.total_utterances:,} |",
                f"| Avg Episode Duration | {summary.avg_duration_minutes:.0f} minutes |",
                f"| Keyword Matches | {summary.keyword_matches:,} |",
                f"| Episodes with Keywords | {summary.episodes_with_keywords}/{summary.total_episodes} ({keyword_pct:.0f}%) |",
                f"| Classification Coverage | {summary.classification_coverage:,}/{summary.total_utterances:,} non-empty ({classification_pct:.0f}%) |",
                f"| Hate Speech Flags (LLM) | {summary.hate_speech_count:,} |",
                f"| Ad Segments (ML Model) | {summary.ad_count:,} |",
                "\n",
            ]
        )

        # document labels section.
        if summary.overall_document_label != "UNKNOWN":
            pos_pct = (summary.positive_episodes / summary.total_episodes * 100) if summary.total_episodes > 0 else 0
            neg_pct = (summary.negative_episodes / summary.total_episodes * 100) if summary.total_episodes > 0 else 0

            lines.extend(
                [
                    "#### Document-Level Assessment\n",
                    f"- **Overall Label**: {summary.overall_document_label}",
                    f"- **Positive Episodes**: {summary.positive_episodes} ({pos_pct:.0f}%)",
                    f"- **Negative Episodes**: {summary.negative_episodes} ({neg_pct:.0f}%)",
                    f"- **Positive Labels**: {summary.total_positive_labels:,}",
                    f"- **Negative Labels**: {summary.total_negative_labels:,}",
                    "\n",
                ]
            )

        if summary.top_categories:
            lines.extend(
                [
                    "#### Content Analysis - Keywords\n",
                ]
            )
            for cat, count in list(summary.top_categories.items())[:5]:
                lines.append(f"- {cat}: {count:,} matches")
            lines.append("\n")

            # create keyword category visualization.
            if len(summary.top_categories) >= 3:
                cat_chart = create_category_distribution_chart(
                    summary.top_categories,
                    title=f"{summary.show_name} - Top Keyword Categories",
                    max_categories=10,
                )
                safe_show_name = sanitize_name(summary.show_name)
                cat_viz_path = viz_dir / f"{safe_show_name}_keywords.svg"
                export_console_to_svg(cat_chart, cat_viz_path)

                lines.extend(
                    [
                        f"![{summary.show_name} - Keyword category distribution showing top 10 categories by match count]({viz_dir.name}/{safe_show_name}_keywords.svg)\n",
                        f"*Keyword category distribution for {summary.show_name}. Shows top categories by total match count.*\n",
                        "\n",
                    ]
                )

        # speaker distribution.
        if summary.speaker_distribution:
            lines.extend(
                [
                    "#### Speaker Analysis\n",
                ]
            )
            for speaker, pct in list(summary.speaker_distribution.items())[:5]:
                lines.append(f"- {speaker}: {pct:.1f}% of total words")
            lines.append("\n")

            # create speaker visualization.
            if len(summary.speaker_distribution) >= 2:
                speaker_chart = create_speaker_distribution_chart(
                    summary.speaker_distribution,
                    title=f"{summary.show_name} - Speaker Distribution",
                    max_speakers=8,
                )
                safe_show_name = sanitize_name(summary.show_name)
                speaker_viz_path = viz_dir / f"{safe_show_name}_speakers.svg"
                export_console_to_svg(speaker_chart, speaker_viz_path)

                lines.extend(
                    [
                        f"![{summary.show_name} - Speaker distribution showing top 8 speakers by word count percentage]({viz_dir.name}/{safe_show_name}_speakers.svg)\n",
                        f"*Speaker distribution for {summary.show_name}. Percentages represent proportion of total words spoken.*\n",
                        "\n",
                    ]
                )

        # top classification labels from document labels.
        if summary.top_classification_labels:
            lines.extend(
                [
                    "#### Top Classification Labels\n",
                ]
            )
            for label, count in list(summary.top_classification_labels.items())[:8]:
                lines.append(f"- {label}: {count:,}")
            lines.append("\n")

        # llm annotations (if available).
        if summary.target_groups:
            lines.extend(
                [
                    "#### LLM-Detected Target Groups\n",
                ]
            )
            for target, count in list(summary.target_groups.items())[:5]:
                lines.append(f"- {target}: {count} segments")
            lines.append("\n")

    # comparison table.
    if len(summaries) > 1:
        lines.extend(
            [
                "---\n",
                "## Show Comparison\n",
                "| Show | Episodes | Duration (h) | Words | Utterances | Keywords | Classifications | Doc Label | Positive Eps | Negative Eps | Hate Speech | Ads |",
                "|------|----------|--------------|-------|------------|----------|-----------------|-----------|--------------|--------------|-------------|-----|",
            ]
        )

        for summary in summaries:
            classification_pct = (summary.classification_coverage / summary.total_utterances * 100) if summary.total_utterances > 0 else 0
            keyword_pct = (summary.episodes_with_keywords / summary.total_episodes * 100) if summary.total_episodes > 0 else 0
            pos_pct = (summary.positive_episodes / summary.total_episodes * 100) if summary.total_episodes > 0 else 0
            neg_pct = (summary.negative_episodes / summary.total_episodes * 100) if summary.total_episodes > 0 else 0

            lines.append(
                f"| {summary.show_name} | {summary.total_episodes} | "
                f"{summary.total_duration_hours:.1f} | {summary.total_words:,} | "
                f"{summary.total_utterances:,} | {keyword_pct:.0f}% | "
                f"{classification_pct:.0f}% | {summary.overall_document_label} | "
                f"{pos_pct:.0f}% | {neg_pct:.0f}% | "
                f"{summary.hate_speech_count} | {summary.ad_count} |"
            )

        lines.append("\n")

    # problematic content section.
    problematic = get_problematic_content_summary()

    if problematic.total_hate_speech > 0:
        lines.extend(
            [
                "---\n",
                "## Problematic Content Analysis\n",
                f"**Total Hate Speech Detections**: {problematic.total_hate_speech}\n",
            ]
        )

        if problematic.by_show:
            lines.extend(
                [
                    "### By Show\n",
                    "| Show | Count | Percentage |",
                    "|------|-------|------------|",
                ]
            )
            total = sum(problematic.by_show.values())
            for show, count in sorted(
                problematic.by_show.items(), key=lambda x: x[1], reverse=True
            ):
                pct = (count / total) * 100 if total > 0 else 0
                lines.append(f"| {show} | {count} | {pct:.1f}% |")
            lines.append("\n")

        if problematic.by_target_group:
            lines.extend(
                [
                    "### By Target Group\n",
                    "| Target | Count | Percentage |",
                    "|--------|-------|------------|",
                ]
            )
            total = sum(problematic.by_target_group.values())
            for target, count in sorted(
                problematic.by_target_group.items(), key=lambda x: x[1], reverse=True
            )[:10]:
                pct = (count / total) * 100 if total > 0 else 0
                lines.append(f"| {target} | {count} | {pct:.1f}% |")
            lines.append("\n")

    # methodology section.
    lines.extend(
        [
            "---\n",
            "## Methodology\n",
            "### Data Sources\n",
            "This report analyzes podcast data through multiple processing pipelines:\n",
            "1. **Audio Collection**: RSS feeds harvested and downloaded",
            "2. **Transcription**: Speaker-diarized transcripts with timestamps and word counts",
            "3. **Keyword Analysis**: Pattern matching against dehumanization taxonomy",
            "4. **Classification**: 8 ML models for hate speech and content detection",
            "5. **Document Labels**: Episode-level aggregated sentiment (positive/negative)",
            "6. **LLM Annotations**: Optional GPT-based hate speech detection\n",
            "\n### Pipeline Details\n",
            "**Keyword Analysis (File-Level)**",
            "- Matches transcript segments against keyword patterns in dehumanization categories",
            "- Coverage: Episodes with at least one keyword match",
            "- Formula: `(episodes_with_keywords / total_episodes) × 100`\n",
            "**Classification (Utterance-Level)**",
            "- Applies 8 ML models to each utterance/segment",
            "- Coverage: Utterances analyzed by all 8 models",
            "- Formula: `(utterances_with_8_models / total_utterances) × 100`\n",
            "**Document Labels (Segment-Level)**",
            "- Aggregates classification results into episode-level labels",
            "- Positive/negative classification based on label threshold",
            "- Excludes ad segments from analysis\n",
            "\n### Ad Detection\n",
            "- **ML Model**: `ad_content_detection` classifier (LABEL_1 = advertisement)",
            "- **LLM Annotations**: GPT `has_advertisement` field (if pipeline was run)",
            "- **Reporting Priority**: ML model results used when available\n",
            "\n### Classification Models\n",
            "Each utterance is analyzed by 8 models:",
            "1. **hate_speech_detection** - Binary hate speech classification",
            "2. **fine_grained_hate_speech_detection** - Multi-class hate categories",
            "3. **hate_against_minorities** - Detects bias against minority groups",
            "4. **hostile_content** - Identifies hostile/aggressive language",
            "5. **ad_content_detection** - Advertisement vs. content classification",
            "6. **multilingual_hate_speech** - Cross-lingual hate speech detection",
            "7. **hate_speech_multilabel_bert** - Multi-label hate taxonomy",
            "8. **beto_contextualized_hate_speech** - Contextualized Spanish hate speech\n",
            "\n### Data Quality Notes\n",
            "- **Transcription**: 100% coverage across all episodes",
            f"- **Keyword Analysis**: {(total_episodes_with_keywords / total_episodes * 100) if total_episodes > 0 else 0:.1f}% of episodes contain keyword matches",
            f"- **Classification**: {(total_classified_utterances / total_utterances * 100) if total_utterances > 0 else 0:.0f}% non-empty utterance coverage",
            "- **Empty Segments**: Segments without text (silence, pauses) are excluded from classification metrics",
            f"- **LLM Annotations**: {'Complete' if total_hate_speech > 0 else 'Not yet run on this corpus'}",
            "- **Safe Labels**: Excluded from label distribution charts (LABEL_0, NOT-HATE, SAFE, Appropriate)",
            "\n---\n",
            f"*Report generated by Terminal Data Visualizer on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*",
        ]
    )

    # write file.
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _generate_json_report(shows: list[str], title: str, output_file: Path) -> None:
    """Generate JSON report."""
    report_data: dict = {
        "title": title,
        "generated_at": datetime.now().isoformat(),
        "shows_analyzed": len(shows),
        "summaries": [],
        "totals": {
            "episodes": 0,
            "duration_hours": 0.0,
            "words": 0,
            "utterances": 0,
            "hate_speech": 0,
            "ads": 0,
            "keywords": 0,
            "episodes_with_keywords": 0,
        },
        "global_speakers": {},
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"Analyzing {len(shows)} shows...", total=len(shows))

        for show in shows:
            summary = generate_show_summary(show)

            summary_dict = {
                "show_name": summary.show_name,
                "total_episodes": summary.total_episodes,
                "total_duration_hours": summary.total_duration_hours,
                "total_words": summary.total_words,
                "total_segments": summary.total_segments,
                "total_utterances": summary.total_utterances,
                "avg_duration_minutes": summary.avg_duration_minutes,
                "avg_words_per_episode": summary.avg_words_per_episode,
                "keyword_matches": summary.keyword_matches,
                "episodes_with_keywords": summary.episodes_with_keywords,
                "keyword_coverage_percent": (summary.episodes_with_keywords / summary.total_episodes * 100) if summary.total_episodes > 0 else 0,
                "classification_coverage": summary.classification_coverage,
                "classification_coverage_percent": (summary.classification_coverage / summary.total_utterances * 100) if summary.total_utterances > 0 else 0,
                "hate_speech_count": summary.hate_speech_count,
                "ad_count": summary.ad_count,
                "top_categories": summary.top_categories,
                "target_groups": summary.target_groups,
                "speaker_distribution": summary.speaker_distribution,
                # document labels.
                "overall_document_label": summary.overall_document_label,
                "positive_episodes": summary.positive_episodes,
                "negative_episodes": summary.negative_episodes,
                "unknown_episodes": summary.unknown_episodes,
                "total_positive_labels": summary.total_positive_labels,
                "total_negative_labels": summary.total_negative_labels,
                "segments_analyzed_for_labels": summary.segments_analyzed_for_labels,
                "ad_segments_skipped": summary.ad_segments_skipped,
                "top_classification_labels": summary.top_classification_labels,
            }

            report_data["summaries"].append(summary_dict)

            report_data["totals"]["episodes"] += summary.total_episodes
            report_data["totals"]["duration_hours"] += summary.total_duration_hours
            report_data["totals"]["words"] += summary.total_words
            report_data["totals"]["utterances"] += summary.total_utterances
            report_data["totals"]["hate_speech"] += summary.hate_speech_count
            report_data["totals"]["ads"] += summary.ad_count
            report_data["totals"]["keywords"] += summary.keyword_matches
            report_data["totals"]["episodes_with_keywords"] += summary.episodes_with_keywords

            # aggregate speaker data.
            for speaker, pct in summary.speaker_distribution.items():
                words_for_speaker = int((pct / 100) * summary.total_words)
                if speaker not in report_data["global_speakers"]:
                    report_data["global_speakers"][speaker] = 0
                report_data["global_speakers"][speaker] += words_for_speaker

            progress.advance(task)

    # calculate global speaker percentages.
    if report_data["global_speakers"] and report_data["totals"]["words"] > 0:
        total_words = report_data["totals"]["words"]
        report_data["global_speaker_percentages"] = {
            speaker: (words / total_words) * 100
            for speaker, words in report_data["global_speakers"].items()
        }

    # add problematic content.
    problematic = get_problematic_content_summary()
    report_data["problematic_content"] = {
        "total_hate_speech": problematic.total_hate_speech,
        "by_show": problematic.by_show,
        "by_target_group": problematic.by_target_group,
        "classifier_flags": problematic.classifier_flags,
    }

    # add global document labels if available.
    global_labels_path = OUTPUTS_PATH / DOCUMENT_LABELS_DIR / "global_summary.md"
    global_doc_labels = parse_global_document_labels(global_labels_path)
    if global_doc_labels:
        report_data["document_labels"] = {
            "total_shows": global_doc_labels.total_shows,
            "positive_shows": global_doc_labels.positive_shows,
            "negative_shows": global_doc_labels.negative_shows,
            "unknown_shows": global_doc_labels.unknown_shows,
            "total_episodes": global_doc_labels.total_episodes,
            "positive_episodes": global_doc_labels.positive_episodes,
            "negative_episodes": global_doc_labels.negative_episodes,
            "total_segments_analyzed": global_doc_labels.total_segments_analyzed,
            "total_ad_segments_skipped": global_doc_labels.total_ad_segments_skipped,
            "total_positive_labels": global_doc_labels.total_positive_labels,
            "total_negative_labels": global_doc_labels.total_negative_labels,
        }

    # write file.
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)


def generate_hate_speech_report(shows: list[str]) -> Path:
    """Generate a focused hate speech analysis report."""
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = REPORTS_DIR / f"hate_speech_report_{timestamp}.md"

    lines: list[str] = [
        "# Hate Speech Analysis Report",
        f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\n**Shows Analyzed**: {', '.join(shows)}",
        "\n---\n",
    ]

    # get problematic content.
    problematic = get_problematic_content_summary()

    lines.extend(
        [
            "## Summary\n",
            f"- **Total Hate Speech Detections**: {problematic.total_hate_speech}",
            f"- **Shows with Detections**: {len(problematic.by_show)}",
            f"- **Unique Target Groups**: {len(problematic.by_target_group)}",
            "\n---\n",
        ]
    )

    # by show breakdown.
    if problematic.by_show:
        lines.extend(
            [
                "## Breakdown by Show\n",
                "| Show | Hate Speech Count | Percentage |",
                "|------|-------------------|------------|",
            ]
        )

        total = sum(problematic.by_show.values())
        for show, count in sorted(problematic.by_show.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total) * 100 if total > 0 else 0
            lines.append(f"| {show} | {count} | {pct:.1f}% |")

        lines.append("\n---\n")

    # by target group.
    if problematic.by_target_group:
        lines.extend(
            [
                "## Target Groups\n",
                "| Target Group | Count | Percentage |",
                "|--------------|-------|------------|",
            ]
        )

        total = sum(problematic.by_target_group.values())
        for target, count in sorted(
            problematic.by_target_group.items(), key=lambda x: x[1], reverse=True
        ):
            pct = (count / total) * 100 if total > 0 else 0
            lines.append(f"| {target} | {count} | {pct:.1f}% |")

        lines.append("\n---\n")

    # classifier flags.
    if problematic.classifier_flags:
        lines.extend(
            [
                "## Classifier Flags\n",
                "| Classification | Count |",
                "|----------------|-------|",
            ]
        )

        for label, count in sorted(
            problematic.classifier_flags.items(), key=lambda x: x[1], reverse=True
        ):
            lines.append(f"| {label} | {count} |")

        lines.append("\n")

    lines.extend(
        [
            "---\n",
            f"*Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*",
        ]
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return output_file


def report_generator_menu() -> None:
    """Interactive report generator menu."""
    while True:
        console.clear()
        console.print(
            Panel(
                f"[bold {COLOR_PRIMARY}]📝 Report Generator[/bold {COLOR_PRIMARY}]\n"
                "[dim]Generate publication-ready reports[/dim]",
                border_style="blue",
            )
        )

        table = Table(show_header=False, box=None)
        table.add_column("Option", style=COLOR_WARNING)
        table.add_column("Description")

        table.add_row("1", "📊 Comprehensive report (all shows)")
        table.add_row("2", "📈 Single show report")
        table.add_row("3", "⚠️  Hate speech focused report")
        table.add_row("4", "📤 Export as JSON")
        table.add_row("5", "📂 View existing reports")
        table.add_row(KEY_BACK, "⬅️  Back")

        console.print(table)
        console.print()

        choice = Prompt.ask(f"[{COLOR_WARNING}]Select option[/{COLOR_WARNING}]", default=KEY_BACK)

        match choice.lower():
            case "1":
                _generate_comprehensive()
            case "2":
                _generate_single_show()
            case "3":
                _generate_hate_speech()
            case "4":
                _generate_json()
            case "5":
                _view_existing_reports()
            case _ if choice.lower() == KEY_BACK:
                break


def _get_available_shows() -> list[str]:
    """Get list of available shows from transcripts directory."""
    from terminal_data_visualizer.config import TRANSCRIPTS_DIR

    transcripts_path = OUTPUTS_PATH / TRANSCRIPTS_DIR
    if not transcripts_path.exists():
        return []
    return sorted([d.name for d in transcripts_path.iterdir() if d.is_dir()])


def _generate_comprehensive() -> None:
    """Generate comprehensive report for all shows."""
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]📊 Comprehensive Report[/bold {COLOR_PRIMARY}]\n")

    shows = _get_available_shows()
    if not shows:
        console.print(f"[{COLOR_WARNING}]No shows found[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    console.print(f"[dim]Generating report for {len(shows)} shows...[/dim]\n")

    output_file = generate_comprehensive_report(shows, "Podcast Analysis Report", "markdown")

    console.print(f"\n[{COLOR_SUCCESS}]✓ Report generated: {output_file}[/{COLOR_SUCCESS}]")
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def _generate_single_show() -> None:
    """Generate report for a single show."""
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]📈 Single Show Report[/bold {COLOR_PRIMARY}]\n")

    shows = _get_available_shows()
    if not shows:
        console.print(f"[{COLOR_WARNING}]No shows found[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    console.print(f"[{COLOR_PRIMARY}]Available shows:[/{COLOR_PRIMARY}]\n")
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
            show = shows[idx]
            console.print(f"\n[dim]Generating report for {show}...[/dim]\n")

            output_file = generate_comprehensive_report(
                [show], f"{show} Analysis Report", "markdown"
            )

            console.print(f"\n[{COLOR_SUCCESS}]✓ Report generated: {output_file}[/{COLOR_SUCCESS}]")
        else:
            console.print(f"[{COLOR_ERROR}]Invalid selection[/{COLOR_ERROR}]")
    except ValueError:
        console.print(f"[{COLOR_ERROR}]Invalid input[/{COLOR_ERROR}]")

    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def _generate_hate_speech() -> None:
    """Generate hate speech focused report."""
    console.clear()
    console.print(f"[bold {COLOR_WARNING}]⚠️ Hate Speech Report[/bold {COLOR_WARNING}]\n")

    shows = _get_available_shows()
    if not shows:
        console.print(f"[{COLOR_WARNING}]No shows found[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    console.print("[dim]Generating hate speech report...[/dim]\n")

    output_file = generate_hate_speech_report(shows)

    console.print(f"\n[{COLOR_SUCCESS}]✓ Report generated: {output_file}[/{COLOR_SUCCESS}]")
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def _generate_json() -> None:
    """Generate JSON format report."""
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]📤 JSON Report[/bold {COLOR_PRIMARY}]\n")

    shows = _get_available_shows()
    if not shows:
        console.print(f"[{COLOR_WARNING}]No shows found[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    console.print(f"[dim]Generating JSON report for {len(shows)} shows...[/dim]\n")

    output_file = generate_comprehensive_report(shows, "Podcast Analysis Report", "json")

    console.print(f"\n[{COLOR_SUCCESS}]✓ Report generated: {output_file}[/{COLOR_SUCCESS}]")
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def _view_existing_reports() -> None:
    """View list of existing reports."""
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]📂 Existing Reports[/bold {COLOR_PRIMARY}]\n")

    if not REPORTS_DIR.exists():
        console.print(f"[{COLOR_WARNING}]No reports directory found[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    reports = sorted(REPORTS_DIR.glob("*.*"), key=lambda x: x.stat().st_mtime, reverse=True)

    if not reports:
        console.print(f"[{COLOR_WARNING}]No reports found[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", style="cyan", width=4)
    table.add_column("Filename", width=40)
    table.add_column("Size", justify="right", width=10)
    table.add_column("Modified", width=20)

    for i, report in enumerate(reports[:20], 1):
        size = report.stat().st_size
        size_str = f"{size / 1024:.1f} KB" if size > 1024 else f"{size} B"

        modified = datetime.fromtimestamp(report.stat().st_mtime)
        modified_str = modified.strftime("%Y-%m-%d %H:%M")

        table.add_row(str(i), report.name, size_str, modified_str)

    console.print(table)
    console.print(f"\n[dim]Reports directory: {REPORTS_DIR}[/dim]")

    Prompt.ask("\n[dim]Press Enter to continue[/dim]")
