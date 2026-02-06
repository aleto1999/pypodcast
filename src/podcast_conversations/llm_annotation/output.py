"""Output formatting and saving for LLM annotations."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def save_annotation_results(
    show_name: str,
    results: list[dict[str, Any]],
    output_dir: Path,
    config_summary: dict[str, Any] | None = None,
) -> Path:
    """
    Save annotation results summary for a show.

    Args:
        show_name: Name of the podcast show
        results: List of per-file annotation statistics
        output_dir: Directory to save results
        config_summary: Optional configuration summary to include

    Returns:
        Path to saved summary file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # aggregate statistics.
    total_segments = sum(r.get("segments_annotated", 0) for r in results)
    total_hate_speech = 0
    total_advertisements = 0
    target_group_counts: dict[str, int] = {}
    hate_type_counts: dict[str, int] = {}
    topic_counts: dict[str, int] = {}

    for result in results:
        annotations = result.get("annotations", [])
        for ann in annotations:
            if ann.get("has_hate_speech"):
                total_hate_speech += 1
                target = ann.get("target_group")
                if target:
                    target_group_counts[target] = target_group_counts.get(target, 0) + 1
                hate_type = ann.get("hate_speech_type")
                if hate_type:
                    hate_type_counts[hate_type] = hate_type_counts.get(hate_type, 0) + 1
            if ann.get("has_advertisement"):
                total_advertisements += 1
            topic = ann.get("main_topic")
            if topic:
                # normalize topic for counting.
                topic_lower = topic.lower().strip()
                topic_counts[topic_lower] = topic_counts.get(topic_lower, 0) + 1

    # create summary.
    summary = {
        "show_name": show_name,
        "generated_at": datetime.now().isoformat(),
        "statistics": {
            "total_files": len(results),
            "total_segments": total_segments,
            "hate_speech_count": total_hate_speech,
            "advertisement_count": total_advertisements,
            "hate_speech_rate": (total_hate_speech / total_segments if total_segments > 0 else 0),
            "advertisement_rate": (
                total_advertisements / total_segments if total_segments > 0 else 0
            ),
        },
        "target_group_distribution": target_group_counts,
        "hate_speech_type_distribution": hate_type_counts,
        "top_topics": dict(sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:20]),
        "config": config_summary,
    }

    # save summary.
    summary_path = output_dir / f"{show_name}_llm_annotation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary_path


def generate_annotation_report(
    results: list[dict[str, Any]],
    output_path: Path,
    show_name: str | None = None,
) -> None:
    """
    Generate a markdown report of annotation results.

    Args:
        results: List of per-file annotation statistics
        output_path: Path to save markdown report
        show_name: Optional show name for report title
    """
    total_segments = sum(r.get("segments_annotated", 0) for r in results)
    total_hate_speech = 0
    total_advertisements = 0

    for result in results:
        annotations = result.get("annotations", [])
        for ann in annotations:
            if ann.get("has_hate_speech"):
                total_hate_speech += 1
            if ann.get("has_advertisement"):
                total_advertisements += 1

    title = f"LLM Annotation Report - {show_name}" if show_name else "LLM Annotation Report"
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""# {title}

Generated: {date_str}

## Summary

| Metric | Value |
|--------|-------|
| Total Files | {len(results)} |
| Total Segments | {total_segments:,} |
| Hate Speech Detected | {total_hate_speech:,} |
| Advertisements Detected | {total_advertisements:,} |
| Hate Speech Rate | {total_hate_speech / total_segments * 100:.2f}% |
| Advertisement Rate | {total_advertisements / total_segments * 100:.2f}% |

## Processing Details

"""
    for i, result in enumerate(results[:20], 1):
        file_name = result.get("file_name", f"File {i}")
        segments = result.get("segments_annotated", 0)
        report += f"- {file_name}: {segments} segments annotated\n"

    if len(results) > 20:
        report += f"\n... and {len(results) - 20} more files\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
