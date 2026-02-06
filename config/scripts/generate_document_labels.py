#!/usr/bin/env python3
"""
generates per-episode and per-show labels by aggregating classification labels.

this script parses all labeled transcripts and assigns:
- per-episode labels based on majority voting across all segments
- per-show labels based on majority voting across all episodes

additionally, it generates comprehensive statistics for all labels present
in the classification data, including distribution and frequency analysis.

utterances labeled as AD content are excluded from the aggregation.
"""

import json
import logging
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional

import click


# label mappings: maps model labels to positive (True) or negative (False).
LABEL_MAPPINGS = {
    "hate_speech_detection": {
        "NOT-HATE": True,
        "HATE": False,
    },
    "fine_grained_hate_speech_detection": {
        # label_0 typically means "no hate" in fine-grained models.
        "LABEL_0": True,
        "LABEL_1": False,
        "LABEL_2": False,
        "LABEL_3": False,
    },
    "hostile_content": {
        "NOT-OFFENSIVE": True,
        "OFFENSIVE": False,
    },
}

# for hate_against_minorities, the label is always "toxic" but confidence varies.
# low confidence (< 0.5) means "non_toxic" (positive), high confidence means "toxic" (negative).
HATE_MINORITIES_THRESHOLD = 0.5


def is_ad_segment(classifications: list[dict]) -> bool:
    """check if a segment is classified as an ad."""
    for clf in classifications:
        if clf.get("model_name") == "ad_content_detection":
            if clf.get("label") == "LABEL_1":
                return True
    return False


def get_segment_labels(classifications: list[dict]) -> tuple[int, int, dict[str, Counter]]:
    """
    extract positive and negative label counts from a segment's classifications.

    returns: (positive_count, negative_count, label_counts_by_model)
    """
    positive = 0
    negative = 0
    label_counts: dict[str, Counter] = defaultdict(Counter)

    for clf in classifications:
        model_name = clf.get("model_name", "")
        label = clf.get("label", "")
        confidence = clf.get("confidence", 0.0)

        # skip ad detection model - it's only used for filtering.
        if model_name == "ad_content_detection":
            # still track ad detection labels.
            label_counts[model_name][label] += 1
            continue

        # special handling for hate_against_minorities (confidence-based).
        if model_name == "hate_against_minorities":
            if label == "toxic":
                if confidence >= HATE_MINORITIES_THRESHOLD:
                    negative += 1
                    label_counts[model_name]["toxic (high conf)"] += 1
                else:
                    positive += 1
                    label_counts[model_name]["toxic (low conf)"] += 1
            else:
                positive += 1
                label_counts[model_name][label] += 1
            continue

        # track all labels for the model.
        label_counts[model_name][label] += 1

        # standard label mapping for other models.
        if model_name in LABEL_MAPPINGS:
            mapping = LABEL_MAPPINGS[model_name]
            if label in mapping:
                if mapping[label]:
                    positive += 1
                else:
                    negative += 1

    return positive, negative, label_counts


def merge_label_counts(
    target: dict[str, Counter],
    source: dict[str, Counter],
) -> None:
    """merge source label counts into target (in-place)."""
    for model_name, counts in source.items():
        target[model_name].update(counts)


def process_episode(file_path: Path) -> Optional[dict]:
    """
    process a single episode file and return label statistics.

    returns: dict with episode stats or none if file couldn't be processed.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"  error reading {file_path}: {e}")
        return None

    segments = data.get("segments", [])

    total_positive = 0
    total_negative = 0
    total_segments = 0
    ad_segments = 0
    episode_label_counts: dict[str, Counter] = defaultdict(Counter)

    for segment in segments:
        classifications = segment.get("classifications", [])

        if not classifications:
            continue

        # skip ad segments for positive/negative counting but still track labels.
        if is_ad_segment(classifications):
            ad_segments += 1
            # still collect label statistics from ad segments.
            _, _, segment_label_counts = get_segment_labels(classifications)
            merge_label_counts(episode_label_counts, segment_label_counts)
            continue

        positive, negative, segment_label_counts = get_segment_labels(classifications)
        total_positive += positive
        total_negative += negative
        total_segments += 1
        merge_label_counts(episode_label_counts, segment_label_counts)

    # determine episode label based on majority.
    if total_positive + total_negative == 0:
        episode_label = "UNKNOWN"
    elif total_positive >= total_negative:
        episode_label = "POSITIVE"
    else:
        episode_label = "NEGATIVE"

    return {
        "file_name": file_path.name,
        "total_segments": total_segments,
        "ad_segments_skipped": ad_segments,
        "positive_labels": total_positive,
        "negative_labels": total_negative,
        "episode_label": episode_label,
        "label_counts": {k: dict(v) for k, v in episode_label_counts.items()},
    }


def format_label_counts_table(label_counts: dict[str, dict[str, int]], indent: str = "") -> str:
    """format label counts as a markdown table."""
    if not label_counts:
        return f"{indent}*No label data available*\n"

    lines = []

    for model_name in sorted(label_counts.keys()):
        counts = label_counts[model_name]
        total = sum(counts.values())

        lines.append(f"{indent}**{model_name}** (total: {total:,})")
        lines.append("")
        lines.append(f"{indent}| Label | Count | Percentage |")
        lines.append(f"{indent}|-------|------:|------------|")

        for label in sorted(counts.keys()):
            count = counts[label]
            pct = (count / total * 100) if total > 0 else 0
            lines.append(f"{indent}| {label} | {count:,} | {pct:.1f}% |")

        lines.append("")

    return "\n".join(lines)


def process_show(show_dir: Path, output_dir: Path) -> dict:
    """
    process all episodes in a show directory.

    returns: dict with show-level statistics.
    """
    show_name = show_dir.name
    print(f"processing show: {show_name}")

    episode_results = []

    # process all json files in the show directory.
    json_files = sorted(show_dir.glob("*.json"))

    for json_file in json_files:
        result = process_episode(json_file)
        if result:
            episode_results.append(result)

    # calculate show-level statistics.
    total_positive_episodes = sum(1 for r in episode_results if r["episode_label"] == "POSITIVE")
    total_negative_episodes = sum(1 for r in episode_results if r["episode_label"] == "NEGATIVE")
    total_unknown_episodes = sum(1 for r in episode_results if r["episode_label"] == "UNKNOWN")

    # determine show label based on majority of episode labels.
    if total_positive_episodes + total_negative_episodes == 0:
        show_label = "UNKNOWN"
    elif total_positive_episodes >= total_negative_episodes:
        show_label = "POSITIVE"
    else:
        show_label = "NEGATIVE"

    # aggregate label counts across all episodes.
    total_positive_labels = sum(r["positive_labels"] for r in episode_results)
    total_negative_labels = sum(r["negative_labels"] for r in episode_results)
    total_segments = sum(r["total_segments"] for r in episode_results)
    total_ad_segments = sum(r["ad_segments_skipped"] for r in episode_results)

    # aggregate all label counts.
    show_label_counts: dict[str, Counter] = defaultdict(Counter)
    for result in episode_results:
        for model_name, counts in result["label_counts"].items():
            show_label_counts[model_name].update(counts)

    show_stats = {
        "show_name": show_name,
        "show_label": show_label,
        "total_episodes": len(episode_results),
        "positive_episodes": total_positive_episodes,
        "negative_episodes": total_negative_episodes,
        "unknown_episodes": total_unknown_episodes,
        "total_positive_labels": total_positive_labels,
        "total_negative_labels": total_negative_labels,
        "total_segments_analyzed": total_segments,
        "total_ad_segments_skipped": total_ad_segments,
        "label_counts": {k: dict(v) for k, v in show_label_counts.items()},
    }

    # write results to output directory.
    show_output_dir = output_dir / show_name
    show_output_dir.mkdir(parents=True, exist_ok=True)

    # write episode-level results.
    episode_output_path = show_output_dir / "episode_labels.md"
    with open(episode_output_path, "w", encoding="utf-8") as f:
        f.write(f"# Episode Labels: {show_name}\n\n")
        f.write(f"**Show-level label:** `{show_label}`\n\n")
        f.write("## Summary\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|------:|\n")
        f.write(f"| Total episodes | {len(episode_results)} |\n")
        f.write(f"| Positive episodes | {total_positive_episodes} |\n")
        f.write(f"| Negative episodes | {total_negative_episodes} |\n")
        f.write(f"| Unknown episodes | {total_unknown_episodes} |\n\n")

        f.write("---\n\n")
        f.write("## Per-Episode Breakdown\n\n")

        for result in episode_results:
            f.write(f"### {result['file_name']}\n\n")
            f.write(f"- **Label:** `{result['episode_label']}`\n")
            f.write(f"- **Segments analyzed:** {result['total_segments']}\n")
            f.write(f"- **Ad segments skipped:** {result['ad_segments_skipped']}\n")
            f.write(f"- **Positive labels:** {result['positive_labels']}\n")
            f.write(f"- **Negative labels:** {result['negative_labels']}\n\n")

            if result["label_counts"]:
                f.write("<details>\n<summary>Label Distribution</summary>\n\n")
                f.write(format_label_counts_table(result["label_counts"]))
                f.write("</details>\n\n")

    # write show summary.
    summary_output_path = show_output_dir / "show_summary.md"
    with open(summary_output_path, "w", encoding="utf-8") as f:
        f.write(f"# Show Summary: {show_name}\n\n")
        f.write(f"**Overall Label:** `{show_label}`\n\n")

        f.write("## Episode Statistics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|------:|\n")
        f.write(f"| Total episodes | {len(episode_results)} |\n")
        f.write(f"| Positive episodes | {total_positive_episodes} |\n")
        f.write(f"| Negative episodes | {total_negative_episodes} |\n")
        f.write(f"| Unknown episodes | {total_unknown_episodes} |\n\n")

        f.write("## Aggregate Label Statistics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|------:|\n")
        f.write(f"| Total positive labels | {total_positive_labels:,} |\n")
        f.write(f"| Total negative labels | {total_negative_labels:,} |\n")
        f.write(f"| Total segments analyzed | {total_segments:,} |\n")
        f.write(f"| Total ad segments skipped | {total_ad_segments:,} |\n\n")

        f.write("## Label Distribution by Model\n\n")
        f.write(format_label_counts_table({k: dict(v) for k, v in show_label_counts.items()}))

    print(f"  processed {len(episode_results)} episodes -> {show_label}")
    print(f"  output saved to: {show_output_dir}")

    return show_stats


def main():
    """main entry point."""
    # define paths.
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / "outputs" / "transcripts_with_diarization_labels_postprocessed_with_utterance_and_document_labels"
    output_dir = base_dir / "outputs" / "document_labels"

    # create output directory.
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"input directory: {input_dir}")
    print(f"output directory: {output_dir}")
    print()

    # process all shows.
    all_show_stats = []

    for show_dir in sorted(input_dir.iterdir()):
        if show_dir.is_dir():
            stats = process_show(show_dir, output_dir)
            all_show_stats.append(stats)

    # aggregate global label counts.
    global_label_counts: dict[str, Counter] = defaultdict(Counter)
    for stats in all_show_stats:
        for model_name, counts in stats["label_counts"].items():
            global_label_counts[model_name].update(counts)

    # write global summary.
    global_summary_path = output_dir / "global_summary.md"
    with open(global_summary_path, "w", encoding="utf-8") as f:
        f.write("# Global Document Labels Summary\n\n")

        # count show labels.
        positive_shows = sum(1 for s in all_show_stats if s["show_label"] == "POSITIVE")
        negative_shows = sum(1 for s in all_show_stats if s["show_label"] == "NEGATIVE")
        unknown_shows = sum(1 for s in all_show_stats if s["show_label"] == "UNKNOWN")

        total_episodes = sum(s["total_episodes"] for s in all_show_stats)
        total_positive_episodes = sum(s["positive_episodes"] for s in all_show_stats)
        total_negative_episodes = sum(s["negative_episodes"] for s in all_show_stats)
        total_segments = sum(s["total_segments_analyzed"] for s in all_show_stats)
        total_ad_segments = sum(s["total_ad_segments_skipped"] for s in all_show_stats)
        total_positive_labels = sum(s["total_positive_labels"] for s in all_show_stats)
        total_negative_labels = sum(s["total_negative_labels"] for s in all_show_stats)

        f.write("## Overview\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|------:|\n")
        f.write(f"| Total shows | {len(all_show_stats)} |\n")
        f.write(f"| Positive shows | {positive_shows} |\n")
        f.write(f"| Negative shows | {negative_shows} |\n")
        f.write(f"| Unknown shows | {unknown_shows} |\n")
        f.write(f"| Total episodes | {total_episodes:,} |\n")
        f.write(f"| Positive episodes | {total_positive_episodes:,} |\n")
        f.write(f"| Negative episodes | {total_negative_episodes:,} |\n")
        f.write(f"| Total segments analyzed | {total_segments:,} |\n")
        f.write(f"| Total ad segments skipped | {total_ad_segments:,} |\n")
        f.write(f"| Total positive labels | {total_positive_labels:,} |\n")
        f.write(f"| Total negative labels | {total_negative_labels:,} |\n\n")

        f.write("---\n\n")
        f.write("## Global Label Distribution by Model\n\n")
        f.write("This section shows the aggregate distribution of all classification labels across all shows and episodes.\n\n")
        f.write(format_label_counts_table({k: dict(v) for k, v in global_label_counts.items()}))

        f.write("---\n\n")
        f.write("## Per-Show Breakdown\n\n")
        f.write("| Show | Label | Episodes | Positive | Negative | Unknown | Pos Labels | Neg Labels |\n")
        f.write("|------|-------|----------|----------|----------|---------|------------|------------|\n")

        for stats in sorted(all_show_stats, key=lambda x: x["show_name"]):
            f.write(
                f"| {stats['show_name']} | `{stats['show_label']}` | "
                f"{stats['total_episodes']} | {stats['positive_episodes']} | "
                f"{stats['negative_episodes']} | {stats['unknown_episodes']} | "
                f"{stats['total_positive_labels']:,} | {stats['total_negative_labels']:,} |\n"
            )

        f.write("\n---\n\n")
        f.write("## Per-Show Label Distribution\n\n")

        for stats in sorted(all_show_stats, key=lambda x: x["show_name"]):
            f.write(f"### {stats['show_name']}\n\n")
            f.write(f"**Label:** `{stats['show_label']}`\n\n")
            f.write(format_label_counts_table(stats["label_counts"]))

    print()
    print(f"global summary saved to: {global_summary_path}")
    print("done!")


if __name__ == "__main__":
    main()
