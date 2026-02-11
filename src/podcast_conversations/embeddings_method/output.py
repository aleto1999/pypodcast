"""output generation and statistics."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def save_analysis_results(
    show_name: str, results: list[dict[str, Any]], threshold: float, model_name: str
) -> None:
    """save full results and summary statistics."""

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_dir = Path(f"outputs/analysis/embeddings_method/{show_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # full results.
    full_results = {
        "show_name": show_name,
        "analysis_timestamp": datetime.now().isoformat(),
        "threshold": threshold,
        "embedding_model": model_name,
        "total_files": len(results),
        "files_analyzed": results,
    }

    full_path = output_dir / f"full_results_{timestamp}.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    # summary statistics.
    summary = generate_summary_statistics(results, threshold, model_name, show_name)
    summary_path = output_dir / f"summary_{timestamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ results saved to {output_dir}")
    print(f"  • full results: {full_path.name}")
    print(f"  • summary: {summary_path.name}")


def generate_summary_statistics(
    results: list[dict[str, Any]],
    threshold: float,
    model_name: str,
    show_name: str,
) -> dict[str, Any]:
    """aggregate statistics from analysis results."""

    total_matches = sum(r["matches_found"] for r in results)
    all_matches = [match for r in results for match in r["matches"]]

    # group by target group.
    target_groups: dict[str, int] = {}
    for match in all_matches:
        group = match["target_group"]
        target_groups[group] = target_groups.get(group, 0) + 1

    # group by strategy.
    strategies: dict[str, int] = {}
    for match in all_matches:
        strategy = match["othering_strategy"]
        strategies[strategy] = strategies.get(strategy, 0) + 1

    # calculate average similarity.
    avg_similarity = (
        sum(m["similarity_score"] for m in all_matches) / len(all_matches)
        if all_matches
        else 0.0
    )

    return {
        "analysis_timestamp": datetime.now().isoformat(),
        "show_name": show_name,
        "threshold_used": threshold,
        "embedding_model": model_name,
        "summary_statistics": {
            "total_files_analyzed": len(results),
            "total_matches_found": total_matches,
            "matches_by_target_group": target_groups,
            "matches_by_strategy": strategies,
            "average_similarity_score": round(avg_similarity, 3),
        },
    }
