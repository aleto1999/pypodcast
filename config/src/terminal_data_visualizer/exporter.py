"""Export functionality for terminal data visualizer."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()

EXPORT_PATH = Path("outputs/terminal_viewer")


def ensure_export_dir() -> None:
    """Ensure the export directory exists."""
    EXPORT_PATH.mkdir(parents=True, exist_ok=True)


def generate_export_filename(prefix: str = "output") -> Path:
    """Generate a timestamped export filename."""
    ensure_export_dir()
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    return EXPORT_PATH / f"{prefix}_{timestamp}.json"


def export_to_json(data: dict[str, Any], prefix: str = "output") -> Path:
    """Export data to a JSON file with timestamp."""
    filepath = generate_export_filename(prefix)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    console.print(f"[green]✓ Exported to:[/green] {filepath}")
    return filepath


def format_matches_for_export(
    matches: list[tuple[Path, Any]],
    keywords: list[str],
    podcast_name: str = "",
    context_n: int = 0,
) -> dict[str, Any]:
    """Format keyword matches for JSON export."""
    export_data = {
        "export_type": "keyword_matches",
        "exported_at": datetime.now().isoformat(),
        "search_parameters": {
            "keywords": keywords,
            "podcast": podcast_name,
            "context_segments": context_n,
        },
        "total_matches": len(matches),
        "matches": [],
    }

    for file_path, match in matches:
        match_data = {
            "file": str(file_path),
            "file_name": file_path.stem,
            "keyword": match.keyword,
            "matched_text": match.matched_text,
            "category": match.category,
            "speaker": match.speaker,
            "start_time": match.start_time,
            "end_time": match.end_time,
            "confidence_tier": match.confidence_tier,
            "segment_id": match.segment_id,
            "context_before": match.context_before,
            "context_after": match.context_after,
        }
        export_data["matches"].append(match_data)

    return export_data


def format_utterance_for_export(
    file_path: Path,
    match: Any,
    segments: list[Any],
    target_segment_id: int,
    context_n: int,
    transcript_file: str = "",
) -> dict[str, Any]:
    """Format utterance detail with context for JSON export."""
    export_data = {
        "export_type": "utterance_detail",
        "exported_at": datetime.now().isoformat(),
        "file": str(file_path),
        "file_name": file_path.stem,
        "transcript_file": transcript_file,
        "match": {
            "keyword": match.keyword,
            "matched_text": match.matched_text,
            "category": match.category,
            "speaker": match.speaker,
            "start_time": match.start_time,
            "end_time": match.end_time,
            "confidence_tier": match.confidence_tier,
            "segment_id": match.segment_id,
        },
        "context_settings": {
            "n": context_n,
            "target_segment_id": target_segment_id,
        },
        "segments": [],
    }

    if segments:
        start_idx = max(0, target_segment_id - context_n)
        end_idx = min(len(segments), target_segment_id + context_n + 1)

        for idx in range(start_idx, end_idx):
            seg = segments[idx]
            seg_data = {
                "segment_id": idx,
                "is_target": idx == target_segment_id,
                "text": seg.text.strip(),
                "speaker": seg.speaker,
                "start_time": seg.start_time,
                "end_time": seg.end_time,
            }
            export_data["segments"].append(seg_data)

    return export_data


def format_folder_stats_for_export(stats: list[Any], title: str = "") -> dict[str, Any]:
    """Format folder statistics for JSON export."""
    export_data = {
        "export_type": "folder_stats",
        "exported_at": datetime.now().isoformat(),
        "title": title,
        "total_folders": len(stats),
        "folders": [],
    }

    total_files = 0
    total_size = 0

    for stat in stats:
        folder_data = {
            "name": stat.name,
            "path": str(stat.path),
            "file_count": stat.file_count,
            "size_bytes": stat.size_bytes,
            "size_human": stat.size_human,
        }
        export_data["folders"].append(folder_data)
        total_files += stat.file_count
        total_size += stat.size_bytes

    export_data["totals"] = {
        "total_files": total_files,
        "total_size_bytes": total_size,
    }

    return export_data


def format_keywords_for_export(
    keywords: set[str], title: str = "", podcast: str = ""
) -> dict[str, Any]:
    """Format keywords list for JSON export."""
    return {
        "export_type": "keywords_list",
        "exported_at": datetime.now().isoformat(),
        "title": title,
        "podcast": podcast,
        "total_keywords": len(keywords),
        "keywords": sorted(keywords),
    }


def format_categories_for_export(
    categories: set[str], title: str = "", podcast: str = ""
) -> dict[str, Any]:
    """Format categories list for JSON export."""
    return {
        "export_type": "categories_list",
        "exported_at": datetime.now().isoformat(),
        "title": title,
        "podcast": podcast,
        "total_categories": len(categories),
        "categories": sorted(categories),
    }


def strip_rich_markup(text: str) -> str:
    """Remove Rich markup tags from text for plain export."""
    return re.sub(r"\[/?[^\]]+\]", "", text)
