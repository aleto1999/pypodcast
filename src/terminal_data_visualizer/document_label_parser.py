"""Parser for document label summary files."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DocumentLabelStats:
    """Document-level label statistics for a show."""
    
    show_name: str
    overall_label: str = "UNKNOWN"  # POSITIVE, NEGATIVE, or UNKNOWN
    
    # episode statistics.
    total_episodes: int = 0
    positive_episodes: int = 0
    negative_episodes: int = 0
    unknown_episodes: int = 0
    
    # aggregate label statistics.
    total_positive_labels: int = 0
    total_negative_labels: int = 0
    total_segments_analyzed: int = 0
    total_ad_segments_skipped: int = 0
    
    # label distribution by model (dict of model_name -> dict of label -> count).
    label_distribution: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass
class GlobalDocumentLabelStats:
    """Global document-level label statistics across all shows."""
    
    total_shows: int = 0
    positive_shows: int = 0
    negative_shows: int = 0
    unknown_shows: int = 0
    
    total_episodes: int = 0
    positive_episodes: int = 0
    negative_episodes: int = 0
    
    total_segments_analyzed: int = 0
    total_ad_segments_skipped: int = 0
    total_positive_labels: int = 0
    total_negative_labels: int = 0
    
    # global label distribution by model.
    label_distribution: dict[str, dict[str, int]] = field(default_factory=dict)


def parse_show_document_labels(summary_file: Path) -> DocumentLabelStats | None:
    """Parse a show's document label summary file.
    
    Args:
        summary_file: Path to show_summary.md file
        
    Returns:
        DocumentLabelStats object or None if parsing fails
    """
    if not summary_file.exists():
        return None
    
    try:
        content = summary_file.read_text(encoding="utf-8")
        lines = content.split("\n")
        
        # extract show name from first line.
        show_name = "unknown"
        for line in lines:
            if line.startswith("# Show Summary:"):
                show_name = line.replace("# Show Summary:", "").strip()
                break
        
        stats = DocumentLabelStats(show_name=show_name)
        
        # extract overall label.
        for line in lines:
            if "**Overall Label:**" in line:
                match = re.search(r'`([A-Z]+)`', line)
                if match:
                    stats.overall_label = match.group(1)
                break
        
        # parse episode statistics.
        in_episode_stats = False
        for i, line in enumerate(lines):
            if "## Episode Statistics" in line:
                in_episode_stats = True
                continue
            
            if in_episode_stats and "|" in line and "---" not in line and "Metric" not in line:
                # parse table rows.
                if "Total episodes" in line:
                    stats.total_episodes = _extract_number(line)
                elif "Positive episodes" in line:
                    stats.positive_episodes = _extract_number(line)
                elif "Negative episodes" in line:
                    stats.negative_episodes = _extract_number(line)
                elif "Unknown episodes" in line:
                    stats.unknown_episodes = _extract_number(line)
            
            if in_episode_stats and "##" in line and i > 0:
                in_episode_stats = False
        
        # parse aggregate label statistics.
        in_aggregate_stats = False
        for i, line in enumerate(lines):
            if "## Aggregate Label Statistics" in line:
                in_aggregate_stats = True
                continue
            
            if in_aggregate_stats and "|" in line and "---" not in line and "Metric" not in line:
                if "Total positive labels" in line:
                    stats.total_positive_labels = _extract_number(line)
                elif "Total negative labels" in line:
                    stats.total_negative_labels = _extract_number(line)
                elif "Total segments analyzed" in line:
                    stats.total_segments_analyzed = _extract_number(line)
                elif "Total ad segments skipped" in line:
                    stats.total_ad_segments_skipped = _extract_number(line)
            
            if in_aggregate_stats and "##" in line and i > 0:
                in_aggregate_stats = False
        
        # parse label distribution by model.
        current_model = None
        in_distribution_table = False
        
        for line in lines:
            # detect model header: **model_name** (total: N).
            model_match = re.match(r'\*\*([a-z_]+)\*\* \(total: ([\d,]+)\)', line)
            if model_match:
                current_model = model_match.group(1)
                stats.label_distribution[current_model] = {}
                in_distribution_table = True
                continue
            
            # parse distribution table rows.
            if in_distribution_table and current_model and "|" in line:
                if "---" in line or "Label" in line or "Count" in line:
                    continue
                
                # parse: | LABEL_NAME | count | percentage |.
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 2:
                    label_name = parts[0]
                    count_str = parts[1].replace(",", "")
                    try:
                        count = int(count_str)
                        stats.label_distribution[current_model][label_name] = count
                    except ValueError:
                        pass
            
            # exit distribution table when we hit another section.
            if in_distribution_table and line.startswith("**") and "(total:" not in line:
                in_distribution_table = False
        
        return stats
        
    except (IOError, UnicodeDecodeError):
        return None


def parse_global_document_labels(global_summary_file: Path) -> GlobalDocumentLabelStats | None:
    """Parse the global document label summary file.
    
    Args:
        global_summary_file: Path to global_summary.md file
        
    Returns:
        GlobalDocumentLabelStats object or None if parsing fails
    """
    if not global_summary_file.exists():
        return None
    
    try:
        content = global_summary_file.read_text(encoding="utf-8")
        lines = content.split("\n")
        
        stats = GlobalDocumentLabelStats()
        
        # parse overview section.
        in_overview = False
        for i, line in enumerate(lines):
            if "## Overview" in line:
                in_overview = True
                continue
            
            if in_overview and "|" in line and "---" not in line and "Metric" not in line:
                if "Total shows" in line:
                    stats.total_shows = _extract_number(line)
                elif "Positive shows" in line:
                    stats.positive_shows = _extract_number(line)
                elif "Negative shows" in line:
                    stats.negative_shows = _extract_number(line)
                elif "Unknown shows" in line:
                    stats.unknown_shows = _extract_number(line)
                elif "Total episodes" in line:
                    stats.total_episodes = _extract_number(line)
                elif "Positive episodes" in line:
                    stats.positive_episodes = _extract_number(line)
                elif "Negative episodes" in line:
                    stats.negative_episodes = _extract_number(line)
                elif "Total segments analyzed" in line:
                    stats.total_segments_analyzed = _extract_number(line)
                elif "Total ad segments skipped" in line:
                    stats.total_ad_segments_skipped = _extract_number(line)
                elif "Total positive labels" in line:
                    stats.total_positive_labels = _extract_number(line)
                elif "Total negative labels" in line:
                    stats.total_negative_labels = _extract_number(line)
            
            if in_overview and "---" in line and i > 5:
                in_overview = False
        
        # parse global label distribution.
        current_model = None
        in_distribution_table = False
        
        for line in lines:
            # detect model header.
            model_match = re.match(r'\*\*([a-z_]+)\*\* \(total: ([\d,]+)\)', line)
            if model_match:
                current_model = model_match.group(1)
                stats.label_distribution[current_model] = {}
                in_distribution_table = True
                continue
            
            # parse distribution table rows.
            if in_distribution_table and current_model and "|" in line:
                if "---" in line or "Label" in line or "Count" in line:
                    continue
                
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 2:
                    label_name = parts[0]
                    count_str = parts[1].replace(",", "")
                    try:
                        count = int(count_str)
                        stats.label_distribution[current_model][label_name] = count
                    except ValueError:
                        pass
            
            if in_distribution_table and line.startswith("**") and "(total:" not in line:
                in_distribution_table = False
        
        return stats
        
    except (IOError, UnicodeDecodeError):
        return None


def _extract_number(line: str) -> int:
    """Extract a number from a markdown table row.
    
    Args:
        line: Markdown table row like "| Metric | 1,234 |"
        
    Returns:
        Extracted integer, or 0 if parsing fails
    """
    parts = [p.strip() for p in line.split("|") if p.strip()]
    if len(parts) >= 2:
        # get the last part (the value column).
        value_str = parts[-1].replace(",", "").strip()
        try:
            return int(value_str)
        except ValueError:
            return 0
    return 0
