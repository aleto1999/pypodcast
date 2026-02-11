"""Statistics calculation for LLM annotations."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from terminal_data_visualizer.models import LLMAnnotation


def load_llm_annotations_from_file(file_path: Path) -> list[LLMAnnotation]:
    """
    load all LLM annotations from an annotated transcript file.
    
    args:
        file_path: path to JSON file with LLM annotations.
        
    returns:
        list of LLMAnnotation objects.
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        
        annotations = []
        for segment in data.get("segments", []):
            llm_data = segment.get("llm_annotation")
            if llm_data:
                annotations.append(LLMAnnotation.from_dict(llm_data))
        
        return annotations
    except Exception:
        return []


def calculate_hate_speech_distribution(annotations: list[LLMAnnotation]) -> dict[str, int]:
    """
    calculate hate speech detection distribution.
    
    args:
        annotations: list of LLM annotations.
        
    returns:
        dict with counts of hate speech vs non-hate speech.
    """
    hate_count = sum(1 for ann in annotations if ann.has_hate_speech)
    non_hate_count = len(annotations) - hate_count
    
    return {
        "has_hate_speech": hate_count,
        "no_hate_speech": non_hate_count,
        "total": len(annotations),
        "hate_speech_rate": (hate_count / len(annotations) * 100) if annotations else 0
    }


def calculate_target_group_stats(annotations: list[LLMAnnotation]) -> dict[str, int]:
    """
    calculate target group statistics.
    
    args:
        annotations: list of LLM annotations.
        
    returns:
        dict mapping target groups to counts.
    """
    target_groups = [ann.target_group for ann in annotations if ann.target_group]
    return dict(Counter(target_groups))


def calculate_hate_speech_type_stats(annotations: list[LLMAnnotation]) -> dict[str, int]:
    """
    calculate hate speech type statistics.
    
    args:
        annotations: list of LLM annotations.
        
    returns:
        dict mapping hate speech types to counts.
    """
    hate_types = [ann.hate_speech_type for ann in annotations if ann.hate_speech_type]
    return dict(Counter(hate_types))


def calculate_topic_frequency(annotations: list[LLMAnnotation]) -> dict[str, int]:
    """
    calculate main topic frequency.
    
    args:
        annotations: list of LLM annotations.
        
    returns:
        dict mapping topics to counts.
    """
    topics = [ann.main_topic for ann in annotations if ann.main_topic]
    return dict(Counter(topics))


def calculate_advertisement_distribution(annotations: list[LLMAnnotation]) -> dict[str, int]:
    """
    calculate advertisement detection distribution.
    
    args:
        annotations: list of LLM annotations.
        
    returns:
        dict with counts of advertisement vs non-advertisement.
    """
    ad_count = sum(1 for ann in annotations if ann.has_advertisement)
    non_ad_count = len(annotations) - ad_count
    
    return {
        "has_advertisement": ad_count,
        "no_advertisement": non_ad_count,
        "total": len(annotations),
        "advertisement_rate": (ad_count / len(annotations) * 100) if annotations else 0
    }


def calculate_comprehensive_stats(annotations: list[LLMAnnotation]) -> dict[str, Any]:
    """
    calculate comprehensive statistics for all annotation fields.
    
    args:
        annotations: list of LLM annotations.
        
    returns:
        dict with all statistics.
    """
    return {
        "total_annotations": len(annotations),
        "hate_speech": calculate_hate_speech_distribution(annotations),
        "advertisement": calculate_advertisement_distribution(annotations),
        "target_groups": calculate_target_group_stats(annotations),
        "hate_speech_types": calculate_hate_speech_type_stats(annotations),
        "topics": calculate_topic_frequency(annotations),
    }


def export_annotation_dataset(
    annotations: list[LLMAnnotation],
    output_path: Path
) -> None:
    """
    export LLM annotations to JSON file.
    
    args:
        annotations: list of LLM annotations.
        output_path: path to output JSON file.
    """
    data = []
    for i, ann in enumerate(annotations):
        data.append({
            "segment_id": i,
            "has_hate_speech": ann.has_hate_speech,
            "has_advertisement": ann.has_advertisement,
            "target_group": ann.target_group,
            "hate_speech_type": ann.hate_speech_type,
            "main_topic": ann.main_topic,
        })
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def export_annotation_csv(
    annotations: list[LLMAnnotation],
    output_path: Path
) -> None:
    """
    export LLM annotations to CSV file.
    
    args:
        annotations: list of LLM annotations.
        output_path: path to output CSV file.
    """
    import csv
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # header.
        writer.writerow([
            "segment_id",
            "has_hate_speech",
            "has_advertisement",
            "target_group",
            "hate_speech_type",
            "main_topic",
        ])
        
        # data rows.
        for i, ann in enumerate(annotations):
            writer.writerow([
                i,
                ann.has_hate_speech,
                ann.has_advertisement,
                ann.target_group or "",
                ann.hate_speech_type or "",
                ann.main_topic,
            ])
