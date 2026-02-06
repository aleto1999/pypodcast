"""Statistics calculation for utterance classifications."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from terminal_data_visualizer.models import Classification


def load_classifications_from_file(file_path: Path) -> list[list[Classification]]:
    """
    load all classifications from a classified transcript file.
    
    args:
        file_path: path to JSON file with classifications.
        
    returns:
        list of classification lists (one per segment).
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        
        all_classifications = []
        for segment in data.get("segments", []):
            segment_classifications = []
            for cls_data in segment.get("classifications", []):
                segment_classifications.append(Classification.from_dict(cls_data))
            all_classifications.append(segment_classifications)
        
        return all_classifications
    except Exception:
        return []


def calculate_label_distribution(
    classifications: list[list[Classification]]
) -> dict[str, dict[str, int]]:
    """
    calculate label distribution per model.
    
    args:
        classifications: list of classification lists.
        
    returns:
        dict mapping model_name -> {label: count}.
    """
    distribution: dict[str, Counter] = defaultdict(Counter)
    
    for segment_classifications in classifications:
        for cls in segment_classifications:
            distribution[cls.model_name][cls.label] += 1
    
    # convert Counter to regular dict.
    return {model: dict(labels) for model, labels in distribution.items()}


def calculate_confidence_distribution(
    classifications: list[list[Classification]],
    model_name: str | None = None,
    bins: int = 10
) -> dict[str, list[int]]:
    """
    calculate confidence score distribution.
    
    args:
        classifications: list of classification lists.
        model_name: filter by specific model (None = all models).
        bins: number of bins for histogram.
        
    returns:
        dict mapping bin ranges to counts.
    """
    confidences = []
    
    for segment_classifications in classifications:
        for cls in segment_classifications:
            if model_name and cls.model_name != model_name:
                continue
            if cls.confidence is not None:
                confidences.append(cls.confidence)
    
    if not confidences:
        return {}
    
    # create histogram bins.
    bin_edges = [i / bins for i in range(bins + 1)]
    histogram = [0] * bins
    
    for confidence in confidences:
        bin_idx = min(int(confidence * bins), bins - 1)
        histogram[bin_idx] += 1
    
    # create readable bin labels.
    result = {}
    for i in range(bins):
        label = f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}"
        result[label] = histogram[i]
    
    return result


def calculate_model_agreement(
    classifications: list[list[Classification]],
    models: list[str] | None = None
) -> dict[str, Any]:
    """
    calculate agreement statistics between models.
    
    args:
        classifications: list of classification lists.
        models: list of model names to compare (None = all models).
        
    returns:
        dict with agreement statistics.
    """
    if not classifications:
        return {}
    
    # get all unique models if not specified.
    if models is None:
        all_models = set()
        for segment_classifications in classifications:
            for cls in segment_classifications:
                all_models.add(cls.model_name)
        models = sorted(all_models)
    
    if len(models) < 2:
        return {"error": "need at least 2 models for agreement analysis"}
    
    # count agreements and disagreements.
    total_segments = len(classifications)
    agreements = 0
    
    # simple agreement: all models have same label.
    for segment_classifications in classifications:
        # group by model.
        model_labels = {}
        for cls in segment_classifications:
            if cls.model_name in models:
                model_labels[cls.model_name] = cls.label
        
        # check if all models present and agree.
        if len(model_labels) == len(models):
            labels = set(model_labels.values())
            if len(labels) == 1:
                agreements += 1
    
    agreement_rate = agreements / total_segments if total_segments > 0 else 0
    
    return {
        "models_compared": models,
        "total_segments": total_segments,
        "agreements": agreements,
        "disagreements": total_segments - agreements,
        "agreement_rate": agreement_rate,
    }


def calculate_label_statistics(
    classifications: list[list[Classification]]
) -> dict[str, Any]:
    """
    calculate comprehensive label statistics.
    
    args:
        classifications: list of classification lists.
        
    returns:
        dict with various label statistics.
    """
    distribution = calculate_label_distribution(classifications)
    
    # calculate totals and percentages.
    stats = {}
    for model, labels in distribution.items():
        total = sum(labels.values())
        stats[model] = {
            "total_classifications": total,
            "unique_labels": len(labels),
            "label_counts": labels,
            "label_percentages": {
                label: (count / total * 100) if total > 0 else 0
                for label, count in labels.items()
            }
        }
    
    return stats


def export_classification_matrix(
    classifications: list[list[Classification]],
    output_path: Path
) -> None:
    """
    export classification matrix to CSV.
    
    args:
        classifications: list of classification lists.
        output_path: path to output CSV file.
    """
    import csv
    
    if not classifications:
        return
    
    # get all models.
    all_models = set()
    for segment_classifications in classifications:
        for cls in segment_classifications:
            all_models.add(cls.model_name)
    
    models = sorted(all_models)
    
    # write CSV.
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # header.
        writer.writerow(["segment_id"] + [f"{model}_label" for model in models] + 
                       [f"{model}_confidence" for model in models])
        
        # data rows.
        for segment_id, segment_classifications in enumerate(classifications):
            # group by model.
            model_data = {}
            for cls in segment_classifications:
                model_data[cls.model_name] = cls
            
            row = [segment_id]
            # labels.
            for model in models:
                row.append(model_data.get(model).label if model in model_data else "")
            # confidences.
            for model in models:
                conf = model_data.get(model).confidence if model in model_data else None
                row.append(f"{conf:.4f}" if conf is not None else "")
            
            writer.writerow(row)
