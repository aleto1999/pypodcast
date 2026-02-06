#!/usr/bin/env python3
"""Check classification coverage discrepancy."""

import json
from pathlib import Path

# Find a sample file
sample_dir = Path("outputs/transcripts_with_diarization_labels_postprocessed_with_utterance_and_document_labels")
shows = list(sample_dir.iterdir())

if not shows:
    print("No shows found")
    exit(1)

show = shows[0]
files = list(show.glob("*.json"))

if not files:
    print("No files found")
    exit(1)

sample_file = files[0]

print(f"Analyzing: {sample_file}")
print(f"Show: {show.name}\n")

with open(sample_file) as f:
    data = json.load(f)

segments = data.get("segments", [])
print(f"Total segments: {len(segments)}\n")

empty_with_0 = 0
empty_with_classifications = 0
nonempty_with_8 = 0
nonempty_with_less = 0
nonempty_with_0 = 0

for seg in segments:
    text = seg.get("text", "").strip()
    classifications = seg.get("classifications", [])
    num_classifications = len(classifications)
    
    if not text:
        # Empty segment
        if num_classifications == 0:
            empty_with_0 += 1
        else:
            empty_with_classifications += 1
            print(f"⚠️  Empty segment has {num_classifications} classifications: {seg.get('text', '')[:50]}")
    else:
        # Non-empty segment
        if num_classifications == 8:
            nonempty_with_8 += 1
        elif num_classifications == 0:
            nonempty_with_0 += 1
        else:
            nonempty_with_less += 1
            # Show model names
            models = [c.get("model_name") for c in classifications]
            print(f"⚠️  Non-empty segment with {num_classifications}/8 labels")
            print(f"    Text: {text[:80]}")
            print(f"    Models: {models}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"\nEmpty segments:")
print(f"  With 0 classifications: {empty_with_0}")
print(f"  With >0 classifications: {empty_with_classifications}")

print(f"\nNon-empty segments:")
print(f"  With 8 classifications: {nonempty_with_8}")
print(f"  With 1-7 classifications: {nonempty_with_less}")
print(f"  With 0 classifications: {nonempty_with_0}")

print(f"\nTotal segments: {len(segments)}")
print(f"Segments with text: {nonempty_with_8 + nonempty_with_less + nonempty_with_0}")
print(f"Empty segments: {empty_with_0 + empty_with_classifications}")

# Calculate coverage different ways
print("\n" + "="*60)
print("COVERAGE CALCULATIONS")
print("="*60)

# Pipeline method (ignores empty)
pipeline_total = nonempty_with_8 + nonempty_with_less + nonempty_with_0
pipeline_complete = nonempty_with_8
pipeline_coverage = (pipeline_complete / pipeline_total * 100) if pipeline_total > 0 else 0
print(f"\nPipeline method (non-empty only):")
print(f"  Complete: {pipeline_complete}/{pipeline_total} = {pipeline_coverage:.1f}%")

# Dashboard method (counts all)
dashboard_total = len(segments)
dashboard_complete = nonempty_with_8  # Only segments with exactly 8
dashboard_coverage = (dashboard_complete / dashboard_total * 100) if dashboard_total > 0 else 0
print(f"\nDashboard method (all segments):")
print(f"  Complete: {dashboard_complete}/{dashboard_total} = {dashboard_coverage:.1f}%")

print(f"\nDifference: {abs(pipeline_coverage - dashboard_coverage):.1f} percentage points")
