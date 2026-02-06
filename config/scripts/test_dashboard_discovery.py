#!/usr/bin/env python3
"""Test dashboard shows discovery."""

import sys
from pathlib import Path

# add src to path.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from terminal_data_visualizer.config import TRANSCRIPTS_DIR, OUTPUTS_PATH
from terminal_data_visualizer.dashboard import get_all_shows

print(f"TRANSCRIPTS_DIR: {TRANSCRIPTS_DIR}")
print(f"Full path: {OUTPUTS_PATH / TRANSCRIPTS_DIR}")
print(f"Exists: {(OUTPUTS_PATH / TRANSCRIPTS_DIR).exists()}")

shows = get_all_shows()
print(f"\nShows found: {len(shows)}")

if shows:
    print(f"\nFirst 5 shows:")
    for show in shows[:5]:
        print(f"  - {show}")
else:
    print("\n⚠️  No shows found!")
