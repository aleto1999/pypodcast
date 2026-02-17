# Terminal Data Visualizer

Interactive terminal-based tool for exploring, analyzing, and visualizing podcast conversation data with parallel processing capabilities and high-resolution screen capture.

## Overview

The Terminal Data Visualizer is a comprehensive CLI application that enables users to browse, search, and visualize podcast analysis outputs. It provides statistical analysis, keyword searching, and terminal-based visualizations for podcast transcripts, speaker diarization, and keyword analysis data. **Every screen can be saved as a high-resolution image file (SVG/JPG) for reports and presentations.**

## Features

### NEW in v2.2: Advanced Analysis Tools

- **🎯 Outputs Dashboard** - Health overview showing completeness of outputs per show
- **📺 Episode View** - All outputs for a single episode in one place
- **🔎 Global Search** - Search across transcripts, keywords, and LLM annotations
- **📋 Summary Generator** - Show-level summaries with cross-show comparison
- **📊 Report Generator** - Research-ready reports in Markdown/JSON
- **📈 Advanced Visualizations** - Timelines, heatmaps, speaker interactions
- **🔖 Bookmarks** - Save and restore filter presets and views

### 1. Data Browsing & Search

- **📊 Scan outputs/ directory** - Get comprehensive folder statistics with parallel processing
- **🌳 View analysis folder structure** - Browse the analysis directory tree
- **📄 Browse analysis files & keywords** - Navigate through keyword analysis by category
- **🔍 Search keywords globally** - Search across all podcasts simultaneously
- **📁 Search specific files** - Find and view individual analysis files

### 2. Statistical Analysis & Visualizations

- **📊 Speaker Distribution Analysis** - Shows participation metrics per speaker
  - Segment counts
  - Word counts
  - Speaking duration
  - Horizontal bar charts with percentages
  
- **📂 Category Distribution** - Visualizes keyword match distribution
  - Match counts per category
  - Percentage distribution
  - Sorted by frequency
  
- **⏱️ Keyword Timeline** - Temporal distribution of keywords
  - Episode divided into time bins
  - Shows when topics appear
  - Identifies conversation patterns
  
- **📈 Episode Comparison** - Compare multiple episodes side-by-side
  - Speaker counts
  - Segment counts
  - Total words
  - Episode duration
  - Summary statistics

### 3. Output Files Viewer

- **📋 Document Labels** - View episode classifications and hate speech detection
  - Episode-level labels (POSITIVE/NEGATIVE)
  - Show-level summaries
  - Label distribution statistics
  - Per-model classification breakdowns
  
- **📊 Features Explorer** - Interactive CSV data exploration with terminal plots
  - **Overview Statistics** - Summary metrics across all episodes
    - Mean, median, min/max for key features
    - Total episodes and shows analyzed
  - **Per-Show Analysis** - Aggregate statistics by podcast show
    - Compare shows across any metric with horizontal bar charts
    - Episode counts and distributions
    - Custom metric selection
    - Terminal-based visualizations for top 15 shows
    - Save plots as HTML or TXT files
  - **Per-Episode Details** - Detailed view of individual episodes
    - All 687 feature columns
    - Core metrics (duration, turns, questions, words)
    - Aggregate features (politeness, switch time, dominance)
    - Per-speaker metrics (speaking time, turns, questions, TTR)
  - **Feature Distributions** - Statistical analysis of feature values
    - Distribution statistics (mean, median, std dev, range)
    - Interactive histograms with 20 bins rendered in terminal
    - Quartile analysis
    - Save histograms as HTML or TXT
  - **Custom Filtering** - Filter episodes by feature criteria
    - Set minimum/maximum thresholds
    - Multi-criterion filtering
    - Export filtered results
  - **Feature Correlations** - Analyze relationships between features
    - Pearson correlation coefficient
    - Terminal-based scatter plots with correlation visualization
    - Correlation strength interpretation
    - Save scatter plots as HTML or TXT
  - **Top/Bottom Rankings** - Identify extreme values
    - Top 10 and bottom 10 episodes per feature
    - Horizontal bar charts for visual comparison
    - Compare across shows
    - Identify outliers and interesting cases
    - Save ranking charts as HTML or TXT
  
- **🤖 LLM Annotations** - Explore LLM-based content analysis
  - Hate speech detection results
  - Advertisement identification
  - Target group identification
  - Classification confidence scores
  - Model-by-model predictions

### 4. Screen Capture

**🖼️ Save Any Screen as High-Resolution Image**

- Press `s` on any visualization or data view screen to save it as an image
- Automatic timestamped filenames
- Outputs saved to `outputs/terminal_viewer/screenshots/`
- Formats:
  - **SVG** (Scalable Vector Graphics) - Always available, perfect quality at any resolution
  - **JPG** (High-resolution bitmap) - Available if system has Cairo library installed
- DPI: 300 (print quality)
- Use cases: Reports, presentations, documentation, sharing results

## Installation

The package is part of the podcast-conversations project. Dependencies are managed via `uv`.

```bash
# install project dependencies.
uv sync

# run the terminal visualizer.
uv run python -m terminal_data_visualizer.main
```

### Optional: High-Resolution JPG Export

For JPG export (in addition to SVG), install the Cairo system library:

```bash
# macOS
brew install cairo

# Ubuntu/Debian
sudo apt-get install libcairo2-dev

# Fedora
sudo dnf install cairo-devel
```

**Note**: SVG format is already available without additional setup and provides perfect quality for all use cases.

## Usage

### Interactive Menu

Launch the interactive menu:

```bash
uv run python -m terminal_data_visualizer.main
```

### Saving Screens

On any visualization or data screen:
1. View the content
2. Press `s` when prompted
3. Image is automatically saved with timestamp
4. File path is displayed
5. Continue viewing other content

Screenshots are saved to: `outputs/terminal_viewer/screenshots/`

### Navigation

```
Main Menu (v2.2)
├─ 1. 🎯 Dashboard (outputs health & completeness)
├─ 2. 📺 Episode View (all outputs for single episode)
├─ 3. 🔎 Global Search (across all output types)
├─ 4. 📋 Summaries (show-level reports)
├─ 5. 📊 Reports (generate research reports)
├─ 6. 📈 Advanced Visualizations
│   ├─ 1. Timeline (hate speech/keywords over time)
│   ├─ 2. Heatmap (category distribution)
│   └─ 3. Interactions (speaker analysis)
├─ 7. 🔖 Bookmarks (saved presets)
├─ 8. 🔬 Features Explorer (CSV data analysis)
│   ├─ 1. Overview Statistics
│   ├─ 2. Per-Show Analysis
│   ├─ 3. Per-Episode Details
│   ├─ 4. Feature Distributions
│   ├─ 5. Custom Filtering
│   ├─ 6. Feature Correlations
│   └─ 7. Top/Bottom Rankings
├─ 9. 📁 Legacy Menu (original features)
│   ├─ 1. Scan outputs/ directory
│   ├─ 2. View analysis folder structure
│   ├─ 3. Browse analysis files & keywords
│   ├─ 4. Search keywords across all podcasts
│   ├─ 5. Search and view specific file
│   ├─ 6. Generate statistics & visualizations
│   └─ 7. View output files
└─ q. Quit
```

### Programmatic Usage

Use the statistics module directly in your code:

```python
from pathlib import Path
from terminal_data_visualizer.statistics import (
    calculate_episode_stats,
    compare_episodes,
    display_category_distribution,
    display_keyword_timeline,
    display_speaker_distribution,
)

# speaker distribution.
transcript = Path("outputs/transcripts_postprocessed/podcast/episode.json")
stats = calculate_episode_stats(transcript)
if stats:
    display_speaker_distribution(stats)

# category distribution.
keyword_file = Path("outputs/analysis/keyword_analysis/podcast/episode.json")
display_category_distribution(keyword_file)

# keyword timeline.
display_keyword_timeline(keyword_file)

# compare episodes.
transcript_files = list(Path("outputs/transcripts_postprocessed/podcast").glob("*.json"))
compare_episodes(transcript_files)
```

## Data Sources

The visualizer works with existing data in the `outputs/` directory:

- **`outputs/transcripts_postprocessed/`** - Transcripts with speaker labels for speaker statistics
- **`outputs/analysis/keyword_analysis/`** - Keyword matches for category and timeline visualizations
- **`outputs/diarizations/`** - Speaker diarization data (RTTM format)
- **`outputs/document_labels/`** - Episode and show-level classification labels
- **`outputs/features/`** - Extracted features CSV (687 columns per episode)
- **`outputs/analysis/llm_annotation_method/`** - LLM-based content analysis (hate speech, ads)

## Example Output

### Speaker Distribution

```
Speaker Distribution - episode_name

┏━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ Speaker      ┃ Segments ┃  Words ┃ Duration ┃ Chart                ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ SPEAKER_01   │      145 │   8234 │   45.2m  │ ███████████ 55.2%    │
│ SPEAKER_02   │      118 │   5891 │   37.1m  │ █████████ 44.8%      │
└──────────────┴──────────┴────────┴──────────┴──────────────────────┘
```

### Document Labels Summary

```
Show Summary: the_joe_rogan_experience

Overall Label: NEGATIVE

Episode Statistics
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Metric           ┃ Value ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ Total episodes   │ 2625  │
│ Positive episodes│  326  │
│ Negative episodes│ 2299  │
└──────────────────┴───────┘
```

### LLM Annotations

```
Segment 1:
  Speaker: SPEAKER_00
  Time: 0.0s - 72.2s
  Text: I'm just a human being with a soul...
  Hate speech: False
  Advertisement: False
  Classifications:
    hate_speech_detection: NOT-HATE (99.89%)
    hostile_content: OFFENSIVE (99.99%)
    ad_content_detection: LABEL_0 (96.96%)
```

### Category Distribution

```
Category Distribution - episode_name

┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Category                 ┃  Matches ┃ Chart                          ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ women                    │       93 │ ███████████████████ 44.1%      │
│ racial_ethnic_minorities │       83 │ ████████████████ 39.3%         │
│ lgbtq                    │       15 │ ███ 7.1%                       │
└──────────────────────────┴──────────┴────────────────────────────────┘
```

### Keyword Timeline

```
Keyword Timeline - episode_name

┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Time Range         ┃  Matches ┃ Distribution                     ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│   0.0 -  20.0m     │        9 │ ▓▓▓▓▓▓▓▓▓▓                       │
│  20.0 -  40.0m     │       26 │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓    │
│  40.0 -  60.0m     │       35 │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │
└────────────────────┴──────────┴──────────────────────────────────┘
```

## Architecture

### Package Structure

```
src/terminal_data_visualizer/
├── __init__.py              # Package initialization
├── main.py                  # Entry point & interactive menu
├── models.py                # Type-safe dataclasses
├── config.py                # Configuration constants
├── cache.py                 # Performance caching layer
├── state.py                 # Session persistence
├── selectors.py             # Reusable UI selectors
├── statistics.py            # Visualization & statistics module
├── outputs_viewer.py        # Output files viewer (labels, features, LLM)
├── features_explorer.py     # NEW: Interactive CSV data explorer with aggregations
├── screen_capture.py        # Screen capture to SVG/JPG
├── keyword_browser.py       # Keyword analysis data handling
├── scanner.py               # Directory scanning with parallel processing
├── tree_viewer.py           # Tree structure visualization
├── visualizer.py            # Core display functions
├── exporter.py              # JSON export functionality
├── data_export.py           # CSV/JSON export utilities
├── batch_processor.py       # Batch analysis tools
│
│   # NEW in v2.2
├── dashboard.py             # Outputs health dashboard
├── episode_view.py          # Episode-centric viewer
├── summary_generator.py     # Show-level summaries
├── global_search.py         # Cross-output search
├── bookmarks.py             # Bookmark management
├── report_generator.py      # Report generation
├── classification_viewer.py # Classification data viewer
├── classification_stats.py  # Classification statistics
├── llm_annotation_viewer.py # LLM annotation viewer
├── llm_annotation_stats.py  # LLM annotation statistics
│
├── visualizations/          # Advanced visualizations
│   ├── __init__.py          # Package exports
│   ├── timeline.py          # Timeline charts
│   ├── heatmap.py           # Category heatmaps
│   └── interactions.py      # Speaker interaction graphs
│
└── README.md                # This file
```

### Design Principles

1. **Terminal-based only** - All visualizations render in terminal using Unicode characters
2. **Uses existing data** - No new data generation, works with outputs/ directory
3. **Minimal dependencies** - Pillow already in project, cairosvg optional for JPG
4. **Consistent styling** - Unified color scheme across all features
5. **Parallel processing** - ThreadPoolExecutor for fast data scanning
6. **Simple and focused** - Essential features without over-engineering
7. **High-quality exports** - SVG (vector) and optional JPG (300 DPI) screen capture

### Color Scheme

- **Cyan** - Headers, titles, prompts
- **Green** - Speaker distribution bars, success messages
- **Yellow** - Category distribution bars, warnings
- **Blue** - Keyword timeline bars, info
- **Magenta** - Table headers
- **Red** - Errors

## Technical Details

### Screen Capture

Screen capture uses Rich's built-in SVG export with optional conversion to JPG:

```python
from terminal_data_visualizer.screen_capture import save_current_screen

# console must have record=True
console = Console(record=True)

# ... render your content ...

# save screen
saved_path = save_current_screen(console, screen_name="my_visualization")
```

**Export Pipeline**:
1. Console records all output (with `record=True`)
2. Export to SVG using Rich's `export_svg()`
3. If cairosvg available: Convert SVG → PNG @ 300 DPI
4. If Pillow available: Convert PNG → JPG (95% quality)
5. Fallback: Keep as SVG (perfect quality, smaller file size)

### Data Classes

```python
@dataclass
class SpeakerStats:
    """Statistics for a speaker."""
    speaker: str
    segment_count: int
    total_words: int
    total_duration: float
    avg_segment_length: float

@dataclass
class EpisodeStats:
    """Statistics for an episode."""
    filename: str
    total_segments: int
    total_duration: float
    speaker_count: int
    total_words: int
    speakers: dict[str, SpeakerStats]
```

### Core Functions

#### Statistics Module

- `calculate_speaker_stats()` - Process transcript segments into speaker statistics
- `calculate_episode_stats()` - Build complete episode statistics
- `display_speaker_distribution()` - Render speaker participation chart
- `display_category_distribution()` - Show keyword category distribution
- `display_keyword_timeline()` - Temporal keyword distribution
- `compare_episodes()` - Side-by-side episode comparison

#### Keyword Browser

- `get_subfolders()` - Get available podcast folders
- `get_all_keywords_parallel()` - Extract keywords with parallel processing
- `get_all_categories_parallel()` - Extract categories with parallel processing
- `search_keywords_parallel()` - Search across files concurrently
- `display_keyword_matches()` - Render search results

### Chart Rendering

- Uses Unicode block characters: `█` (full), `▓` (shaded)
- Auto-scaling based on terminal width (configurable)
- Percentage calculations with 1 decimal precision
- Duration formatting in minutes for readability

## Development

### Code Quality

- **Linting**: 100% pass with ruff (`uv run ruff check`)
- **Formatting**: Consistent with ruff formatter (`uv run ruff format`)
- **Type hints**: Full type annotations throughout
- **Comments**: Lowercase with periods (project convention)
- **Naming**: Descriptive snake_case

### Testing

```bash
# test all imports.
uv run python -c "
from terminal_data_visualizer.statistics import *
print('✅ All imports successful')
"

# run demo.
uv run python -c "
from pathlib import Path
from terminal_data_visualizer.statistics import calculate_episode_stats, display_speaker_distribution

transcript = Path('outputs/transcripts_postprocessed/podcast_name/episode.json')
stats = calculate_episode_stats(transcript)
if stats:
    display_speaker_distribution(stats)
"
```

### Adding New Visualizations

1. Add function to `src/terminal_data_visualizer/statistics.py`
2. Add menu option to `visualizations_menu()` in `main.py`
3. Create corresponding `*_menu()` function in `main.py`
4. Follow existing patterns (see any current menu function)

Example pattern:

```python
def new_visualization_menu() -> None:
    """Display new visualization."""
    clear_screen()
    console.print("[bold cyan]New Visualization[/bold cyan]\n")
    
    # 1. get available data.
    subfolders = get_subfolders()
    if not subfolders:
        console.print("[yellow]No data found[/yellow]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return
    
    # 2. let user select.
    # ... selection logic ...
    
    # 3. display visualization.
    display_new_visualization(selected_data)
    
    # 4. wait for user.
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")
```

## Features NOT Implemented

The following features were intentionally excluded as they require data not available in the current codebase:

- ❌ **Toxicity heatmaps** - No toxicity scores in outputs
- ❌ **Sentiment analysis graphs** - No sentiment data available
- ❌ **Political orientation charts** - Not applicable
- ❌ **Speaker profiles** - Data not present

> **Note**: SVG/JPG export IS implemented - press `s` on any visualization screen to save.

## Performance

- **Parallel processing** - Uses ThreadPoolExecutor throughout
- **Lazy loading** - Data loaded only when needed
- **Efficient scanning** - Concurrent file operations
- **Memory efficient** - Streaming approach for large datasets

## Export Functionality

Export search results and analysis data to JSON:

```python
from terminal_data_visualizer.exporter import export_to_json

# export search results.
export_to_json(matches, "search_results.json")

# default export path: outputs/terminal_viewer/
```

## Requirements

- Python 3.11+
- Rich library (for terminal UI and SVG export)
- Pillow (for image processing) - already in project dependencies
- pathlib (stdlib)
- json (stdlib)
- csv (stdlib)
- collections (stdlib)
- concurrent.futures (stdlib)

**Optional**:
- cairosvg - For JPG export (requires Cairo system library)
- Cairo system library - For high-resolution bitmap conversion

All core dependencies are managed via `pyproject.toml` and installed with `uv sync`.


### Design Decisions

- **Terminal-only rendering** - Avoided adding Plotly/Matplotlib dependencies
- **Adapted from reference** - Took concepts from other codebase but simplified
- **Data-driven** - Only implemented features with available data
- **No 1:1 replication** - Created custom visualizations suited to this project
- **Minimal complexity** - 4 focused visualizations vs. many complex ones

## Contributing

When adding features:

1. Follow existing code patterns
2. Use pathlib for all paths
3. Add type hints to all functions
4. Keep functions focused and single-purpose
5. Use Rich library for all terminal output
6. Test with ruff linting and formatting
7. Update documentation (this README)

## License

Part of the podcast-conversations project.

## Contact & Support

For issues, questions, or contributions, refer to the main project repository.

## See Also

- Main project [README.md](../../README.md)
- Main project [claude-docs](../../claude-docs/README.md) - Development guidance
