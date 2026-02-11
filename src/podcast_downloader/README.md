# Podcast Downloader

Download podcast episodes from RSS feeds and YouTube with automatic feed discovery, concurrent downloads, and metadata management.

## Overview

This package provides a CLI tool and Python API for downloading podcast episodes. It supports automatic podcast discovery via iTunes/Podcast Index, RSS feed parsing, concurrent downloads, date filtering, random sampling, and metadata tracking.

## Features

- **Feed Discovery**: Find podcasts by name using iTunes Search API
- **RSS Parsing**: Parse standard RSS/Atom podcast feeds
- **YouTube Support**: Download from YouTube podcast channels
- **Concurrent Downloads**: Parallel downloads with configurable concurrency
- **Episode Filtering**: Filter by date range, max count, or random sample
- **Audio Validation**: Verify downloaded files are valid audio
- **Metadata Storage**: Track downloaded content with JSON metadata
- **Rate Limiting**: Configurable delays between requests
- **Retry Logic**: Exponential backoff for failed downloads
- **Rich CLI**: Progress bars and formatted output

## Installation

The package is automatically installed with the main project:

```bash
uv sync
```

## CLI Usage

### Discover Podcasts

Find podcast feeds by searching:

```bash
# Search for podcasts
uv run python -m podcast_downloader discover "The Daily"
uv run python -m podcast_downloader discover "true crime" --limit 20
```

### Download Episodes

```bash
# Download by podcast name (uses iTunes Search API)
uv run python -m podcast_downloader download "The Daily"

# Download with max episodes
uv run python -m podcast_downloader download "The Daily" --max-episodes 10

# Download from direct RSS feed URL
uv run python -m podcast_downloader download --feed-url https://example.com/feed.xml

# Download with date filtering
uv run python -m podcast_downloader download "The Ezra Klein Show" \
    --start-date 2025-01-01 \
    --end-date 2025-06-30

# Random sample of episodes
uv run python -m podcast_downloader download "Lex Fridman Podcast" \
    --random-sample 5

# Multiple podcasts from file
uv run python -m podcast_downloader download -f podcasts.txt --max-episodes 5

# Custom output directory
uv run python -m podcast_downloader download "The Daily" -o downloads/podcasts

# Increase concurrent downloads
uv run python -m podcast_downloader download "The Daily" --concurrent 8
```

### List Downloaded Content

```bash
# List downloaded podcasts
uv run python -m podcast_downloader list

# List from custom metadata directory
uv run python -m podcast_downloader list --metadata-dir ./my-metadata
```

### Cleanup

```bash
# Clean up small/incomplete files (dry run)
uv run python -m podcast_downloader cleanup --dry-run

# Clean up files smaller than 500KB
uv run python -m podcast_downloader cleanup --min-size 500
```

## Python API

### Basic Download

```python
import asyncio
from pathlib import Path
from podcast_downloader import (
    DownloadConfig,
    EpisodeFilter,
    download_podcast,
    download_podcasts,
)

async def main():
    # Configure download
    config = DownloadConfig(
        output_dir=Path("downloads"),
        skip_existing=True,
        max_concurrent_downloads=4,
    )

    # Episode filter
    episode_filter = EpisodeFilter(
        max_episodes=10,
        start_date=None,
        end_date=None,
    )

    # Download single podcast
    stats, results = await download_podcast(
        "The Daily",
        config,
        episode_filter,
    )

    print(f"Downloaded: {stats.episodes_downloaded}")
    print(f"Skipped: {stats.episodes_skipped}")
    print(f"Failed: {stats.episodes_failed}")

asyncio.run(main())
```

### Download Multiple Podcasts

```python
import asyncio
from podcast_downloader import DownloadConfig, EpisodeFilter, download_podcasts

async def main():
    config = DownloadConfig(output_dir=Path("downloads"))
    episode_filter = EpisodeFilter(max_episodes=5)

    podcasts = [
        "The Daily",
        "The Ezra Klein Show",
        "Lex Fridman Podcast",
    ]

    stats, results = await download_podcasts(
        podcasts,
        config,
        episode_filter,
    )

    print(f"Podcasts processed: {stats.podcasts_processed}")
    print(f"Total downloaded: {stats.episodes_downloaded}")

asyncio.run(main())
```

### Discover Feeds

```python
import asyncio
from podcast_downloader import discover_feeds

async def main():
    feeds = await discover_feeds("machine learning", limit=10)

    for feed in feeds:
        print(f"{feed.title}")
        print(f"  Author: {feed.author}")
        print(f"  Episodes: {feed.episode_count}")
        print(f"  Feed: {feed.feed_url}")
        print()

asyncio.run(main())
```

### Parse RSS Feed

```python
from podcast_downloader import parse_feed

async def main():
    podcast = await parse_feed("https://example.com/feed.xml")

    print(f"Title: {podcast.title}")
    print(f"Episodes: {len(podcast.episodes)}")

    for episode in podcast.episodes[:5]:
        print(f"  - {episode.title} ({episode.published})")

asyncio.run(main())
```

## CLI Options

### `download` Command

```
USAGE: podcast-dl download [OPTIONS] [PODCASTS]...

Arguments:
  PODCASTS       Podcast names to download

Options:
  -f, --file PATH           File with podcast names (one per line)
  -u, --feed-url URL        Direct RSS feed URL
  -n, --max-episodes INT    Maximum episodes per podcast
  -r, --random-sample INT   Random sample N episodes
  --start-date YYYY-MM-DD   Earliest episode date
  --end-date YYYY-MM-DD     Latest episode date
  -o, --output-dir PATH     Output directory (default: downloads)
  --skip-existing/--overwrite   Skip/overwrite existing files (default: skip)
  -c, --concurrent INT      Concurrent downloads (default: 4)
```

### `discover` Command

```
USAGE: podcast-dl discover [OPTIONS] QUERY

Arguments:
  QUERY          Search query

Options:
  -n, --limit INT   Maximum results (default: 10)
```

### `list` Command

```
USAGE: podcast-dl list [OPTIONS]

Options:
  -m, --metadata-dir PATH   Metadata directory (default: metadata)
```

### `cleanup` Command

```
USAGE: podcast-dl cleanup [OPTIONS]

Options:
  -o, --output-dir PATH   Downloads directory (default: downloads)
  --min-size INT          Minimum file size in KB (default: 100)
  --dry-run               Show what would be deleted
```

## Configuration

### DownloadConfig

```python
class DownloadConfig(BaseModel):
    output_dir: Path = Path("downloads")
    metadata_dir: Path = Path("metadata")
    max_concurrent_downloads: int = 4  # 1-20
    rate_limit_delay: float = 1.0      # seconds
    max_retries: int = 3
    retry_backoff_base: float = 2.0
    connect_timeout: float = 15.0
    read_timeout: float = 300.0
    enable_feed_fallback: bool = True
    max_alternative_feeds: int = 3
    skip_existing: bool = True
    audio_format: str = "mp3"  # mp3, m4a, wav, ogg, flac
    validate_audio: bool = True
```

### EpisodeFilter

```python
class EpisodeFilter(BaseModel):
    max_episodes: int | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    random_sample: int | None = None
```

## Output Structure

Downloaded files are organized by podcast:

```
downloads/
├── the_daily/
│   ├── 2025-01-15_episode-title.mp3
│   ├── 2025-01-14_another-episode.mp3
│   └── ...
├── lex_fridman_podcast/
│   ├── 2025-01-10_guest-name.mp3
│   └── ...
└── ...

metadata/
├── the_daily.json
├── lex_fridman_podcast.json
└── ...
```

### Metadata Format

```json
{
  "title": "The Daily",
  "feed_url": "https://...",
  "last_updated": "2025-01-15T12:00:00Z",
  "episodes": [
    {
      "guid": "unique-id",
      "title": "Episode Title",
      "published": "2025-01-15T06:00:00Z",
      "audio_url": "https://...",
      "file_path": "downloads/the_daily/2025-01-15_episode-title.mp3",
      "downloaded_at": "2025-01-15T12:00:00Z"
    }
  ]
}
```

## Feed Discovery

Podcasts are discovered using these sources (in order):

1. **iTunes Search API**: Primary source for most podcasts
2. **Direct URL**: If a feed URL is provided
3. **Fallback feeds**: Alternative feeds if primary fails

### Search Tips

- Use exact podcast names for best results
- Popular podcasts have higher match confidence
- Use `--limit` to see more results when searching

## Architecture

### Components

- **cli.py**: CLI commands (download, discover, list, cleanup)
- **config.py**: Configuration models (DownloadConfig, EpisodeFilter)
- **discovery.py**: Feed discovery via iTunes API
- **downloader.py**: Async download pipeline
- **feed_parser.py**: RSS/Atom feed parsing
- **metadata.py**: Metadata storage and tracking
- **retry.py**: Retry logic with exponential backoff
- **providers/**: Download providers (RSS, YouTube)

### Data Flow

```
Podcast Name/URL
      ↓
discover_feeds() → Search iTunes API
      ↓
select_best_feed() → Choose highest confidence match
      ↓
parse_feed() → Parse RSS/Atom feed
      ↓
filter_episodes() → Apply date/count filters
      ↓
download_episodes_concurrent() → Parallel downloads
      ↓
validate_audio_file() → Verify audio integrity
      ↓
metadata_store.save() → Save metadata
```

## Transcript Placeholders

When episodes are downloaded, the downloader can optionally create empty placeholder JSON files in the `transcripts_dir` (default: `outputs/transcripts`). These files serve several purposes:

1.  **Pipeline Orchestration**: They signal to subsequent pipeline stages (like transcription) which episodes are available for processing, even before the actual audio transcription is complete.
2.  **Skipping Duplicates**: They allow the transcription pipeline to easily identify and skip episodes that have already been "queued" for transcription, preventing redundant processing.
3.  **Consistent Structure**: Ensures a consistent directory structure for all processed episodes from the very beginning of the pipeline.

These placeholder files contain only basic metadata and are later populated with actual transcript data by the transcription stage.

## Troubleshooting

### No Feeds Found

**Issue:** `no feeds found` when searching

**Fix:**
1. Try exact podcast name
2. Check spelling
3. Use direct feed URL with `--feed-url`

### Download Failures

**Issue:** Episodes fail to download

**Fix:**
1. Increase retries in config
2. Check network connection
3. Verify feed URL is accessible
4. Try with `--concurrent 1` for rate-limited servers

### Validation Errors

**Issue:** `downloaded file failed audio validation`

**Fix:**
1. Re-download the episode
2. Check if source file is valid
3. Disable validation: `config.validate_audio = False`

### Rate Limiting

**Issue:** Downloads fail with 429 errors

**Fix:**
1. Reduce concurrency: `--concurrent 2`
2. Increase rate limit delay in config

## Credits

Built with:
- [httpx](https://www.python-httpx.org/) - Async HTTP client
- [feedparser](https://feedparser.readthedocs.io/) - RSS/Atom parser
- [Pydantic](https://docs.pydantic.dev/) - Data validation
- [Rich](https://rich.readthedocs.io/) - Terminal UI
- [iTunes Search API](https://developer.apple.com/library/archive/documentation/AudioVideo/Conceptual/iTuneSearchAPI/) - Podcast discovery
