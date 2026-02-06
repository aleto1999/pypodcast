"""cli commands for podcast downloader."""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table

from podcast_downloader.config import DownloadConfig, DownloadStats, EpisodeFilter, FeedSearchResult
from podcast_downloader.discovery import discover_feeds, select_best_feed
from podcast_downloader.downloader import download_podcast, download_podcasts
from podcast_downloader.feed_parser import fetch_feed_metadata
from podcast_downloader.metadata import MetadataStore

console = Console()
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False, debug: bool = False, log_file: Optional[str] = None) -> None:
    """configure logging based on verbosity level."""
    level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING)
    
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def select_feed_interactive(
    feeds: list[FeedSearchResult],
    query: str,
    max_choices: int = 3,
) -> FeedSearchResult | None:
    """display top feed matches and let user select one interactively.

    args:
        feeds: list of feed search results to choose from.
        query: original search query for display.
        max_choices: maximum number of choices to display.

    returns:
        selected feed or None if user cancels.
    """
    if not feeds:
        console.print("[yellow]no feeds found[/yellow]")
        return None

    # limit to max_choices.
    display_feeds = feeds[:max_choices]

    console.print()
    console.print(f"[cyan]search results for:[/cyan] {query}")
    console.print()

    # build selection table.
    table = Table(show_header=True, header_style="bold")
    table.add_column("#", style="dim", width=3)
    table.add_column("title", style="cyan", max_width=35)
    table.add_column("author", style="green", max_width=20)
    table.add_column("episodes*", justify="right", width=10)
    table.add_column("source", width=12)
    table.add_column("feed url", style="dim", max_width=40)

    for idx, feed in enumerate(display_feeds, start=1):
        # format episode count.
        ep_count = str(feed.episode_count) if feed.episode_count else "?"

        # truncate feed url for display.
        feed_url_display = feed.feed_url
        if len(feed_url_display) > 40:
            feed_url_display = feed_url_display[:37] + "..."

        table.add_row(
            str(idx),
            feed.title[:35] if feed.title else "Unknown",
            (feed.author or "")[:20],
            ep_count,
            feed.source,
            feed_url_display,
        )

    console.print(table)
    console.print("[dim]*Episode count from search API; actual RSS/YouTube feed may contain fewer.[/dim]")
    console.print()

    # build prompt choices.
    valid_choices = [str(i) for i in range(1, len(display_feeds) + 1)]
    valid_choices.append("c")  # cancel option.

    choice_hint = ", ".join(valid_choices[:-1]) + ", or [c]ancel"
    console.print(f"[dim]enter choice ({choice_hint}):[/dim]")

    choice = Prompt.ask(
        "select podcast",
        choices=valid_choices,
        default="1",
        show_choices=False,
    )

    if choice.lower() == "c":
        console.print("[yellow]cancelled[/yellow]")
        return None

    selected_idx = int(choice) - 1
    selected_feed = display_feeds[selected_idx]
    console.print(f"[green]✓[/green] selected: {selected_feed.title}")

    return selected_feed


def select_feeds_batch(
    show_names: list[str],
    max_choices: int = 3,
    auto_select: bool = False,
) -> dict[str, str | None]:
    """select feeds for multiple shows in batch mode.

    args:
        show_names: list of podcast names to search.
        max_choices: max choices to display per show.
        auto_select: if True, auto-select best feed without prompting.

    returns:
        dict mapping show names to selected feed urls (or None if skipped).
    """
    selected_feeds: dict[str, str | None] = {}

    async def discover_all():
        results = {}
        for name in show_names:
            feeds = await discover_feeds(name, limit=max(max_choices, 10))
            results[name] = feeds
        return results

    console.print(f"\n[bold blue]🔍 RSS Feed Discovery[/bold blue]")
    console.print(f"Searching for {len(show_names)} podcast(s)...\n")

    feed_results = asyncio.run(discover_all())

    for i, (name, feeds) in enumerate(feed_results.items(), 1):
        console.print(f"[cyan]({i}/{len(show_names)}) {name}[/cyan]")

        if not feeds:
            console.print(f"  [red]❌ No feeds found[/red]\n")
            selected_feeds[name] = None
            continue

        if auto_select:
            # auto-select best feed by episode count and confidence.
            best_feed = select_best_feed(feeds, query=name)
            if best_feed:
                console.print(
                    f"  [green]✅ Auto-selected: {best_feed.title} "
                    f"({best_feed.episode_count or '?'} episodes)[/green]\n"
                )
                selected_feeds[name] = best_feed.feed_url
            else:
                selected_feeds[name] = None
        else:
            selected = select_feed_interactive(feeds, name, max_choices=max_choices)
            selected_feeds[name] = selected.feed_url if selected else None
            console.print()

    return selected_feeds


def parse_date(date_str: str) -> datetime:
    """parse date string in YYYY-MM-DD format."""
    return datetime.strptime(date_str, "%Y-%m-%d")


@click.group()
@click.version_option(version="0.1.0")
@click.option("--verbose", "-v", is_flag=True, help="enable verbose logging")
@click.option("--debug", is_flag=True, help="enable debug logging")
@click.option("--log-file", type=click.Path(), help="log file path")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, debug: bool, log_file: Optional[str]):
    """podcast downloader - download episodes from rss feeds."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["debug"] = debug
    setup_logging(verbose, debug, log_file)


@cli.command()
@click.argument("podcasts", nargs=-1, required=False)
@click.option("--shows", "-s", type=str, help="comma-separated list of podcast names")
@click.option("--file", "-f", type=click.Path(exists=True), help="file with podcast names")
@click.option("--feed-url", "-u", help="direct rss feed url")
@click.option("--max-episodes", "-n", type=int, help="max episodes per podcast")
@click.option("--random-sample", "-r", type=int, help="random sample n episodes")
@click.option("--start-date", type=str, help="earliest episode date (YYYY-MM-DD)")
@click.option("--end-date", type=str, help="latest episode date (YYYY-MM-DD)")
@click.option("--output-dir", "-o", type=click.Path(), default="outputs/downloads", help="output directory")
@click.option("--transcripts-dir", type=click.Path(), default="outputs/transcripts", help="transcripts output directory")
@click.option("--skip-existing/--overwrite", default=True, help="skip already downloaded")
@click.option("--concurrent", "-c", type=int, default=4, help="concurrent downloads")
@click.option("--parallel", "-p", is_flag=True, help="enable parallel processing (same as default)")
@click.option("--no-confirm", "-y", is_flag=True, help="skip interactive selection, use best match")
@click.option("--interactive-feeds", is_flag=True, help="enable interactive rss feed selection")
@click.option("--auto-select-feeds", is_flag=True, help="automatically select best rss feed for each show")
@click.option("--matches", "-m", type=int, default=3, help="number of matches to show (default: 3)")
@click.pass_context
def download(
    ctx: click.Context,
    podcasts: tuple[str, ...],
    shows: str | None,
    file: str | None,
    feed_url: str | None,
    max_episodes: int | None,
    random_sample: int | None,
    start_date: str | None,
    end_date: str | None,
    output_dir: str,
    transcripts_dir: str,
    skip_existing: bool,
    concurrent: int,
    parallel: bool,
    no_confirm: bool,
    interactive_feeds: bool,
    auto_select_feeds: bool,
    matches: int,
):
    """download podcast episodes.

    examples:
        podcast-dl download "The Daily"
        podcast-dl download --shows "The Daily,Crime Junkie" --auto-select-feeds
        podcast-dl download "The Ezra Klein Show" --matches 5
        podcast-dl download --feed-url https://example.com/feed.xml
        podcast-dl download -f podcasts.txt --max-episodes 10 --no-confirm
    """
    # validate conflicting options.
    if interactive_feeds and auto_select_feeds:
        console.print("[red]error: cannot use both --interactive-feeds and --auto-select-feeds[/red]")
        console.print("[dim]choose either interactive or automatic selection, not both[/dim]")
        raise SystemExit(1)

    # collect podcast names from all sources.
    podcast_names = list(podcasts)

    # add from --shows option (comma-separated).
    if shows:
        podcast_names.extend(name.strip() for name in shows.split(",") if name.strip())

    # add from file.
    if file:
        with open(file) as f:
            podcast_names.extend(line.strip() for line in f if line.strip() and not line.startswith("#"))

    if not podcast_names and not feed_url:
        console.print("[red]error: provide podcast names, --shows, --file, or --feed-url[/red]")
        raise SystemExit(1)

    # check for known popular podcasts with direct rss feeds.
    from podcast_downloader.popular_podcasts import get_rss_feed_for_popular_podcast

    # build config.
    config = DownloadConfig(
        output_dir=Path(output_dir),
        transcripts_dir=Path(transcripts_dir),
        skip_existing=skip_existing,
        max_concurrent_downloads=concurrent,
    )

    # build filter.
    episode_filter = EpisodeFilter(
        max_episodes=max_episodes,
        random_sample=random_sample,
        start_date=parse_date(start_date) if start_date else None,
        end_date=parse_date(end_date) if end_date else None,
    )

    # resolve feed urls based on mode.
    resolved_feeds: dict[str, str] = {}  # podcast_name -> feed_url.

    if feed_url:
        # direct feed url provided - use it.
        podcast_name = podcast_names[0] if podcast_names else "podcast"
        resolved_feeds[podcast_name] = feed_url

    elif interactive_feeds or auto_select_feeds:
        # batch feed selection mode.
        resolved_feeds = {
            k: v for k, v in select_feeds_batch(
                podcast_names,
                max_choices=matches,
                auto_select=auto_select_feeds,
            ).items() if v is not None
        }

        if not resolved_feeds:
            console.print("[yellow]no podcasts selected[/yellow]")
            return

    elif no_confirm:
        # auto-select best feed without prompting.
        console.print(f"[blue]🔍 Auto-discovering feeds for {len(podcast_names)} podcast(s)...[/blue]")
        resolved_feeds = {
            k: v for k, v in select_feeds_batch(
                podcast_names,
                max_choices=matches,
                auto_select=True,
            ).items() if v is not None
        }

    else:
        # default: interactive selection for each podcast.
        for podcast_name in podcast_names:
            console.print(f"\n[cyan]searching for:[/cyan] {podcast_name}")

            # check popular podcasts database for a known feed.
            known_feed_url = get_rss_feed_for_popular_podcast(podcast_name)

            # run feed discovery and metadata fetching in parallel.
            async def _discover_and_fetch_metadata():
                tasks = [_discover_feeds_async(podcast_name, limit=max(matches, 10))]
                if known_feed_url:
                    tasks.append(fetch_feed_metadata(known_feed_url))
                return await asyncio.gather(*tasks)

            results = asyncio.run(_discover_and_fetch_metadata())
            feeds = results[0]
            known_metadata = results[1] if known_feed_url and len(results) > 1 else None

            # if we have a known feed and it's not already in results, add it first.
            if known_feed_url:
                # check if already present in discovered feeds.
                known_in_results = any(f.feed_url == known_feed_url for f in feeds)
                if not known_in_results:
                    # prepend the known feed as first option with fetched metadata.
                    if known_metadata:
                        known_feed = FeedSearchResult(
                            title=f"{known_metadata.title} ★",
                            feed_url=known_feed_url,
                            author=known_metadata.author or "",
                            description=known_metadata.description or "From popular podcasts database",
                            episode_count=known_metadata.episode_count,
                            source="popular_db",
                            confidence=1.0,
                        )
                    else:
                        # fallback if metadata fetch failed.
                        known_feed = FeedSearchResult(
                            title=f"{podcast_name} (Popular Podcasts DB)",
                            feed_url=known_feed_url,
                            author="",
                            description="Pre-configured feed from popular podcasts database",
                            episode_count=None,
                            source="popular_db",
                            confidence=1.0,
                        )
                    feeds.insert(0, known_feed)

            if not feeds:
                console.print(f"[yellow]no feeds found for '{podcast_name}', skipping[/yellow]")
                continue

            selected = select_feed_interactive(feeds, podcast_name, max_choices=matches)

            if selected:
                resolved_feeds[podcast_name] = selected.feed_url
            else:
                console.print(f"[yellow]skipping '{podcast_name}'[/yellow]")

        if not resolved_feeds:
            console.print("[yellow]no podcasts selected[/yellow]")
            return

    # run download.
    async def run():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            if feed_url:
                # single feed download.
                podcast_name = podcast_names[0] if podcast_names else "podcast"
                stats, results = await download_podcast(
                    podcast_name,
                    config,
                    episode_filter,
                    feed_url=feed_url,
                    progress=progress,
                )
            elif resolved_feeds:
                # download with resolved feed urls.
                total_stats = DownloadStats()
                all_results: list = []

                for podcast_name, resolved_url in resolved_feeds.items():
                    stats, results = await download_podcast(
                        podcast_name,
                        config,
                        episode_filter,
                        feed_url=resolved_url,
                        progress=progress,
                    )
                    # aggregate stats.
                    total_stats.podcasts_processed += stats.podcasts_processed
                    total_stats.episodes_found += stats.episodes_found
                    total_stats.episodes_downloaded += stats.episodes_downloaded
                    total_stats.episodes_skipped += stats.episodes_skipped
                    total_stats.episodes_failed += stats.episodes_failed
                    total_stats.total_bytes += stats.total_bytes
                    total_stats.errors.extend(stats.errors)
                    all_results.extend(results)

                return total_stats, all_results
            else:
                # no-confirm mode: use automatic best match.
                stats, results = await download_podcasts(
                    podcast_names,
                    config,
                    episode_filter,
                    progress=progress,
                )

            return stats, results

    stats, results = asyncio.run(run())

    # print summary.
    console.print()
    console.print(f"[green]✓[/green] podcasts processed: {stats.podcasts_processed}")
    console.print(f"[green]✓[/green] episodes found: {stats.episodes_found}")
    console.print(f"[green]✓[/green] episodes downloaded: {stats.episodes_downloaded}")
    console.print(f"[yellow]○[/yellow] episodes skipped: {stats.episodes_skipped}")
    console.print(f"[red]✗[/red] episodes failed: {stats.episodes_failed}")
    console.print(f"[blue]↓[/blue] total downloaded: {stats.total_bytes / (1024*1024):.1f} MB")

    if stats.errors:
        console.print()
        console.print("[red]errors:[/red]")
        for error in stats.errors[:10]:
            console.print(f"  • {error}")
        if len(stats.errors) > 10:
            console.print(f"  ... and {len(stats.errors) - 10} more")


async def _discover_feeds_async(query: str, limit: int = 10) -> list[FeedSearchResult]:
    """async wrapper for discover_feeds."""
    return await discover_feeds(query, limit=limit)


@cli.command()
@click.argument("query")
@click.option("--limit", "-n", type=int, default=10, help="max results to show")
def discover(query: str, limit: int):
    """discover podcast feeds by searching.

    examples:
        podcast-dl discover "true crime"
        podcast-dl discover "the daily" --limit 5
    """
    async def run():
        return await discover_feeds(query, limit=limit)

    console.print(f"[cyan]searching for:[/cyan] {query}")

    feeds = asyncio.run(run())

    if not feeds:
        console.print("[yellow]no feeds found[/yellow]")
        return

    table = Table(title=f"found {len(feeds)} feeds")
    table.add_column("title", style="cyan", max_width=35)
    table.add_column("author", style="green", max_width=20)
    table.add_column("episodes*", justify="right")
    table.add_column("source", width=12)
    table.add_column("feed url", style="dim", max_width=50)

    for feed in feeds:
        table.add_row(
            feed.title[:35],
            (feed.author or "")[:20],
            str(feed.episode_count or "?"),
            feed.source,
            feed.feed_url[:50] + "..." if len(feed.feed_url) > 50 else feed.feed_url,
        )

    console.print(table)
    console.print("[dim]*Episode count from search API; actual RSS/YouTube feed may contain fewer.[/dim]")


@cli.command("list")
@click.option("--metadata-dir", "-m", type=click.Path(), default="outputs/metadata", help="metadata dir")
def list_content(metadata_dir: str):
    """list downloaded content.

    examples:
        podcast-dl list
        podcast-dl list --metadata-dir ./my-metadata
    """
    store = MetadataStore(Path(metadata_dir))
    podcasts = store.list_podcasts()

    if not podcasts:
        console.print("[yellow]no downloaded content found[/yellow]")
        return

    table = Table(title="downloaded podcasts")
    table.add_column("podcast", style="cyan")
    table.add_column("episodes", justify="right")
    table.add_column("last updated")

    for podcast in podcasts:
        table.add_row(
            podcast.title[:40],
            str(len(podcast.episodes)),
            podcast.last_updated.strftime("%Y-%m-%d %H:%M"),
        )

    console.print(table)

    # show stats.
    stats = store.get_stats()
    console.print()
    console.print(f"total: {stats['total_podcasts']} podcasts, {stats['total_episodes']} episodes")
    console.print(f"size: {stats['total_size_mb']:.1f} MB")


@cli.command()
@click.option("--output-dir", "-o", type=click.Path(), default="outputs/downloads", help="downloads dir")
@click.option("--min-size", type=int, default=100, help="min file size in KB")
@click.option("--dry-run", is_flag=True, help="show what would be deleted")
def cleanup(output_dir: str, min_size: int, dry_run: bool):
    """clean up incomplete or small downloads.

    examples:
        podcast-dl cleanup --dry-run
        podcast-dl cleanup --min-size 500
    """
    downloads_path = Path(output_dir)
    if not downloads_path.exists():
        console.print("[yellow]downloads directory not found[/yellow]")
        return

    min_bytes = min_size * 1024
    files_to_delete = []

    for file in downloads_path.rglob("*.mp3"):
        if file.stat().st_size < min_bytes:
            files_to_delete.append(file)

    if not files_to_delete:
        console.print("[green]no files to clean up[/green]")
        return

    console.print(f"found {len(files_to_delete)} files smaller than {min_size}KB")

    for file in files_to_delete:
        size_kb = file.stat().st_size / 1024
        if dry_run:
            console.print(f"  would delete: {file} ({size_kb:.1f}KB)")
        else:
            file.unlink()
            console.print(f"  deleted: {file} ({size_kb:.1f}KB)")

    if not dry_run:
        # remove empty directories.
        for dir_path in sorted(downloads_path.rglob("*"), reverse=True):
            if dir_path.is_dir() and not any(dir_path.iterdir()):
                dir_path.rmdir()
                console.print(f"  removed empty dir: {dir_path}")


@cli.command("discover-feeds")
@click.option("--shows", "-s", type=str, help="comma-separated list of podcast names")
@click.option("--file", "-f", type=click.Path(exists=True), help="file with podcast names")
@click.option("--output-format", "-o", type=click.Choice(["table", "json"]), default="table", help="output format")
def discover_feeds_cmd(shows: str | None, file: str | None, output_format: str):
    """discover available rss feeds for podcast shows without downloading.

    examples:
        podcast-dl discover-feeds --shows "The Daily,Crime Junkie"
        podcast-dl discover-feeds -f podcasts.txt --output-format json
    """
    import json as json_lib

    # collect show names.
    show_names: list[str] = []

    if shows:
        show_names.extend(name.strip() for name in shows.split(",") if name.strip())

    if file:
        with open(file) as f:
            show_names.extend(line.strip() for line in f if line.strip() and not line.startswith("#"))

    if not show_names:
        console.print("[red]error: provide --shows or --file[/red]")
        raise SystemExit(1)

    console.print(f"[green]discovering rss feeds for {len(show_names)} podcast show(s)[/green]")

    # discover feeds for all shows.
    async def discover_all():
        results = {}
        for name in show_names:
            feeds = await discover_feeds(name, limit=10)
            results[name] = feeds
        return results

    feed_mapping = asyncio.run(discover_all())

    if output_format == "json":
        json_output = {}
        for show_name, feeds in feed_mapping.items():
            json_output[show_name] = [
                {
                    "title": feed.title,
                    "feed_url": feed.feed_url,
                    "episode_count": feed.episode_count,
                    "author": feed.author,
                    "confidence": feed.confidence,
                    "source": feed.source,
                }
                for feed in feeds
            ]
        console.print(json_lib.dumps(json_output, indent=2))
    else:
        # table format.
        total_feeds = 0
        for show_name, feeds in feed_mapping.items():
            console.print(f"\n[cyan bold]{show_name}[/cyan bold]")

            if not feeds:
                console.print("  [red]No feeds found[/red]")
                continue

            table = Table(show_header=True, header_style="bold dim")
            table.add_column("#", style="dim", width=3)
            table.add_column("title", style="cyan", max_width=30)
            table.add_column("author", max_width=15)
            table.add_column("episodes", justify="right", width=8)
            table.add_column("source", width=12)
            table.add_column("feed url", style="dim", max_width=40)

            for idx, feed in enumerate(feeds[:5], 1):
                table.add_row(
                    str(idx),
                    feed.title[:30],
                    (feed.author or "")[:15],
                    str(feed.episode_count or "?"),
                    feed.source,
                    feed.feed_url[:40] + "..." if len(feed.feed_url) > 40 else feed.feed_url,
                )
                total_feeds += 1

            console.print(table)

        console.print(f"\n[bold]Summary:[/bold] {total_feeds} feeds discovered for {len(show_names)} shows")


@cli.command()
@click.option("--count", "-k", type=int, default=10, help="number of top podcasts (1-50, default: 10)")
@click.option("--genre", "-g", type=str, help="filter by genre (e.g., Comedy, True Crime, News)")
def popular(count: int, genre: str | None):
    """display top k most popular podcasts in the usa.

    examples:
        podcast-dl popular
        podcast-dl popular --count 20
        podcast-dl popular --genre "True Crime"
    """
    from podcast_downloader.popular_podcasts import (
        get_popular_podcasts_by_rank,
        get_podcasts_by_genre,
        get_all_genres,
    )

    # validate count.
    if count < 1:
        console.print("[red]error: count must be at least 1[/red]")
        return
    if count > 50:
        console.print("[yellow]note: maximum 50 podcasts available, showing top 50[/yellow]")
        count = 50

    # get podcasts based on criteria.
    if genre:
        podcasts = get_podcasts_by_genre(genre)
        if not podcasts:
            console.print(f"[red]no podcasts found for genre: {genre}[/red]")
            console.print("[dim]available genres:[/dim]")
            genres = get_all_genres()
            for g in genres:
                console.print(f"  • {g}")
            return

        podcasts = podcasts[:count]
        title = f"Top {len(podcasts)} {genre} Podcasts in the USA"
    else:
        podcasts = get_popular_podcasts_by_rank(count)
        title = f"Top {len(podcasts)} Most Popular Podcasts in the USA"

    # create display table.
    table = Table(title=title)
    table.add_column("Rank", justify="right", style="cyan", width=6)
    table.add_column("Podcast Title", style="bold", width=35)
    table.add_column("Host", style="dim", width=25)
    table.add_column("Genre", style="green", width=15)
    table.add_column("RSS", justify="center", width=5)

    for podcast in podcasts:
        rss_status = "✅" if podcast.rss_feed else "❌"
        table.add_row(
            str(podcast.rank),
            podcast.title[:35],
            podcast.host[:25],
            podcast.genre,
            rss_status,
        )

    console.print(table)

    # show summary.
    available_rss = sum(1 for p in podcasts if p.rss_feed)
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"• Total shown: {len(podcasts)} podcasts")
    console.print(f"• RSS feeds available: {available_rss}/{len(podcasts)}")

    if genre:
        console.print(f"• Filtered by genre: {genre}")

    console.print(f"\n[dim]💡 Tip: Use these exact titles with the download command[/dim]")
    console.print(f"[dim]💡 Use --genre option to filter by specific genres[/dim]")

    # show available genres if not filtered.
    if not genre:
        genres = get_all_genres()
        console.print(f"\n[dim]Available genres: {', '.join(genres[:8])}{'...' if len(genres) > 8 else ''}[/dim]")


@cli.command("create-config")
@click.option("--output", "-o", type=click.Path(), help="output file for configuration")
def create_config(output: str | None):
    """create a default configuration file.

    examples:
        podcast-dl create-config
        podcast-dl create-config --output my-config.yaml
    """
    import yaml

    config_data = {
        "output_dir": "outputs/downloads",
        "transcripts_dir": "outputs/transcripts",
        "metadata_dir": "outputs/metadata",
        "max_concurrent_downloads": 4,
        "skip_existing": True,
        "validate_audio": True,
        "retry_attempts": 3,
        "rate_limit_delay": 1.0,
        "episode_filter": {
            "max_episodes": None,
            "random_sample": None,
            "start_date": None,
            "end_date": None,
        },
        "deep_validation": False,
        "min_file_size": 1024,
        "max_file_size": 500 * 1024 * 1024,
        "enable_youtube_fallback": True,
        "youtube_min_confidence": 0.7,
        "enable_fuzzy_matching": True,
        "fuzzy_min_score": 0.6,
    }

    config_path = Path(output) if output else Path("download_config.yaml")

    with open(config_path, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

    console.print(f"[green]✓ Configuration file created:[/green] {config_path}")
    console.print("[dim]Edit this file to customize download settings[/dim]")


if __name__ == "__main__":
    cli()
