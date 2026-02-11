"""core download pipeline with async generator pattern."""

import asyncio
import random
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Callable

from rich.progress import Progress, TaskID

from podcast_downloader.audio import validate_audio_file
from podcast_downloader.config import DownloadConfig, DownloadStats, EpisodeFilter
from podcast_downloader.discovery import discover_feeds, select_best_feed
from podcast_downloader.errors import DownloadError
from podcast_downloader.feed_parser import Episode, Podcast, parse_feed
from podcast_downloader.metadata import MetadataStore
from podcast_downloader.providers import DownloadResult, get_provider
from podcast_downloader.statistics import StatisticsCollector
from podcast_downloader.validation import (
    ValidationConfig,
    check_disk_space,
    validate_download,
)
from podcast_downloader.youtube_fallback import fallback_download_episode
from podcast_downloader.youtube_feed import is_youtube_url, parse_youtube_feed


def filter_episodes(
    episodes: list[Episode],
    episode_filter: EpisodeFilter,
) -> list[Episode]:
    """apply filters to episode list."""
    filtered = episodes.copy()

    # filter by date range.
    if episode_filter.start_date:
        start_date = episode_filter.start_date.replace(tzinfo=None)
        filtered = [
            e for e in filtered
            if e.published and e.published.replace(tzinfo=None) >= start_date
        ]

    if episode_filter.end_date:
        end_date = episode_filter.end_date.replace(tzinfo=None)
        filtered = [
            e for e in filtered
            if e.published and e.published.replace(tzinfo=None) <= end_date
        ]

    # random sample or take newest.
    random.seed(42)
    if episode_filter.random_sample:
        count = min(episode_filter.random_sample, len(filtered))
        filtered = random.sample(filtered, count)
    elif episode_filter.max_episodes:
        filtered = filtered[: episode_filter.max_episodes]

    return filtered


async def download_episode(
    episode: Episode,
    output_dir: Path,
    config: DownloadConfig,
    podcast_name: str = "",
    progress_callback: Callable[[int, int], None] | None = None,
    stats_collector: StatisticsCollector | None = None,
) -> DownloadResult:
    """download a single episode using the appropriate provider.

    includes enhanced validation and youtube fallback when enabled.

    args:
        episode: the episode to download.
        output_dir: directory to save the file.
        config: download configuration.
        podcast_name: name of the podcast (for youtube fallback).
        progress_callback: optional progress callback.
        stats_collector: optional statistics collector.

    returns:
        download result.
    """
    provider = get_provider(episode)

    if provider is None:
        return DownloadResult(
            episode=episode,
            success=False,
            error=f"no provider available for url: {episode.audio_url}",
        )

    # track download start.
    if stats_collector:
        stats_collector.start_download(episode.id)

    result = await provider.download(episode, output_dir, progress_callback)

    # validate downloaded file with enhanced validation.
    if result.success and config.validate_audio and result.file_path:
        # basic validation first.
        if not validate_audio_file(result.file_path):
            result.file_path.unlink(missing_ok=True)
            result = DownloadResult(
                episode=episode,
                success=False,
                error="downloaded file failed audio validation",
            )
        else:
            # enhanced validation if enabled.
            if config.deep_validation:
                validation_config = ValidationConfig(
                    min_file_size=config.min_file_size,
                    max_file_size=config.max_file_size,
                    use_ffprobe=True,
                )
                validation_result = await validate_download(
                    result.file_path,
                    validation_config,
                )
                if not validation_result.valid:
                    result.file_path.unlink(missing_ok=True)
                    result = DownloadResult(
                        episode=episode,
                        success=False,
                        error=f"deep validation failed: {validation_result.error_message}",
                    )

    # youtube fallback if primary download failed.
    if not result.success and config.enable_youtube_fallback and podcast_name:
        result = await fallback_download_episode(
            episode=episode,
            show_name=podcast_name,
            output_dir=output_dir,
            min_confidence=config.youtube_min_confidence,
            max_retries=config.youtube_max_retries,
        )

    # track download end.
    if stats_collector:
        stats_collector.end_download(
            episode_id=episode.id,
            episode_title=episode.title,
            podcast_name=podcast_name,
            success=result.success,
            bytes_downloaded=result.bytes_downloaded,
            error_message=result.error,
        )

    return result


async def download_episodes_concurrent(
    episodes: list[Episode],
    output_dir: Path,
    config: DownloadConfig,
    podcast: Podcast,
    metadata_store: MetadataStore,
    stats: DownloadStats,
    progress: Progress | None = None,
    task_id: TaskID | None = None,
    stats_collector: StatisticsCollector | None = None,
) -> AsyncIterator[DownloadResult]:
    """download multiple episodes with concurrency control.

    args:
        episodes: list of episodes to download.
        output_dir: directory to save files.
        config: download configuration.
        podcast: the podcast being downloaded.
        metadata_store: metadata store for tracking downloads.
        stats: download stats to update.
        progress: optional rich progress bar.
        task_id: optional progress task id.
        stats_collector: optional statistics collector.

    yields:
        download results as they complete.
    """
    semaphore = asyncio.Semaphore(config.max_concurrent_downloads)

    # check disk space before starting.
    space_result = check_disk_space(output_dir, config.min_disk_space)
    if not space_result.valid:
        for episode in episodes:
            yield DownloadResult(
                episode=episode,
                success=False,
                error=space_result.error_message,
            )
        return

    async def download_with_semaphore(episode: Episode) -> DownloadResult:
        async with semaphore:
            # rate limiting.
            if config.rate_limit_delay > 0:
                await asyncio.sleep(config.rate_limit_delay)

            result = await download_episode(
                episode=episode,
                output_dir=output_dir,
                config=config,
                podcast_name=podcast.title,
                stats_collector=stats_collector,
            )

            # update metadata on success.
            if result.success and result.file_path:
                metadata_store.mark_downloaded(
                    podcast,
                    episode,
                    result.file_path,
                    result.bytes_downloaded,
                )
                stats.episodes_downloaded += 1
                stats.total_bytes += result.bytes_downloaded
            else:
                stats.episodes_failed += 1
                if result.error:
                    stats.add_error(f"{episode.title}: {result.error}")

            # update progress.
            if progress and task_id is not None:
                progress.advance(task_id)

            return result

    # create tasks for all episodes.
    tasks = [download_with_semaphore(episode) for episode in episodes]

    # yield results as they complete.
    for coro in asyncio.as_completed(tasks):
        result = await coro
        yield result


async def download_podcast(
    podcast_name: str,
    config: DownloadConfig | None = None,
    episode_filter: EpisodeFilter | None = None,
    feed_url: str | None = None,
    progress: Progress | None = None,
    stats_collector: StatisticsCollector | None = None,
) -> tuple[DownloadStats, list[DownloadResult]]:
    """
    download episodes from a podcast.

    args:
        podcast_name: name of podcast to search for (or use with feed_url).
        config: download configuration.
        episode_filter: filters for episode selection.
        feed_url: optional direct feed url (skips discovery).
        progress: optional rich progress bar.
        stats_collector: optional statistics collector.

    returns:
        tuple of (stats, list of download results).
    """
    config = config or DownloadConfig()
    episode_filter = episode_filter or config.default_filter
    stats = DownloadStats()
    results: list[DownloadResult] = []

    # initialize statistics collector if enabled.
    if config.collect_statistics and stats_collector is None:
        stats_collector = StatisticsCollector()
        stats_collector.start_session()

    metadata_store = MetadataStore(config.metadata_dir)

    # discover feed if not provided.
    if not feed_url:
        feeds = await discover_feeds(
            podcast_name,
            limit=config.max_alternative_feeds,
            use_fuzzy_matching=config.enable_fuzzy_matching,
        )
        if not feeds:
            stats.add_error(f"no feeds found for: {podcast_name}")
            return stats, results

        best_feed = select_best_feed(feeds, query=podcast_name)
        if not best_feed:
            stats.add_error(f"no valid feed found for: {podcast_name}")
            return stats, results

        feed_url = best_feed.feed_url

    # parse feed (handle youtube urls differently).
    try:
        if is_youtube_url(feed_url):
            podcast, episodes = await parse_youtube_feed(feed_url)
        else:
            podcast, episodes = await parse_feed(feed_url)
    except DownloadError as e:
        stats.add_error(f"failed to parse feed: {e}")

        # try alternative feeds if enabled.
        if config.enable_feed_fallback:
            feeds = await discover_feeds(podcast_name, limit=config.max_alternative_feeds)
            for alt_feed in feeds:
                if alt_feed.feed_url == feed_url:
                    continue
                try:
                    if is_youtube_url(alt_feed.feed_url):
                        podcast, episodes = await parse_youtube_feed(alt_feed.feed_url)
                    else:
                        podcast, episodes = await parse_feed(alt_feed.feed_url)
                    feed_url = alt_feed.feed_url
                    break
                except (DownloadError, RuntimeError):
                    continue
            else:
                return stats, results

        else:
            return stats, results
    except RuntimeError as e:
        # youtube feed parsing error.
        stats.add_error(f"failed to parse youtube feed: {e}")
        return stats, results

    stats.podcasts_processed = 1
    stats.episodes_found = len(episodes)

    # record podcast processed in statistics.
    if stats_collector:
        stats_collector.record_podcast_processed(podcast.title, len(episodes))

    # filter episodes.
    filtered_episodes = filter_episodes(episodes, episode_filter)

    # skip already downloaded.
    if config.skip_existing:
        filtered_episodes = [
            e for e in filtered_episodes
            if not metadata_store.is_downloaded(podcast.title, e.id)
        ]
        stats.episodes_skipped = stats.episodes_found - len(filtered_episodes)
        if stats_collector:
            stats_collector.record_episodes_skipped(stats.episodes_skipped)

    if not filtered_episodes:
        return stats, results

    # create output directories for both audio files and transcripts.
    audio_dir = config.output_dir / podcast.safe_dirname
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    transcripts_dir = config.transcripts_dir / podcast.safe_dirname
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    # setup progress tracking.
    task_id = None
    if progress:
        task_id = progress.add_task(
            f"[cyan]{podcast.title[:30]}...",
            total=len(filtered_episodes),
        )

    # download episodes.
    async for result in download_episodes_concurrent(
        filtered_episodes,
        audio_dir,
        config,
        podcast,
        metadata_store,
        stats,
        progress,
        task_id,
        stats_collector,
    ):
        results.append(result)

    return stats, results


async def download_podcasts(
    podcast_names: list[str],
    config: DownloadConfig | None = None,
    episode_filter: EpisodeFilter | None = None,
    progress: Progress | None = None,
) -> tuple[DownloadStats, list[DownloadResult]]:
    """
    download episodes from multiple podcasts.

    downloads are processed sequentially by podcast but concurrently within each podcast.

    args:
        podcast_names: list of podcast names to download.
        config: download configuration.
        episode_filter: filters for episode selection.
        progress: optional rich progress bar.

    returns:
        tuple of (stats, list of download results).
    """
    config = config or DownloadConfig()
    total_stats = DownloadStats()
    all_results: list[DownloadResult] = []

    # shared statistics collector for all podcasts.
    stats_collector: StatisticsCollector | None = None
    if config.collect_statistics:
        stats_collector = StatisticsCollector()
        stats_collector.start_session()

    for podcast_name in podcast_names:
        stats, results = await download_podcast(
            podcast_name,
            config,
            episode_filter,
            progress=progress,
            stats_collector=stats_collector,
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
