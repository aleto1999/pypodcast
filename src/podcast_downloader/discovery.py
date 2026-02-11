"""feed discovery using itunes, podcast index, and youtube apis."""

import asyncio
import hashlib
import shutil
import time
from typing import AsyncIterator

import httpx

from podcast_downloader.config import FeedSearchResult
from podcast_downloader.errors import FeedDiscoveryError, NetworkError

# itunes search api.
ITUNES_SEARCH_URL = "https://itunes.apple.com/search"

# podcast index api (free, requires api key for full access).
PODCAST_INDEX_SEARCH_URL = "https://api.podcastindex.org/api/1.0/search/byterm"


async def search_itunes(
    query: str,
    limit: int = 10,
    client: httpx.AsyncClient | None = None,
) -> list[FeedSearchResult]:
    """search for podcasts using itunes api."""
    should_close = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=30.0)

    try:
        response = await client.get(
            ITUNES_SEARCH_URL,
            params={
                "term": query,
                "media": "podcast",
                "limit": limit,
            },
        )

        if response.status_code != 200:
            raise FeedDiscoveryError(f"itunes api error: {response.status_code}")

        data = response.json()
        results = []

        for item in data.get("results", []):
            feed_url = item.get("feedUrl")
            if not feed_url:
                continue

            results.append(
                FeedSearchResult(
                    title=item.get("collectionName", "Unknown"),
                    feed_url=feed_url,
                    author=item.get("artistName"),
                    description=item.get("description"),
                    artwork_url=item.get("artworkUrl600") or item.get("artworkUrl100"),
                    episode_count=item.get("trackCount"),
                    source="itunes",
                    confidence=1.0,
                )
            )

        return results

    except httpx.TimeoutException:
        raise NetworkError("timeout searching itunes")
    except httpx.RequestError as e:
        raise NetworkError(f"network error searching itunes: {e}")
    finally:
        if should_close:
            await client.aclose()


async def search_podcast_index(
    query: str,
    api_key: str | None = None,
    api_secret: str | None = None,
    limit: int = 10,
    client: httpx.AsyncClient | None = None,
) -> list[FeedSearchResult]:
    """
    search for podcasts using podcast index api.

    requires api_key and api_secret for authenticated requests.
    get free keys at https://api.podcastindex.org/
    """
    if not api_key or not api_secret:
        return []  # skip if no credentials.

    should_close = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=30.0)

    try:
        # podcast index requires auth headers.
        epoch_time = int(time.time())
        auth_hash = hashlib.sha1(
            f"{api_key}{api_secret}{epoch_time}".encode()
        ).hexdigest()

        headers = {
            "X-Auth-Key": api_key,
            "X-Auth-Date": str(epoch_time),
            "Authorization": auth_hash,
            "User-Agent": "PodcastDownloader/1.0",
        }

        response = await client.get(
            PODCAST_INDEX_SEARCH_URL,
            params={"q": query, "max": limit},
            headers=headers,
        )

        if response.status_code != 200:
            raise FeedDiscoveryError(f"podcast index api error: {response.status_code}")

        data = response.json()
        results = []

        for item in data.get("feeds", []):
            feed_url = item.get("url")
            if not feed_url:
                continue

            results.append(
                FeedSearchResult(
                    title=item.get("title", "Unknown"),
                    feed_url=feed_url,
                    author=item.get("author"),
                    description=item.get("description"),
                    artwork_url=item.get("artwork"),
                    episode_count=item.get("episodeCount"),
                    source="podcastindex",
                    confidence=1.0,
                )
            )

        return results

    except httpx.TimeoutException:
        raise NetworkError("timeout searching podcast index")
    except httpx.RequestError as e:
        raise NetworkError(f"network error searching podcast index: {e}")
    finally:
        if should_close:
            await client.aclose()


async def search_youtube(
    query: str,
    limit: int = 5,
) -> list[FeedSearchResult]:
    """
    search for podcast channels/playlists on youtube using yt-dlp.

    returns channels and playlists that match the query.
    requires yt-dlp to be installed.
    """
    if not shutil.which("yt-dlp"):
        return []  # yt-dlp not available.

    try:
        # use yt-dlp to search youtube for channels/playlists.
        # search for channels first, then playlists.
        results: list[FeedSearchResult] = []

        # search for channels (podcasts often have dedicated channels).
        channel_cmd = [
            "yt-dlp",
            "--flat-playlist",
            "--dump-json",
            "--no-warnings",
            "--quiet",
            f"ytsearch{limit}:{query} podcast",
        ]

        process = await asyncio.create_subprocess_exec(
            *channel_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=30.0)

        if stdout:
            import json
            for line in stdout.decode().strip().split("\n"):
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    video_id = item.get("id")
                    title = item.get("title", "Unknown")
                    channel = item.get("channel") or item.get("uploader") or ""
                    channel_id = item.get("channel_id") or item.get("uploader_id")
                    duration = item.get("duration")

                    # skip very short videos (likely not podcast episodes).
                    if duration and duration < 300:  # less than 5 minutes.
                        continue

                    if video_id:
                        # for individual videos, use the video url.
                        video_url = f"https://www.youtube.com/watch?v={video_id}"
                        results.append(
                            FeedSearchResult(
                                title=title,
                                feed_url=video_url,
                                author=channel,
                                description=f"YouTube video: {title}",
                                episode_count=1,
                                source="youtube",
                                confidence=0.8,
                            )
                        )

                        # also try to get the channel url if available.
                        if channel_id and len(results) < limit:
                            channel_url = f"https://www.youtube.com/channel/{channel_id}/videos"
                            # check if we already added this channel.
                            if not any(r.feed_url == channel_url for r in results):
                                results.append(
                                    FeedSearchResult(
                                        title=f"{channel} (Channel)",
                                        feed_url=channel_url,
                                        author=channel,
                                        description=f"YouTube channel: {channel}",
                                        episode_count=None,
                                        source="youtube",
                                        confidence=0.9,
                                    )
                                )
                except json.JSONDecodeError:
                    continue

        # deduplicate by url.
        seen_urls: set[str] = set()
        unique_results: list[FeedSearchResult] = []
        for r in results:
            if r.feed_url not in seen_urls:
                seen_urls.add(r.feed_url)
                unique_results.append(r)

        return unique_results[:limit]

    except asyncio.TimeoutError:
        return []  # timeout, skip youtube results.
    except Exception:
        return []  # any error, skip youtube results.


async def _get_feed_episode_count(feed_url: str) -> int | None:
    """fetch episode count from an RSS feed without downloading full metadata."""
    try:
        from podcast_downloader.feed_parser import parse_feed
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            podcast, episodes = await parse_feed(feed_url, client)
            return len(episodes)
    except Exception:
        return None  # failed to fetch, return None.


async def discover_feeds(
    query: str,
    limit: int = 10,
    podcast_index_key: str | None = None,
    podcast_index_secret: str | None = None,
    include_youtube: bool = True,
    use_fuzzy_matching: bool = True,
    use_query_variations: bool = True,
) -> list[FeedSearchResult]:
    """
    discover podcast feeds by searching multiple apis.

    searches itunes, podcast index (if credentials provided), and youtube.
    results are deduplicated, ranked by fuzzy match confidence, and sorted.

    args:
        query: search query string.
        limit: maximum number of results to return.
        podcast_index_key: optional podcast index api key.
        podcast_index_secret: optional podcast index api secret.
        include_youtube: whether to include youtube results.
        use_fuzzy_matching: whether to use fuzzy matching for ranking.
        use_query_variations: whether to try query variations.

    returns:
        list of feed search results.
    """
    from podcast_downloader.fuzzy_matcher import (
        fuzzy_match,
        generate_query_variations,
    )
    from podcast_downloader.popular_podcasts import search_popular_podcasts

    # generate query variations if enabled.
    queries = [query]
    if use_query_variations:
        queries = generate_query_variations(query)[:3]  # limit to 3 variations.

    all_results: list[FeedSearchResult] = []
    seen_urls: set[str] = set()

    # check popular podcasts database first and add with actual episode counts.
    popular_matches = search_popular_podcasts(query)[:3]  # limit to top 3.
    for podcast in popular_matches:
        if podcast.rss_feed and podcast.rss_feed not in seen_urls:
            # fetch actual episode count from RSS feed.
            episode_count = await _get_feed_episode_count(podcast.rss_feed)
            
            all_results.append(
                FeedSearchResult(
                    title=podcast.title,
                    feed_url=podcast.rss_feed,
                    author=podcast.host,
                    description=podcast.description,
                    episode_count=episode_count,
                    source="popular_db",
                    confidence=1.0,  # high confidence for popular db matches.
                )
            )
            seen_urls.add(podcast.rss_feed)

    async with httpx.AsyncClient(timeout=30.0) as client:
        for q in queries:
            # search apis in parallel.
            itunes_task = search_itunes(q, limit, client)
            podcast_index_task = search_podcast_index(
                q, podcast_index_key, podcast_index_secret, limit, client
            )

            tasks = [itunes_task, podcast_index_task]

            # include youtube search if enabled (only for first query).
            if include_youtube and q == queries[0]:
                youtube_task = search_youtube(q, limit=min(limit, 5))
                tasks.append(youtube_task)

            results = await asyncio.gather(
                *tasks,
                return_exceptions=True,
            )

            for result in results:
                if isinstance(result, Exception):
                    continue  # skip failed apis.
                for feed in result:
                    # deduplicate by feed url.
                    if feed.feed_url not in seen_urls:
                        seen_urls.add(feed.feed_url)

                        # apply fuzzy matching to improve confidence.
                        if use_fuzzy_matching:
                            match_result = fuzzy_match(query, feed.title)
                            # blend original confidence with fuzzy score.
                            blended_confidence = (
                                feed.confidence * 0.5 + match_result.score * 0.5
                            )
                            feed = FeedSearchResult(
                                title=feed.title,
                                feed_url=feed.feed_url,
                                author=feed.author,
                                description=feed.description,
                                artwork_url=feed.artwork_url,
                                episode_count=feed.episode_count,
                                source=feed.source,
                                confidence=blended_confidence,
                            )

                        all_results.append(feed)

            # if we have enough results, stop trying variations.
            if len(all_results) >= limit:
                break

    # sort by confidence and episode count (rss feeds first, then youtube).
    all_results.sort(
        key=lambda f: (
            0 if f.source == "youtube" else 1,  # prefer rss feeds.
            f.confidence,
            f.episode_count or 0,
        ),
        reverse=True,
    )

    return all_results[:limit]


async def discover_feeds_iter(
    query: str,
    limit: int = 10,
) -> AsyncIterator[FeedSearchResult]:
    """async iterator version of discover_feeds."""
    results = await discover_feeds(query, limit)
    for result in results:
        yield result


def select_best_feed(
    feeds: list[FeedSearchResult],
    query: str | None = None,
) -> FeedSearchResult | None:
    """select the best feed from a list of candidates.

    args:
        feeds: list of feed search results.
        query: optional original query for additional fuzzy matching.

    returns:
        best matching feed or none.
    """
    if not feeds:
        return None

    # if query provided, use fuzzy matching for additional scoring.
    if query:
        from podcast_downloader.fuzzy_matcher import fuzzy_match

        scored_feeds: list[tuple[FeedSearchResult, float]] = []
        for feed in feeds:
            match_result = fuzzy_match(query, feed.title)
            # combine confidence, fuzzy score, and episode count.
            score = (
                feed.confidence * 0.4
                + match_result.score * 0.4
                + min((feed.episode_count or 0) / 1000, 0.2)
            )
            scored_feeds.append((feed, score))

        scored_feeds.sort(key=lambda x: x[1], reverse=True)
        return scored_feeds[0][0]

    # fallback: prefer feeds with more episodes and higher confidence.
    return max(
        feeds,
        key=lambda f: (f.confidence, f.episode_count or 0),
    )
