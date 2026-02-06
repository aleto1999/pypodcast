"""tests for popular_podcasts module."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest

from podcast_downloader.popular_podcasts import (
    PopularPodcast,
    TOP_US_PODCASTS,
    get_popular_podcasts_by_rank,
    get_podcasts_by_genre,
    get_all_genres,
    search_popular_podcasts,
    get_podcast_by_title,
    get_rss_feed_for_popular_podcast,
)


class TestPopularPodcast:
    """tests for PopularPodcast dataclass."""

    def test_create_podcast(self):
        """should create podcast with required fields."""
        podcast = PopularPodcast(
            rank=1,
            title="Test Podcast",
            host="Test Host",
            genre="Comedy",
            description="A test podcast",
        )
        assert podcast.rank == 1
        assert podcast.title == "Test Podcast"
        assert podcast.host == "Test Host"
        assert podcast.genre == "Comedy"
        assert podcast.rss_feed is None
        assert podcast.search_aliases == []

    def test_podcast_with_rss_feed(self):
        """should store rss feed url."""
        podcast = PopularPodcast(
            rank=1,
            title="Test",
            host="Host",
            genre="News",
            description="Test",
            rss_feed="https://example.com/feed.xml",
        )
        assert podcast.rss_feed == "https://example.com/feed.xml"

    def test_podcast_with_aliases(self):
        """should store search aliases."""
        podcast = PopularPodcast(
            rank=1,
            title="Test",
            host="Host",
            genre="News",
            description="Test",
            search_aliases=["test", "tst"],
        )
        assert podcast.search_aliases == ["test", "tst"]


class TestTopUsPodcasts:
    """tests for TOP_US_PODCASTS list."""

    def test_has_50_podcasts(self):
        """should contain 50 podcasts."""
        assert len(TOP_US_PODCASTS) == 50

    def test_podcasts_are_ranked(self):
        """should have podcasts in rank order."""
        for i, podcast in enumerate(TOP_US_PODCASTS, 1):
            assert podcast.rank == i

    def test_the_daily_is_included(self):
        """should include The Daily."""
        titles = [p.title for p in TOP_US_PODCASTS]
        assert "The Daily" in titles

    def test_top_podcasts_have_rss(self):
        """top podcasts should mostly have rss feeds."""
        top_10 = TOP_US_PODCASTS[:10]
        with_rss = sum(1 for p in top_10 if p.rss_feed)
        assert with_rss >= 8  # at least 8 of top 10 should have feeds


class TestGetPopularPodcastsByRank:
    """tests for get_popular_podcasts_by_rank function."""

    def test_get_top_10(self):
        """should return top 10 podcasts by default."""
        result = get_popular_podcasts_by_rank(10)
        assert len(result) == 10
        assert result[0].rank == 1

    def test_get_top_5(self):
        """should return top 5 podcasts."""
        result = get_popular_podcasts_by_rank(5)
        assert len(result) == 5

    def test_max_50(self):
        """should cap at 50 podcasts."""
        result = get_popular_podcasts_by_rank(100)
        assert len(result) == 50


class TestGetPodcastsByGenre:
    """tests for get_podcasts_by_genre function."""

    def test_filter_by_comedy(self):
        """should filter comedy podcasts."""
        result = get_podcasts_by_genre("Comedy")
        assert len(result) > 0
        for podcast in result:
            assert "Comedy" in podcast.genre

    def test_filter_case_insensitive(self):
        """should be case insensitive."""
        upper = get_podcasts_by_genre("COMEDY")
        lower = get_podcasts_by_genre("comedy")
        assert len(upper) == len(lower)

    def test_nonexistent_genre(self):
        """should return empty for unknown genre."""
        result = get_podcasts_by_genre("NonexistentGenre")
        assert len(result) == 0

    def test_filter_by_news(self):
        """should filter news podcasts."""
        result = get_podcasts_by_genre("News")
        assert len(result) > 0


class TestGetAllGenres:
    """tests for get_all_genres function."""

    def test_returns_list(self):
        """should return list of genres."""
        result = get_all_genres()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_genres_are_sorted(self):
        """should return sorted genres."""
        result = get_all_genres()
        assert result == sorted(result)

    def test_common_genres_included(self):
        """should include common genres."""
        result = get_all_genres()
        # at least some of these should be present
        common = ["Comedy", "News", "True Crime", "Business"]
        found = sum(1 for g in common if g in result)
        assert found >= 3


class TestSearchPopularPodcasts:
    """tests for search_popular_podcasts function."""

    def test_search_by_title(self):
        """should find podcast by title."""
        result = search_popular_podcasts("Joe Rogan")
        assert len(result) > 0
        assert any("Rogan" in p.title for p in result)

    def test_search_by_alias(self):
        """should find podcast by alias."""
        result = search_popular_podcasts("jre")
        assert len(result) > 0

    def test_search_case_insensitive(self):
        """should be case insensitive."""
        upper = search_popular_podcasts("THE DAILY")
        lower = search_popular_podcasts("the daily")
        assert len(upper) == len(lower)

    def test_search_no_results(self):
        """should return empty for no matches."""
        result = search_popular_podcasts("xyznonexistent123")
        assert len(result) == 0

    def test_search_partial_match(self):
        """should match partial titles."""
        result = search_popular_podcasts("Daily")
        assert len(result) > 0


class TestGetPodcastByTitle:
    """tests for get_podcast_by_title function."""

    def test_exact_match(self):
        """should find podcast by exact title."""
        result = get_podcast_by_title("The Daily")
        assert result is not None
        assert result.title == "The Daily"

    def test_case_insensitive(self):
        """should be case insensitive."""
        result = get_podcast_by_title("the daily")
        assert result is not None

    def test_not_found(self):
        """should return None for unknown title."""
        result = get_podcast_by_title("Unknown Podcast Title")
        assert result is None


class TestGetRssFeedForPopularPodcast:
    """tests for get_rss_feed_for_popular_podcast function."""

    def test_get_feed_for_known_podcast(self):
        """should return rss feed for known podcast."""
        result = get_rss_feed_for_popular_podcast("The Daily")
        assert result is not None
        assert result.startswith("https://")

    def test_get_feed_by_alias(self):
        """should work with aliases."""
        result = get_rss_feed_for_popular_podcast("huberman")
        assert result is not None
        assert "huberman" in result.lower()

    def test_unknown_podcast(self):
        """should return None for unknown podcast."""
        result = get_rss_feed_for_popular_podcast("Unknown Podcast 12345")
        assert result is None

    def test_podcast_without_feed(self):
        """should return None for podcast without rss feed."""
        # find a podcast without rss feed
        no_feed = next((p for p in TOP_US_PODCASTS if p.rss_feed is None), None)
        if no_feed:
            result = get_rss_feed_for_popular_podcast(no_feed.title)
            assert result is None


class TestIntegration:
    """integration tests for popular podcasts module."""

    def test_workflow_find_and_download(self):
        """should support finding podcast and getting feed url."""
        # search for a podcast
        results = search_popular_podcasts("crime")
        assert len(results) > 0

        # get the rss feed
        feed = get_rss_feed_for_popular_podcast(results[0].title)
        assert feed is not None or results[0].rss_feed is None

    def test_genre_to_podcasts_to_feeds(self):
        """should support filtering by genre and getting feeds."""
        # get true crime podcasts
        podcasts = get_podcasts_by_genre("True Crime")
        assert len(podcasts) > 0

        # count how many have feeds
        with_feeds = [p for p in podcasts if p.rss_feed]
        assert len(with_feeds) > 0
