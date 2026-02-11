"""download providers protocol and registry."""

from pathlib import Path
from typing import Protocol

from podcast_downloader.feed_parser import Episode

# import providers to trigger registration - must be done after protocol/registry definitions.
# see bottom of file.


class DownloadResult:
    """result of a download attempt."""

    def __init__(
        self,
        episode: Episode,
        success: bool,
        file_path: Path | None = None,
        error: str | None = None,
        bytes_downloaded: int = 0,
    ):
        self.episode = episode
        self.success = success
        self.file_path = file_path
        self.error = error
        self.bytes_downloaded = bytes_downloaded


class DownloadProvider(Protocol):
    """protocol for download providers."""

    async def download(
        self,
        episode: Episode,
        output_dir: Path,
    ) -> DownloadResult:
        """download an episode to the output directory."""
        ...

    def can_handle(self, episode: Episode) -> bool:
        """check if this provider can handle the given episode."""
        ...


# provider registry - populated by provider modules.
_providers: list[DownloadProvider] = []


def register_provider(provider: DownloadProvider) -> None:
    """register a download provider."""
    _providers.append(provider)


def get_provider(episode: Episode) -> DownloadProvider | None:
    """get a provider that can handle the given episode."""
    for provider in _providers:
        if provider.can_handle(episode):
            return provider
    return None


def get_all_providers() -> list[DownloadProvider]:
    """get all registered providers."""
    return list(_providers)


# import providers to trigger self-registration.
# must be at bottom to avoid circular imports.
from podcast_downloader.providers import rss  # noqa: F401, E402
from podcast_downloader.providers import youtube  # noqa: F401, E402
