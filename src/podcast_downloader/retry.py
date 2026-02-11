"""retry logic with exponential backoff and error-aware retry counts."""

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from podcast_downloader.errors import DownloadError

T = TypeVar("T")


async def retry_with_backoff(
    func: Callable[[], Awaitable[T]],
    max_retries: int | None = None,
    backoff_base: float = 2.0,
    max_delay: float = 60.0,
    jitter: bool = True,
) -> T:
    """
    retry an async function with exponential backoff.

    if max_retries is None, uses error-category-aware retry counts.
    """
    attempt = 0

    while True:
        try:
            return await func()
        except DownloadError as e:
            # use error-specific retry count if not overridden.
            effective_max = max_retries if max_retries is not None else e.max_retries

            if attempt >= effective_max:
                raise

            attempt += 1
            delay = min(backoff_base**attempt, max_delay)
            if jitter:
                delay = delay * (0.5 + random.random())

            await asyncio.sleep(delay)

        except Exception:
            effective_max = max_retries if max_retries is not None else 1

            if attempt >= effective_max:
                raise

            attempt += 1
            delay = min(backoff_base**attempt, max_delay)
            if jitter:
                delay = delay * (0.5 + random.random())

            await asyncio.sleep(delay)


class RetryContext:
    """context manager for tracking retry state."""

    def __init__(
        self,
        max_retries: int = 3,
        backoff_base: float = 2.0,
        max_delay: float = 60.0,
    ):
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.max_delay = max_delay
        self.attempt = 0
        self.last_error: Exception | None = None

    def should_retry(self, error: Exception) -> bool:
        """check if we should retry after the given error."""
        self.last_error = error
        self.attempt += 1

        if isinstance(error, DownloadError):
            return self.attempt <= error.max_retries

        return self.attempt <= self.max_retries

    async def wait(self) -> None:
        """wait before the next retry attempt."""
        delay = min(self.backoff_base**self.attempt, self.max_delay)
        # add jitter.
        delay = delay * (0.5 + random.random())
        await asyncio.sleep(delay)

    def reset(self) -> None:
        """reset retry state."""
        self.attempt = 0
        self.last_error = None
