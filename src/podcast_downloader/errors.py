"""custom exceptions for podcast downloader with categorization."""

from enum import Enum, auto


class ErrorCategory(Enum):
    """categories of download errors for retry logic."""

    NETWORK = auto()  # timeout, connection errors.
    RATE_LIMIT = auto()  # 429, too many requests.
    NOT_FOUND = auto()  # 404, unavailable.
    FORMAT = auto()  # unsupported format, parsing errors.
    PERMISSION = auto()  # 403, forbidden.
    VALIDATION = auto()  # corrupted audio, invalid file.
    UNKNOWN = auto()  # uncategorized errors.


# retry counts per error category.
RETRY_COUNTS: dict[ErrorCategory, int] = {
    ErrorCategory.NETWORK: 3,
    ErrorCategory.RATE_LIMIT: 2,
    ErrorCategory.NOT_FOUND: 0,
    ErrorCategory.FORMAT: 1,
    ErrorCategory.PERMISSION: 0,
    ErrorCategory.VALIDATION: 1,
    ErrorCategory.UNKNOWN: 1,
}


class DownloadError(Exception):
    """base exception for download errors."""

    category: ErrorCategory = ErrorCategory.UNKNOWN

    def __init__(self, message: str, url: str | None = None):
        self.message = message
        self.url = url
        super().__init__(message)

    @property
    def max_retries(self) -> int:
        """get max retry count for this error type."""
        return RETRY_COUNTS.get(self.category, 1)


class NetworkError(DownloadError):
    """network-related errors: timeout, connection, ssl."""

    category = ErrorCategory.NETWORK


class RateLimitError(DownloadError):
    """rate limiting errors: 429, too many requests."""

    category = ErrorCategory.RATE_LIMIT

    def __init__(self, message: str, url: str | None = None, retry_after: int | None = None):
        super().__init__(message, url)
        self.retry_after = retry_after


class NotFoundError(DownloadError):
    """resource not found: 404, unavailable."""

    category = ErrorCategory.NOT_FOUND


class FormatError(DownloadError):
    """format errors: unsupported audio format, parsing failures."""

    category = ErrorCategory.FORMAT


class PermissionError(DownloadError):
    """permission errors: 403, forbidden."""

    category = ErrorCategory.PERMISSION


class ValidationError(DownloadError):
    """validation errors: corrupted audio, invalid file."""

    category = ErrorCategory.VALIDATION


class FeedParseError(FormatError):
    """error parsing rss feed."""

    pass


class FeedDiscoveryError(DownloadError):
    """error discovering feeds."""

    category = ErrorCategory.NETWORK


def categorize_http_error(status_code: int, url: str, message: str = "") -> DownloadError:
    """create appropriate error based on http status code."""
    if status_code == 404:
        return NotFoundError(message or f"not found: {url}", url)
    elif status_code == 403:
        return PermissionError(message or f"forbidden: {url}", url)
    elif status_code == 429:
        return RateLimitError(message or f"rate limited: {url}", url)
    elif 400 <= status_code < 500:
        return DownloadError(f"client error {status_code}: {url}", url)
    elif 500 <= status_code < 600:
        return NetworkError(f"server error {status_code}: {url}", url)
    else:
        return DownloadError(f"http error {status_code}: {url}", url)
