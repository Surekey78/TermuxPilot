"""Provider-layer error types.

The one bit of policy that matters for fallback: :attr:`ProviderError.retryable`.
* retryable  -> the fallback chain may try the next provider
* non-retryable -> the request itself is bad (or a stream was interrupted after
  visible output), so fail over immediately.
"""

from __future__ import annotations

from typing import Any

#: HTTP statuses where "try the next provider" makes sense:
#: 401/403 (dead key — a keyless local server may still work),
#: 404 (wrong base_url path), 408/429/5xx (outage / rate limit).
DEFAULT_RETRYABLE_STATUS = frozenset({401, 403, 404, 408, 429, 500, 502, 503, 504})


class ProviderError(Exception):
    """Base class for all provider failures."""

    retryable: bool = False

    def __init__(self, message: str, *, provider: str | None = None) -> None:
        super().__init__(message)
        self.provider = provider
        #: attempts recorded by the chain: [(label, error), ...]
        self.attempts: list[tuple[str, str]] = []


class ConnectionFailed(ProviderError):
    """Could not establish a connection (DNS, refused, TLS, ...)."""

    retryable = True


class RequestTimeout(ProviderError):
    """Connect or read timeout."""

    retryable = True


class HttpError(ProviderError):
    """Non-2xx response from the endpoint."""

    def __init__(self, message: str, *, status: int, provider: str | None = None) -> None:
        super().__init__(message, provider=provider)
        self.status = status
        self.retryable = status in DEFAULT_RETRYABLE_STATUS


class ProtocolError(ProviderError):
    """Endpoint returned 200 but the payload is not usable (bad SSE/JSON)."""

    retryable = False


class StreamInterrupted(ProviderError):
    """Stream died after the user already saw partial output.

    Failing over would duplicate visible text, so this is non-retryable:
    the partial content is attached to :attr:`partial`.
    """

    retryable = False

    def __init__(self, message: str, *, provider: str | None = None,
                 partial: Any | None = None) -> None:
        super().__init__(message, provider=provider)
        self.partial = partial
