"""Provider layer: OpenAI-compatible client + fallback chain."""

from ..config import ProviderSettings
from .client import ChatResult, OpenAICompatibleClient, ToolCall
from .errors import (
    ConnectionFailed,
    HttpError,
    ProtocolError,
    ProviderError,
    RequestTimeout,
    StreamInterrupted,
    DEFAULT_RETRYABLE_STATUS,
)
from .fallback import FailoverHook, ProviderChain
from .sse import DONE, iter_sse_data, parse_sse_chunks

__all__ = [
    "ChatResult",
    "ConnectionFailed",
    "DEFAULT_RETRYABLE_STATUS",
    "FailoverHook",
    "HttpError",
    "OpenAICompatibleClient",
    "ProtocolError",
    "ProviderChain",
    "ProviderError",
    "ProviderSettings",
    "RequestTimeout",
    "StreamInterrupted",
    "ToolCall",
    "iter_sse_data",
    "parse_sse_chunks",
    "DONE",
]
