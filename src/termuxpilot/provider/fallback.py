"""Fallback chain across OpenAI-compatible providers.

``ProviderChain`` holds an ordered list of endpoints (primary first, then the
configured fallbacks) and tries them in order until one succeeds.

Policy:
* only *retryable* failures (timeouts, connection errors, 401/403/404,
  408/429, 5xx) trigger failover — see :mod:`termuxpilot.provider.errors`;
* a stream that dies **after** visible output does NOT fail over
  (duplicating text the user already saw is worse than an error);
* an optional ``failover`` hook is invoked as ``failover(failed_label,
  next_label, error)`` so the UI can announce the switch.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from ..config import ProviderSettings
from .client import ChatResult, OpenAICompatibleClient
from .errors import ProviderError

FailoverHook = Callable[[str, str, ProviderError], None]


class ProviderChain:
    def __init__(
        self,
        providers: Sequence[ProviderSettings],
        *,
        failover: FailoverHook | None = None,
    ) -> None:
        if not providers:
            raise ValueError("ProviderChain needs at least one provider")
        self.providers: list[ProviderSettings] = list(providers)
        self.failover = failover

    @property
    def primary(self) -> ProviderSettings:
        return self.providers[0]

    def __len__(self) -> int:
        return len(self.providers)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        names = " -> ".join(p.label for p in self.providers)
        return f"ProviderChain({names})"

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> ChatResult:
        """Try each provider in order; return the first successful result.

        On total failure the *last* error is raised, with every attempt
        recorded in ``exc.attempts`` as ``(label, message)`` pairs.
        """
        attempts: list[tuple[str, str]] = []
        for index, settings in enumerate(self.providers):
            client = OpenAICompatibleClient(settings)
            try:
                result = client.chat(messages, **kwargs)
            except ProviderError as exc:
                client.close()
                attempts.append((settings.label, str(exc)))
                exc.attempts = list(attempts)
                if exc.retryable and index + 1 < len(self.providers):
                    self._announce(settings.label, self.providers[index + 1].label, exc)
                    continue
                raise
            client.close()
            result.provider = settings.label
            return result
        raise AssertionError("unreachable")  # pragma: no cover

    def _announce(self, failed: str, next_label: str, error: ProviderError) -> None:
        if self.failover is not None:
            self.failover(failed, next_label, error)
