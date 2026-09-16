from __future__ import annotations

import pytest

from conftest import make_config_text
from termuxpilot.config import load_config, select_profile
from termuxpilot.provider import (
    HttpError,
    ProviderChain,
    ProviderError,
    StreamInterrupted,
)
from mockserver import MockServer

MESSAGES = [{"role": "user", "content": "ping"}]


def chain_from_config(cfg_text: str, **overrides) -> ProviderChain:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "config.yaml"
        path.write_text(cfg_text, encoding="utf-8")
        cfg = load_config(path)
        profile = select_profile(cfg, None, **overrides)
    events: list[tuple[str, str]] = []
    return ProviderChain(
        profile.chain(), failover=lambda a, b, e: events.append((a, b))
    ), events


def test_failover_to_fallback_on_503():
    good = MockServer().start()
    flaky = MockServer(fail_next=1, fail_status=503).start()
    try:
        cfg_text = make_config_text(
            base_url=flaky.base_url, fallbacks=[good.base_url]
        )
        chain, events = chain_from_config(cfg_text)
        result = chain.chat(MESSAGES)
        assert result.content.startswith("Mock reply")
        assert result.provider == "fallback-1"
        assert flaky.request_count == 1
        assert good.request_count == 1
        assert events == [("default", "fallback-1")]
    finally:
        flaky.stop()
        good.stop()


def test_no_failover_on_non_retryable_400():
    bad = MockServer(fail_next=9999, fail_status=400).start()
    good = MockServer().start()
    try:
        cfg_text = make_config_text(
            base_url=bad.base_url, fallbacks=[good.base_url]
        )
        chain, events = chain_from_config(cfg_text)
        with pytest.raises(HttpError) as exc:
            chain.chat(MESSAGES)
        assert exc.value.status == 400
        assert good.request_count == 0
        assert events == []
        assert [label for label, _ in exc.value.attempts] == ["default"]
    finally:
        bad.stop()
        good.stop()


def test_all_providers_fail_reports_all_attempts():
    a = MockServer(fail_next=9999, fail_status=503).start()
    b = MockServer(fail_next=9999, fail_status=502).start()
    try:
        cfg_text = make_config_text(base_url=a.base_url, fallbacks=[b.base_url])
        chain, events = chain_from_config(cfg_text)
        with pytest.raises(ProviderError) as exc:
            chain.chat(MESSAGES)
        assert [label for label, _ in exc.value.attempts] == ["default", "fallback-1"]
        assert events == [("default", "fallback-1")]
    finally:
        a.stop()
        b.stop()


def test_auth_failure_fails_over_to_keyless_local():
    good = MockServer().start()
    locked = MockServer(require_key="sekret").start()
    try:
        # primary sends no key -> 401 -> should fall over to keyless local
        cfg_text = make_config_text(
            base_url=locked.base_url, fallbacks=[good.base_url]
        )
        chain, events = chain_from_config(cfg_text)
        result = chain.chat(MESSAGES)
        assert result.provider == "fallback-1"
        assert events == [("default", "fallback-1")]
    finally:
        locked.stop()
        good.stop()


def test_mid_stream_interrupt_does_not_failover():
    flaky = MockServer(drop_after=2).start()
    good = MockServer().start()
    try:
        cfg_text = make_config_text(
            base_url=flaky.base_url, fallbacks=[good.base_url]
        )
        chain, events = chain_from_config(cfg_text)
        with pytest.raises(StreamInterrupted) as exc:
            chain.chat(MESSAGES)
        # partial output was preserved and no duplicate request was sent
        assert exc.value.partial is not None
        assert exc.value.partial.content
        assert good.request_count == 0
        assert events == []
    finally:
        flaky.stop()
        good.stop()


def test_chain_requires_providers():
    with pytest.raises(ValueError):
        ProviderChain([])


@pytest.mark.parametrize("failure", [KeyboardInterrupt(), RuntimeError("callback failed")])
def test_client_is_closed_on_non_provider_exception(monkeypatch, failure):
    from termuxpilot.config import ProviderSettings
    from termuxpilot.provider import fallback

    closed = []

    class Client:
        def __init__(self, settings):
            pass

        def chat(self, *args, **kwargs):
            raise failure

        def close(self):
            closed.append(True)

    monkeypatch.setattr(fallback, "OpenAICompatibleClient", Client)
    chain = ProviderChain([ProviderSettings(label="test", base_url="http://example.test/v1", model="m")])
    with pytest.raises(type(failure)):
        chain.chat(MESSAGES)
    assert closed == [True]
