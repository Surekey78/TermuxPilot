from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from mockserver import MockServer  # noqa: E402


@pytest.fixture
def mock_server():
    server = MockServer().start()
    yield server
    server.stop()


@pytest.fixture
def write_config(tmp_path):
    def _write(text: str) -> Path:
        path = tmp_path / "config.yaml"
        path.write_text(text, encoding="utf-8")
        return path

    return _write


def make_config_text(
    *,
    base_url: str,
    model: str = "mock-model-1",
    api_key: str | None = None,
    fallbacks: list[str] | None = None,
    profile_block: str | None = None,
) -> str:
    lines = ["provider:"]
    lines.append(f"  base_url: \"{base_url}\"")
    if api_key is not None:
        lines.append(f'  api_key: "{api_key}"')
    lines.append(f'  model: "{model}"')
    lines.append("  temperature: 0.2")
    if fallbacks:
        lines.append("fallback:")
        for url in fallbacks:
            lines.append(f"  - base_url: \"{url}\"")
            lines.append("    model: \"mock-model-1\"")
    if profile_block:
        lines.append(profile_block)
    return "\n".join(lines) + "\n"
