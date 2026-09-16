from __future__ import annotations

import pytest

from termuxpilot.config import (
    ConfigError,
    expand_env,
    load_config,
    resolve_config_path,
    select_profile,
    write_sample_config,
)


def test_expand_env_var():
    import os

    os.environ["TP_TEST_KEY"] = "secret123"
    try:
        assert expand_env("${TP_TEST_KEY}") == "secret123"
        assert expand_env("${TP_TEST_KEY:-fallback}") == "secret123"
        assert expand_env("a${TP_TEST_KEY}b") == "asecret123b"
    finally:
        del os.environ["TP_TEST_KEY"]


def test_expand_env_unset_and_default():
    assert expand_env("${TP_DEFINITELY_UNSET_VAR_XYZ}") == ""
    assert expand_env("${TP_DEFINITELY_UNSET_VAR_XYZ:-default}") == "default"
    assert expand_env({"k": ["${NOPE_XYZ:-d}"]}) == {"k": ["d"]}
    assert expand_env(3.14) == 3.14


def test_missing_config_and_no_env_raises(write_config, monkeypatch, tmp_path):
    missing = tmp_path / "nope.yaml"
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    with pytest.raises(ConfigError, match="tp config init"):
        load_config(missing)


def test_env_fallback_when_no_config(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_BASE", "https://example.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "envkey")
    monkeypatch.setenv("OPENAI_MODEL", "env-model")
    cfg = load_config(tmp_path / "nope.yaml")
    prof = cfg.get_profile("default")
    assert prof.primary.base_url == "https://example.test/v1"
    assert prof.primary.model == "env-model"
    assert prof.primary.api_key == "envkey"
    assert prof.fallbacks == []


def test_basic_config_with_single_fallback_dict(write_config):
    text = (
        "provider:\n"
        '  base_url: "https://api.groq.com/openai/v1"\n'
        '  api_key: "k1"\n'
        '  model: "llama-3.3-70b-versatile"\n'
        "  temperature: 0.3\n"
        "  max_tokens: 4096\n"
        "fallback:\n"
        '  base_url: "http://192.168.1.5:11434/v1"\n'
        '  model: "qwen2.5-coder:14b"\n'
    )
    cfg = load_config(write_config(text))
    prof = cfg.get_profile("default")
    assert prof.primary.base_url == "https://api.groq.com/openai/v1"
    assert prof.primary.model == "llama-3.3-70b-versatile"
    assert prof.primary.temperature == 0.3
    assert prof.primary.max_tokens == 4096
    assert prof.primary.has_api_key()
    assert len(prof.fallbacks) == 1
    assert prof.fallbacks[0].base_url == "http://192.168.1.5:11434/v1"
    assert prof.fallbacks[0].model == "qwen2.5-coder:14b"
    assert not prof.fallbacks[0].has_api_key()  # no key -> local server
    assert [p.label for p in prof.chain()] == ["default", "fallback-1"]


def test_fallback_list_and_string_entries(write_config):
    text = (
        "provider:\n"
        '  base_url: "https://a.test/v1"\n'
        '  model: "m1"\n'
        "fallback:\n"
        '  - base_url: "http://b.test/v1"\n'
        '    name: lan\n'
        '    model: "m2"\n'
        '  - "http://c.test/v1"\n'
    )
    cfg = load_config(write_config(text))
    prof = cfg.get_profile("default")
    labels = [p.label for p in prof.chain()]
    assert labels == ["default", "lan", "fallback-2"]
    assert prof.fallbacks[1].base_url == "http://c.test/v1"
    assert prof.fallbacks[1].model is None


def test_profiles_inherit_top_level(write_config):
    text = (
        "provider:\n"
        '  base_url: "https://a.test/v1"\n'
        '  api_key: "k1"\n'
        '  model: "m1"\n'
        "  temperature: 0.5\n"
        "fallback:\n"
        '  - base_url: "http://b.test/v1"\n'
        "profiles:\n"
        "  local:\n"
        "    provider:\n"
        '      base_url: "http://127.0.0.1:8080/v1"\n'
        '      api_key: "none"\n'
        '      model: "m2"\n'
        "    fallback: []\n"
    )
    cfg = load_config(write_config(text))
    local = cfg.get_profile("local")
    # inherited
    assert local.primary.temperature == 0.5
    # overridden
    assert local.primary.base_url == "http://127.0.0.1:8080/v1"
    assert local.primary.model == "m2"
    assert not local.primary.has_api_key()  # "none" -> no auth header
    assert local.fallbacks == []  # explicit [] disables inherited chain


def test_profile_without_fallback_inherits_chain(write_config):
    text = (
        "provider:\n"
        '  base_url: "https://a.test/v1"\n'
        '  model: "m1"\n'
        "fallback:\n"
        '  - base_url: "http://b.test/v1"\n'
        "profiles:\n"
        "  alt:\n"
        "    provider:\n"
        '      base_url: "https://c.test/v1"\n'
    )
    cfg = load_config(write_config(text))
    alt = cfg.get_profile("alt")
    assert alt.primary.base_url == "https://c.test/v1"
    assert len(alt.fallbacks) == 1
    assert alt.fallbacks[0].base_url == "http://b.test/v1"


def test_default_profile_selection(write_config):
    text = (
        "default_profile: cloud\n"
        "profiles:\n"
        "  cloud:\n"
        "    provider:\n"
        '      base_url: "https://a.test/v1"\n'
        '      model: "m1"\n'
        "  local:\n"
        "    provider:\n"
        '      base_url: "http://b.test/v1"\n'
        '      model: "m2"\n'
    )
    cfg = load_config(write_config(text))
    assert cfg.default_profile == "cloud"
    assert cfg.get_profile("cloud").primary.model == "m1"
    assert cfg.get_profile("local").primary.model == "m2"


def test_unknown_default_profile_raises(write_config):
    text = (
        "default_profile: nope\n"
        "provider:\n"
        '  base_url: "https://a.test/v1"\n'
        '  model: "m1"\n'
    )
    with pytest.raises(ConfigError, match="not defined"):
        load_config(write_config(text))


def test_unknown_profile_selection():
    import yaml

    from termuxpilot.config import _parse_config, expand_env

    from pathlib import Path

    text = (
        "provider:\n"
        '  base_url: "https://a.test/v1"\n'
        '  model: "m1"\n'
    )
    cfg = _parse_config(expand_env(yaml.safe_load(text)), Path("cfg"))
    with pytest.raises(ConfigError, match="unknown profile"):
        cfg.get_profile("ghost")


def test_select_profile_cli_overrides():
    import yaml

    from pathlib import Path

    from termuxpilot.config import _parse_config, expand_env

    text = (
        "provider:\n"
        '  base_url: "https://a.test/v1"\n'
        '  api_key: "k"\n'
        '  model: "m1"\n'
        "  temperature: 0.5\n"
    )
    cfg = _parse_config(expand_env(yaml.safe_load(text)), Path("cfg"))
    prof = select_profile(cfg, None, model="m9", temperature=0.1, base_url="https://z.test/v1")
    assert prof.primary.model == "m9"
    assert prof.primary.temperature == 0.1
    assert prof.primary.base_url == "https://z.test/v1"
    assert prof.primary.api_key == "k"  # untouched
    # default profile unchanged
    assert cfg.get_profile("default").primary.model == "m1"


def test_base_url_trailing_slash_normalized(write_config):
    text = (
        "provider:\n"
        '  base_url: "https://a.test/v1///"\n'
        '  model: "m1"\n'
    )
    cfg = load_config(write_config(text))
    assert cfg.get_profile("default").primary.base_url == "https://a.test/v1"


def test_invalid_configs_raise(write_config):
    with pytest.raises(ConfigError):
        load_config(write_config("provider: []"))
    with pytest.raises(ConfigError, match="base_url"):
        load_config(write_config("provider:\n  model: m1\n"))
    with pytest.raises(ConfigError, match="temperature"):
        load_config(
            write_config("provider:\n  base_url: u\n  model: m\n  temperature: hot\n")
        )
    with pytest.raises(ConfigError, match="not defined"):
        load_config(
            write_config(
                "provider:\n  base_url: u\n  model: m\n"
                "default_profile: ghost\n"
            )
        )


def test_profiles_only_requires_default_profile(write_config):
    text = (
        "profiles:\n"
        "  a:\n"
        "    provider:\n"
        '      base_url: "https://a.test/v1"\n'
        '      model: "m1"\n'
        "  b:\n"
        "    provider:\n"
        '      base_url: "https://b.test/v1"\n'
        '      model: "m2"\n'
    )
    with pytest.raises(ConfigError, match="default_profile"):
        load_config(write_config(text))
    cfg = load_config(write_config("default_profile: b\n" + text))
    assert cfg.default_profile == "b"
    assert cfg.get_profile("b").primary.base_url == "https://b.test/v1"


def test_resolve_config_path_explicit_and_env(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit.yaml"
    assert resolve_config_path(explicit) == explicit
    envp = tmp_path / "env.yaml"
    monkeypatch.setenv("TERMUXPILOT_CONFIG", str(envp))
    assert resolve_config_path(None) == envp
    monkeypatch.delenv("TERMUXPILOT_CONFIG")
    assert resolve_config_path(None) == (
        __import__("pathlib").Path.home() / ".termuxpilot" / "config.yaml"
    )


def test_sample_config_round_trips(tmp_path, monkeypatch):
    # sample must not blow up on the unset GROQ_API_KEY (expands to "")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    path = write_sample_config(tmp_path / "config.yaml")
    cfg = load_config(path)
    assert cfg.default_profile == "cloud"
    assert set(cfg.profiles) == {"default", "cloud", "local", "ollama-lan"}
    cloud = cfg.get_profile("cloud")
    assert cloud.primary.model == "llama-3.3-70b-versatile"
    assert len(cloud.fallbacks) == 2  # inherited from top level
    local = cfg.get_profile("local")
    assert local.fallbacks == []
    assert not local.primary.has_api_key()
    assert cfg.get_profile("ollama-lan").fallbacks[0].model == (
        "qwen2.5-coder-7b-instruct-q4_K_M"
    )
    with pytest.raises(ConfigError, match="already exists"):
        write_sample_config(path)
    write_sample_config(path, force=True)


@pytest.mark.parametrize("block,key", [
    ("tools:\n  shell:\n    timeout: .nan\n", "timeout"),
    ("tools:\n  shell:\n    timeout: .inf\n", "timeout"),
    ("tools:\n  shell:\n    timeout: true\n", "timeout"),
    ("tools:\n  shell:\n    max_timeout: 0\n", "max_timeout"),
    ("tools:\n  shell:\n    timeout: 10\n    max_timeout: 5\n", "max_timeout"),
    ("tools:\n  shell:\n    kill_grace: -1\n", "kill_grace"),
    ("tools:\n  shell:\n    kill_grace: 31\n", "kill_grace"),
    ("tools:\n  max_output_chars: 127\n", "max_output_chars"),
    ("tools:\n  max_output_chars: 1000001\n", "max_output_chars"),
    ("tools:\n  max_output_chars: true\n", "max_output_chars"),
    ("agent:\n  max_tool_rounds: true\n", "max_tool_rounds"),
    ("agent:\n  max_tool_calls: 0\n", "max_tool_calls"),
    ("agent:\n  max_context_chars: -1\n", "max_context_chars"),
])
def test_invalid_runtime_limits_raise_config_error(write_config, block, key):
    with pytest.raises(ConfigError, match=key):
        load_config(write_config("provider:\n  base_url: https://example.test/v1\n  model: m\n" + block))


def test_runtime_limits_are_loaded_and_wired_to_session(write_config):
    from termuxpilot.cli import build_session
    from termuxpilot.provider import ProviderChain

    cfg = load_config(write_config(
        "provider:\n  base_url: https://example.test/v1\n  model: m\n"
        "tools:\n  max_output_chars: 8192\n  shell:\n"
        "    timeout: 900\n    max_timeout: 7200\n    kill_grace: 0\n"
        "agent:\n  max_tool_rounds: 24\n  max_tool_calls: 80\n  max_context_chars: 100000\n"
    ))
    profile = cfg.get_profile("default")
    router, agent = build_session(cfg, profile, ProviderChain(profile.chain()), mode=None, dry_run=False)
    assert router.ctx.shell_timeout == 900
    assert router.ctx.shell_max_timeout == 7200
    assert router.ctx.shell_kill_grace == 0
    assert router.ctx.max_output_chars == 8192
    assert agent.max_rounds == 24
    assert agent.max_tool_calls == 80
    assert agent.max_context_chars == 100000


def test_existing_long_timeout_config_gets_a_compatible_ceiling(write_config):
    cfg = load_config(write_config(
        "provider:\n  base_url: https://example.test/v1\n  model: m\n"
        "tools:\n  shell:\n    timeout: 7200\n"
    ))
    assert cfg.tools.shell_max_timeout == 7200
