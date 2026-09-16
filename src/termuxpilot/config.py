"""TermuxPilot configuration.

Config file: ``~/.termuxpilot/config.yaml`` (override with ``$TERMUXPILOT_CONFIG``
or ``--config-path``).

Layout
------
* A top-level ``provider:`` block plus optional ``fallback:`` block form the
  *default profile* (name: ``default``) — exactly the shape shown in the
  project spec.
* Named profiles live under ``profiles:``.  A profile **inherits** the
  top-level ``provider``/``fallback`` blocks and overrides only the keys it
  sets, so small profiles stay small.  ``fallback: []`` disables fallback.
* ``fallback`` accepts a single provider mapping, a *list* of mappings (the
  fallback chain, tried in order), or a list of plain ``base_url`` strings.
* String values support environment expansion: ``${VAR}`` and
  ``${VAR:-default}``.  An unset variable with no default expands to the
  empty string; an empty ``api_key`` means "send no Authorization header".

Precedence (highest wins): CLI flags > selected profile > top-level defaults.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

APP_DIR_NAME = ".termuxpilot"
CONFIG_FILE_NAME = "config.yaml"
CONFIG_ENV_VAR = "TERMUXPILOT_CONFIG"

DEFAULT_TIMEOUT = 60.0
DEFAULT_PROFILE_NAME = "default"

#: api_key values that mean "no authentication" (local servers)
NO_KEY_VALUES = {"", "none", "null", "empty", "ollama", "local", "dummy"}

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


class ConfigError(Exception):
    """Raised for missing/unreadable/invalid configuration."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ProviderSettings:
    """One OpenAI-compatible endpoint (base_url + api_key + model)."""

    label: str
    base_url: str
    model: str | None = None
    api_key: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    timeout: float = DEFAULT_TIMEOUT
    extra_headers: dict[str, str] = field(default_factory=dict)

    def has_api_key(self) -> bool:
        return bool(self.api_key) and self.api_key.strip().lower() not in NO_KEY_VALUES

    def describe(self) -> str:
        """One-line human description (key masked)."""
        key = "key=set" if self.has_api_key() else "no-key"
        parts = [self.label]
        if self.model:
            parts.append(f"model={self.model}")
        parts.append(self.base_url.rstrip("/"))
        parts.append(key)
        return " ".join(parts)

    def masked_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "base_url": self.base_url,
            "model": self.model,
            "api_key": "***set***" if self.has_api_key() else None,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "extra_headers": {
                k: ("***" if "key" in k.lower() or "token" in k.lower() else v)
                for k, v in self.extra_headers.items()
            },
        }


@dataclass
class Profile:
    name: str
    primary: ProviderSettings
    fallbacks: list[ProviderSettings] = field(default_factory=list)

    def chain(self) -> list[ProviderSettings]:
        return [self.primary, *self.fallbacks]


@dataclass
class AppConfig:
    profiles: dict[str, Profile]
    default_profile: str
    system_prompt: str | None = None
    path: Path | None = None

    def get_profile(self, name: str) -> Profile:
        try:
            return self.profiles[name]
        except KeyError:
            known = ", ".join(sorted(self.profiles)) or "<none>"
            raise ConfigError(f"unknown profile '{name}'. Known profiles: {known}") from None


# ---------------------------------------------------------------------------
# Environment expansion
# ---------------------------------------------------------------------------


def expand_env(value: Any) -> Any:
    """Recursively expand ``${VAR}`` / ``${VAR:-default}`` in strings."""

    if isinstance(value, str):
        def _repl(match: re.Match[str]) -> str:
            var, default = match.group(1), match.group(2)
            raw = os.environ.get(var)
            if raw is None or raw == "":
                return default if default is not None else ""
            return raw

        return _ENV_PATTERN.sub(_repl, value)
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _as_dict(block: Any, where: str) -> dict:
    if block is None:
        return {}
    if isinstance(block, dict):
        return block
    raise ConfigError(f"{where}: expected a mapping, got {type(block).__name__}")


def _normalize_fallback(fallback: Any, where: str) -> list[dict]:
    """Accept None / single mapping / list of mappings-or-URL-strings."""
    if fallback is None:
        return []
    if isinstance(fallback, dict):
        return [fallback]
    if isinstance(fallback, list):
        entries: list[dict] = []
        for i, item in enumerate(fallback):
            if isinstance(item, str):
                entries.append({"base_url": item})
            elif isinstance(item, dict):
                entries.append(item)
            else:
                raise ConfigError(f"{where}[{i}]: expected a mapping or a base_url string")
        return entries
    raise ConfigError(f"{where}: expected a mapping or a list, got {type(fallback).__name__}")


def _parse_provider(
    label: str,
    block: dict,
    *,
    base: dict | None = None,
    where: str = "provider",
) -> ProviderSettings:
    merged = {**(base or {}), **block}
    base_url = merged.get("base_url")
    if not base_url or not isinstance(base_url, str):
        raise ConfigError(f"{where}: 'base_url' is required (profile '{label}')")

    model = merged.get("model")
    if model is not None and not isinstance(model, str):
        raise ConfigError(f"{where}: 'model' must be a string (profile '{label}')")

    api_key = merged.get("api_key")
    if api_key is not None and not isinstance(api_key, str):
        raise ConfigError(f"{where}: 'api_key' must be a string (profile '{label}')")

    temperature = merged.get("temperature")
    if temperature is not None and not isinstance(temperature, (int, float)):
        raise ConfigError(f"{where}: 'temperature' must be a number")

    max_tokens = merged.get("max_tokens")
    if max_tokens is not None and not isinstance(max_tokens, int):
        raise ConfigError(f"{where}: 'max_tokens' must be an integer")

    timeout = merged.get("timeout", DEFAULT_TIMEOUT)
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ConfigError(f"{where}: 'timeout' must be a positive number of seconds")

    extra_headers = _as_dict(merged.get("headers"), f"{where}.headers")
    extra_headers = {str(k): str(v) for k, v in extra_headers.items()}

    return ProviderSettings(
        label=label,
        base_url=base_url.rstrip("/"),
        model=model,
        api_key=api_key,
        temperature=float(temperature) if temperature is not None else None,
        max_tokens=max_tokens,
        timeout=float(timeout),
        extra_headers=extra_headers,
    )


def _parse_profile(name: str, block: dict) -> Profile:
    provider_block = _as_dict(block.get("provider"), f"profiles.{name}.provider")
    if not provider_block:
        raise ConfigError(f"profiles.{name}: 'provider' block is required")
    primary = _parse_provider(name, provider_block, where=f"profiles.{name}.provider")
    fallbacks: list[ProviderSettings] = []
    for i, fb in enumerate(_normalize_fallback(block.get("fallback"), f"profiles.{name}.fallback")):
        label = str(fb.get("name") or f"{name}-fallback-{i + 1}")
        fallbacks.append(
            _parse_provider(label, fb, where=f"profiles.{name}.fallback[{i}]")
        )
    return Profile(name=name, primary=primary, fallbacks=fallbacks)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def config_dir() -> Path:
    return Path.home() / APP_DIR_NAME


def default_config_path() -> Path:
    return config_dir() / CONFIG_FILE_NAME


def resolve_config_path(explicit: str | Path | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get(CONFIG_ENV_VAR)
    if env:
        return Path(env).expanduser()
    return default_config_path()


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load config from *path* (see :func:`resolve_config_path`).

    If no config file exists, falls back to the conventional OpenAI
    environment variables (``OPENAI_API_BASE``/``OPENAI_BASE_URL``,
    ``OPENAI_API_KEY``, ``OPENAI_MODEL``); if those are absent too, raises
    :class:`ConfigError` with a pointer at ``tp config init``.
    """
    resolved = resolve_config_path(path)
    if not resolved.exists():
        env_config = _config_from_env(resolved)
        if env_config is not None:
            return env_config
        raise ConfigError(
            f"no config file at {resolved} and no OPENAI_* environment "
            "variables set.\n"
            "  Create one with:  tp config init\n"
            f"  or set {CONFIG_ENV_VAR} to point at an existing config."
        )
    try:
        text = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config file {resolved}: {exc}") from exc
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {resolved}: {exc}") from exc
    if doc is None:
        doc = {}
    if not isinstance(doc, dict):
        raise ConfigError(f"{resolved}: top level must be a mapping")
    return _parse_config(expand_env(doc), resolved)


def _config_from_env(path: Path) -> AppConfig | None:
    base_url = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
    model = os.environ.get("OPENAI_MODEL")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not base_url or not model:
        return None
    provider = ProviderSettings(
        label=DEFAULT_PROFILE_NAME,
        base_url=base_url.rstrip("/"),
        model=model,
        api_key=api_key or None,
    )
    return AppConfig(
        profiles={DEFAULT_PROFILE_NAME: Profile(DEFAULT_PROFILE_NAME, provider)},
        default_profile=DEFAULT_PROFILE_NAME,
        path=path,
    )


def _parse_config(doc: dict, path: Path) -> AppConfig:
    top_provider = _as_dict(doc.get("provider"), "provider")
    top_fallback_raw = _normalize_fallback(doc.get("fallback"), "fallback")

    profiles: dict[str, Profile] = {}
    if top_provider or top_fallback_raw:
        primary = _parse_provider(DEFAULT_PROFILE_NAME, top_provider, where="provider")
        fallbacks = [
            _parse_provider(
                str(fb.get("name") or f"fallback-{i + 1}"),
                fb,
                where=f"fallback[{i}]",
            )
            for i, fb in enumerate(top_fallback_raw)
        ]
        profiles[DEFAULT_PROFILE_NAME] = Profile(DEFAULT_PROFILE_NAME, primary, fallbacks)

    for name, pblock in _as_dict(doc.get("profiles"), "profiles").items():
        name = str(name)
        pblock = _as_dict(pblock, f"profiles.{name}")
        # A named profile inherits the top-level provider/fallback and
        # overrides only what it sets.
        provider_block = {
            **top_provider,
            **_as_dict(pblock.get("provider"), f"profiles.{name}.provider"),
        }
        if not provider_block:
            raise ConfigError(f"profiles.{name}: 'provider' block is required")
        if "fallback" in pblock:
            fallback_raw = _normalize_fallback(pblock.get("fallback"), f"profiles.{name}.fallback")
        else:
            fallback_raw = top_fallback_raw
        primary = _parse_provider(name, provider_block, where=f"profiles.{name}.provider")
        fallbacks = [
            _parse_provider(
                str(fb.get("name") or f"{name}-fallback-{i + 1}"),
                fb,
                where=f"profiles.{name}.fallback[{i}]",
            )
            for i, fb in enumerate(fallback_raw)
        ]
        profiles[name] = Profile(name, primary, fallbacks)

    if not profiles:
        raise ConfigError(
            f"{path}: empty config — define a 'provider:' block and/or profiles."
        )

    has_top_level = bool(top_provider or top_fallback_raw)
    if not has_top_level and not doc.get("default_profile"):
        known = ", ".join(sorted(profiles))
        raise ConfigError(
            "no top-level 'provider' block — set 'default_profile' to one of: "
            f"{known}"
        )
    default_name = str(doc.get("default_profile") or DEFAULT_PROFILE_NAME)
    if default_name not in profiles:
        known = ", ".join(sorted(profiles))
        raise ConfigError(f"default_profile '{default_name}' is not defined. Known: {known}")

    system_prompt = doc.get("system_prompt")
    if system_prompt is not None and not isinstance(system_prompt, str):
        raise ConfigError("system_prompt must be a string")

    return AppConfig(
        profiles=profiles,
        default_profile=default_name,
        system_prompt=system_prompt or None,
        path=path,
    )


# ---------------------------------------------------------------------------
# Profile selection / CLI overrides
# ---------------------------------------------------------------------------


def select_profile(
    config: AppConfig,
    name: str | None = None,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
) -> Profile:
    """Return the requested profile with CLI overrides applied to the primary."""
    name = name or config.default_profile
    profile = config.get_profile(name)
    primary = profile.primary
    changed = {
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
    }
    if any(v is not None for v in changed.values()):
        kwargs = {k: v for k, v in changed.items() if v is not None}
        primary = replace(primary, **kwargs)
    return replace(profile, primary=primary)


# ---------------------------------------------------------------------------
# Sample config for `tp config init`
# ---------------------------------------------------------------------------

SAMPLE_CONFIG = """\
# TermuxPilot configuration
#
# - String values support env expansion: ${VAR} or ${VAR:-default}
# - api_key: "none" (or omitted) sends no Authorization header (local servers)
# - The top-level `provider`/`fallback` blocks below are profile "default"
#   (used when no --profile is given) AND the inheritance base: named
#   profiles override only the keys they set.
# - `fallback` may be a single provider OR a list (tried in order).
# - Switch profiles with:  tp --profile <name>

default_profile: cloud

# Global system prompt (optional — TermuxPilot ships a sensible default).
# system_prompt: |
#   You are TermuxPilot, an assistant on my Android phone, running in Termux.

# Cloud provider (the "default" profile / inheritance base) ----------------
provider:
  base_url: "https://api.groq.com/openai/v1"
  api_key: "${GROQ_API_KEY}"
  model: "llama-3.3-70b-versatile"
  temperature: 0.3
  max_tokens: 4096
  timeout: 60

# Fallback chain: tried in order if the primary fails or times out.
fallback:
  - name: lan-ollama
    base_url: "http://192.168.1.5:11434/v1"
    model: "qwen2.5-coder:14b"
  - name: local-llamacpp
    base_url: "http://localhost:8080/v1"
    api_key: "none"
    model: "qwen2.5-coder-7b-instruct-q4_K_M"

# Named profiles -----------------------------------------------------------
profiles:
  # Identical to the top-level blocks above, just given a name so that
  # `tp --profile cloud` works as you'd expect.
  cloud: {}

  local:
    provider:
      base_url: "http://localhost:8080/v1"
      api_key: "none"
      model: "qwen2.5-coder-7b-instruct-q4_K_M"
    fallback: []            # no cloud fallback when intentionally local

  ollama-lan:
    provider:
      base_url: "http://192.168.1.5:11434/v1"
      model: "qwen2.5-coder:14b"
    fallback:
      - base_url: "http://localhost:8080/v1"
        api_key: "none"
        model: "qwen2.5-coder-7b-instruct-q4_K_M"
"""


def write_sample_config(path: str | Path, *, force: bool = False) -> Path:
    path = Path(path).expanduser()
    if path.exists() and not force:
        raise ConfigError(f"{path} already exists (use --force to overwrite)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SAMPLE_CONFIG, encoding="utf-8")
    return path
