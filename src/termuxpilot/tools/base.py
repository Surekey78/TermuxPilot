"""Tool layer primitives: the Tool spec, results, and execution context.

A *tool* is a discrete, named capability the agent may invoke via function
calling — never free-form shell execution.  The router
(:class:`termuxpilot.tools.router.ToolRouter`) sits between the agent and the
handlers and enforces the permission mode, allow/block lists, dry-run,
auditing, and secret redaction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..safety import RiskAssessment, is_protected_path
from .output import BoundedText, DEFAULT_OUTPUT_CHARS

# Tool categories — the permission model keys off these:
#   read    -> always allowed
#   write   -> gated: confirmation in standard/safe modes
#   execute -> gated + risk-classified (shell)
READ, WRITE, EXECUTE = "read", "write", "execute"


@dataclass
class ExecutionContext:
    """Config knobs available to tool handlers (built by the CLI/REPL)."""

    shell_timeout: float = 60.0
    shell_workdir: str | None = None
    shell_max_timeout: float = 3600.0
    shell_kill_grace: float = 1.0
    max_output_chars: int = DEFAULT_OUTPUT_CHARS
    redact_secrets: bool = True
    protected_paths: tuple[str, ...] = ()
    dry_run: bool = False

    def guard_path(self, path: str) -> str | None:
        if not self.protected_paths:
            return None
        return is_protected_path(str(path), self.protected_paths)


@dataclass
class ToolResult:
    ok: bool
    output: str  # text fed back to the model (redacted + truncated)
    exit_code: int | None = None
    denied: bool = False
    risk: RiskAssessment | None = None
    duration: float | None = None
    timed_out: bool = False
    truncated: bool = False

    def model_json(self) -> str:
        import json

        payload: dict[str, Any] = {"ok": self.ok, "output": self.output}
        if self.timed_out:
            payload["timed_out"] = True
        if self.truncated:
            payload["truncated"] = True
        if self.denied:
            payload["denied"] = True
        if self.exit_code is not None:
            payload["exit_code"] = self.exit_code
        if self.risk is not None and self.risk.reasons:
            payload["risk"] = {"level": self.risk.level, "reasons": self.risk.reasons}
        if self.duration is not None:
            payload["duration_s"] = round(self.duration, 2)
        return json.dumps(payload, ensure_ascii=False)


@dataclass
class ToolRequest:
    name: str
    args: dict[str, Any]
    risk: RiskAssessment = field(default_factory=RiskAssessment)
    state: dict[str, Any] = field(default_factory=dict)  # private preview/execution bookkeeping


Handler = Callable[[ToolRequest, ExecutionContext], ToolResult]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for the arguments
    category: str  # READ | WRITE | EXECUTE
    handler: Handler
    #: optional pre-check: returns (risk, preview_text) before execution
    preview: Callable[[ToolRequest, ExecutionContext], tuple[RiskAssessment, str]] | None = None

    def openai_spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def resolve_args(self, raw: Any) -> dict[str, Any]:
        """Accept either a parsed dict or a JSON string (as sent by models)."""
        if isinstance(raw, str):
            import json

            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                raise ValueError(f"tool '{self.name}' arguments are not valid JSON") from None
        if not isinstance(raw, dict):
            raise ValueError(f"tool '{self.name}' arguments must be a JSON object")
        # Validate the subset used by our built-in tool schemas before any
        # preview or side effect. In particular, bool is not an integer/number
        # and NaN/Infinity must never disable a timeout or bounds check.
        for name in self.parameters.get("required", []):
            if name not in raw:
                raise ValueError(f"tool '{self.name}' requires argument '{name}'")
        properties = self.parameters.get("properties", {})
        if self.parameters.get("additionalProperties") is False:
            if set(raw) - properties.keys():
                raise ValueError(f"tool '{self.name}' received unknown arguments")
        for name, value in raw.items():
            spec = properties.get(name, {})
            kind = spec.get("type")
            matches = {
                "string": isinstance(value, str),
                "integer": type(value) is int,
                "number": type(value) in (int, float),
                "boolean": type(value) is bool,
                "object": isinstance(value, dict),
                "array": isinstance(value, list),
            }
            if kind in matches and not matches[kind]:
                raise ValueError(f"argument '{name}' must be a {kind}")
            if kind in ("integer", "number"):
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError(f"argument '{name}' must be finite")
                if "minimum" in spec and value < spec["minimum"]:
                    raise ValueError(f"argument '{name}' must be >= {spec['minimum']}")
                if "exclusiveMinimum" in spec and value <= spec["exclusiveMinimum"]:
                    raise ValueError(f"argument '{name}' must be > {spec['exclusiveMinimum']}")
                if "maximum" in spec and value > spec["maximum"]:
                    raise ValueError(f"argument '{name}' must be <= {spec['maximum']}")
            if kind == "string":
                if len(value) < spec.get("minLength", 0):
                    raise ValueError(f"argument '{name}' must not be empty")
                if "maxLength" in spec and len(value) > spec["maxLength"]:
                    raise ValueError(f"argument '{name}' exceeds the size limit")
        return raw


def truncate_output(text: str, limit: int = DEFAULT_OUTPUT_CHARS) -> str:
    """Keep head + tail, including the truncation note within the limit."""
    if len(text) <= limit:
        return text
    buffer = BoundedText(limit)
    buffer.append(text)
    return buffer.text()


def ensure_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def expand(path: Any) -> Path:
    return Path(ensure_str(path)).expanduser()
