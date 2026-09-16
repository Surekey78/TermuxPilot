"""Tool layer primitives: the Tool spec, results, and execution context.

A *tool* is a discrete, named capability the agent may invoke via function
calling — never free-form shell execution.  The router
(:class:`termuxpilot.tools.router.ToolRouter`) sits between the agent and the
handlers and enforces the permission mode, allow/block lists, dry-run,
auditing, and secret redaction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..safety import RiskAssessment, is_protected_path

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

    def model_json(self) -> str:
        import json

        payload: dict[str, Any] = {"ok": self.ok, "output": self.output}
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
        return raw


def truncate_output(text: str, limit: int = 30_000) -> str:
    """Keep context-sized output: head + tail with a truncation note."""
    if len(text) <= limit:
        return text
    head = limit - 2_000
    tail = 2_000
    return (
        text[:head]
        + f"\n… [truncated {len(text) - head - tail:,} chars] …\n"
        + text[-tail:]
    )


def ensure_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def expand(path: Any) -> Path:
    return Path(ensure_str(path)).expanduser()
