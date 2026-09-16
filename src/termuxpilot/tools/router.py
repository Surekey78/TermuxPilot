"""Single permission, sanitization, and audit gate for every tool call.

Argument validation and previews happen before execution. Blocklists always
win; dry-run previews never need execution permission. Only a conservative
subset of inspection commands auto-runs in safe/standard mode. This policy
is not an OS sandbox and assumes trusted tools/binaries on the host.
"""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from ..audit import AuditLog
from ..safety import (
    RiskAssessment, assess_command, compile_rules, is_read_only, match_rules,
    redact_data, redact_secrets,
)
from .base import EXECUTE, ExecutionContext, READ, Tool, ToolRequest, ToolResult, truncate_output

MODES = ("safe", "standard", "yolo")

MODE_DESCRIPTIONS = {
    "safe": "read tools and a conservative subset of inspection commands only (not an OS sandbox)",
    "standard": "unknown commands and writes need your confirmation after a preview",
    "yolo": "fully automatic: the agent executes tools without asking (blocklist still applies)",
}


class ConfirmFn(Protocol):
    def __call__(self, name: str, preview: str, risk: RiskAssessment) -> bool: ...


@dataclass
class RouterDecision:
    executed: bool
    result: ToolResult
    reason: str = ""


class ToolRouter:
    def __init__(
        self,
        tools: list[Tool],
        ctx: ExecutionContext,
        *,
        mode: str = "standard",
        blocklist: list[str] | None = None,
        allowlist: list[str] | None = None,
        audit: AuditLog | None = None,
        confirm: ConfirmFn | None = None,
        announce: Callable[[str], None] | None = None,
        request_hook: Callable[[str, dict, RiskAssessment, str], None] | None = None,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.tools = {t.name: t for t in tools}
        self.ctx = ctx
        self.mode = mode
        self._block_rules = compile_rules(blocklist or [])
        self._allow_rules = compile_rules(allowlist or [])
        self.audit = audit or AuditLog()
        self.confirm = confirm
        self.announce = announce
        self.request_hook = request_hook

    def specs(self) -> list[dict[str, Any]]:
        return [t.openai_spec() for t in self.tools.values()]

    def names(self) -> list[str]:
        return list(self.tools)

    def execute(self, name: str, raw_args: Any) -> ToolResult:
        return self.decide(name, raw_args).result

    def sanitize(self, value: Any) -> Any:
        """For UI/model-bound copies only; never change actual tool inputs."""
        return redact_data(value) if self.ctx.redact_secrets else deepcopy(value)

    def decide(self, name: str, raw_args: Any) -> RouterDecision:
        started = time.monotonic()
        args = raw_args

        def finish(result: ToolResult, reason: str, *, executed: bool = False) -> RouterDecision:
            return self._finish(name, args, result, reason, executed, started)

        tool = self.tools.get(name)
        if tool is None:
            return finish(ToolResult(ok=False, denied=True, output=f"unknown tool: {name}"), "unknown tool")
        try:
            args = tool.resolve_args(raw_args)
        except ValueError as exc:
            return finish(ToolResult(ok=False, denied=True, output=str(exc)), "bad arguments")

        request = ToolRequest(name=name, args=args)
        try:
            risk, preview = self._assess(tool, request)
        except Exception as exc:
            # An unreadable/oversized preview must not crash the entire agent
            # or turn into an unpreviewed write.
            return finish(
                ToolResult(ok=False, denied=True, output=f"cannot preview '{name}': {exc}"),
                "preview failed",
            )
        request.risk = risk
        display_risk = self._safe_risk(risk)
        preview = truncate_output(self.sanitize(preview), self.ctx.max_output_chars)
        if self.request_hook is not None:
            self.request_hook(name, self.sanitize(args), display_risk, preview)

        denial = self._check_lists(request)
        if denial:
            why = "blocklist" if "blocklist" in denial else "allowlist"
            return finish(ToolResult(ok=False, denied=True, output=denial, risk=risk), why)

        if self.ctx.dry_run and tool.category != READ:
            self._announce(f"[dry-run] {name}: {preview.splitlines()[0] if preview else ''}")
            return finish(
                ToolResult(ok=True, output=f"dry-run: would call '{name}' with {self.sanitize(args)} "
                           "(nothing was done)", risk=risk),
                "dry-run",
            )

        readonly = tool.category == READ or (
            tool.category == EXECUTE and name == "run_shell"
            and is_read_only(args.get("cmd", "")) and risk.score == 0
        )
        if not readonly:
            if self.mode == "safe":
                return finish(
                    ToolResult(ok=False, denied=True, risk=risk, output=(
                        f"denied: '{name}' is not a vetted read-only operation in 'safe' mode. "
                        "Use read tools, or ask the user to switch to standard mode for approval."
                    )), "safe mode",
                )
            if self.mode == "standard":
                if self.confirm is None or not self.confirm(name, preview, display_risk):
                    return finish(
                        ToolResult(ok=False, denied=True, risk=risk, output=(
                            f"denied: '{name}' needs user confirmation and it was not given. "
                            "Explain the proposed operation or use a read-only alternative."
                        )), "confirmation declined",
                    )
                self._announce(f"approved by user: {name}")

        try:
            result = tool.handler(request, self.ctx)
        except KeyboardInterrupt:
            finish(ToolResult(ok=False, risk=risk, output="interrupted; side effects may be partial"),
                   "interrupted", executed=True)
            raise
        except Exception as exc:
            result = ToolResult(ok=False, output=f"tool '{name}' crashed: {exc!r}", risk=risk)
        if result.risk is None:
            result.risk = risk
        return finish(result, "executed", executed=True)

    def _assess(self, tool: Tool, request: ToolRequest) -> tuple[RiskAssessment, str]:
        if tool.preview is not None:
            risk, preview = tool.preview(request, self.ctx)
        else:
            risk, preview = RiskAssessment(), f"{tool.name} {request.args}"
        if tool.category == EXECUTE and request.name == "run_shell":
            cmd = request.args.get("cmd", "")
            shell_risk = assess_command(cmd)
            if not is_read_only(cmd):
                shell_risk.escalate("medium", "not in the vetted read-only command subset")
            risk.merge(shell_risk)
        return risk, preview

    def _check_lists(self, request: ToolRequest) -> str | None:
        if request.name != "run_shell":
            return None
        cmd = request.args.get("cmd", "")
        for raw in match_rules(cmd, self._block_rules):
            return f"denied: command matches blocklist rule {raw!r}"
        if self._allow_rules and not match_rules(cmd, self._allow_rules):
            return "denied: command does not match any allowlist rule"
        return None

    def _safe_risk(self, risk: RiskAssessment | None) -> RiskAssessment | None:
        if risk is None:
            return None
        return RiskAssessment(level=risk.level, reasons=self.sanitize(list(risk.reasons)))

    def _announce(self, text: str) -> None:
        if self.announce is not None:
            self.announce(self.sanitize(text))

    def _finish(
        self, name: str, args: Any, result: ToolResult, reason: str,
        executed: bool, started: float,
    ) -> RouterDecision:
        # Every exit path, including denials and preview/handler failures, goes
        # through the same sanitization boundary. Audit redaction is mandatory
        # even if the user opted out of model/UI output redaction.
        output = self.sanitize(result.output)
        result.truncated = result.truncated or len(output) > self.ctx.max_output_chars
        result.output = truncate_output(output, self.ctx.max_output_chars)
        result.risk = self._safe_risk(result.risk)
        result.duration = time.monotonic() - started
        self.audit.record(
            tool=redact_secrets(name),
            args=_redact_args(args),
            mode=self.mode,
            dry_run=self.ctx.dry_run,
            executed=executed,
            ok=result.ok,
            denied=result.denied,
            exit_code=result.exit_code,
            timed_out=result.timed_out,
            truncated=result.truncated,
            risk=result.risk.level if result.risk else None,
            reason=reason,
            duration_s=round(result.duration, 3),
        )
        return RouterDecision(executed, result, reason=reason)


def _redact_args(args: Any) -> Any:
    return redact_data(args)
