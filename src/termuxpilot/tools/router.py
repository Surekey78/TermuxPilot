"""ToolRouter — the single enforcement point between the agent and the tools.

Permission model (enforced HERE, not inside individual tools):

  mode    read tools        write/execute tools
  ------  ----------------  ---------------------------------
  safe    allowed           denied (read-only session)
  standard allowed          dry-run preview + user confirmation
  yolo    allowed           executed automatically (still audited; the
                            config blocklist always wins)

Additional gates, in order:
  1. tool exists & arguments parse
  2. shell allowlist (if configured, the command must match)
  3. shell blocklist (always denies — any mode)
  4. mode permission (above) — with dry-run mode short-circuiting execution
  5. confirmation (interactive prompt; non-interactive sessions auto-deny)

Every decision (allowed / denied / dry-run) is appended to the audit log.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from ..audit import AuditLog
from ..safety import RiskAssessment, assess_command, compile_rules, is_read_only, match_rules
from .base import EXECUTE, ExecutionContext, READ, Tool, ToolRequest, ToolResult, WRITE

MODES = ("safe", "standard", "yolo")

MODE_DESCRIPTIONS = {
    "safe": "read-only: commands may inspect but never modify the system",
    "standard": "writes and commands need your confirmation after a preview",
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
        announce: Callable[[str, str], None] | None = None,
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
        #: called once per request, after assessment, before any gate/execution
        self.request_hook = request_hook

    # ------------------------------------------------------------------ api

    def specs(self) -> list[dict[str, Any]]:
        return [t.openai_spec() for t in self.tools.values()]

    def names(self) -> list[str]:
        return list(self.tools)

    def execute(self, name: str, raw_args: Any) -> ToolResult:
        decision = self.decide(name, raw_args)
        return decision.result

    def decide(self, name: str, raw_args: Any) -> RouterDecision:
        started = time.monotonic()
        tool = self.tools.get(name)
        if tool is None:
            result = ToolResult(ok=False, denied=True, output=f"unknown tool: {name}")
            self._audit(name, None, result, reason="unknown tool", duration=time.monotonic() - started)
            return RouterDecision(False, result, reason="unknown tool")

        try:
            args = tool.resolve_args(raw_args)
        except ValueError as exc:
            result = ToolResult(ok=False, denied=True, output=str(exc))
            self._audit(name, raw_args, result, reason="bad arguments",
                        duration=time.monotonic() - started)
            return RouterDecision(False, result, reason="bad arguments")

        request = ToolRequest(name=name, args=args)
        risk, preview = self._assess(tool, request)
        if self.request_hook is not None:
            self.request_hook(name, args, risk, preview)

        # --- gates ---------------------------------------------------------
        denial = self._check_lists(request, risk)
        if denial:
            result = ToolResult(ok=False, denied=True, output=denial, risk=risk)
            why = "blocklist" if "blocklist" in denial else "allowlist"
            self._audit(name, args, result, reason=why, duration=time.monotonic() - started)
            return RouterDecision(False, result, reason=why)

        if tool.category == EXECUTE:
            cmd = str(request.args.get("cmd", ""))
            readonly = is_read_only(cmd) and risk.score == 0  # risk level "low"
            gate_reason = None
            if self.mode == "safe" and not readonly:
                gate_reason = (
                    "denied: session is in 'safe' (read-only) mode and this "
                    "command would modify the system. Ask the user to switch "
                    "modes (e.g. `/mode standard` or `--mode standard`), or "
                    "suggest a read-only alternative."
                )
            elif self.mode == "standard" and not readonly:
                if self.confirm is None or not self.confirm(name, preview, risk):
                    gate_reason = (
                        "denied: this command needs user confirmation and it was "
                        "not given (non-interactive session or the user said no). "
                        "Adapt: explain what you were about to do and why."
                    )
                else:
                    self._announce(f"approved by user: {name}")
            if gate_reason:
                result = ToolResult(ok=False, denied=True, output=gate_reason, risk=risk)
                why = "safe mode" if self.mode == "safe" else "confirmation declined"
                self._audit(name, args, result, reason=why,
                            duration=time.monotonic() - started)
                return RouterDecision(False, result, reason=why)
        elif tool.category == WRITE:
            gate_reason = None
            if self.mode == "safe":
                gate_reason = (
                    f"denied: session is in 'safe' (read-only) mode and "
                    f"'{name}' would modify files. Ask the user to switch modes "
                    "(e.g. `/mode standard` or `--mode standard`)."
                )
            elif self.mode == "standard":
                if self.confirm is None or not self.confirm(name, preview, risk):
                    gate_reason = (
                        f"denied: '{name}' needs user confirmation and it was "
                        "not given (non-interactive session or the user said no). "
                        "Adapt: explain what you were about to do and why."
                    )
                else:
                    self._announce(f"approved by user: {name}")
            if gate_reason:
                result = ToolResult(ok=False, denied=True, output=gate_reason, risk=risk)
                why = "safe mode" if self.mode == "safe" else "confirmation declined"
                self._audit(name, args, result, reason=why,
                            duration=time.monotonic() - started)
                return RouterDecision(False, result, reason=why)

        if self.ctx.dry_run and tool.category != READ:
            result = ToolResult(
                ok=True,
                output=f"dry-run: would call '{name}' with {args} (nothing was done)",
                risk=risk,
            )
            self._audit(name, args, result, reason="dry-run",
                        duration=time.monotonic() - started)
            self._announce(f"[dry-run] {name}: {preview.splitlines()[0]}")
            return RouterDecision(False, result, reason="dry-run")

        # --- execute ---------------------------------------------------------
        try:
            result = tool.handler(request, self.ctx)
        except Exception as exc:  # noqa: BLE001 - surfaced to the model
            result = ToolResult(ok=False, output=f"tool '{name}' crashed: {exc!r}", risk=risk)
        if result.risk is None:
            result.risk = risk
        result.duration = time.monotonic() - started
        self._audit(name, args, result, reason="executed",
                    duration=time.monotonic() - started)
        return RouterDecision(True, result, reason="executed")

    # -------------------------------------------------------------- helpers

    def _assess(self, tool: Tool, request: ToolRequest) -> tuple[RiskAssessment, str]:
        if tool.preview is not None:
            risk, preview = tool.preview(request, self.ctx)
        else:
            risk, preview = RiskAssessment(), f"{tool.name} {request.args}"
        if tool.category == "execute" and request.name == "run_shell":
            cmd = str(request.args.get("cmd", ""))
            shell_risk = assess_command(cmd)
            if not is_read_only(cmd):
                shell_risk.escalate("medium", "mutates system state")
            risk.merge(shell_risk)
            if not preview.strip().startswith("$"):
                preview = f"$ {cmd}"
        return risk, preview

    def _check_lists(self, request: ToolRequest, risk: RiskAssessment) -> str | None:
        if request.name != "run_shell":
            return None
        cmd = str(request.args.get("cmd", ""))
        for raw in match_rules(cmd, self._block_rules):
            return f"denied: command matches blocklist rule {raw!r}"
        if self._allow_rules:
            if not match_rules(cmd, self._allow_rules):
                return (
                    "denied: command does not match any allowlist rule "
                    f"({', '.join(repr(r) for r, _ in self._allow_rules)})"
                )
        return None

    def _announce(self, text: str) -> None:
        if self.announce is not None:
            self.announce(text)

    def _audit(
        self,
        name: str,
        args: Any,
        result: ToolResult,
        *,
        reason: str,
        duration: float,
    ) -> None:
        self.audit.record(
            tool=name,
            args=_redact_args(args),
            mode=self.mode,
            dry_run=self.ctx.dry_run,
            executed=result.denied is False and not (
                isinstance(result.output, str) and result.output.startswith("dry-run:")
            ),
            ok=result.ok,
            denied=result.denied,
            exit_code=result.exit_code,
            risk=result.risk.level if result.risk else None,
            reason=reason,
            duration_s=round(duration, 3),
        )


def _redact_args(args: Any) -> Any:
    """Mask likely secrets in audited arguments (e.g. pasted tokens)."""
    from ..safety import redact_secrets

    if isinstance(args, dict):
        return {k: (redact_secrets(str(v)) if isinstance(v, str) else v) for k, v in args.items()}
    if isinstance(args, str):
        return redact_secrets(args)
    return args
