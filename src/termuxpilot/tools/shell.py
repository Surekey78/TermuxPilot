"""Shell executor tool — the agent's hands.

Sandboxing strategy for v0.2 (Termux-friendly, no mandatory chroot):
* the permission mode + risk classifier + allow/block list gate every command
  in the *router*, not here;
* output is secret-redacted and size-truncated before it reaches the model;
* a hard timeout kills runaway commands.
"""

from __future__ import annotations

import subprocess
import time

from ..safety import assess_command, is_read_only, redact_secrets
from .base import (
    ExecutionContext,
    EXECUTE,
    Tool,
    ToolRequest,
    ToolResult,
    ensure_str,
    truncate_output,
)

DESCRIPTION = (
    "Run a shell command on the device (Termux home directory by default) and "
    "return its exit code and output. Use for any system operation that the "
    "file tools don't cover: package management (`pkg`), Termux:API tools "
    "(`termux-battery-status`, ...), process control, git, etc. "
    "Output is redacted for secrets and truncated if very long. "
    "Destructive commands may be blocked or require the user's confirmation "
    "depending on the session's permission mode."
)

PARAMETERS = {
    "type": "object",
    "properties": {
        "cmd": {
            "type": "string",
            "description": "The shell command to run (may include pipes, &&, ;).",
        },
        "timeout": {
            "type": "number",
            "description": "Optional timeout in seconds (default from config).",
        },
    },
    "required": ["cmd"],
}


def _preview(request: ToolRequest, ctx: ExecutionContext) -> tuple:
    cmd = ensure_str(request.args.get("cmd"))
    risk = request.risk or assess_command(cmd)
    return risk, f"$ {cmd}"


def _run(request: ToolRequest, ctx: ExecutionContext) -> ToolResult:
    cmd = ensure_str(request.args.get("cmd"))
    if not cmd.strip():
        return ToolResult(ok=False, output="empty command")
    timeout = request.args.get("timeout")
    timeout = float(timeout) if timeout else ctx.shell_timeout

    started = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=ctx.shell_workdir or None,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            ok=False,
            output=f"command timed out after {timeout:g}s: {cmd}",
            exit_code=None,
            risk=request.risk,
            duration=time.monotonic() - started,
        )
    except OSError as exc:
        return ToolResult(ok=False, output=f"failed to start command: {exc}", risk=request.risk)

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if ctx.redact_secrets:
        stdout = redact_secrets(stdout)
        stderr = redact_secrets(stderr)
    parts = []
    if stdout.strip():
        parts.append(stdout.rstrip("\n"))
    if stderr.strip():
        parts.append(f"[stderr]\n{stderr.rstrip(chr(10))}")
    output = "\n".join(parts) if parts else "(no output)"
    return ToolResult(
        ok=proc.returncode == 0,
        output=truncate_output(output),
        exit_code=proc.returncode,
        risk=request.risk,
        duration=time.monotonic() - started,
    )


def build_shell_tool() -> Tool:
    return Tool(
        name="run_shell",
        description=DESCRIPTION,
        parameters=PARAMETERS,
        category=EXECUTE,
        handler=_run,
        preview=_preview,
    )


def shell_preview_text(cmd: str) -> str:
    return f"$ {cmd}"


def classify_for_mode(cmd: str):
    """Convenience for the router: (risk, read_only)."""
    return assess_command(cmd), is_read_only(cmd)
