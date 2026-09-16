"""Foreground shell execution with bounded output and POSIX group cleanup.

The router owns permissions; this module owns the lifetime of a command. Pipes
are drained together, stdin is closed, and ordinary descendants in the same
process group are terminated on timeout, interruption, or executor failure.
This is not an isolation boundary against processes that deliberately detach.
"""

from __future__ import annotations

import os
import selectors
import signal
import subprocess
import time

from ..safety import assess_command, is_read_only
from .base import ExecutionContext, EXECUTE, Tool, ToolRequest, ToolResult, truncate_output
from .output import StreamCapture

READ_CHUNK_BYTES = 16_384

DESCRIPTION = (
    "Run a foreground shell command and return its exit code and bounded output. "
    "Use for package management, git, builds, and other operations not covered "
    "by file tools. Output retains the beginning and end; very long lines are "
    "omitted. Optional timeout is limited by tools.shell.max_timeout. "
    "Stdin is closed: use non-interactive command flags. Background daemons "
    "are not supported. Unknown or mutating commands require confirmation "
    "in standard mode and are denied in safe mode."
)

PARAMETERS = {
    "type": "object",
    "properties": {
        "cmd": {
            "type": "string",
            "minLength": 1,
            "description": "The shell command to run (may include pipes, &&, ;).",
        },
        "timeout": {
            "type": "number",
            "exclusiveMinimum": 0,
            "description": "Seconds; defaults to config and cannot exceed the configured maximum.",
        },
    },
    "required": ["cmd"],
    "additionalProperties": False,
}


def _preview(request: ToolRequest, ctx: ExecutionContext) -> tuple:
    if not request.args["cmd"].strip():
        raise ValueError("empty command")
    if request.args.get("timeout", ctx.shell_timeout) > ctx.shell_max_timeout:
        raise ValueError(f"timeout exceeds tools.shell.max_timeout ({ctx.shell_max_timeout:g}s); "
                         "request a shorter timeout or ask the user to change the limit")
    return assess_command(request.args["cmd"]), f"$ {request.args['cmd']}"


def _signal_group(proc: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        pass


def _terminate_group(proc: subprocess.Popen, grace: float) -> None:
    """Give the whole group a bounded TERM grace period, then force cleanup."""
    _signal_group(proc, signal.SIGTERM)
    deadline = time.monotonic() + grace
    try:
        while time.monotonic() < deadline:
            proc.poll()  # reap the leader even if its children are still alive
            try:
                os.killpg(proc.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
    finally:
        _signal_group(proc, signal.SIGKILL)
        proc.wait()


def _drain_ready(selector: selectors.BaseSelector, wait: float) -> None:
    for key, _ in selector.select(wait):
        try:
            data = os.read(key.fd, READ_CHUNK_BYTES)
        except BlockingIOError:
            continue
        if data:
            key.data.feed(data)
        else:
            selector.unregister(key.fileobj)


def _run(request: ToolRequest, ctx: ExecutionContext) -> ToolResult:
    cmd = request.args["cmd"]
    if not cmd.strip():
        return ToolResult(ok=False, output="empty command")
    timeout = request.args.get("timeout", ctx.shell_timeout)
    if timeout > ctx.shell_max_timeout:
        return ToolResult(
            ok=False,
            output=f"timeout exceeds tools.shell.max_timeout ({ctx.shell_max_timeout:g}s); "
            "request a shorter timeout or ask the user to change the limit",
        )
    if os.name != "posix":
        return ToolResult(ok=False, output="run_shell requires a POSIX host (Termux/Linux/macOS)")

    # Reserve space for stream labels and a timeout notice so a noisy stderr
    # does not discard stdout's tail (or vice versa) in the final result.
    capture_limit = max(128, (ctx.max_output_chars - 128) // 2)
    stdout = StreamCapture(capture_limit, redact=ctx.redact_secrets)
    stderr = StreamCapture(capture_limit, redact=ctx.redact_secrets)
    started = time.monotonic()
    timed_out = False
    try:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            start_new_session=True,
            cwd=os.path.expanduser(ctx.shell_workdir) if ctx.shell_workdir else None,
        )
    except OSError as exc:
        return ToolResult(ok=False, output=f"failed to start command: {exc}", risk=request.risk)

    try:
        with selectors.DefaultSelector() as selector:
            assert proc.stdout is not None and proc.stderr is not None
            for pipe, capture in ((proc.stdout, stdout), (proc.stderr, stderr)):
                os.set_blocking(pipe.fileno(), False)
                selector.register(pipe, selectors.EVENT_READ, capture)
            deadline = started + timeout
            while selector.get_map() or proc.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    _terminate_group(proc, ctx.shell_kill_grace)
                    # Capture bytes already in the pipes, but never wait on a
                    # deliberately detached child holding a pipe open.
                    drain_deadline = time.monotonic() + 0.1
                    while selector.get_map() and time.monotonic() < drain_deadline:
                        if not selector.select(0):
                            break
                        _drain_ready(selector, 0)
                    break
                _drain_ready(selector, min(remaining, 0.1))
            proc.wait()
    except BaseException:
        # KeyboardInterrupt and cancellation must not leave the workload alive.
        _terminate_group(proc, ctx.shell_kill_grace)
        raise
    finally:
        # No unmanaged background jobs, including descendants whose shell
        # leader exited successfully after redirecting/closing its pipes.
        _signal_group(proc, signal.SIGKILL)
        proc.wait()
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()

    stdout.finish()
    stderr.finish()
    parts: list[str] = []
    if timed_out:
        parts.append(f"command timed out after {timeout:g}s (process group terminated)")
    if stdout.text().strip():
        parts.append(stdout.text().rstrip("\n"))
    if stderr.text().strip():
        parts.append(f"[stderr]\n{stderr.text().rstrip(chr(10))}")
    output = "\n".join(parts) if parts else "(no output)"
    truncated = stdout.truncated or stderr.truncated or len(output) > ctx.max_output_chars
    return ToolResult(
        ok=not timed_out and proc.returncode == 0,
        output=truncate_output(output, ctx.max_output_chars),
        exit_code=None if timed_out else proc.returncode,
        risk=request.risk,
        duration=time.monotonic() - started,
        timed_out=timed_out,
        truncated=truncated,
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
