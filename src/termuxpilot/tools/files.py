"""File operations: read / write / edit / move / diff — with diff previews.

Every mutating operation produces a unified-diff preview shown (and confirmed)
by the router before anything hits disk.  Writes into protected system paths
are escalated to "high" risk by the router via ``ExecutionContext.guard_path``.
"""

from __future__ import annotations

import difflib
import os
import shutil
import stat
import tempfile
from contextlib import contextmanager
from itertools import islice
from pathlib import Path

from ..safety import LineRedactor, RiskAssessment, redact_secrets
from .base import (
    ExecutionContext,
    READ,
    Tool,
    ToolRequest,
    ToolResult,
    WRITE,
    ensure_str,
    expand,
    truncate_output,
)
from .output import BoundedText

MAX_READ_BYTES = 1_000_000
MAX_LINE_BYTES = 65_536
MAX_READ_LINES = 10_000
DEFAULT_WINDOW_LINES = 200
PREVIEW_MAX_LINES = 200


@contextmanager
def _open_regular(path: Path):
    # O_NONBLOCK prevents a FIFO from hanging before we can inspect fstat.
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"not a regular file: {path}")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            yield stream
    finally:
        os.close(fd)


def _read_text(path: Path) -> str:
    """Bound previews, edits and diffs too, not just read_file results."""
    with _open_regular(path) as stream:
        data = stream.read(MAX_READ_BYTES + 1)
    if len(data) > MAX_READ_BYTES:
        raise ValueError(f"file too large for an in-memory edit/diff: {path}; use a streaming shell tool")
    return data.decode("utf-8", errors="replace")


def _snapshot(path: Path) -> tuple:
    resolved = path.resolve()
    try:
        info = resolved.stat()
    except FileNotFoundError:
        return (str(resolved), None)
    return (str(resolved), info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_mode)


def _atomic_write(path: Path, content: str, expected: tuple) -> None:
    """Same-directory replace; detected changes since approval fail closed.

    This protects against torn writes, not arbitrary concurrent writers: a
    filesystem does not provide compare-and-swap for the final rename.
    """
    if _snapshot(path) != expected:
        raise ValueError("file changed since preview; read it again before writing")
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.tp-", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            if expected[1] is not None:
                os.fchmod(stream.fileno(), stat.S_IMODE(expected[-1]))
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if _snapshot(path) != expected:
            raise ValueError("file changed since preview; read it again before writing")
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _diff_label(prefix: str, path: str) -> str:
    return f"{prefix}{path}" if path.startswith("/") else f"{prefix}/{path}"


def _unified_diff(old: str, new: str, label_old: str, label_new: str, *, redact: bool = False) -> str:
    if redact:
        changed = old != new
        old, new = redact_secrets(old), redact_secrets(new)
        if changed and old == new:
            return "(redacted content changes; sensitive values hidden)"
    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=label_old,
        tofile=label_new,
    )
    lines = list(islice(diff, PREVIEW_MAX_LINES + 1))
    if len(lines) > PREVIEW_MAX_LINES:
        lines = lines[:PREVIEW_MAX_LINES] + ["\n... [diff truncated; more lines omitted] ...\n"]
    return "".join(lines) or "(no changes)"


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


def _preview_read(request: ToolRequest, ctx: ExecutionContext) -> tuple[RiskAssessment, str]:
    path = ensure_str(request.args.get("path"))
    return RiskAssessment(), f"read {path}"


def _run_read(request: ToolRequest, ctx: ExecutionContext) -> ToolResult:
    path = expand(request.args.get("path"))
    if not path.exists():
        return ToolResult(ok=False, output=f"no such file: {path}")
    start = request.args.get("start_line", 1)
    end = request.args.get("end_line")
    windowed = "start_line" in request.args or "end_line" in request.args
    if end is not None and end < start:
        return ToolResult(ok=False, output="end_line must be >= start_line")
    if end is not None and end - start + 1 > MAX_READ_LINES:
        return ToolResult(ok=False, output=f"request at most {MAX_READ_LINES:,} lines per window")
    if end is None:
        end = start + (DEFAULT_WINDOW_LINES if windowed else MAX_READ_LINES) - 1

    output = BoundedText(ctx.max_output_chars)
    redactor = LineRedactor() if ctx.redact_secrets else None
    line_no = last = selected_bytes = 0
    limited = False
    with _open_regular(path) as stream:
        size = os.fstat(stream.fileno()).st_size
        if size > MAX_READ_BYTES and not windowed:
            return ToolResult(ok=False, output=(
                f"file too large to read in one go ({size:,} bytes); "
                "request a start_line/end_line window"
            ))
        while line_no < end:
            raw = stream.readline(MAX_LINE_BYTES + 1)
            if not raw:
                break
            line_no += 1
            if len(raw) > MAX_LINE_BYTES:
                return ToolResult(ok=False, output=(
                    f"line {line_no} exceeds {MAX_LINE_BYTES:,} bytes; "
                    "use a bounded byte-processing shell command instead"
                ))
            text = raw.decode("utf-8", errors="replace")
            # Observe skipped lines too, so a requested window inside a PEM
            # private-key block does not disclose its body.
            if redactor:
                text = redactor.redact(text)
            if line_no < start:
                continue
            if selected_bytes + len(raw) > MAX_READ_BYTES:
                limited = True
                break
            selected_bytes += len(raw)
            last = line_no
            if text:
                output.append(f"{line_no:>6}\t{text.rstrip(chr(10)).rstrip(chr(13))}\n")
        more = stream.tell() < size or limited

    if not last:
        if line_no == 0:
            return ToolResult(ok=True, output=f"[{path} — empty file]")
        return ToolResult(ok=False, output=f"start_line {start} is beyond end of file ({line_no} lines)")
    limited = limited or (not windowed and more)
    total = f" of {last}" if not more else ""
    note = f"[{path} — lines {start}-{last}{total}]\n"
    if limited:
        note += f"[read limit reached; continue with start_line={last + 1}]\n"
    return ToolResult(ok=True, output=note + output.text(), truncated=limited or output.truncated)


def build_read_tool() -> Tool:
    return Tool(
        name="read_file",
        description=(
            "Read a text file. Returns numbered lines. Optional 1-based "
            "start_line/end_line window works even for large files. start_line "
            "alone reads up to 200 lines. Each read is bounded to 1 MB / 10,000 "
            "lines; lines over 64 KiB and non-regular files are rejected."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "start_line": {"type": "integer", "minimum": 1},
                "end_line": {"type": "integer", "minimum": 1},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        category=READ,
        handler=_run_read,
        preview=_preview_read,
    )


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------


def _write_preview(request: ToolRequest, ctx: ExecutionContext) -> tuple[RiskAssessment, str]:
    path = expand(request.args.get("path"))
    content = ensure_str(request.args.get("content"))
    risk = RiskAssessment()
    protected = ctx.guard_path(str(path))
    if protected:
        risk.escalate("high", f"target is inside protected path {protected}")
    if len(content.encode("utf-8")) > MAX_READ_BYTES:
        raise ValueError("content exceeds the 1 MB text-write limit")
    request.state["snapshot"] = _snapshot(path)
    if path.exists():
        diff = _unified_diff(_read_text(path), content, _diff_label("a", str(path)),
                             _diff_label("b", str(path)), redact=ctx.redact_secrets)
    else:
        diff = _unified_diff("", content, "/dev/null", _diff_label("b", str(path)),
                             redact=ctx.redact_secrets)
    return risk, f"write {path}\n{diff}"


def _run_write(request: ToolRequest, ctx: ExecutionContext) -> ToolResult:
    path = expand(request.args.get("path"))
    content = ensure_str(request.args.get("content"))
    _atomic_write(path, content, request.state["snapshot"])
    return ToolResult(ok=True, output=f"wrote {len(content):,} chars to {path}")


def build_write_tool() -> Tool:
    return Tool(
        name="write_file",
        description=(
            "Create or overwrite a file with the given full content. A unified "
            "diff preview is shown to the user (and confirmed unless the "
            "session is in yolo mode). For small changes prefer edit_file."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "content": {"type": "string", "maxLength": MAX_READ_BYTES, "description": "Full new file content (up to 1 MB UTF-8)."},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        category=WRITE,
        handler=_run_write,
        preview=_write_preview,
    )


# ---------------------------------------------------------------------------
# edit_file
# ---------------------------------------------------------------------------


def _preview_edit(request: ToolRequest, ctx: ExecutionContext) -> tuple[RiskAssessment, str]:
    path = expand(request.args.get("path"))
    old_text = ensure_str(request.args.get("old_text"))
    new_text = ensure_str(request.args.get("new_text"))
    risk = RiskAssessment()
    protected = ctx.guard_path(str(path))
    if protected:
        risk.escalate("high", f"target is inside protected path {protected}")
    if not path.exists():
        return risk, f"edit {path} (file does not exist)"
    request.state["snapshot"] = _snapshot(path)
    current = _read_text(path)
    updated = current.replace(old_text, new_text, 1)
    if len(updated.encode("utf-8")) > MAX_READ_BYTES:
        raise ValueError("edited content exceeds the 1 MB text-write limit")
    diff = _unified_diff(current, updated, _diff_label("a", str(path)),
                         _diff_label("b", str(path)), redact=ctx.redact_secrets)
    return risk, f"edit {path}\n{diff}"


def _run_edit(request: ToolRequest, ctx: ExecutionContext) -> ToolResult:
    path = expand(request.args.get("path"))
    old_text = ensure_str(request.args.get("old_text"))
    new_text = ensure_str(request.args.get("new_text"))
    if not path.exists():
        return ToolResult(ok=False, output=f"no such file: {path}")
    if not old_text:
        return ToolResult(ok=False, output="old_text must be non-empty")
    current = _read_text(path)
    count = current.count(old_text)
    if count == 0:
        return ToolResult(
            ok=False,
            output="old_text not found in the file — re-read the file and copy the "
            "exact text (whitespace is significant)",
        )
    if count > 1:
        return ToolResult(
            ok=False,
            output=f"old_text matches {count} places in the file — include more "
            "surrounding context to make it unique",
        )
    _atomic_write(path, current.replace(old_text, new_text, 1), request.state["snapshot"])
    return ToolResult(ok=True, output=f"edited {path} (1 replacement)")


def build_edit_tool() -> Tool:
    return Tool(
        name="edit_file",
        description=(
            "Replace the first exact occurrence of old_text with new_text in a "
            "file. old_text must match exactly (whitespace-sensitive) and be "
            "unique in the file. A diff preview is shown to the user."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "old_text": {"type": "string", "minLength": 1, "description": "Exact text to find (must be unique)."},
                "new_text": {"type": "string", "description": "Replacement text."},
            },
            "required": ["path", "old_text", "new_text"],
            "additionalProperties": False,
        },
        category=WRITE,
        handler=_run_edit,
        preview=_preview_edit,
    )


# ---------------------------------------------------------------------------
# move_file
# ---------------------------------------------------------------------------


def _preview_move(request: ToolRequest, ctx: ExecutionContext) -> tuple[RiskAssessment, str]:
    src = expand(request.args.get("src"))
    dst = expand(request.args.get("dst"))
    risk = RiskAssessment()
    for p in (src, dst):
        protected = ctx.guard_path(str(p))
        if protected:
            risk.escalate("high", f"path {p} is inside protected {protected}")
    note = " (dest exists — will be overwritten)" if dst.exists() else ""
    return risk, f"move {src} -> {dst}{note}"


def _run_move(request: ToolRequest, ctx: ExecutionContext) -> ToolResult:
    src = expand(request.args.get("src"))
    dst = expand(request.args.get("dst"))
    if not src.exists():
        return ToolResult(ok=False, output=f"no such source: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    overwrite = bool(request.args.get("overwrite"))
    if dst.exists() and not overwrite:
        return ToolResult(
            ok=False,
            output=f"destination exists: {dst} — pass overwrite=true to replace it",
        )
    shutil.move(str(src), str(dst))
    return ToolResult(ok=True, output=f"moved {src} -> {dst}")


def build_move_tool() -> Tool:
    return Tool(
        name="move_file",
        description=(
            "Move or rename a file or directory. Refuses to overwrite an "
            "existing destination unless overwrite=true. A confirmation is "
            "shown to the user."
        ),
        parameters={
            "type": "object",
            "properties": {
                "src": {"type": "string", "minLength": 1},
                "dst": {"type": "string", "minLength": 1},
                "overwrite": {"type": "boolean"},
            },
            "required": ["src", "dst"],
            "additionalProperties": False,
        },
        category=WRITE,
        handler=_run_move,
        preview=_preview_move,
    )


# ---------------------------------------------------------------------------
# diff_files
# ---------------------------------------------------------------------------


def _preview_diff(request: ToolRequest, ctx: ExecutionContext) -> tuple[RiskAssessment, str]:
    a = ensure_str(request.args.get("a"))
    b = ensure_str(request.args.get("b"))
    return RiskAssessment(), f"diff {a} vs {b}"


def _run_diff(request: ToolRequest, ctx: ExecutionContext) -> ToolResult:
    a = expand(request.args.get("a"))
    b = expand(request.args.get("b"))
    if not a.exists():
        return ToolResult(ok=False, output=f"no such file: {a}")
    if not b.exists():
        return ToolResult(ok=False, output=f"no such file: {b}")
    text_a = _read_text(a)
    text_b = _read_text(b)
    if text_a == text_b:
        return ToolResult(ok=True, output="files are identical")
    diff = _unified_diff(text_a, text_b, _diff_label("a", str(a)),
                         _diff_label("b", str(b)), redact=ctx.redact_secrets)
    return ToolResult(ok=True, output=truncate_output(diff))


def build_diff_tool() -> Tool:
    return Tool(
        name="diff_files",
        description="Show the unified diff between two files (read-only).",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "string", "minLength": 1},
                "b": {"type": "string", "minLength": 1},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        },
        category=READ,
        handler=_run_diff,
        preview=_preview_diff,
    )
