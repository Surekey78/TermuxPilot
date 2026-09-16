"""File operations: read / write / edit / move / diff — with diff previews.

Every mutating operation produces a unified-diff preview shown (and confirmed)
by the router before anything hits disk.  Writes into protected system paths
are escalated to "high" risk by the router via ``ExecutionContext.guard_path``.
"""

from __future__ import annotations

import difflib
import shutil
from pathlib import Path

from ..safety import RiskAssessment
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

MAX_READ_BYTES = 1_000_000
PREVIEW_MAX_LINES = 200


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _diff_label(prefix: str, path: str) -> str:
    return f"{prefix}{path}" if path.startswith("/") else f"{prefix}/{path}"


def _unified_diff(old: str, new: str, label_old: str, label_new: str) -> str:
    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=label_old,
        tofile=label_new,
    )
    lines = list(diff)
    if len(lines) > PREVIEW_MAX_LINES:
        lines = lines[:PREVIEW_MAX_LINES] + [
            f"... [diff truncated, {len(lines) - PREVIEW_MAX_LINES} more lines] ..."
        ]
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
    if path.stat().st_size > MAX_READ_BYTES:
        return ToolResult(
            ok=False,
            output=f"file too large to read in one go ({path.stat().st_size:,} bytes); "
            "use start_line/end_line or split it up",
        )
    text = _read_text(path)
    start = request.args.get("start_line")
    end = request.args.get("end_line")
    lines = text.splitlines()
    total = len(lines)
    lo = int(start) - 1 if start else 0
    hi = int(end) if end else total
    selected = lines[lo:hi]
    numbered = "\n".join(f"{i + lo + 1:>6}\t{line}" for i, line in enumerate(selected))
    note = f"[{path} — lines {lo + 1}-{min(hi, total)} of {total}]\n"
    return ToolResult(ok=True, output=truncate_output(note + numbered))


def build_read_tool() -> Tool:
    return Tool(
        name="read_file",
        description=(
            "Read a text file. Returns numbered lines. Optional 1-based "
            "start_line/end_line window for large files. Binary files are read "
            "as replacement text — prefer run_shell (hexdump) for binaries."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer", "minimum": 1},
                "end_line": {"type": "integer", "minimum": 1},
            },
            "required": ["path"],
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
    if path.exists():
        diff = _unified_diff(_read_text(path), content, _diff_label("a", str(path)), _diff_label("b", str(path)))
    else:
        diff = _unified_diff("", content, "/dev/null", _diff_label("b", str(path)))
    return risk, f"write {path}\n{diff}"


def _run_write(request: ToolRequest, ctx: ExecutionContext) -> ToolResult:
    path = expand(request.args.get("path"))
    content = ensure_str(request.args.get("content"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
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
                "path": {"type": "string"},
                "content": {"type": "string", "description": "Full new file content."},
            },
            "required": ["path", "content"],
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
    current = _read_text(path)
    updated = current.replace(old_text, new_text, 1)
    diff = _unified_diff(current, updated, _diff_label("a", str(path)), _diff_label("b", str(path)))
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
    path.write_text(current.replace(old_text, new_text, 1), encoding="utf-8")
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
                "path": {"type": "string"},
                "old_text": {"type": "string", "description": "Exact text to find (must be unique)."},
                "new_text": {"type": "string", "description": "Replacement text."},
            },
            "required": ["path", "old_text", "new_text"],
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
                "src": {"type": "string"},
                "dst": {"type": "string"},
                "overwrite": {"type": "boolean"},
            },
            "required": ["src", "dst"],
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
    diff = _unified_diff(text_a, text_b, _diff_label("a", str(a)), _diff_label("b", str(b)))
    return ToolResult(ok=True, output=truncate_output(diff))


def build_diff_tool() -> Tool:
    return Tool(
        name="diff_files",
        description="Show the unified diff between two files (read-only).",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"},
            },
            "required": ["a", "b"],
        },
        category=READ,
        handler=_run_diff,
        preview=_preview_diff,
    )
