from __future__ import annotations

import pytest

from termuxpilot.audit import AuditLog
from termuxpilot.tools import build_default_tools
from termuxpilot.tools.base import ExecutionContext
from termuxpilot.tools.router import ToolRouter


@pytest.fixture
def ctx(tmp_path):
    return ExecutionContext(
        shell_timeout=10.0,
        shell_workdir=str(tmp_path),
        redact_secrets=True,
        protected_paths=("/etc", "/dev"),
        dry_run=False,
    )


@pytest.fixture
def audit_path(tmp_path, monkeypatch):
    path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("TERMUXPILOT_AUDIT", str(path))
    return path


def make_router(ctx, *, mode="standard", blocklist=None, allowlist=None,
                confirm=None, dry_run=False, audit_path=None):
    if dry_run:
        ctx.dry_run = True
    return ToolRouter(
        build_default_tools(),
        ctx,
        mode=mode,
        blocklist=blocklist or [],
        allowlist=allowlist or [],
        audit=AuditLog(),
        confirm=confirm,
    )


# ---------------------------------------------------------------- shell tool


def test_shell_readonly_auto_in_standard(ctx, audit_path):
    router = make_router(ctx, mode="standard")
    result = router.execute("run_shell", {"cmd": "echo hello-from-shell"})
    assert result.ok
    assert "hello-from-shell" in result.output
    assert result.exit_code == 0


def test_shell_write_requires_confirmation(ctx, audit_path):
    calls = []
    router = make_router(ctx, mode="standard",
                         confirm=lambda n, p, r: (calls.append(n), True)[1])
    tmp = ctx.shell_workdir
    result = router.execute("run_shell", {"cmd": "echo data > out.txt"})
    assert result.ok
    assert calls == ["run_shell"]
    assert open(f"{tmp}/out.txt").read() == "data\n"


def test_shell_declined_confirmation_not_executed(ctx, audit_path, tmp_path):
    router = make_router(ctx, mode="standard", confirm=lambda n, p, r: False)
    target = tmp_path / "should_not_exist.txt"
    result = router.execute("run_shell", {"cmd": f"echo x > {target}"})
    assert result.denied
    assert not target.exists()
    assert "confirmation" in result.output


def test_shell_safe_mode_denies_writes(ctx, audit_path, tmp_path):
    router = make_router(ctx, mode="safe")
    ok = router.execute("run_shell", {"cmd": "echo hello"})
    assert ok.ok  # read-only command allowed in safe mode
    target = tmp_path / "nope.txt"
    denied = router.execute("run_shell", {"cmd": f"echo x > {target}"})
    assert denied.denied
    assert "safe" in denied.output
    assert not target.exists()


def test_shell_yolo_skips_confirmation(ctx, audit_path, tmp_path):
    confirmed = []
    router = make_router(ctx, mode="yolo",
                         confirm=lambda n, p, r: confirmed.append(1) or False)
    target = tmp_path / "yolo.txt"
    result = router.execute("run_shell", {"cmd": f"echo yolo > {target}"})
    assert result.ok
    assert confirmed == []  # never asked
    assert target.read_text() == "yolo\n"


def test_blocklist_beats_yolo(ctx, audit_path, tmp_path):
    router = make_router(ctx, mode="yolo", blocklist=[r"rm\s+-rf"])
    result = router.execute("run_shell", {"cmd": "rm -rf /tmp/whatever"})
    assert result.denied
    assert "blocklist" in result.output


def test_allowlist_restrains_yolo(ctx, audit_path):
    router = make_router(ctx, mode="yolo", allowlist=[r"^echo\b"])
    ok = router.execute("run_shell", {"cmd": "echo fine"})
    assert ok.ok
    denied = router.execute("run_shell", {"cmd": "ls -la"})
    assert denied.denied
    assert "allowlist" in denied.output


def test_dry_run_executes_nothing(ctx, audit_path, tmp_path):
    target = tmp_path / "dry.txt"
    router = make_router(ctx, mode="yolo", dry_run=True)
    result = router.execute("run_shell", {"cmd": f"echo x > {target}"})
    assert result.ok
    assert result.output.startswith("dry-run:")
    assert not target.exists()


def test_shell_timeout(ctx, audit_path):
    router = make_router(ctx, mode="yolo")
    result = router.execute("run_shell", {"cmd": "sleep 5", "timeout": 0.3})
    assert not result.ok
    assert "timed out" in result.output


def test_shell_output_redacted(ctx, audit_path):
    router = make_router(ctx, mode="yolo")
    result = router.execute(
        "run_shell",
        {"cmd": "echo 'token sk-abcdefghijklmnop1234567890ABC done'"},
    )
    assert result.ok
    assert "sk-abcdefghijklmnop1234567890ABC" not in result.output
    assert "[REDACTED:openai-key]" in result.output


def test_shell_stderr_captured(ctx, audit_path):
    router = make_router(ctx, mode="yolo")
    result = router.execute("run_shell", {"cmd": "echo oops 1>&2; exit 3"})
    assert not result.ok
    assert result.exit_code == 3
    assert "oops" in result.output
    assert "[stderr]" in result.output


# -------------------------------------------------------------- file tools


def test_read_write_edit_cycle(ctx, tmp_path, audit_path):
    confirmed = []
    router = make_router(ctx, mode="standard",
                         confirm=lambda n, p, r: (confirmed.append(n), True)[1])
    target = tmp_path / "note.txt"

    written = router.execute("write_file", {
        "path": str(target), "content": "line1\nline2\nline3\n",
    })
    assert written.ok
    assert "write_file" in confirmed

    read = router.execute("read_file", {"path": str(target)})
    assert read.ok
    assert "line1" in read.output and "line3" in read.output

    edited = router.execute("edit_file", {
        "path": str(target), "old_text": "line2", "new_text": "LINE2",
    })
    assert edited.ok
    assert "edit_file" in confirmed
    assert target.read_text() == "line1\nLINE2\nline3\n"


def test_edit_rejects_ambiguous_match(ctx, tmp_path, audit_path):
    router = make_router(ctx, mode="yolo")
    target = tmp_path / "dup.txt"
    target.write_text("aaa\nbbb\naaa\n")
    result = router.execute("edit_file", {
        "path": str(target), "old_text": "aaa", "new_text": "ccc",
    })
    assert not result.ok
    assert "2 places" in result.output
    assert target.read_text() == "aaa\nbbb\naaa\n"  # untouched


def test_edit_requires_exact_text(ctx, tmp_path, audit_path):
    router = make_router(ctx, mode="yolo")
    target = tmp_path / "x.txt"
    target.write_text("hello world\n")
    result = router.execute("edit_file", {
        "path": str(target), "old_text": "HELLO", "new_text": "y",
    })
    assert not result.ok
    assert "not found" in result.output


def test_write_preview_shows_diff(ctx, tmp_path, audit_path):
    seen_previews = []
    target = tmp_path / "d.txt"
    target.write_text("old content\n")
    router = make_router(ctx, mode="standard", confirm=lambda n, p, r: True)
    router.request_hook = lambda name, args, risk, preview: seen_previews.append(preview)
    router.execute("write_file", {"path": str(target), "content": "new content\n"})
    assert any("old content" in p and "new content" in p for p in seen_previews)


def test_move_and_diff(ctx, tmp_path, audit_path):
    router = make_router(ctx, mode="yolo")
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("alpha\n")
    b.write_text("beta\n")

    moved = router.execute("move_file", {"src": str(a), "dst": str(tmp_path / "c.txt")})
    assert moved.ok
    assert not a.exists()
    assert (tmp_path / "c.txt").read_text() == "alpha\n"

    refused = router.execute("move_file", {"src": str(b), "dst": str(tmp_path / "c.txt")})
    assert not refused.ok
    assert "overwrite" in refused.output

    diff = router.execute("diff_files", {"a": str(b), "b": str(tmp_path / "c.txt")})
    assert diff.ok
    assert "+alpha" in diff.output and "-beta" in diff.output


def test_write_protected_path_flagged_high(ctx, audit_path, tmp_path):
    # The guard must flag the target as high risk BEFORE execution, regardless
    # of whether the OS lets the (non-root) process actually write to /etc.
    risks = []
    router = make_router(ctx, mode="standard", confirm=lambda n, p, r: True)
    router.request_hook = lambda name, args, risk, preview: risks.append(risk)
    result = router.execute("write_file", {"path": "/etc/motd_test", "content": "x"})
    assert risks and risks[-1].level == "high"
    assert "protected" in " ".join(risks[-1].reasons)
    if result.ok is False and not result.denied:
        # executed but blocked by the OS (non-root sandbox): acceptable
        assert "PermissionError" in result.output or "crashed" in result.output


def test_safe_mode_denies_file_writes(ctx, tmp_path, audit_path):
    router = make_router(ctx, mode="safe")
    target = tmp_path / "nope.md"
    result = router.execute("write_file", {"path": str(target), "content": "x"})
    assert result.denied
    assert "safe" in result.output
    assert not target.exists()
    read = router.execute("read_file", {"path": str(tmp_path / "nonexistent.md")})
    assert read.ok is False and "no such file" in read.output


# ------------------------------------------------------------------ audit


def test_audit_log_records_risk_level(ctx, audit_path):
    router = make_router(ctx, mode="yolo")
    router.execute("run_shell", {"cmd": "rm -rf ./build"})
    entries = AuditLog(audit_path).tail(5)
    assert entries, "audit log empty"
    last = entries[-1]
    assert last["tool"] == "run_shell"
    assert last["risk"] in ("high", "critical")


def test_audit_log_records_decisions(ctx, audit_path):
    router = make_router(ctx, mode="yolo", blocklist=[r"rm\s+-rf"])
    router.execute("run_shell", {"cmd": "echo audited"})
    router.execute("run_shell", {"cmd": "rm -rf /"})
    entries = AuditLog().tail(10)
    assert len(entries) == 2
    assert entries[0]["tool"] == "run_shell"
    assert entries[0]["ok"] is True
    assert entries[1]["denied"] is True
    assert entries[1]["reason"] == "blocklist"


def test_unknown_tool_denied(ctx, audit_path):
    router = make_router(ctx, mode="yolo")
    result = router.execute("hack_the_planet", {"cmd": "x"})
    assert result.denied
    assert "unknown tool" in result.output


def test_bad_arguments_denied(ctx, audit_path):
    router = make_router(ctx, mode="yolo")
    result = router.execute("run_shell", "{not json")
    assert result.denied
    assert "JSON" in result.output
