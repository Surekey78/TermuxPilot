from __future__ import annotations

import re
import shlex
import sys

import pytest

from termuxpilot.audit import AuditLog
from termuxpilot.safety import is_protected_path, is_read_only, redact_data, redact_secrets
from termuxpilot.tools import build_default_tools
from termuxpilot.tools.base import ExecutionContext, READ, Tool, ToolResult
from termuxpilot.tools.router import ToolRouter

FAKE_KEY = "sk-abcdefghijklmnop1234567890ABC"


@pytest.fixture
def router(tmp_path):
    return ToolRouter(
        build_default_tools(), ExecutionContext(), mode="safe",
        audit=AuditLog(tmp_path / "audit.jsonl"),
    )


@pytest.mark.parametrize("command", [
    "python -c 'print(1)'", "sh -c ls", "node -e '1'", "unknown-tool",
    "git status", "git diff --output=out", "find . -delete",
    "awk 'BEGIN {system(\"id\")}'", "sed -n '1w out' input", "sort -o out input",
    "echo $(python -c pass)", "echo `id`", "cat <(id)", "echo > out",
    "ls; python -c pass", "ls\npython -c pass", "ls &", "ls *", "X=1 ls",
    "printf -v PATH %s /tmp && ls", "./ls", "/tmp/ls", "ls |", "| ls", "ls &&", "ls &&& cat x", "echo 'unterminated",
])
def test_unknown_or_complex_commands_are_not_automatic(command):
    assert not is_read_only(command)


@pytest.mark.parametrize("mode", ["safe", "standard"])
def test_interpreter_cannot_write_without_permission(router, tmp_path, mode):
    target = tmp_path / "must-not-exist"
    code = f"from pathlib import Path; Path({str(target)!r}).write_text('unauthorized')"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"
    router.mode = mode
    router._allow_rules = [(".*", re.compile(".*"))]
    result = router.execute("run_shell", {"cmd": command})
    assert result.denied
    assert not target.exists()
    assert router.audit.tail(1)[0]["executed"] is False


def test_file_read_diff_preview_and_audit_are_redacted(router, tmp_path):
    path, other = tmp_path / "a", tmp_path / "b"
    path.write_text(f"api_key={FAKE_KEY}\n")
    other.write_text("ordinary\n")
    for name, args in (("read_file", {"path": str(path)}),
                       ("diff_files", {"a": str(path), "b": str(other)})):
        result = router.execute(name, args)
        assert result.ok
        assert FAKE_KEY not in result.output
    seen = []
    router.mode = "standard"
    router.request_hook = lambda *args: seen.append(args)
    router.confirm = lambda *args: seen.append(args) or True
    result = router.execute("write_file", {"path": str(path), "content": FAKE_KEY})
    assert result.ok
    assert path.read_text() == FAKE_KEY  # redaction must NOT corrupt the write
    assert FAKE_KEY not in repr(seen)
    assert FAKE_KEY not in router.audit.path.read_text()


def test_errors_and_unknown_tool_names_are_sanitized(router):
    assert FAKE_KEY not in router.execute(FAKE_KEY, {}).output

    def crash(*args):
        raise ValueError(FAKE_KEY)

    tool = Tool(name="crash", description="test", parameters={"type": "object"}, category=READ, handler=crash)
    router.tools[tool.name] = tool
    result = router.execute("crash", {})
    assert not result.ok
    assert FAKE_KEY not in result.output
    tool.preview = crash
    result = router.execute("crash", {})
    assert result.denied
    assert FAKE_KEY not in result.output
    assert router.audit.tail(1)[0]["executed"] is False


def test_recursive_redaction_and_explicit_opt_out(router):
    original = {"nested": [{"password": "short", "value": FAKE_KEY}], "normal": 3}
    result = redact_data(original)
    assert FAKE_KEY not in repr(result) and "short" not in repr(result)
    assert original["nested"][0]["password"] == "short"
    assert result["normal"] == 3
    assert FAKE_KEY not in repr(redact_data({FAKE_KEY: "value"}))
    router.ctx.redact_secrets = False
    tool = Tool(
        name="inspect", description="test", parameters={"type": "object"}, category=READ,
        handler=lambda request, ctx: ToolResult(ok=True, output=FAKE_KEY),
    )
    router.tools[tool.name] = tool
    assert router.execute("inspect", original).output == FAKE_KEY
    assert FAKE_KEY not in router.audit.path.read_text()  # audit is always redacted
    assert "short" not in router.audit.path.read_text()


def test_unterminated_private_key_block_is_redacted():
    output = redact_secrets("prefix\n-----BEGIN PRIVATE KEY-----\nPRIVATE_BODY")
    assert "prefix" in output
    assert "PRIVATE_BODY" not in output


@pytest.mark.parametrize("name,args", [
    ("run_shell", {}), ("run_shell", {"cmd": 42}),
    ("write_file", {"path": "missing-parent/file"}),
    ("write_file", {"path": "", "content": "x"}),
    ("move_file", {"src": "a", "dst": "b", "overwrite": "false"}),
])
def test_invalid_arguments_never_reach_preview(router, name, args):
    router.tools[name].preview = lambda *args: pytest.fail("preview should not run")
    result = router.execute(name, args)
    assert result.denied
    assert router.audit.tail(1)[0]["reason"] == "bad arguments"


@pytest.mark.parametrize("mode", ["safe", "standard", "yolo"])
def test_dry_run_needs_no_confirmation_but_still_obeys_blocklist(router, tmp_path, mode):
    router.mode = mode
    router.ctx.dry_run = True
    router.confirm = lambda *args: pytest.fail("dry-run must not request execution approval")
    target = tmp_path / "would-write"
    result = router.execute("write_file", {"path": str(target), "content": "preview only"})
    assert result.ok and result.output.startswith("dry-run:")
    assert not target.exists()
    entry = router.audit.tail(1)[0]
    assert entry["executed"] is False and entry["reason"] == "dry-run"
    blocked = ToolRouter(
        build_default_tools(), router.ctx, mode=mode, blocklist=["touch"], audit=router.audit,
    ).execute("run_shell", {"cmd": f"touch {shlex.quote(str(target))}"})
    assert blocked.denied
    assert not target.exists()


def test_protected_paths_resolve_dotdot_relative_paths_and_symlinks(tmp_path, monkeypatch):
    protected = tmp_path / "protected"
    protected.mkdir()
    (protected / "sub").mkdir()
    link = tmp_path / "link"
    link.symlink_to(protected, target_is_directory=True)
    roots = (str(protected),)
    monkeypatch.chdir(tmp_path)
    assert is_protected_path("link/key", roots) == str(protected)
    assert is_protected_path("protected/sub/../key", roots) == str(protected)
    assert is_protected_path("protected/../ordinary", roots) is None
    assert is_protected_path(str(tmp_path), ("/",)) == "/"
