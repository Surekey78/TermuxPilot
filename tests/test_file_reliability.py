from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from termuxpilot.audit import AuditLog
from termuxpilot.tools import build_default_tools, files
from termuxpilot.tools.base import ExecutionContext
from termuxpilot.tools.router import ToolRouter


@pytest.fixture
def router(tmp_path):
    return ToolRouter(
        build_default_tools(), ExecutionContext(max_output_chars=4096),
        mode="yolo", audit=AuditLog(tmp_path / "audit.jsonl"),
    )


def test_line_window_on_sparse_gigabyte_file_does_not_read_whole_file(router, tmp_path, monkeypatch):
    path = tmp_path / "large.log"
    with path.open("wb") as stream:
        stream.write(b"first\nsecond\nthird\n")
        stream.seek(1024 ** 3)
        stream.write(b"\n")

    def forbidden(*args, **kwargs):
        raise AssertionError("whole-file read is forbidden")

    monkeypatch.setattr(Path, "read_text", forbidden)
    result = router.execute("read_file", {"path": str(path), "start_line": 2, "end_line": 3})
    assert result.ok, result.output
    assert "second" in result.output and "third" in result.output
    assert "first" not in result.output
    assert "lines 2-3" in result.output
    assert not result.truncated
    refused = router.execute("read_file", {"path": str(path)})
    assert not refused.ok and "window" in refused.output


def test_start_line_alone_has_bounded_default_window(router, tmp_path):
    path = tmp_path / "many.log"
    path.write_text("".join(f"entry-{i}\n" for i in range(1, 1001)))
    result = router.execute("read_file", {"path": str(path), "start_line": 500})
    assert result.ok
    assert "lines 500-699" in result.output
    assert "entry-700\n" not in result.output


@pytest.mark.parametrize("args", [
    {"start_line": 0}, {"end_line": -1}, {"start_line": True},
    {"end_line": "2"}, {"start_line": 1.5}, {"start_line": None},
])
def test_invalid_line_types_and_bounds_are_denied(router, tmp_path, args):
    path = tmp_path / "a.txt"
    path.write_text("one\ntwo\n")
    result = router.execute("read_file", {"path": str(path), **args})
    assert result.denied


def test_reversed_and_excessive_windows_fail_cleanly(router, tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("one\ntwo\n")
    for args in ({"start_line": 3, "end_line": 2}, {"end_line": 10001}):
        result = router.execute("read_file", {"path": str(path), **args})
        assert not result.ok


def test_read_window_eof_empty_and_crlf(router, tmp_path):
    path = tmp_path / "a.txt"
    path.write_bytes(b"one\r\ntwo")
    result = router.execute("read_file", {"path": str(path), "start_line": 2, "end_line": 9})
    assert result.ok and "lines 2-2 of 2" in result.output
    assert "\r" not in result.output
    beyond = router.execute("read_file", {"path": str(path), "start_line": 10})
    assert not beyond.ok and "beyond end" in beyond.output
    path.write_text("")
    assert "empty file" in router.execute("read_file", {"path": str(path)}).output


def test_overlong_line_is_rejected_without_unbounded_read(router, tmp_path):
    path = tmp_path / "one-long-line.log"
    with path.open("wb") as stream:
        stream.seek(1024 ** 3)
        stream.write(b"\n")
    result = router.execute("read_file", {"path": str(path), "end_line": 1})
    assert not result.ok
    assert "line 1 exceeds" in result.output


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO")
def test_non_regular_file_does_not_block(router, tmp_path):
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    result = router.execute("read_file", {"path": str(fifo)})
    assert not result.ok and "not a regular file" in result.output


def test_window_and_output_limits_are_explicit(router, tmp_path):
    path = tmp_path / "many.log"
    with path.open("w") as stream:
        for _ in range(2000):
            stream.write("x" * 1000 + "\n")
    result = router.execute("read_file", {"path": str(path), "end_line": 2000})
    assert result.ok and result.truncated
    assert "continue with start_line=" in result.output
    assert len(result.output) <= router.ctx.max_output_chars


def test_read_window_inside_private_key_is_redacted(router, tmp_path):
    path = tmp_path / "synthetic.pem"
    path.write_text("-----BEGIN RSA PRIVATE KEY-----\nPRIVATE_BODY\nPRIVATE_BODY_2\n"
                    "-----END RSA PRIVATE KEY-----\nnormal\n")
    result = router.execute("read_file", {"path": str(path), "start_line": 2, "end_line": 5})
    assert result.ok
    assert "PRIVATE_BODY" not in result.output
    assert "normal" in result.output


def test_oversized_edit_preview_does_not_execute(router, tmp_path):
    path = tmp_path / "large.txt"
    path.write_bytes(b"x" * (files.MAX_READ_BYTES + 1))
    result = router.execute("edit_file", {"path": str(path), "old_text": "x", "new_text": "y"})
    assert result.denied and "too large" in result.output
    assert path.stat().st_size == files.MAX_READ_BYTES + 1
    assert router.audit.tail(1)[0]["executed"] is False


def test_atomic_replace_failure_preserves_original_and_removes_temp(router, tmp_path, monkeypatch):
    path = tmp_path / "important.txt"
    path.write_text("original")

    def full_disk(*args):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(files.os, "replace", full_disk)
    result = router.execute("write_file", {"path": str(path), "content": "replacement"})
    assert not result.ok
    assert path.read_text() == "original"
    assert not list(tmp_path.glob(".important.txt.tp-*"))


def test_atomic_write_preserves_permissions_and_symlink(router, tmp_path):
    target = tmp_path / "script.sh"
    target.write_text("old")
    target.chmod(0o700)
    link = tmp_path / "link"
    link.symlink_to(target)
    result = router.execute("write_file", {"path": str(link), "content": "new"})
    assert result.ok
    assert link.is_symlink()
    assert target.read_text() == "new"
    assert stat.S_IMODE(target.stat().st_mode) == 0o700


@pytest.mark.parametrize("tool,args", [
    ("write_file", {"content": "replacement"}),
    ("edit_file", {"old_text": "original", "new_text": "replacement"}),
])
def test_file_changed_during_confirmation_is_not_overwritten(router, tmp_path, tool, args):
    path = tmp_path / "a.txt"
    path.write_text("original")
    router.mode = "standard"

    def confirm(*args):
        path.write_text("changed by another writer")
        return True

    router.confirm = confirm
    result = router.execute(tool, {"path": str(path), **args})
    assert not result.ok
    assert path.read_text() == "changed by another writer"


def test_symlink_retargeted_during_confirmation_is_not_followed(router, tmp_path):
    a, b, link = tmp_path / "a", tmp_path / "b", tmp_path / "link"
    a.write_text("a")
    b.write_text("b")
    link.symlink_to(a)
    router.mode = "standard"

    def confirm(*args):
        link.unlink()
        link.symlink_to(b)
        return True

    router.confirm = confirm
    result = router.execute("write_file", {"path": str(link), "content": "replacement"})
    assert not result.ok and "changed since preview" in result.output
    assert a.read_text() == "a" and b.read_text() == "b"


def test_diff_does_not_leak_private_key_body_outside_pem_header_context(router, tmp_path):
    a, b = tmp_path / "a.pem", tmp_path / "b.pem"
    old = "-----BEGIN RSA PRIVATE KEY-----\n" + "UNCHANGED_PRIVATE_BODY\n" * 10
    old += "OLD_SECRET_BODY\n" + "UNCHANGED_PRIVATE_BODY\n" * 10 + "-----END RSA PRIVATE KEY-----\n"
    a.write_text(old)
    b.write_text(old.replace("OLD_SECRET_BODY", "NEW_SECRET_BODY"))
    result = router.execute("diff_files", {"a": str(a), "b": str(b)})
    assert result.ok
    assert "SECRET_BODY" not in result.output
    assert "redacted content changes" in result.output
