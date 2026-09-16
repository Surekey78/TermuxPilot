from __future__ import annotations

from pathlib import Path

from termuxpilot.audit import AuditLog


def test_audit_tail_streams_instead_of_loading_whole_file(tmp_path, monkeypatch):
    audit = AuditLog(tmp_path / "audit.jsonl")
    for index in range(1000):
        audit.record(index=index)

    def forbidden(*args, **kwargs):
        raise AssertionError("audit tail must not use read_text")

    monkeypatch.setattr(Path, "read_text", forbidden)
    assert [entry["index"] for entry in audit.tail(3)] == [997, 998, 999]
    assert audit.tail(0) == []
    assert audit.tail(-1) == []


def test_audit_tail_tolerates_partial_final_line(tmp_path):
    path = tmp_path / "audit.jsonl"
    path.write_text('{"tool": "read_file"}\n{"tool":')
    assert AuditLog(path).tail(2) == [{"tool": "read_file"}]
