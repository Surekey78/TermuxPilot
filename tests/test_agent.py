from __future__ import annotations

import pytest

from termuxpilot.agent import AgentLoop, parse_agent_json
from termuxpilot.audit import AuditLog
from mockserver import MockServer
from termuxpilot.tools import build_default_tools
from termuxpilot.tools.base import ExecutionContext
from termuxpilot.tools.router import ToolRouter


@pytest.fixture
def audit_path(tmp_path, monkeypatch):
    path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("TERMUXPILOT_AUDIT", str(path))
    return path


def make_chain_and_router(server, *, mode="standard", function_calling="auto",
                          blocklist=None, confirm=None, tmp_path=None):
    from termuxpilot.config import ProviderSettings
    from termuxpilot.provider import ProviderChain

    settings = ProviderSettings(label="mock", base_url=server.base_url, model="mock-model")
    chain = ProviderChain([settings])
    ctx = ExecutionContext(
        shell_timeout=10, shell_workdir=str(tmp_path) if tmp_path else None,
        redact_secrets=True, protected_paths=("/etc", "/dev"), dry_run=False,
    )
    router = ToolRouter(
        build_default_tools(), ctx, mode=mode, blocklist=blocklist or [],
        audit=AuditLog(), confirm=confirm,
    )
    agent = AgentLoop(
        chain, router,
        max_rounds=8,
        function_calling=function_calling,
    )
    return chain, router, agent


def test_native_multi_round_agent(tmp_path, audit_path):
    server = MockServer(tool_script=[
        {"tool": "run_shell", "args": {"cmd": "echo agent-works"}},
        {"text": "Done: I ran the command and it printed agent-works."},
    ]).start()
    try:
        _, _, agent = make_chain_and_router(
            server, mode="yolo", tmp_path=tmp_path,
        )
        outcome = agent.run("You are a test agent.", [{"role": "user", "content": "run a check"}])
        assert outcome.final_text.startswith("Done:")
        assert outcome.rounds == 2
        assert len(outcome.tool_calls) == 1
        tc = outcome.tool_calls[0]
        assert tc.name == "run_shell"
        assert tc.ok is True
        assert tc.exit_code == 0
        assert "agent-works" in tc.output
        # the transcript carries assistant + tool messages for context
        roles = [m["role"] for m in outcome.transcript]
        assert roles == ["assistant", "tool", "assistant"]
        # and the real shell was used (audit records it)
        entries = AuditLog().tail(5)
        assert entries[-1]["tool"] == "run_shell"
        assert server.request_count == 2
    finally:
        server.stop()


def test_json_mode_agent(tmp_path, audit_path):
    server = MockServer(tool_script=[
        {"tool": "run_shell", "args": {"cmd": "echo json-mode"}},
        {"text": "All good in JSON mode."},
    ]).start()
    try:
        _, _, agent = make_chain_and_router(
            server, mode="yolo", function_calling="json", tmp_path=tmp_path,
        )
        outcome = agent.run("You are a test agent.", [{"role": "user", "content": "go"}])
        assert outcome.final_text == "All good in JSON mode."
        assert outcome.tool_calls[0].name == "run_shell"
        assert "json-mode" in outcome.tool_calls[0].output
        # tool result was fed back as a user message
        assert any(
            m.get("role") == "user" and m["content"].startswith("[tool_result run_shell]")
            for m in outcome.transcript
        )
    finally:
        server.stop()


def test_auto_degrades_to_json_mode(tmp_path, audit_path):
    server = MockServer(
        tool_script=[
            {"tool": "run_shell", "args": {"cmd": "echo degraded"}},
            {"text": "Recovered via JSON mode."},
        ],
        reject_tools=True,
    ).start()
    try:
        _, _, agent = make_chain_and_router(server, mode="yolo", tmp_path=tmp_path)
        assert agent.mode == "native"
        outcome = agent.run(
            "You are a test agent.",
            [{"role": "user", "content": "go"}],
            on_degrade=lambda: None,
        )
        assert agent.mode == "json"  # degraded mid-run
        assert outcome.final_text == "Recovered via JSON mode."
        assert outcome.tool_calls[0].name == "run_shell"
        assert "degraded" in outcome.tool_calls[0].output
    finally:
        server.stop()


def test_tool_denied_feeds_back_and_loop_recovers(tmp_path, audit_path):
    server = MockServer(tool_script=[
        {"tool": "run_shell", "args": {"cmd": "echo x > forbidden.txt"}},
        {"text": "I could not write the file (denied), here is what happened."},
    ]).start()
    try:
        _, router, agent = make_chain_and_router(
            server, mode="standard", tmp_path=tmp_path,
            confirm=lambda n, p, r: False,  # user says no
        )
        outcome = agent.run(
            "You are a test agent.", [{"role": "user", "content": "try to write"}]
        )
        assert outcome.final_text.startswith("I could not write")
        assert outcome.tool_calls[0].ok is False
        assert not (tmp_path / "forbidden.txt").exists()
        # the denial was fed back to the model as a tool message
        tool_msgs = [m for m in outcome.transcript if m["role"] == "tool"]
        assert tool_msgs and "denied" in tool_msgs[0]["content"]
    finally:
        server.stop()


def test_max_rounds_stops_loop(tmp_path, audit_path):
    # script that always answers with a tool call
    server = MockServer(tool_script=[
        {"tool": "run_shell", "args": {"cmd": "echo 1"}},
        {"tool": "run_shell", "args": {"cmd": "echo 2"}},
        {"tool": "run_shell", "args": {"cmd": "echo 3"}},
        {"tool": "run_shell", "args": {"cmd": "echo 4"}},
        {"tool": "run_shell", "args": {"cmd": "echo 5"}},
    ]).start()
    try:
        _, _, agent = make_chain_and_router(server, mode="yolo", tmp_path=tmp_path)
        agent.max_rounds = 3
        outcome = agent.run("You are a test agent.", [{"role": "user", "content": "loop"}])
        assert outcome.truncated is True
        assert outcome.rounds == 3
        assert len(outcome.tool_calls) == 3
    finally:
        server.stop()


def test_parse_agent_json_variants():
    assert parse_agent_json('{"tool": "x", "args": {}}') == {"tool": "x", "args": {}}
    assert parse_agent_json('```json\n{"tool": "x"}\n```') == {"tool": "x"}
    assert parse_agent_json('Sure! {"tool": "x", "args": {"a": 1}} hope that helps') == {
        "tool": "x", "args": {"a": 1},
    }
    assert parse_agent_json("no json here") is None
    assert parse_agent_json("") is None
