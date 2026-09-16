from __future__ import annotations

import io
import json
from copy import deepcopy

import pytest
from rich.console import Console

from termuxpilot.agent import AgentLoop
from termuxpilot.audit import AuditLog
from termuxpilot.config import AppConfig, Profile, ProviderSettings
from termuxpilot.provider import ChatResult, HttpError, RequestTimeout, ToolCall
from termuxpilot.repl import Repl
from termuxpilot.tools import build_default_tools
from termuxpilot.tools.base import ExecutionContext, ToolResult
from termuxpilot.tools.router import ToolRouter


class ScriptedChain:
    def __init__(self, *steps):
        self.steps = iter(steps)
        self.requests = []

    def chat(self, messages, **kwargs):
        self.requests.append((deepcopy(messages), kwargs))
        step = next(self.steps)
        if isinstance(step, BaseException):
            raise step
        return step


def call(identifier="call-1", cmd="echo test"):
    return ToolCall(id=identifier, name="run_shell", arguments=json.dumps({"cmd": cmd}))


@pytest.fixture
def router(tmp_path):
    return ToolRouter(
        build_default_tools(), ExecutionContext(shell_workdir=str(tmp_path)),
        mode="yolo", audit=AuditLog(tmp_path / "audit.jsonl"),
    )


def test_usage_is_aggregated_across_rounds_and_nested_details(router):
    chain = ScriptedChain(
        ChatResult(content="", tool_calls=[call()], usage={
            "prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12,
            "prompt_tokens_details": {"cached_tokens": 3},
        }),
        ChatResult(content="Done", usage={
            "prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24,
            "prompt_tokens_details": {"cached_tokens": 5},
        }),
    )
    outcome = AgentLoop(chain, router).run("test", [])
    assert outcome.completed and outcome.status == "completed"
    assert outcome.usage == {
        "prompt_tokens": 30, "completion_tokens": 6, "total_tokens": 36,
        "prompt_tokens_details": {"cached_tokens": 8},
    }


def test_native_response_containing_json_is_not_executed(router, tmp_path):
    response = json.dumps({"tool": "run_shell", "args": {"cmd": "touch must-not-exist"}})
    outcome = AgentLoop(ScriptedChain(ChatResult(content=response)), router).run("test", [])
    assert outcome.completed
    assert outcome.final_text == response
    assert outcome.tool_calls == []
    assert not (tmp_path / "must-not-exist").exists()


def test_native_mode_does_not_silently_degrade(router):
    chain = ScriptedChain(HttpError("tools unsupported", status=400))
    agent = AgentLoop(chain, router, function_calling="native")
    with pytest.raises(HttpError):
        agent.run("test", [])
    assert agent.mode == "native"
    assert len(chain.requests) == 1


def test_auto_negotiation_does_not_consume_tool_round(router):
    chain = ScriptedChain(
        HttpError("tools unsupported", status=400),
        ChatResult(content='{"tool": null, "response": "done"}'),
    )
    agent = AgentLoop(chain, router, max_rounds=1)
    outcome = agent.run("test", [])
    assert outcome.completed and outcome.rounds == 1
    assert len(chain.requests) == 2
    assert chain.requests[1][1]["json_mode"] is True


def test_tool_batch_is_rejected_before_any_execution_when_over_budget(router, tmp_path):
    chain = ScriptedChain(ChatResult(content="", tool_calls=[
        call("one", "touch must-not-exist"), call("two", "touch also-must-not-exist"),
    ]))
    outcome = AgentLoop(chain, router, max_tool_calls=1).run("test", [])
    assert not outcome.completed and outcome.stop_reason == "max_tool_calls"
    assert outcome.tool_calls == []
    assert outcome.transcript == []  # no unanswered tool call IDs
    assert not (tmp_path / "must-not-exist").exists()


def test_context_guard_stops_before_contacting_provider(router):
    chain = ScriptedChain()
    outcome = AgentLoop(chain, router, max_context_chars=100).run("x" * 1000, [])
    assert outcome.stop_reason == "max_context_chars"
    assert outcome.rounds == 0 and not outcome.completed
    assert chain.requests == []


@pytest.mark.parametrize("reason", ["length", "interrupted", "error", "content_filter"])
def test_incomplete_provider_responses_never_execute_tools(router, tmp_path, reason):
    result = ChatResult(content="partial", tool_calls=[call(cmd="touch must-not-exist")], finish_reason=reason)
    outcome = AgentLoop(ScriptedChain(result), router).run("test", [])
    assert outcome.stop_reason == "incomplete_response"
    assert not outcome.completed
    assert outcome.transcript == [{"role": "assistant", "content": "partial"}]
    assert not (tmp_path / "must-not-exist").exists()


def test_empty_response_is_not_success(router):
    outcome = AgentLoop(ScriptedChain(ChatResult(content="")), router).run("test", [])
    assert outcome.stop_reason == "empty_response"
    assert not outcome.completed


def test_provider_failure_preserves_completed_tool_progress(router):
    chain = ScriptedChain(ChatResult(content="", tool_calls=[call()]), RequestTimeout("offline"))
    agent = AgentLoop(chain, router)
    with pytest.raises(RequestTimeout):
        agent.run("test", [])
    assert agent.last_outcome.stop_reason == "provider_error"
    assert [m["role"] for m in agent.last_outcome.transcript] == ["assistant", "tool"]
    assert agent.last_outcome.tool_calls[0].ok


@pytest.mark.parametrize("interrupt_in_callback", [False, True])
def test_interrupted_batch_has_results_for_all_tool_ids(router, monkeypatch, interrupt_in_callback):
    chain = ScriptedChain(ChatResult(content="", tool_calls=[call("one"), call("two")]))
    attempted = []

    def execute(name, args):
        attempted.append(name)
        if not interrupt_in_callback:
            raise KeyboardInterrupt
        return ToolResult(ok=True, output="executed")

    def callback(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(router, "execute", execute)
    agent = AgentLoop(chain, router)
    with pytest.raises(KeyboardInterrupt):
        agent.run("test", [], on_tool_result=callback)
    assert attempted == ["run_shell"]
    outcome = agent.last_outcome
    assert outcome.stop_reason == "interrupted"
    assert [m["tool_call_id"] for m in outcome.transcript if m["role"] == "tool"] == ["one", "two"]
    assert outcome.tool_calls[1].denied
    assert outcome.tool_calls[0].ok is interrupt_in_callback


def make_repl(router, chain):
    settings = ProviderSettings(label="mock", base_url="http://example.test/v1", model="mock")
    profile = Profile(name="default", primary=settings)
    config = AppConfig(profiles={"default": profile}, default_profile="default")
    output = io.StringIO()
    repl = Repl(config, profile, chain, router, AgentLoop(chain, router),
                console=Console(file=output), plain=True, stream=False)
    return repl, output


def test_repl_does_not_duplicate_final_answer_in_history(router):
    chain = ScriptedChain(ChatResult(content="answer"), ChatResult(content="next answer"))
    repl, _ = make_repl(router, chain)
    assert repl.ask("hello")
    assert repl.conversation.history() == [
        {"role": "user", "content": "hello"}, {"role": "assistant", "content": "answer"},
    ]
    assert repl.ask("next")
    messages = chain.requests[1][0]
    assert sum(m.get("content") == "answer" for m in messages) == 1


def test_repl_preserves_tool_progress_on_provider_error_and_reports_plain_error(router, capsys):
    chain = ScriptedChain(ChatResult(content="", tool_calls=[call()]), RequestTimeout("offline"))
    repl, output = make_repl(router, chain)
    assert not repl.ask("go")
    assert repl.last_exit_code == 1
    assert "PROVIDER ERROR: offline" in capsys.readouterr().out
    assert any(m["role"] == "tool" for m in repl.conversation.history())


def test_repl_incomplete_turn_has_distinct_exit_code(router):
    chain = ScriptedChain(ChatResult(content="", tool_calls=[call()]))
    repl, _ = make_repl(router, chain)
    repl.agent.max_rounds = 1
    assert not repl.ask("go")
    assert repl.last_exit_code == 3


def test_tool_record_arguments_are_redacted_without_changing_execution(router, tmp_path):
    secret = "sk-abcdefghijklmnop1234567890ABC"
    path = tmp_path / "synthetic-secret"
    tool_call = ToolCall(id="one", name="write_file", arguments=json.dumps({"path": str(path), "content": secret}))
    chain = ScriptedChain(ChatResult(content="", tool_calls=[tool_call]), ChatResult(content="done"))
    outcome = AgentLoop(chain, router).run("test", [])
    assert outcome.completed and path.read_text() == secret
    assert secret not in repr(outcome.tool_calls[0].args)


def test_configured_hundred_step_workflow_completes_without_hardcoded_round_cap(router, monkeypatch):
    steps = [ChatResult(content="", tool_calls=[call(str(i))], usage={"total_tokens": 10}) for i in range(100)]
    steps.append(ChatResult(content="verified final answer", usage={"total_tokens": 10}))
    chain = ScriptedChain(*steps)
    monkeypatch.setattr(router, "execute", lambda *args: ToolResult(ok=True, output="verified step"))
    agent = AgentLoop(chain, router, max_rounds=101, max_tool_calls=100)
    outcome = agent.run("test", [{"role": "user", "content": "long workflow"}])
    assert outcome.completed and outcome.rounds == 101
    assert len(outcome.tool_calls) == 100
    assert outcome.usage["total_tokens"] == 1010


def test_context_limit_mid_workflow_keeps_completed_tool_result(router, monkeypatch):
    chain = ScriptedChain(ChatResult(content="", tool_calls=[call()]))
    monkeypatch.setattr(router, "execute", lambda *args: ToolResult(ok=True, output="x" * 20_000))
    outcome = AgentLoop(chain, router, max_context_chars=20_000).run("test", [])
    assert not outcome.completed and outcome.stop_reason == "max_context_chars"
    assert len(chain.requests) == 1
    assert outcome.tool_calls[0].ok
    assert outcome.tool_calls[0].truncated
    assert len(outcome.tool_calls[0].output) <= 500
    assert [m["role"] for m in outcome.transcript] == ["assistant", "tool"]


def test_piped_input_cannot_supply_interactive_confirmation(router, monkeypatch):
    repl, _ = make_repl(router, ScriptedChain())
    source = io.StringIO("yes\n")
    monkeypatch.setattr("sys.stdin", source)
    from termuxpilot.safety import RiskAssessment

    assert not repl._confirm("run_shell", "touch file", RiskAssessment())
    assert source.read() == "yes\n"  # did not consume another REPL prompt as approval
