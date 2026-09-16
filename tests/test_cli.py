from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from conftest import make_config_text
from mockserver import MockServer

TP = [sys.executable, "-m", "termuxpilot"]


def run_tp(args: list[str], *, stdin: str | None = None, env: dict | None = None,
           timeout: int = 30) -> subprocess.CompletedProcess:
    full_env = dict(os.environ)
    # keep subprocess runs from writing to the developer's real audit log
    full_env.setdefault(
        "TERMUXPILOT_AUDIT",
        os.path.join(tempfile.gettempdir(), "tp_test_audit.jsonl"),
    )
    if env:
        full_env.update(env)
    return subprocess.run(
        TP + args,
        input=stdin,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=full_env,
    )


def test_version():
    proc = run_tp(["--version"])
    assert proc.returncode == 0
    assert proc.stdout.startswith("tp ")


def test_one_shot_and_streaming_to_stdout(tmp_path):
    server = MockServer(delay=0.0).start()
    try:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(make_config_text(base_url=server.base_url))
        proc = run_tp(["--config-path", str(cfg), "hello world"])
        assert proc.returncode == 0, proc.stderr
        assert "Mock reply" in proc.stdout
        assert "hello world" in proc.stdout
    finally:
        server.stop()


def test_no_stream_flag(tmp_path):
    server = MockServer().start()
    try:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(make_config_text(base_url=server.base_url))
        proc = run_tp(["--config-path", str(cfg), "--no-stream", "hello"])
        assert proc.returncode == 0, proc.stderr
        assert "Mock reply" in proc.stdout
        # server saw a non-streaming request
        assert server.state.last_payload.get("stream") is False
    finally:
        server.stop()


def test_json_output(tmp_path):
    server = MockServer().start()
    try:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(make_config_text(base_url=server.base_url))
        proc = run_tp(["--config-path", str(cfg), "--json", "ping"])
        assert proc.returncode == 0, proc.stderr
        data = json.loads(proc.stdout)
        assert data["ok"] is True
        assert data["content"].startswith("Mock reply")
        assert data["provider"] == "default"
        assert "rounds" in data
        assert isinstance(data["tool_calls"], list)
    finally:
        server.stop()


def test_json_output_with_tool_call(tmp_path):
    server = MockServer(tool_script=[
        {"tool": "run_shell", "args": {"cmd": "echo e2e-tool"}},
        {"text": "Ran the command successfully."},
    ]).start()
    try:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(make_config_text(base_url=server.base_url))
        proc = run_tp(["--config-path", str(cfg), "--json", "--mode", "yolo", "run it"])
        assert proc.returncode == 0, proc.stderr
        data = json.loads(proc.stdout)
        assert data["ok"] is True
        assert data["content"] == "Ran the command successfully."
        assert data["rounds"] == 2
        assert data["tool_calls"][0]["name"] == "run_shell"
        assert data["tool_calls"][0]["ok"] is True
        assert "e2e-tool" in data["tool_calls"][0]["output"]
    finally:
        server.stop()


def test_dry_run_flag_executes_nothing(tmp_path):
    server = MockServer(tool_script=[
        {"tool": "run_shell", "args": {"cmd": "echo x > /tmp/tp_dry_check.txt"}},
        {"text": "Previewed only."},
    ]).start()
    try:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(make_config_text(base_url=server.base_url))
        proc = run_tp(["--config-path", str(cfg), "--dry-run", "--mode", "yolo",
                       "--json", "run it"])
        assert proc.returncode == 0, proc.stderr
        data = json.loads(proc.stdout)
        assert data["tool_calls"][0]["ok"] is True
        assert data["tool_calls"][0]["output"].startswith("dry-run:")
        assert not Path("/tmp/tp_dry_check.txt").exists()
    finally:
        server.stop()


def test_safe_mode_one_shot_denies_writes(tmp_path):
    server = MockServer(tool_script=[
        {"tool": "run_shell", "args": {"cmd": "echo x > /tmp/tp_safe_check.txt"}},
        {"text": "The write was denied; safe mode is read-only."},
    ]).start()
    try:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(make_config_text(base_url=server.base_url))
        proc = run_tp(["--config-path", str(cfg), "--mode", "safe", "--json", "run it"])
        assert proc.returncode == 0, proc.stderr
        data = json.loads(proc.stdout)
        assert data["tool_calls"][0]["ok"] is False
        assert "safe" in data["tool_calls"][0]["output"]
        assert not Path("/tmp/tp_safe_check.txt").exists()
    finally:
        server.stop()


def test_json_failure_reports_attempts(tmp_path):
    server = MockServer(fail_next=9999, fail_status=503).start()
    try:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(make_config_text(base_url=server.base_url))
        proc = run_tp(["--config-path", str(cfg), "--json", "ping"])
        assert proc.returncode == 1
        data = json.loads(proc.stdout)
        assert data["ok"] is False
        assert "503" in data["error"]
    finally:
        server.stop()


def test_e2e_failover_chain(tmp_path):
    good = MockServer().start()
    flaky = MockServer(fail_next=2, fail_status=503).start()
    try:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(make_config_text(base_url=flaky.base_url,
                                        fallbacks=[good.base_url]))
        proc = run_tp(["--config-path", str(cfg), "hi"])
        assert proc.returncode == 0, proc.stderr
        assert "Mock reply" in proc.stdout
        assert "fallback-1" in proc.stdout  # failover was announced
        assert flaky.request_count == 1
        assert good.request_count == 1
    finally:
        flaky.stop()
        good.stop()


def test_piped_stdin_attached_as_context(tmp_path):
    server = MockServer().start()
    try:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(make_config_text(base_url=server.base_url))
        proc = run_tp(
            ["--config-path", str(cfg), "why is this failing?"],
            stdin="Traceback (most recent call last):\n  boom\n",
        )
        assert proc.returncode == 0, proc.stderr
        assert "Context pasted from stdin" in proc.stdout
        assert "boom" in proc.stdout
    finally:
        server.stop()


def test_interactive_repl_via_piped_stdin(tmp_path):
    server = MockServer().start()
    try:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(make_config_text(base_url=server.base_url))
        proc = run_tp(
            ["-i", "--config-path", str(cfg)],
            stdin="hello there\n/about\n/exit\n",
        )
        assert proc.returncode == 0, proc.stderr
        assert "Mock reply" in proc.stdout
        assert "TermuxPilot" in proc.stdout
        assert "profile default" in proc.stdout
    finally:
        server.stop()


def test_interactive_repl_profile_switch(tmp_path):
    a = MockServer().start()
    b = MockServer().start()
    try:
        cfg_text = (
            "provider:\n"
            f'  base_url: "{a.base_url}"\n'
            '  model: "mock-model-1"\n'
            "profiles:\n"
            "  second:\n"
            "    provider:\n"
            f'      base_url: "{b.base_url}"\n'
            '      model: "mock-model-1"\n'
            "    fallback: []\n"
        )
        cfg = tmp_path / "config.yaml"
        cfg.write_text(cfg_text)
        proc = run_tp(
            ["-i", "--config-path", str(cfg)],
            stdin="/profile second\nhi\n/about\n/exit\n",
        )
        assert proc.returncode == 0, proc.stderr
        assert "profile second" in proc.stdout
        assert a.request_count == 0
        assert b.request_count == 1
    finally:
        a.stop()
        b.stop()


def test_list_profiles_plain(tmp_path):
    a = MockServer().start()
    try:
        cfg_text = (
            "provider:\n"
            f'  base_url: "{a.base_url}"\n'
            '  model: "m1"\n'
            "profiles:\n"
            "  other:\n"
            "    provider:\n"
            f'      base_url: "{a.base_url}"\n'
            '      model: "m2"\n'
            "    fallback: []\n"
        )
        cfg = tmp_path / "config.yaml"
        cfg.write_text(cfg_text)
        proc = run_tp(["--config-path", str(cfg), "--list-profiles", "--plain"])
        assert proc.returncode == 0, proc.stderr
        assert "default" in proc.stdout
        assert "other" in proc.stdout
        assert "m2" in proc.stdout
    finally:
        a.stop()


def test_no_prompt_no_stdin_errors():
    proc = run_tp(["--config-path", "/nonexistent/config.yaml"], stdin="")
    # missing config -> exit 2 (config error) before the prompt check
    assert proc.returncode == 2


def test_config_init_writes_sample(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    proc = run_tp(["config", "init"], env={"HOME": str(home)})
    assert proc.returncode == 0, proc.stderr
    written = home / ".termuxpilot" / "config.yaml"
    assert written.exists()
    assert "fallback:" in written.read_text()
    # second run without --force fails
    proc2 = run_tp(["config", "init"], env={"HOME": str(home)})
    assert proc2.returncode == 2
    proc3 = run_tp(["config", "init", "--force"], env={"HOME": str(home)})
    assert proc3.returncode == 0


def test_config_show(tmp_path):
    a = MockServer().start()
    try:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(make_config_text(base_url=a.base_url))
        proc = run_tp(["--config-path", str(cfg), "config", "show"])
        assert proc.returncode == 0, proc.stderr
        assert "mock-model-1" in proc.stdout
        assert a.base_url in proc.stdout
    finally:
        a.stop()


def test_json_round_limit_is_not_success(tmp_path):
    server = MockServer(tool_script=[{"tool": "run_shell", "args": {"cmd": "echo checkpoint"}}]).start()
    try:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(make_config_text(base_url=server.base_url) + "agent:\n  max_tool_rounds: 1\n")
        proc = run_tp(["--config-path", str(cfg), "--json", "go"], stdin="")
        assert proc.returncode == 3, proc.stderr
        data = json.loads(proc.stdout)
        assert data["ok"] is False
        assert data["status"] == "incomplete"
        assert data["truncated"] is True
        assert data["stop_reason"] == "max_tool_rounds"
        assert data["tool_calls"][0]["ok"] is True
        assert server.state.last_payload["stream"] is False
    finally:
        server.stop()


def test_plain_round_limit_has_distinct_exit_code(tmp_path):
    server = MockServer(tool_script=[{"tool": "run_shell", "args": {"cmd": "echo checkpoint"}}]).start()
    try:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(make_config_text(base_url=server.base_url) + "agent:\n  max_tool_rounds: 1\n")
        proc = run_tp(["--config-path", str(cfg), "--plain", "go"], stdin="")
        assert proc.returncode == 3, proc.stderr
        assert "maximum number of tool rounds" in proc.stdout
    finally:
        server.stop()


def test_oversized_stdin_fails_before_provider_call(tmp_path):
    server = MockServer().start()
    try:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(make_config_text(base_url=server.base_url))
        proc = run_tp(["--config-path", str(cfg), "--json", "summarize"], stdin="x" * 1_000_001)
        assert proc.returncode == 2
        data = json.loads(proc.stdout)
        assert data["ok"] is False and "line windows" in data["error"]
        assert server.request_count == 0
    finally:
        server.stop()


def test_context_limit_is_reported_in_json_before_provider_call(tmp_path):
    server = MockServer().start()
    try:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(make_config_text(base_url=server.base_url) + "agent:\n  max_context_chars: 100\n")
        proc = run_tp(["--config-path", str(cfg), "--json", "go"], stdin="")
        assert proc.returncode == 3
        assert json.loads(proc.stdout)["stop_reason"] == "max_context_chars"
        assert server.request_count == 0
    finally:
        server.stop()
