from __future__ import annotations

import os
import shlex
import signal
import sys
import time
import tracemalloc
from pathlib import Path

import pytest

from termuxpilot.audit import AuditLog
from termuxpilot.tools import build_default_tools
from termuxpilot.tools import shell
from termuxpilot.tools.base import ExecutionContext
from termuxpilot.tools.router import ToolRouter

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")


@pytest.fixture
def router(tmp_path):
    ctx = ExecutionContext(
        shell_workdir=str(tmp_path), shell_timeout=5, shell_max_timeout=5,
        shell_kill_grace=0.05, max_output_chars=4096,
    )
    return ToolRouter(build_default_tools(), ctx, mode="yolo", audit=AuditLog(tmp_path / "audit.jsonl"))


def python_cmd(code: str) -> str:
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


def _running(pid: int) -> bool:
    # A killed orphan may briefly remain a zombie until PID 1 reaps it.
    stat_file = Path(f"/proc/{pid}/stat")
    if stat_file.exists():
        try:
            if stat_file.read_text().rsplit(")", 1)[1].split()[0] == "Z":
                return False
        except FileNotFoundError:
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def assert_stopped(pid: int) -> None:
    deadline = time.monotonic() + 2
    while _running(pid) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not _running(pid)


def test_noisy_stdout_and_stderr_are_drained_with_bounded_memory(router):
    command = python_cmd(
        "import os\n"
        "os.write(1, b'STDOUT_HEAD\\n'); os.write(2, b'STDERR_HEAD\\n')\n"
        "for _ in range(300):\n"
        " os.write(1, b'ordinary stdout line\\n' * 500)\n"
        " os.write(2, b'ordinary stderr line\\n' * 500)\n"
        "os.write(1, b'STDOUT_TAIL\\n'); os.write(2, b'STDERR_TAIL\\n')\n"
    )
    # Tracing regex/line processing is slower than normal execution.
    router.ctx.shell_timeout = router.ctx.shell_max_timeout = 60
    tracemalloc.start()
    try:
        result = router.execute("run_shell", {"cmd": command})
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result.ok, result.output
    assert result.truncated
    assert len(result.output) <= router.ctx.max_output_chars
    for marker in ("STDOUT_HEAD", "STDOUT_TAIL", "STDERR_HEAD", "STDERR_TAIL"):
        assert marker in result.output
    assert peak < 2_000_000


def test_timeout_preserves_partial_output_and_kills_term_ignoring_child(router, tmp_path):
    command = python_cmd(
        "import os, signal, time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "Path('child.pid').write_text(str(os.getpid()))\n"
        "print('partial result before timeout', flush=True)\n"
        "time.sleep(30)\n"
    )
    result = router.execute("run_shell", {"cmd": command, "timeout": 1})
    pid = int((tmp_path / "child.pid").read_text())
    assert not result.ok and result.timed_out
    assert result.exit_code is None
    assert "partial result before timeout" in result.output
    assert "timed out" in result.output
    assert result.duration < 4
    assert_stopped(pid)
    assert router.audit.tail(1)[0]["timed_out"] is True


@pytest.mark.parametrize("exception", [KeyboardInterrupt, RuntimeError])
def test_interrupt_and_executor_error_cleanup(router, monkeypatch, exception):
    processes = []
    original_popen = shell.subprocess.Popen

    def tracked_popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        processes.append(process)
        return process

    def fail(*args, **kwargs):
        raise exception("test interruption")

    monkeypatch.setattr(shell.subprocess, "Popen", tracked_popen)
    monkeypatch.setattr(shell, "_drain_ready", fail)
    if exception is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt):
            router.execute("run_shell", {"cmd": python_cmd("import time; time.sleep(30)")})
        assert router.audit.tail(1)[0]["reason"] == "interrupted"
    else:
        result = router.execute("run_shell", {"cmd": python_cmd("import time; time.sleep(30)")})
        assert not result.ok
    assert len(processes) == 1
    assert processes[0].poll() is not None
    with pytest.raises(ProcessLookupError):
        os.killpg(processes[0].pid, signal.SIGCONT)


def test_success_cleans_up_unmanaged_background_children(router, tmp_path):
    command = python_cmd(
        "import subprocess, sys\n"
        "from pathlib import Path\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], "
        "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        "Path('background.pid').write_text(str(p.pid))\n"
        "print('foreground finished')\n"
    )
    result = router.execute("run_shell", {"cmd": command})
    assert result.ok
    assert_stopped(int((tmp_path / "background.pid").read_text()))


def test_stdin_is_closed_and_invalid_utf8_does_not_crash(router):
    command = python_cmd(
        "import os, sys\n"
        "assert sys.stdin.read() == ''\n"
        "os.write(1, b'closed stdin \\xff\\n')\n"
    )
    result = router.execute("run_shell", {"cmd": command})
    assert result.ok
    assert "closed stdin \ufffd" in result.output


@pytest.mark.parametrize("timeout", [0, -1, True, "3", None, float("inf"), float("nan")])
def test_invalid_timeouts_are_denied_before_execution(router, tmp_path, timeout):
    result = router.execute("run_shell", {"cmd": "touch should-not-exist", "timeout": timeout})
    assert result.denied
    assert not (tmp_path / "should-not-exist").exists()
    assert router.audit.tail(1)[0]["executed"] is False


def test_model_cannot_raise_timeout_ceiling(router, tmp_path):
    result = router.execute("run_shell", {"cmd": "touch should-not-exist", "timeout": 6})
    assert not result.ok
    assert "max_timeout" in result.output
    assert result.denied
    assert router.audit.tail(1)[0]["executed"] is False
    assert not (tmp_path / "should-not-exist").exists()
