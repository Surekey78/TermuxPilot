"""Session audit log — one JSON line per tool decision/execution.

Best-effort: a failure to write the audit log never breaks the agent.
Location: ``~/.termuxpilot/audit.jsonl`` (override: ``$TERMUXPILOT_AUDIT``).
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Any

from .config import APP_DIR_NAME


def default_audit_path() -> Path:
    env = os.environ.get("TERMUXPILOT_AUDIT")
    if env:
        return Path(env).expanduser()
    return Path.home() / APP_DIR_NAME / "audit.jsonl"


class AuditLog:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_audit_path()

    def record(self, **entry: Any) -> None:
        entry.setdefault("ts", time.time())
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass  # never block the agent on audit bookkeeping

    def tail(self, n: int = 10) -> list[dict[str, Any]]:
        if n <= 0 or not self.path.exists():
            return []
        try:
            with self.path.open(encoding="utf-8", errors="replace") as stream:
                lines = deque(stream, maxlen=n)
        except OSError:
            return []
        out = []
        for line in lines:
            try:
                entry = json.loads(line)
                if isinstance(entry, dict):
                    out.append(entry)
            except json.JSONDecodeError:
                continue
        return out
