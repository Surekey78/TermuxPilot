"""Safety primitives: risk classification, allow/block lists, secret redaction.

Everything that touches the shell or the filesystem is judged HERE, before the
tool router enforces the permission mode.  The classifier is deliberately
conservative: a false "high" just costs a confirmation; a false "low" could
brick a phone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

RiskLevel = str  # "low" | "medium" | "high" | "critical"

RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}

# (pattern, reason, level) — first match per pattern wins; levels are max-ed.
DESTRUCTIVE_PATTERNS: list[tuple[re.Pattern[str], str, RiskLevel]] = [
    (re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;?\s*:?", re.S), "fork bomb", "critical"),
    (re.compile(r"\bmkfs(\.\w+)?\b"), "filesystem format (mkfs)", "critical"),
    (re.compile(r"\bdd\s+.*\bof=/dev/"), "raw write to a block device (dd of=/dev/…)", "critical"),
    (re.compile(r">\s*/dev/(sd|nvme|mmc|hd)[a-z0-9]*"), "raw overwrite of a disk device", "critical"),
    (re.compile(r"\b(shred|wipefs)\b"), "secure erase / wipe tool", "critical"),
    (re.compile(r"\brm\s+(-[a-z]+\s+)*-[a-z]*[rf][a-z]*\s+(-[a-z]+\s+)*/(\s|$|;|&|\||`)"),
     "rm -rf of /", "critical"),
    (re.compile(
        r"\brm\s+(-[a-z]+\s+)*-[a-z]*[rf][a-z]*\s+(-[a-z]+\s+)*/"
        r"(etc|usr|bin|sbin|boot|vendor|system|dev|data|lib|lib64|opt|storage|app)(/|\s|$|;|&|\||`)"
    ),
    "rm -rf of a system path", "critical"),
    (re.compile(
        r"\brm\s+(-[a-z]+\s+)*-[a-z]*[rf][a-z]*\s+(-[a-z]+\s+)*(~|\$HOME|\$\{HOME\})(/|\s|$|;|&|\||`|/\s*$|/\s*;.*)"
    ),
    "rm -rf of the home directory", "critical"),
    (re.compile(r"\brm\s+(-[a-z]+\s+)*-[a-z]*[rf][a-z]*"),
     "recursive forceful rm", "high"),
    (re.compile(r"\b(fdisk|parted|sgdisk)\b"), "disk partitioning tool", "critical"),
    (re.compile(r"\b(termux-(reboot|poweroff|shutdown)|reboot|poweroff|halt|shutdown)\b"),
     "device power action", "high"),
    (re.compile(r"\bkill(all)?\s+(-\w+\s+)*-?\d*\s*1\b"), "kill of PID 1", "high"),
    (re.compile(r"\bsudo\b"), "privilege escalation (sudo)", "high"),
    (re.compile(r"(curl|wget)\s[^|;]*\|\s*(sudo\s+)?(ba|z|da|k)?sh\b"),
     "pipe downloaded script into a shell", "high"),
    (re.compile(r"\bchmod\s+(-\w+\s+)*(-\w+\s+)*777\s+(/|~|\$HOME)(\s|$|/)"),
     "world-writable chmod on a home/root path", "high"),
    (re.compile(r">\s*/etc/"), "write into /etc via redirect", "high"),
    (re.compile(r"\bmount\b"), "mounting a filesystem", "high"),
    (re.compile(r"\bip6?tables\s+-F\b"), "flushing firewall rules", "high"),
    (re.compile(r"\brm\s+.*\s/(\s|$|;|&|\||`|/)"), "rm targeting a system path", "high"),
    (re.compile(r"\bcrontab\s+-r\b"), "removing all cron jobs", "medium"),
    (re.compile(r"\bgit\s+push\s+(-\w+\s+)*(-f|--force|--force-with-lease)\b"),
     "forceful git push", "medium"),
    (re.compile(r"\bgit\s+reset\s+--hard\b"), "git reset --hard (discards local changes)", "medium"),
    (re.compile(r"\bgit\s+clean\s+-\w*x\b"), "git clean of untracked files", "medium"),
    (re.compile(r"\bmv\s+.*\s/dev/null\b"), "mv into /dev/null (deletion)", "medium"),
    (re.compile(r"\bpkill\s+-9\b"), "kill -9 via pkill", "medium"),
    (re.compile(r"\bhistory\s+-c\b"), "clearing shell history", "medium"),
    (re.compile(r"\bpkg\s+(remove|purge)\b"), "removing an installed package", "medium"),
    (re.compile(r"\buseradd\b|\buserdel\b"), "user account change", "medium"),
]

#: substrings/indicators that make a command non-read-only
WRITE_COMMANDS = (
    "rm ", "mv ", "cp ", "dd ", "mkdir", "touch ", "ln ", "install ",
    "chmod", "chown", "mkfs", "sed -i", "tee ", "printf > ", "echo >",
    "echo >>", "wget -O", "curl -o", "curl --output", "pkg install",
    "pkg remove", "pip install", "pip uninstall", "git push", "git commit",
    "git reset", "git checkout", "git clone", "crontab", "reboot", "shutdown",
    "mount", "umount", "apt ", "dpkg ", "systemctl",
)
WRITE_REDIRECT_RE = re.compile(r"(?<!\d)>+\s*\S|&>")


@dataclass
class RiskAssessment:
    level: RiskLevel = "low"
    reasons: list[str] = field(default_factory=list)

    @property
    def score(self) -> int:
        return RISK_ORDER.get(self.level, 0)

    def escalate(self, level: RiskLevel, reason: str) -> None:
        if RISK_ORDER[level] > self.score:
            self.level = level
        if reason not in self.reasons:
            self.reasons.append(reason)

    def merge(self, other: "RiskAssessment") -> None:
        for reason in other.reasons:
            self.escalate(other.level, reason)


def assess_command(command: str) -> RiskAssessment:
    """Classify a shell command string by destructive potential."""
    assessment = RiskAssessment()
    for pattern, reason, level in DESTRUCTIVE_PATTERNS:
        if pattern.search(command):
            assessment.escalate(level, reason)

    write_indicators: list[str] = []
    if WRITE_REDIRECT_RE.search(command):
        write_indicators.append("output redirect")
    lowered = " " + command.lower() + " "
    for cmd in WRITE_COMMANDS:
        if cmd in lowered:
            write_indicators.append(cmd.strip())
    # `pkg install` etc. mutate the system; even low-risk text counts as a write
    if write_indicators:
        assessment.escalate(
            "medium" if assessment.level == "low" else assessment.level,
            "mutates system state: " + ", ".join(sorted(set(write_indicators)))[:120],
        )
    return assessment


def is_read_only(command: str) -> bool:
    """True when the command has no obvious write side-effects."""
    if WRITE_REDIRECT_RE.search(command):
        return False
    lowered = " " + command.lower() + " "
    for cmd in WRITE_COMMANDS:
        if cmd in lowered:
            return False
    return True


# ---------------------------------------------------------------------------
# Allowlist / blocklist (config-driven regexes, applied to shell commands)
# ---------------------------------------------------------------------------


def compile_rules(patterns: list[str]) -> list[tuple[str, re.Pattern[str]]]:
    rules: list[tuple[str, re.Pattern[str]]] = []
    for raw in patterns or []:
        try:
            rules.append((raw, re.compile(raw)))
        except re.error as exc:
            raise ValueError(f"invalid allow/blocklist pattern {raw!r}: {exc}") from exc
    return rules


def match_rules(command: str, rules: list[tuple[str, re.Pattern[str]]]) -> list[str]:
    return [raw for raw, pattern in rules if pattern.search(command)]


# ---------------------------------------------------------------------------
# Secret redaction (applied to tool output before it reaches the model)
# ---------------------------------------------------------------------------

_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("private-key-block", re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S)),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{4,}\b")),
    ("openai-key", re.compile(r"\bsk-(?!ant-)[A-Za-z0-9_-]{20,}\b")),
    ("anthropic-key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("github-token", re.compile(r"\b(?:ghp|gho|ghs|ghu)_[A-Za-z0-9]{20,}\b|\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("aws-key-id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b")),
    ("groq-key", re.compile(r"\bgsk_[A-Za-z0-9]{20,}\b")),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("bearer-token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{16,}\b")),
    ("key-value-secret", re.compile(
        r"(?i)\b(api[_-]?key|apikey|access[_-]?token|auth[_-]?token|secret|password|passwd|"
        r"client[_-]?secret|token)\b\s*[:=]\s*(['\"]?)(\S{8,})\2")),
]


def redact_secrets(text: str) -> str:
    """Mask obvious secrets in command/file output before it reaches the model."""
    if not text:
        return text
    for name, pattern in _SECRET_PATTERNS:
        if name == "key-value-secret":
            def _kv(m: re.Match[str]) -> str:
                return f"{m.group(1)}={m.group(2)}[REDACTED:{name}]{m.group(2)}"

            text = pattern.sub(_kv, text)
        else:
            text = pattern.sub(f"[REDACTED:{name}]", text)
    return text


# ---------------------------------------------------------------------------
# Path guards (file tools)
# ---------------------------------------------------------------------------

DEFAULT_PROTECTED_PATHS = (
    "/etc", "/dev", "/boot", "/system", "/vendor", "/bin", "/sbin", "/usr",
    "/data/data/com.termux/files/usr",
)


def is_protected_path(path: str, protected: tuple[str, ...] | list[str]) -> str | None:
    """Return the protected prefix *path* falls under, else None."""
    normalized = path.rstrip("/") or "/"
    for prefix in protected:
        prefix = prefix.rstrip("/")
        if normalized == prefix or normalized.startswith(prefix + "/"):
            return prefix
    return None
