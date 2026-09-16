"""Interactive REPL (``tp -i``) and one-shot turn engine.

Since v0.2 the turn engine is an :class:`AgentLoop`: the model can call tools
(shell, file ops) through the :class:`ToolRouter`, which enforces the
permission mode, shows dry-run previews / confirmations, redacts secrets, and
audits every call.  The same :meth:`Repl.ask` drives the interactive loop and
the one-shot ``tp "prompt"`` path.

Rendering:
* TTY      -> rich Live markdown panel per model round, tool panels in between
* non-TTY  -> plain text (pipe-safe)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Callable

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from . import __version__
from .agent import AgentLoop, AgentOutcome
from .audit import AuditLog
from .config import AppConfig, Profile
from .prompts import DEFAULT_SYSTEM_PROMPT
from .provider import ProviderChain, ProviderError
from .safety import RiskAssessment
from .tools.base import ToolResult
from .tools.router import MODES, MODE_DESCRIPTIONS, ToolRouter

SLASH_COMMANDS: dict[str, str] = {
    "/help": "show this help",
    "/exit": "exit (same as Ctrl+D)",
    "/quit": "exit (same as Ctrl+D)",
    "/reset": "clear the conversation (keep settings)",
    "/clear": "clear the terminal screen",
    "/profile <name>": "switch provider profile (live)",
    "/profiles": "list configured profiles",
    "/config": "show the resolved config of the current profile",
    "/mode [safe|standard|yolo]": "show or set the permission mode",
    "/dry-run [on|off]": "preview-only mode: nothing is executed",
    "/tools": "list available tools",
    "/audit [n]": "show the last n audited tool calls (default 10)",
    "/model <name>": "override the model for this session",
    "/base-url <url>": "override the base URL for this session",
    "/about": "version and active provider info",
}

RELOAD_HOOK = Callable[[str], Profile]  # name -> Profile (cli applies overrides)


class Conversation:
    """Rolling message list for one session (long-term memory lands in v0.4)."""

    def __init__(self, system_prompt: str) -> None:
        self._system = system_prompt
        self._messages: list[dict[str, Any]] = []

    @property
    def system(self) -> str:
        return self._system

    def add_user(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self._messages.append({"role": "assistant", "content": text})

    def extend(self, messages: list[dict[str, Any]]) -> None:
        self._messages.extend(messages)

    def history(self) -> list[dict[str, Any]]:
        return list(self._messages)

    def reset(self) -> None:
        self._messages.clear()

    def set_system(self, prompt: str) -> None:
        self._system = prompt

    def messages(self) -> list[dict[str, Any]]:
        return [{"role": "system", "content": self._system}, *self._messages]

    @property
    def user_turns(self) -> int:
        return sum(1 for m in self._messages if m["role"] == "user")


class Repl:
    def __init__(
        self,
        config: AppConfig,
        profile: Profile,
        chain: ProviderChain,
        router: ToolRouter,
        agent: AgentLoop,
        *,
        console: Console | None = None,
        reload_profile: RELOAD_HOOK | None = None,
        plain: bool = False,
        stream: bool = True,
        audit: AuditLog | None = None,
    ) -> None:
        self.config = config
        self.profile = profile
        self.chain = chain
        self.router = router
        self.agent = agent
        self.console = console or Console()
        self.reload_profile = reload_profile
        self.plain = plain or not (
            self.console.is_terminal and self.console.file.isatty()
        )
        self.stream = stream
        self.audit = audit or router.audit
        self.conversation = Conversation(
            config.system_prompt or DEFAULT_SYSTEM_PROMPT
        )
        self._last_outcome: AgentOutcome | None = None

    # ------------------------------------------------------------------ loop

    def run(self) -> int:
        self._banner()
        try:
            if sys.stdin.isatty():
                self._run_prompt_toolkit()
            else:
                self._run_stdin()
        except KeyboardInterrupt:
            self._out("\n(interrupted — exit with /exit or Ctrl+D)")
        return 0

    def _run_prompt_toolkit(self) -> None:
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.completion import Completer, Completion
            from prompt_toolkit.history import FileHistory
        except ImportError:  # pragma: no cover - prompt_toolkit is a hard dep
            self._run_stdin()
            return

        history_path = Path.home() / ".termuxpilot" / "history"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        commands = list(SLASH_COMMANDS)

        class _SlashCompleter(Completer):
            def get_completions(self, document, complete_kind):
                text = document.text_before_cursor
                if text.startswith("/") and " " not in text and "\n" not in text:
                    for cmd in commands:
                        if cmd.startswith(text):
                            yield Completion(cmd, start_index=-len(text))

        session = PromptSession(
            history=FileHistory(str(history_path)),
            completer=_SlashCompleter(),
            enable_history_search=True,
        )
        while True:
            try:
                line = session.prompt("tp ❯ ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            if not self._handle_line(line):
                break

    def _run_stdin(self) -> None:
        """Non-TTY input (piped lines, tests)."""
        self._out("(stdin is not a TTY — reading lines until EOF; /exit to stop)\n")
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            if not self._handle_line(line):
                break

    def _handle_line(self, line: str) -> bool:
        """Process one input line.  Returns False when the REPL should exit."""
        if line.startswith("/"):
            return self._handle_command(line)
        self.ask(line)
        return True

    # --------------------------------------------------------------- commands

    def _handle_command(self, line: str) -> bool:
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/exit", "/quit"):
            return False
        if cmd == "/help":
            self._out("[bold]commands:[/bold]")
            for name, desc in SLASH_COMMANDS.items():
                self._out(f"  [cyan]{name:<30}[/cyan] {desc}")
            return True
        if cmd == "/reset":
            self.conversation.reset()
            self._out("[dim]conversation cleared[/dim]")
            return True
        if cmd == "/clear":
            self.console.clear()
            return True
        if cmd == "/profiles":
            self._show_profiles(current=self.profile.name)
            return True
        if cmd == "/profile":
            if not arg:
                self._out(f"current profile: [bold]{self.profile.name}[/bold]")
            elif self.reload_profile is None:
                self._out("[red]profile switching unavailable in this session[/red]")
            else:
                try:
                    self._switch_profile(arg)
                except Exception as exc:  # noqa: BLE001 - surfaced to the user
                    self._out(f"[red]cannot switch profile:[/red] {exc}")
            return True
        if cmd == "/config":
            self._show_config()
            return True
        if cmd == "/mode":
            return self._handle_mode(arg)
        if cmd == "/dry-run":
            if arg in ("on", "off"):
                self.router.ctx.dry_run = arg == "on"
            self._out(
                f"dry-run: [bold]{'on' if self.router.ctx.dry_run else 'off'}[/bold] "
                f"(nothing is executed while on)"
            )
            return True
        if cmd == "/tools":
            for tool in self.router.tools.values():
                self._out(f"  [cyan]{tool.name:<12}[/cyan] [{tool.category}] {tool.description[:100]}")
            return True
        if cmd == "/audit":
            n = int(arg) if arg.isdigit() else 10
            entries = self.audit.tail(n)
            if not entries:
                self._out("[dim]no audited tool calls yet[/dim]")
            for e in entries:
                status = "DENIED" if e.get("denied") else ("dry-run" if e.get("reason") == "dry-run" else "ok" if e.get("ok") else "fail")
                self._out(
                    f"  [{e.get('mode')}] {e.get('tool'):<12} "
                    f"risk={e.get('risk') or '-':<8} {status:<8} "
                    f"{str(e.get('args'))[:80]}"
                )
            return True
        if cmd == "/model":
            if not arg:
                self._out(f"model: [bold]{self.profile.primary.model or '(unset)'}[/bold]")
            else:
                from dataclasses import replace

                self.profile = replace(
                    self.profile, primary=replace(self.profile.primary, model=arg)
                )
                self._rebuild_chain()
                self._out(f"model set to [bold]{arg}[/bold] for this session")
            return True
        if cmd == "/base-url":
            if not arg:
                self._out(f"base_url: [bold]{self.profile.primary.base_url}[/bold]")
            else:
                from dataclasses import replace

                self.profile = replace(
                    self.profile, primary=replace(self.profile.primary, base_url=arg.rstrip("/"))
                )
                self._rebuild_chain()
                self._out(f"base_url set to [bold]{arg}[/bold] for this session")
            return True
        if cmd == "/about":
            self._out(
                f"TermuxPilot v{__version__} — profile [bold]{self.profile.name}[/bold], "
                f"model [bold]{self.profile.primary.model or '(unset)'}[/bold] @ "
                f"{self.profile.primary.base_url}, mode "
                f"[bold]{self.router.mode}[/bold], {len(self.chain)} provider(s) in chain"
            )
            return True
        self._out(f"[red]unknown command:[/red] {cmd}  (try /help)")
        return True

    def _handle_mode(self, arg: str) -> bool:
        if not arg:
            self._out(
                f"mode: [bold]{self.router.mode}[/bold] — {MODE_DESCRIPTIONS[self.router.mode]}"
            )
            self._out("[dim]switch: /mode safe | /mode standard | /mode yolo[/dim]")
            return True
        if arg not in MODES:
            self._out(f"[red]unknown mode:[/red] {arg} (use {' | '.join(MODES)})")
            return True
        self.router.mode = arg
        if arg == "yolo":
            self._out(
                "[bold red]⚠ yolo mode: the agent now executes tools WITHOUT "
                "asking (blocklist still applies).[/bold red]"
            )
        self._banner()
        return True

    def _rebuild_chain(self) -> None:
        self.chain = ProviderChain(
            self.profile.chain(),
            failover=lambda *a: self._announce_failover(*a),
        )
        self.agent.chain = self.chain  # keep the agent on the new chain

    def _switch_profile(self, name: str) -> None:
        assert self.reload_profile is not None
        self.profile = self.reload_profile(name)
        self._rebuild_chain()
        self._banner()

    def _show_profiles(self, current: str | None = None) -> None:
        from rich.table import Table

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("name")
        table.add_column("model")
        table.add_column("base_url")
        table.add_column("fallbacks")
        for name, prof in self.config.profiles.items():
            marker = " [green]•[/green]" if name == current else ""
            table.add_row(
                f"{name}{marker}",
                prof.primary.model or "-",
                prof.primary.base_url,
                str(len(prof.fallbacks)),
            )
        self.console.print(table)
        if self.config.default_profile:
            self._out(f"[dim]default profile: {self.config.default_profile}[/dim]")

    def _show_config(self) -> None:
        from rich.table import Table

        p = self.profile.primary
        table = Table(title=f"profile '{self.profile.name}'", show_header=False)
        table.add_column("key", style="cyan")
        table.add_column("value")
        for key, value in p.masked_dict().items():
            table.add_row(key, str(value) if value not in (None, "") else "-")
        self.console.print(table)
        if self.profile.fallbacks:
            self._out("[dim]fallback chain:[/dim]")
            for i, fb in enumerate(self.profile.fallbacks, start=1):
                self._out(f"  {i}. {fb.describe()}")

    # ------------------------------------------------------------------ turn

    def ask(self, user_text: str, *, pipe_context: str | None = None) -> bool:
        """Run one user turn through the agent loop.

        Returns True when the assistant produced a final answer.
        """
        message = user_text
        if pipe_context:
            message = (
                "Context pasted from stdin:\n```\n"
                f"{pipe_context.strip()}\n```\n\n"
                + user_text
            )
        self.conversation.add_user(message)
        start = time.monotonic()
        view = _TurnView(self.console, plain=self.plain, live=self.stream)

        def on_round(_n: int) -> None:
            view.begin_round()

        def on_tool_request(name: str, args: dict, risk: RiskAssessment, preview: str) -> None:
            view.close_round()
            self._render_tool_request(name, args, risk, preview)

        def on_tool_result(name: str, result: ToolResult) -> None:
            self._render_tool_result(name, result)

        def on_degrade() -> None:
            self._out(
                "[yellow]⚠ this endpoint rejected native tool calls — "
                "continuing in JSON mode.[/yellow]"
            )

        try:
            outcome = self.agent.run(
                self.conversation.system,
                self.conversation.history(),
                on_delta=view.on_delta,
                on_round=on_round,
                on_tool_request=on_tool_request,
                on_tool_result=on_tool_result,
                on_degrade=on_degrade,
                stream=self.stream,
            )
        except KeyboardInterrupt:
            view.close_round()
            self._out(" [yellow](interrupted)[/yellow]")
            return False
        except ProviderError as exc:
            view.close_round()
            self._show_provider_error(exc)
            return False

        self._last_outcome = outcome
        self.conversation.extend(outcome.transcript)
        self.conversation.add_assistant(outcome.final_text)
        view.close_round()
        view.finish(outcome.final_text)
        self._footer(outcome, start)
        return True

    def _footer(self, outcome: AgentOutcome, start: float) -> None:
        elapsed = time.monotonic() - start
        bits = [
            str(outcome.model or self.profile.primary.model or "unknown model"),
            f"via {outcome.provider or self.profile.name}",
            f"{elapsed:.1f}s",
            f"{len(outcome.tool_calls)} tool call(s)",
        ]
        if outcome.usage:
            tokens = outcome.usage.get("completion_tokens") or outcome.usage.get("total_tokens")
            if tokens:
                bits.append(f"{tokens} tokens")
        self._out(f"[dim]— {' · '.join(bits)}[/dim]")

    def _show_provider_error(self, exc: ProviderError) -> None:
        lines: list[str] = [f"[bold red]✗ {exc}[/bold red]"]
        for label, message in getattr(exc, "attempts", []) or []:
            if message:
                lines.append(f"[dim]  attempted {label}: {message}[/dim]")
        body = "\n".join(lines)
        if self.plain:
            self._out("PROVIDER ERROR: " + exc)
        else:
            self.console.print(Panel(body, title="provider error", border_style="red"))

    def _announce_failover(self, failed: str, next_label: str, error: ProviderError) -> None:
        self._out(
            f"[yellow]⚠ '{failed}' failed ({short_error(error)}) — "
            f"trying '{next_label}'[/yellow]"
        )

    # ------------------------------------------------------- tool rendering

    def _render_tool_request(
        self, name: str, args: dict, risk: RiskAssessment, preview: str
    ) -> None:
        badge = risk_badge(risk)
        if self.router.ctx.dry_run and not (name == "read_file" or name == "diff_files"):
            suffix = " [blue]dry-run[/blue]"
        elif self.router.mode == "yolo":
            suffix = " [magenta]auto (yolo)[/magenta]"
        else:
            suffix = ""
        title = f"tool: {name}  {badge}{suffix}"
        body = f"[{name}]\n{preview[:4000]}"
        if self.plain:
            self._out(f"TOOL {name} (risk: {risk.level})\n{preview[:2000]}")
        else:
            self.console.print(Panel(body, title=title, border_style=risk_border(risk), expand=False))

    def _render_tool_result(self, name: str, result: ToolResult) -> None:
        if result.denied:
            state = "[red]denied[/red]"
        elif not result.ok:
            state = f"[yellow]exit {result.exit_code if result.exit_code is not None else '?'}[/yellow]"
        else:
            state = "[green]ok[/green]"
        duration = f" · {result.duration:.1f}s" if result.duration else ""
        output = result.output
        if len(output) > 1200:
            output = output[:1200] + f"\n… [showing 1200 of {len(result.output)} chars]"
        if self.plain:
            self._out(f"RESULT {name}: {state.strip()} {duration}\n{output}")
        else:
            self.console.print(
                Panel(
                    f"[dim]{state} {duration}[/dim]\n{output}",
                    title=f"result: {name}",
                    border_style="grey50",
                    expand=False,
                )
            )

    def _confirm(self, name: str, preview: str, risk: RiskAssessment) -> bool:
        """Interactive y/N gate (standard mode).  Non-interactive -> False."""
        try:
            answer = input(
                f"  [cyan]tp[/cyan] run {name}? [yellow](risk: {risk.level})[/yellow] [y/N] "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer in ("y", "yes")

    # ----------------------------------------------------------------- layout

    def _banner(self) -> None:
        p = self.profile.primary
        mode_line = f"mode: [bold]{self.router.mode}[/bold] ({MODE_DESCRIPTIONS[self.router.mode]})"
        lines = [
            f"[bold]TermuxPilot[/bold] v{__version__} — profile [bold cyan]{self.profile.name}[/bold cyan]",
            f"model: [bold]{p.model or '(unset)'}[/bold] @ {p.base_url}",
        ]
        if self.profile.fallbacks:
            names = ", ".join(f.label for f in self.profile.fallbacks)
            lines.append(f"fallback: {len(self.profile.fallbacks)} ({names})")
        else:
            lines.append("fallback: none")
        lines.append(mode_line)
        if self.router.ctx.dry_run:
            lines.append("[blue]dry-run: ON (nothing will be executed)[/blue]")
        lines.append("[dim]type /help for commands · Ctrl+D to exit[/dim]")
        if self.plain:
            self._out("\n".join(strip_markup(l) for l in lines) + "\n")
        else:
            self.console.print(Panel("\n".join(lines), border_style="cyan", expand=False))

    def _out(self, text: str) -> None:
        if self.plain:
            print(strip_markup(text))
        else:
            self.console.print(text)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _is_write_tool(name: str) -> bool:
    return name in ("write_file", "edit_file", "move_file")


def risk_badge(risk: RiskAssessment) -> str:
    color = {"low": "green", "medium": "yellow", "high": "red", "critical": "bold red"}.get(
        risk.level, "white"
    )
    reasons = f" — {risk.reasons[0]}" if risk.reasons and risk.level != "low" else ""
    return f"[{color}]risk: {risk.level}{reasons}[/]"


def risk_border(risk: RiskAssessment) -> str:
    return {"low": "green", "medium": "yellow", "high": "red", "critical": "bold red"}.get(
        risk.level, "cyan"
    )


def short_error(exc: ProviderError) -> str:
    text = str(exc)
    return text if len(text) <= 90 else text[:87] + "..."


def strip_markup(text: str) -> str:
    """Crude [tag] removal for plain output (tags are short, no nesting)."""
    import re

    return re.sub(r"\[[^\[\]]*\]", "", text)


class _TurnView:
    """Render agent rounds:

    * ``plain``  -> raw text written straight to stdout (pipe-safe)
    * ``live``   -> rich Live markdown panel per round, growing per delta
    * otherwise  -> no-stream TTY: a status line per round, final panel once
    """

    def __init__(self, console: Console, *, plain: bool, live: bool) -> None:
        self.console = console
        self.plain = plain
        self.live = live and not plain
        self.text = ""
        self._emitted = False
        self._live: Live | None = None
        self._open = False

    def begin_round(self) -> None:
        self.close_round()
        self.text = ""
        self._emitted = False
        self._open = True
        if self.live:
            self._live = Live(
                self._pending(),
                console=self.console,
                vertical_overflow="visible",
                refresh_per_second=12,
            )
            self._live.start()
        elif not self.plain:
            self.console.print(Text("requesting…", style="dim"))

    def close_round(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None
        self._open = False

    def on_delta(self, delta: str) -> None:
        if not self._open and not self.plain:
            self.begin_round()
        self.text += delta
        if self.plain:
            self.console.file.write(delta)
            self.console.file.flush()
            self._emitted = True
            return
        if self._live is not None:
            self._live.update(self._content(self.text))

    def finish(self, final_text: str) -> None:
        if self.plain:
            if final_text and not self._emitted:
                self.console.file.write(final_text + "\n")
            return
        if self._live is not None:
            self._live.update(self._content(final_text or self.text))
        elif not self.live and final_text:
            self.console.print(self._content(final_text))

    def _pending(self) -> Any:
        return Panel(
            Spinner("dots", style="cyan") + Text("  thinking…", style="dim"),
            title="termuxpilot",
            border_style="cyan",
            expand=False,
        )

    def _content(self, text: str) -> Any:
        return Panel(
            Markdown(text) if text else Text("", style="dim"),
            title="termuxpilot",
            border_style="cyan",
            expand=False,
        )
