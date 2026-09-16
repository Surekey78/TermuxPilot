"""Interactive REPL (``tp -i``) and one-shot turn engine.

The same :meth:`Repl.ask` method drives both the interactive loop and the
one-shot ``tp "prompt"`` path, so behaviour (streaming, failover notices,
error reporting) is identical.

Rendering:
* TTY      -> rich ``Live`` markdown panel that grows with each SSE delta
* non-TTY  -> plain text deltas written straight to stdout (pipe-safe)
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
from .config import AppConfig, Profile
from .prompts import DEFAULT_SYSTEM_PROMPT
from .provider import ChatResult, ProviderChain, ProviderError

SLASH_COMMANDS: dict[str, str] = {
    "/help": "show this help",
    "/exit": "exit (same as Ctrl+D)",
    "/quit": "exit (same as Ctrl+D)",
    "/reset": "clear the conversation (keep settings)",
    "/clear": "clear the terminal screen",
    "/profile <name>": "switch provider profile (live)",
    "/profiles": "list configured profiles",
    "/config": "show the resolved config of the current profile",
    "/model <name>": "override the model for this session",
    "/base-url <url>": "override the base URL for this session",
    "/about": "version and active provider info",
}

RELOAD_HOOK = Callable[[str], Profile]  # name -> Profile (cli applies overrides)


class Conversation:
    """Rolling message list for one session (long-term memory lands in v0.4)."""

    def __init__(self, system_prompt: str) -> None:
        self._system = system_prompt
        self._messages: list[dict[str, str]] = []

    def add_user(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self._messages.append({"role": "assistant", "content": text})

    def reset(self) -> None:
        self._messages.clear()

    def set_system(self, prompt: str) -> None:
        self._system = prompt

    def messages(self) -> list[dict[str, str]]:
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
        *,
        console: Console | None = None,
        reload_profile: RELOAD_HOOK | None = None,
        plain: bool = False,
        stream: bool = True,
    ) -> None:
        self.config = config
        self.profile = profile
        self.chain = chain
        self.console = console or Console()
        self.reload_profile = reload_profile
        self.plain = plain or not (
            self.console.is_terminal and self.console.file.isatty()
        )
        self.stream = stream
        self.conversation = Conversation(
            config.system_prompt or DEFAULT_SYSTEM_PROMPT
        )
        self._last_result: ChatResult | None = None

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
                self._out(f"  [cyan]{name:<18}[/cyan] {desc}")
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
        if cmd == "/model":
            if not arg:
                self._out(f"model: [bold]{self.profile.primary.model or '(unset)'}[/bold]")
            else:
                from dataclasses import replace

                self.profile = replace(
                    self.profile, primary=replace(self.profile.primary, model=arg)
                )
                self.chain = ProviderChain(
                    self.profile.chain(), failover=self.chain.failover
                )
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
                self.chain = ProviderChain(
                    self.profile.chain(), failover=self.chain.failover
                )
                self._out(f"base_url set to [bold]{arg}[/bold] for this session")
            return True
        if cmd == "/about":
            self._out(
                f"TermuxPilot v{__version__} — profile [bold]{self.profile.name}[/bold], "
                f"model [bold]{self.profile.primary.model or '(unset)'}[/bold] @ "
                f"{self.profile.primary.base_url}, {len(self.chain)} provider(s) in chain"
            )
            return True
        self._out(f"[red]unknown command:[/red] {cmd}  (try /help)")
        return True

    def _switch_profile(self, name: str) -> None:
        assert self.reload_profile is not None
        profile = self.reload_profile(name)
        self.profile = profile
        self.chain = ProviderChain(
            profile.chain(),
            failover=lambda *a: self._announce_failover(*a),
        )
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
        """Run one user turn.  Returns True when the assistant answered."""
        message = user_text
        if pipe_context:
            message = (
                "Context pasted from stdin:\n```\n"
                f"{pipe_context.strip()}\n```\n\n"
                + user_text
            )
        self.conversation.add_user(message)
        start = time.monotonic()
        result: ChatResult | None = None

        view = _TurnView(self.console, plain=self.plain, live=self.stream)
        with view:
            try:
                result = self.chain.chat(
                    self.conversation.messages(),
                    stream=self.stream,
                    on_delta=view.on_delta,
                )
            except KeyboardInterrupt:
                view.interrupted()
                partial = view.partial_text
                if partial:
                    self.conversation.add_assistant(partial + "\n\n_(interrupted)_")
                self._out(" [yellow](interrupted)[/yellow]")
                return False
            except ProviderError as exc:
                view.failed()
                self._show_provider_error(exc)
                return False

        self._last_result = result
        if not self.stream:
            view.text = result.content
        self.conversation.add_assistant(result.content or json_tools_summary(result))
        view.finish()
        self._footer(result, start)
        return True

    def _footer(self, result: ChatResult, start: float) -> None:
        elapsed = time.monotonic() - start
        bits = [f"{result.model or self.profile.primary.model or 'unknown model'}",
                f"via {result.provider or self.profile.name}", f"{elapsed:.1f}s"]
        if result.usage:
            tokens = result.usage.get("completion_tokens") or result.usage.get("total_tokens")
            if tokens:
                bits.append(f"{tokens} tokens")
        self._out(f"[dim]— {' · '.join(str(b) for b in bits)}[/dim]")

    def _show_provider_error(self, exc: ProviderError) -> None:
        lines: list[str] = [f"[bold red]✗ {exc}[/bold red]"]
        for label, message in getattr(exc, "attempts", []) or []:
            if message:
                lines.append(f"[dim]  attempted {label}: {message}[/dim]")
        body = "\n".join(lines)
        self.console.print(Panel(body, title="provider error", border_style="red"))

    def _announce_failover(self, failed: str, next_label: str, error: ProviderError) -> None:
        self._out(
            f"[yellow]⚠ '{failed}' failed ({short_error(error)}) — "
            f"trying '{next_label}'[/yellow]"
        )

    # ----------------------------------------------------------------- layout

    def _banner(self) -> None:
        p = self.profile.primary
        lines = [
            f"[bold]TermuxPilot[/bold] v{__version__} — profile [bold cyan]{self.profile.name}[/bold cyan]",
            f"model: [bold]{p.model or '(unset)'}[/bold] @ {p.base_url}",
        ]
        if self.profile.fallbacks:
            names = ", ".join(f.label for f in self.profile.fallbacks)
            lines.append(f"fallback: {len(self.profile.fallbacks)} ({names})")
        else:
            lines.append("fallback: none")
        lines.append("[dim]type /help for commands · Ctrl+D to exit[/dim]")
        if self.plain:
            self._out("\n".join(strip_markup(l) for l in lines) + "\n")
        else:
            self.console.print(
                Panel("\n".join(lines), border_style="cyan", expand=False)
            )

    def _out(self, text: str) -> None:
        if self.plain:
            print(strip_markup(text))
        else:
            self.console.print(text)


def json_tools_summary(result: ChatResult) -> str:
    """Human-readable stand-in when a response has only tool calls (v0.2+)."""
    import json

    calls = [
        {"name": tc.name, "args": tc.arguments_dict if tc.arguments else {}}
        for tc in result.tool_calls
    ]
    return "_(requested tool calls: " + json.dumps(calls) + ")_"


def short_error(exc: ProviderError) -> str:
    text = str(exc)
    return text if len(text) <= 90 else text[:87] + "..."


def strip_markup(text: str) -> str:
    """Crude [tag] removal for plain output (tags are short, no nesting)."""
    import re

    return re.sub(r"\[[a-z /=#*]+\]", "", text)


class _TurnView:
    """Render one turn:

    * ``plain``  -> raw text written straight to stdout (pipe-safe)
    * ``live``   -> rich Live markdown panel that grows with each delta
    * otherwise  -> no-stream TTY: a status line, then the final panel once
    """

    def __init__(self, console: Console, *, plain: bool, live: bool) -> None:
        self.console = console
        self.plain = plain
        self.live = live and not plain
        self.text = ""
        self._emitted = False
        self._live: Live | None = None

    def __enter__(self) -> "_TurnView":
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
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    def on_delta(self, delta: str) -> None:
        self.text += delta
        if self.plain:
            self.console.file.write(delta)
            self.console.file.flush()
            self._emitted = True
            return
        if self._live is not None:
            self._live.update(self._content())

    @property
    def partial_text(self) -> str:
        return self.text

    def _pending(self) -> Any:
        return Panel(
            Spinner("dots", style="cyan") + Text("  requesting…", style="dim"),
            title="termuxpilot",
            border_style="cyan",
            expand=False,
        )

    def _content(self) -> Any:
        return Panel(
            Markdown(self.text) if self.text else Text("", style="dim"),
            title="termuxpilot",
            border_style="cyan",
            expand=False,
        )

    def interrupted(self) -> None:
        if self._live is not None and self.text:
            self._live.update(
                self._content_with_footer(Text("(interrupted)", style="yellow"))
            )

    def failed(self) -> None:
        if self._live is not None and not self.text:
            self._live.update(Text("", style="dim"))

    def finish(self) -> None:
        if self.plain:
            if self.text and not self._emitted:
                # no-stream: nothing was written incrementally
                self.console.file.write(self.text)
            if self.text:
                self.console.file.write("\n")
            return
        if self._live is not None and self.text:
            self._live.update(self._content())
        elif not self.live and self.text:
            # no-stream TTY: render the finished answer once
            self.console.print(self._content())

    def _content_with_footer(self, footer: Text) -> Any:
        from rich.console import Group

        return Panel(
            Group(Markdown(self.text), footer),
            title="termuxpilot",
            border_style="yellow",
            expand=False,
        )
