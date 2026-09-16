"""Command-line interface for TermuxPilot (``tp``).

Surface (v0.1):
    tp "prompt"                        one-shot, streamed to the terminal
    cat error.log | tp "why?"          stdin attached as context
    tp -i                              interactive REPL
    tp --profile local -i              switch config profile
    tp --list-profiles                 show all profiles
    tp config init / tp config show    manage ~/.termuxpilot/config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rich.console import Console

from . import __version__
from .config import (
    AppConfig,
    ConfigError,
    Profile,
    default_config_path,
    load_config,
    select_profile,
    write_sample_config,
)
from .provider import ProviderChain, ProviderError
from .repl import Repl

MAX_STDIN_CHARS = 1_000_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tp",
        description="TermuxPilot — CLI-first autonomous AI agent for Termux.",
        epilog=(
            "Examples:\n"
            '  tp "summarize this log"\n'
            '  cat error.log | tp "why is this failing?"\n'
            "  tp -i\n"
            "  tp --profile local -i\n"
            "  tp config init\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("prompt", nargs="*", help="one-shot prompt (joined)")
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="start the interactive REPL (default when no prompt)")
    parser.add_argument("--profile", help="config profile to use (see --list-profiles)")
    parser.add_argument("--model", help="override the model for this run")
    parser.add_argument("--base-url", help="override the provider base URL for this run")
    parser.add_argument("--api-key", help="override the API key for this run")
    parser.add_argument("--temperature", type=float, help="override temperature")
    parser.add_argument("--max-tokens", type=int, help="override max_tokens")
    parser.add_argument("--timeout", type=float, help="request timeout in seconds")
    parser.add_argument("--system-prompt", help="override the system prompt")
    parser.add_argument("--no-stream", action="store_true",
                        help="wait for the full response before printing")
    parser.add_argument("--plain", action="store_true",
                        help="plain-text output (no rich rendering)")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="print the result as JSON (implies --no-stream)")
    parser.add_argument("--list-profiles", action="store_true",
                        help="list configured profiles and exit")
    parser.add_argument("--config-path", help="explicit config file path")
    parser.add_argument("--mode", choices=("safe", "standard", "yolo"),
                        help="permission mode (overrides tools.mode from config)")
    parser.add_argument("--dry-run", action="store_true",
                        help="preview tool calls only; nothing is executed")
    parser.add_argument("--version", action="version", version=f"tp {__version__}")
    return parser


def build_session(config: AppConfig, profile: Profile, chain: ProviderChain, *,
                  mode: str | None, dry_run: bool, confirm=None, announce=None):
    """ExecutionContext + ToolRouter + AgentLoop for one run."""
    from .agent import AgentLoop
    from .audit import AuditLog
    from .tools import build_default_tools
    from .tools.base import ExecutionContext
    from .tools.router import ToolRouter

    t = config.tools
    ctx = ExecutionContext(
        shell_timeout=t.shell_timeout,
        shell_max_timeout=t.shell_max_timeout,
        shell_kill_grace=t.shell_kill_grace,
        max_output_chars=t.max_output_chars,
        shell_workdir=t.shell_workdir,
        redact_secrets=t.redact_secrets,
        protected_paths=t.protected_paths,
        dry_run=dry_run or t.dry_run,
    )
    router = ToolRouter(
        build_default_tools(),
        ctx,
        mode=mode or t.mode,
        blocklist=t.blocklist,
        allowlist=t.allowlist,
        audit=AuditLog(),
        confirm=confirm,
        announce=announce,
    )
    agent = AgentLoop(
        chain,
        router,
        max_rounds=config.agent.max_tool_rounds,
        max_tool_calls=config.agent.max_tool_calls,
        max_context_chars=config.agent.max_context_chars,
        function_calling=config.agent.function_calling,
    )
    return router, agent


def _config_subparser(name: str, help_text: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"tp config {name}", description=help_text)
    parser.add_argument("--config-path", help="explicit config file path")
    if name == "init":
        parser.add_argument("--force", action="store_true",
                            help="overwrite if it exists")
    return parser


def _split_config_argv(argv: list[str]) -> tuple[list[str], list[str]] | None:
    """Split ``tp [--config-path P] config <sub> [flags]`` into (prefix, rest).

    Only ``--config-path`` (flag or ``=`` form) may precede the ``config``
    subcommand; anything else means this is not a config invocation.
    """
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--config-path":
            i += 2
        elif token.startswith("--config-path="):
            i += 1
        elif token == "config":
            return argv[:i], argv[i + 1:]
        else:
            return None
    return None


def _overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "base_url": args.base_url,
        "api_key": args.api_key,
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
    }


def _make_failover_printer(console: Console, plain: bool):
    def failover(failed: str, next_label: str, error: ProviderError) -> None:
        text = (
            f"[yellow]⚠ '{failed}' failed — trying '{next_label}' "
            f"({type(error).__name__})[/yellow]"
        )
        if plain:
            print(f"WARNING: '{failed}' failed — trying '{next_label}'", file=sys.stderr)
        else:
            console.print(text)

    return failover


def _load(args: argparse.Namespace) -> tuple[AppConfig, Profile]:
    try:
        config = load_config(args.config_path)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    try:
        profile = select_profile(config, args.profile, **_overrides_from_args(args))
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    return config, profile


def _list_profiles(config: AppConfig, current: str | None, *, plain: bool) -> int:
    from rich.table import Table

    if plain or not sys.stdout.isatty():
        for name, prof in config.profiles.items():
            marker = "*" if name == current else " "
            fb = f" +{len(prof.fallbacks)} fallback(s)" if prof.fallbacks else ""
            print(f" {marker} {name:<12} {prof.primary.model or '-':<40} "
                  f"{prof.primary.base_url}{fb}")
        print(f"\n  default profile: {config.default_profile}")
        return 0

    table = Table(title="profiles", show_header=True, header_style="bold cyan")
    table.add_column("name")
    table.add_column("model")
    table.add_column("base_url")
    table.add_column("fallbacks")
    for name, prof in config.profiles.items():
        table.add_row(
            f"{name} [green]•[/green]" if name == current else name,
            prof.primary.model or "-",
            prof.primary.base_url,
            str(len(prof.fallbacks)),
        )
    Console().print(table)
    Console().print(f"[dim]default profile: {config.default_profile}[/dim]")
    return 0


def _config_command(argv: list[str]) -> int:
    """Handle ``tp config {init,show}`` — ``--config-path`` may sit anywhere."""
    sub_idx = next(
        (i for i, token in enumerate(argv) if token in ("init", "show")), None
    )
    if sub_idx is None:
        print("usage: tp config {init,show} [--config-path PATH]", file=sys.stderr)
        return 2
    cmd = argv[sub_idx]
    flag_args = [t for i, t in enumerate(argv) if i != sub_idx]
    if cmd == "init":
        ns = _config_subparser(
            "init", "write a sample config to ~/.termuxpilot/config.yaml"
        ).parse_args(flag_args)
        path = (
            Path(ns.config_path).expanduser()
            if ns.config_path
            else default_config_path()
        )
        try:
            written = write_sample_config(path, force=ns.force)
        except ConfigError as exc:
            print(f"config error: {exc}", file=sys.stderr)
            return 2
        print(f"wrote {written}")
        print("edit it, export your API key (export GROQ_API_KEY=...), then run: tp -i")
        return 0
    # cmd == "show"
    ns = _config_subparser("show", "show the resolved config").parse_args(flag_args)
    try:
        config = load_config(ns.config_path)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title=f"config ({config.path})", show_header=False)
    table.add_column("key", style="cyan")
    table.add_column("value")
    for name, prof in config.profiles.items():
        p = prof.primary
        table.add_row(f"[bold]{name}[/bold].model", p.model or "-")
        table.add_row(f"[bold]{name}[/bold].base_url", p.base_url)
        table.add_row(f"[bold]{name}[/bold].api_key",
                      "***set***" if p.has_api_key() else "-")
        table.add_row(f"[bold]{name}[/bold].fallbacks",
                      ", ".join(f.label for f in prof.fallbacks) or "-")
    table.add_row("default_profile", config.default_profile)
    table.add_row("system_prompt", "custom" if config.system_prompt else "built-in")
    for key, value in {
        "tools.mode": config.tools.mode,
        "tools.max_output_chars": config.tools.max_output_chars,
        "tools.shell.timeout": config.tools.shell_timeout,
        "tools.shell.max_timeout": config.tools.shell_max_timeout,
        "tools.shell.kill_grace": config.tools.shell_kill_grace,
        "agent.max_tool_rounds": config.agent.max_tool_rounds,
        "agent.max_tool_calls": config.agent.max_tool_calls,
        "agent.max_context_chars": config.agent.max_context_chars,
    }.items():
        table.add_row(key, str(value))
    console.print(table)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    split = _split_config_argv(argv)
    if split is not None:
        prefix, rest = split
        return _config_command(prefix + rest)

    args = build_parser().parse_args(argv)

    config, profile = _load(args)
    if args.list_profiles:
        return _list_profiles(config, profile.name, plain=args.plain)

    console = Console() if not args.plain else Console(no_color=True, highlight=False)
    chain = ProviderChain(
        profile.chain(),
        failover=None if args.plain else _make_failover_printer(console, args.plain),
    )
    if args.system_prompt:
        config = type(config)(
            profiles=config.profiles,
            default_profile=config.default_profile,
            system_prompt=args.system_prompt,
            path=config.path,
            tools=config.tools,
            agent=config.agent,
        )
    def _announce(text: str) -> None:
        if args.plain:
            return
        if args.as_json:
            print(text, file=sys.stderr)  # keep stdout machine-readable
        else:
            console.print(f"[dim]{text}[/dim]")

    router, agent = build_session(
        config, profile, chain,
        mode=args.mode, dry_run=args.dry_run,
        confirm=None,  # the REPL wires its interactive y/N gate
        announce=_announce,
    )

    pipe_context: str | None = None
    has_prompt = bool(args.prompt)
    if has_prompt and not sys.stdin.isatty():
        data = sys.stdin.read(MAX_STDIN_CHARS + 1)
        if len(data) > MAX_STDIN_CHARS:
            error = "stdin exceeds 1,000,000 characters; pass a file path and use line windows instead"
            if args.as_json:
                print(json.dumps({"ok": False, "status": "error", "error": error}))
            else:
                print(f"usage error: {error}", file=sys.stderr)
            return 2
        if data and data.strip():
            pipe_context = data

    interactive = args.interactive or (not has_prompt and sys.stdin.isatty())

    if args.as_json:
        return _run_json(args, config, profile, chain, router, agent,
                         has_prompt, pipe_context)

    repl = Repl(config, profile, chain, router, agent, console=console,
                plain=args.plain, stream=not args.no_stream,
                reload_profile=_reload_hook(args, config))
    router.confirm = repl._confirm  # interactive y/N gate (standard mode)
    if interactive:
        return repl.run()

    if not has_prompt:
        print(
            "error: no prompt given.\n"
            "  one-shot:  tp \"your question\"\n"
            "  with pipe: cat error.log | tp \"why is this failing?\"\n"
            "  chat:      tp -i",
            file=sys.stderr,
        )
        return 2

    prompt = " ".join(args.prompt)
    repl.ask(prompt, pipe_context=pipe_context)
    return repl.last_exit_code


def _reload_hook(args: argparse.Namespace, config: AppConfig):
    def reload(name: str) -> Profile:
        return select_profile(config, name, **_overrides_from_args(args))

    return reload


def _run_json(
    args: argparse.Namespace,
    config: AppConfig,
    profile: Profile,
    chain: ProviderChain,
    router,
    agent,
    has_prompt: bool,
    pipe_context: str | None,
) -> int:
    if not has_prompt:
        payload = {"ok": False, "error": "no prompt given (pipe requires a prompt)"}
        print(json.dumps(payload))
        return 2
    prompt = " ".join(args.prompt)
    if pipe_context:
        prompt = (
            "Context pasted from stdin:\n```\n"
            f"{pipe_context.strip()}\n```\n\n" + prompt
        )
    history = [{"role": "user", "content": prompt}]
    try:
        outcome = agent.run(
            config.system_prompt or "",
            history,
            on_delta=None,
            stream=False,  # --json promises non-streaming provider requests
        )
    except KeyboardInterrupt:
        print(json.dumps({
            "ok": False, "status": "interrupted",
            "error": "interrupted; inspect partial side effects before retrying",
            "tool_calls": _tool_records(agent.last_outcome),
        }, ensure_ascii=False))
        return 130
    except ProviderError as exc:
        print(json.dumps({
            "ok": False,
            "status": "error",
            "error": str(exc),
            "attempts": [
                {"provider": label, "error": message} for label, message in exc.attempts
            ],
            "tool_calls": _tool_records(agent.last_outcome),
        }, ensure_ascii=False))
        return 1
    print(json.dumps({
        "ok": outcome.completed,
        "status": outcome.status,
        "truncated": outcome.truncated,
        "stop_reason": outcome.stop_reason,
        "content": outcome.final_text,
        "model": outcome.model,
        "provider": outcome.provider,
        "rounds": outcome.rounds,
        "usage": outcome.usage,
        "tool_calls": _tool_records(outcome),
    }, ensure_ascii=False))
    return 0 if outcome.completed else 3


def _tool_records(outcome) -> list[dict[str, Any]]:
    return [asdict(record) for record in outcome.tool_calls] if outcome is not None else []


if __name__ == "__main__":
    sys.exit(main())
