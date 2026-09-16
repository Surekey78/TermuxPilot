"""Agent core — the ReAct loop.

One *round* = one model call.  The loop runs until the model produces a final
answer (no tool call) or ``max_rounds`` is reached.

Function-calling strategy (config ``agent.function_calling``):
* ``native`` — send OpenAI-style ``tools``; never degrade (errors surface).
* ``json``   — never send tools; the model answers with a strict JSON object
               (``{"thought", "tool", "args", "response"}``) parsed here.
* ``auto``   (default) — try native first; if the endpoint rejects the
               ``tools`` parameter with HTTP 400, degrade to JSON mode for the
               rest of the session (announced once via ``on_degrade``).

Message bookkeeping: ``messages = [system] + history + added`` where
``added`` grows with every assistant/tool message this run produced; the
whole ``added`` list is returned as the transcript so the REPL can fold it
into the rolling conversation.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from ..provider import ChatResult, HttpError, ProviderChain, ProviderError
from ..prompts import tool_instructions
from ..tools.base import ToolResult, truncate_output
from ..tools.router import ToolRouter

CodeFence = re.compile(r"^```[a-zA-Z]*\n?|\n?```$")


def parse_agent_json(text: str) -> dict[str, Any] | None:
    """Strict-ish parse of the JSON-mode contract; None when not an object."""
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = CodeFence.sub("", stripped)
        stripped = stripped.strip()
    try:
        data = json.loads(stripped)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(stripped[start : end + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            return None
    return None


@dataclass
class ToolCallRecord:
    name: str
    args: dict[str, Any]
    ok: bool
    exit_code: int | None = None
    output: str = ""
    denied: bool = False
    timed_out: bool = False
    truncated: bool = False


@dataclass
class AgentOutcome:
    final_text: str
    rounds: int
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    truncated: bool = False
    model: str | None = None
    provider: str | None = None
    usage: dict[str, Any] | None = None
    stop_reason: str | None = None

    @property
    def completed(self) -> bool:
        return not self.truncated

    @property
    def status(self) -> str:
        return "completed" if self.completed else "incomplete"


class AgentLoop:
    def __init__(
        self,
        chain: ProviderChain,
        router: ToolRouter,
        *,
        max_rounds: int = 8,
        max_tool_calls: int = 32,
        max_context_chars: int = 200_000,
        function_calling: str = "auto",
    ) -> None:
        if function_calling not in ("auto", "native", "json"):
            raise ValueError("function_calling must be auto|native|json")
        self.chain = chain
        self.router = router
        self.max_rounds = max(1, max_rounds)
        self.max_tool_calls = max(1, max_tool_calls)
        self.max_context_chars = max(1, max_context_chars)
        # In-memory progress is available even when a provider fails or the
        # user interrupts. This is not durable checkpoint/resume storage.
        self.last_outcome: AgentOutcome | None = None
        self.function_calling = function_calling
        self.mode = "json" if function_calling == "json" else "native"

    def run(
        self,
        system_prompt: str,
        history: list[dict[str, Any]],
        *,
        on_delta: Callable[[str], None] | None = None,
        on_round: Callable[[int], None] | None = None,
        on_tool_request: Callable[[str, dict, Any, str], None] | None = None,
        on_tool_result: Callable[[str, ToolResult], None] | None = None,
        on_degrade: Callable[[], None] | None = None,
        stream: bool = True,
    ) -> AgentOutcome:
        self.router.request_hook = on_tool_request
        sysmsg = self._system_message(system_prompt)
        outcome = AgentOutcome(final_text="", rounds=0, truncated=True, stop_reason="running")
        self.last_outcome = outcome
        added = outcome.transcript
        records = outcome.tool_calls

        def stop(reason: str, explanation: str) -> AgentOutcome:
            outcome.stop_reason = reason
            outcome.final_text = f"_(stopped: {explanation})_"
            return outcome

        def current_messages() -> list[dict[str, Any]]:
            return [sysmsg, *history, *added]

        def fits_context() -> bool:
            # This is an explicit serialized-character guard, not a claim to
            # know an arbitrary provider's tokenizer or context window.
            size = len(json.dumps(self.router.specs(), ensure_ascii=False))
            for message in current_messages():
                size += len(json.dumps(message, ensure_ascii=False))
                if size > self.max_context_chars:
                    return False
            return size <= self.max_context_chars

        def remember(call, tool_result: ToolResult) -> None:
            call_id, name, _, args = call
            records.append(ToolCallRecord(
                name=name,
                args=self.router.sanitize(args if isinstance(args, dict) else {}),
                ok=tool_result.ok and not tool_result.denied,
                exit_code=tool_result.exit_code,
                output=truncate_output(tool_result.output, 500),
                denied=tool_result.denied,
                timed_out=tool_result.timed_out,
                truncated=tool_result.truncated or len(tool_result.output) > 500,
            ))
            if self.mode == "native":
                added.append({
                    "role": "tool", "tool_call_id": call_id,
                    "content": tool_result.model_json(),
                })
            else:
                added.append({
                    "role": "user",
                    "content": f"[tool_result {name}]\n{tool_result.model_json()}",
                })

        for round_no in range(self.max_rounds):
            if not fits_context():
                return stop("max_context_chars", "message/tool context exceeds agent.max_context_chars; "
                            "use smaller reads, reset the conversation, or adjust the size limit")
            if on_round:
                on_round(round_no)
            try:
                if self.mode == "native":
                    try:
                        result = self.chain.chat(
                            current_messages(), tools=self.router.specs(),
                            on_delta=on_delta if stream else None, stream=stream,
                        )
                    except HttpError as exc:
                        if self.function_calling != "auto" or not self._is_tools_rejection(exc):
                            raise
                        self.mode = "json"
                        sysmsg = self._system_message(system_prompt)
                        if on_degrade:
                            on_degrade()
                        if not fits_context():
                            return stop("max_context_chars", "JSON-mode context exceeds the size limit")
                        # Capability negotiation does not consume a tool round.
                        result = self.chain.chat(current_messages(), json_mode=True, stream=stream)
                else:
                    result = self.chain.chat(current_messages(), json_mode=True, stream=stream)
            except ProviderError:
                stop("provider_error", "provider failed; earlier tool results are preserved in this session")
                raise
            except KeyboardInterrupt:
                stop("interrupted", "interrupted while waiting for the provider")
                raise

            outcome.rounds = round_no + 1
            outcome.model, outcome.provider = result.model, result.provider
            if result.usage:
                outcome.usage = _merge_usage(outcome.usage or {}, result.usage)
            if result.finish_reason in {"length", "interrupted", "error", "content_filter"}:
                # Never execute possibly incomplete streamed tool arguments.
                if result.content:
                    added.append({"role": "assistant", "content": result.content})
                return stop("incomplete_response", f"provider response ended with {result.finish_reason!r}")

            calls = self._extract_calls(result)
            if not calls:
                final = self._extract_final(result)
                if not final.strip():
                    return stop("empty_response", "the model did not return a final answer")
                added.append(_model_message(self.mode, result))
                outcome.final_text = final.strip()
                outcome.truncated = False
                outcome.stop_reason = None
                return outcome

            if len(records) + len(calls) > self.max_tool_calls:
                # Reject the batch before any side effect. Do not add an
                # assistant message containing unanswered tool-call IDs.
                return stop("max_tool_calls", "this batch would exceed agent.max_tool_calls")
            added.append(_model_message(self.mode, result))
            for index, call in enumerate(calls):
                _, name, raw_args, _ = call
                remembered = False
                try:
                    tool_result = self.router.execute(name, raw_args)
                    remember(call, tool_result)  # persist before UI callbacks
                    remembered = True
                    if on_tool_result:
                        on_tool_result(name, tool_result)
                except KeyboardInterrupt:
                    if not remembered:
                        remember(call, ToolResult(ok=False, output=(
                            "interrupted; execution may have had partial side effects. "
                            "Inspect state and ask before retrying a mutation."
                        )))
                    for pending in calls[index + 1:]:
                        remember(pending, ToolResult(ok=False, denied=True,
                                                    output="not executed: turn interrupted"))
                    stop("interrupted", "tool execution interrupted; inspect partial effects before retrying")
                    raise

        return stop("max_tool_rounds", "reached the maximum number of tool rounds "
                    f"({self.max_rounds}) without a final answer")

    # -------------------------------------------------------------- helpers

    def _system_message(self, base_prompt: str) -> dict[str, str]:
        return {
            "role": "system",
            "content": tool_instructions(base_prompt, self.mode, self.router),
        }

    @staticmethod
    def _is_tools_rejection(exc: HttpError) -> bool:
        return exc.status == 400 and "tool" in str(exc).lower()

    def _extract_calls(self, result: ChatResult) -> list[tuple[str, str, Any, Any]]:
        """Only JSON mode may interpret response text as a tool request."""
        if self.mode == "native":
            return [(tc.id, tc.name, tc.arguments, _safe_args(tc)) for tc in result.tool_calls]
        parsed = parse_agent_json(result.content) or {}
        if not parsed.get("tool"):
            return []
        return [("", str(parsed["tool"]), parsed.get("args"), parsed.get("args"))]

    def _extract_final(self, result: ChatResult) -> str:
        if self.mode == "json":
            parsed = parse_agent_json(result.content)
            # Some compatible servers ignore response_format on the final
            # answer. Preserve the existing plain-text degradation behavior.
            if parsed is None:
                return result.content
            response = parsed.get("response")
            return response if isinstance(response, str) else ""
        return result.content


def _model_message(mode: str, result: ChatResult) -> dict[str, Any]:
    if mode == "native" and result.tool_calls:
        return {
            "role": "assistant",
            "content": result.content or None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in result.tool_calls
            ],
        }
    return {"role": "assistant", "content": result.content}


def _safe_args(tc) -> dict[str, Any]:
    try:
        return tc.arguments_dict
    except Exception:  # noqa: BLE001 - malformed args flow to the router
        return {}


def _merge_usage(total: dict[str, Any], usage: dict[str, Any]) -> dict[str, Any]:
    """Sum reported numeric counters (including nested token details)."""
    merged = dict(total)
    for key, value in usage.items():
        if isinstance(value, dict):
            previous = merged.get(key)
            merged[key] = _merge_usage(previous if isinstance(previous, dict) else {}, value)
        elif type(value) is int or (type(value) is float and math.isfinite(value)):
            previous = merged.get(key, 0)
            merged[key] = (previous if type(previous) in (int, float) else 0) + value
    return merged
