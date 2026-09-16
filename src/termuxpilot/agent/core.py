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
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from ..provider import ChatResult, HttpError, ProviderChain
from ..prompts import tool_instructions
from ..tools.base import ToolResult
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


class AgentLoop:
    def __init__(
        self,
        chain: ProviderChain,
        router: ToolRouter,
        *,
        max_rounds: int = 8,
        function_calling: str = "auto",
    ) -> None:
        if function_calling not in ("auto", "native", "json"):
            raise ValueError("function_calling must be auto|native|json")
        self.chain = chain
        self.router = router
        self.max_rounds = max(1, max_rounds)
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
        added: list[dict[str, Any]] = []
        records: list[ToolCallRecord] = []
        result: ChatResult | None = None

        def current_messages() -> list[dict[str, Any]]:
            return [sysmsg, *history, *added]

        for round_no in range(self.max_rounds):
            if on_round:
                on_round(round_no)

            if self.mode == "native":
                try:
                    result = self.chain.chat(
                        current_messages(),
                        tools=self.router.specs(),
                        on_delta=on_delta if stream else None,
                        stream=stream,
                    )
                except HttpError as exc:
                    if self._is_tools_rejection(exc):
                        self.mode = "json"
                        sysmsg = self._system_message(system_prompt)
                        if on_degrade:
                            on_degrade()
                        continue  # retry this round in JSON mode
                    raise
            else:
                # JSON mode: no streaming — the raw output is a JSON object,
                # and the UI renders the extracted final answer instead.
                result = self.chain.chat(
                    current_messages(), json_mode=True, on_delta=None, stream=stream
                )

            added.append(_model_message(self.mode, result))
            tool_names = self._extract_tool_names(result)

            if not tool_names:  # final answer
                final = self._extract_final(result)
                if not final.strip():
                    final = "_(the model returned an empty response)_"
                return AgentOutcome(
                    final_text=final.strip(),
                    rounds=round_no + 1,
                    tool_calls=records,
                    transcript=list(added),
                    model=result.model,
                    provider=result.provider,
                    usage=result.usage,
                )

            for call_id, name, raw_args, args in self._extract_calls(result):
                args = args if isinstance(args, dict) else {}
                tool_result = self.router.execute(name, raw_args)
                records.append(
                    ToolCallRecord(
                        name=name,
                        args=args,
                        ok=tool_result.ok and not tool_result.denied,
                        exit_code=tool_result.exit_code,
                        output=tool_result.output[:500],
                    )
                )
                if on_tool_result:
                    on_tool_result(name, tool_result)
                if self.mode == "native":
                    added.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": tool_result.model_json(),
                        }
                    )
                else:
                    added.append(
                        {
                            "role": "user",
                            "content": f"[tool_result {name}]\n{tool_result.model_json()}",
                        }
                    )

        return AgentOutcome(
            final_text=(
                "_(stopped: reached the maximum number of tool rounds "
                f"({self.max_rounds}) without a final answer)_"
            ),
            rounds=self.max_rounds,
            tool_calls=records,
            transcript=list(added),
            truncated=True,
            model=result.model if result else None,
            provider=result.provider if result else None,
            usage=result.usage if result else None,
        )

    # -------------------------------------------------------------- helpers

    def _system_message(self, base_prompt: str) -> dict[str, str]:
        return {
            "role": "system",
            "content": tool_instructions(base_prompt, self.mode, self.router),
        }

    @staticmethod
    def _is_tools_rejection(exc: HttpError) -> bool:
        return exc.status == 400 and "tool" in str(exc).lower()

    @staticmethod
    def _extract_tool_names(result: ChatResult) -> list[str]:
        if result.tool_calls:
            return [tc.name for tc in result.tool_calls]
        parsed = parse_agent_json(result.content)
        if not parsed:
            return []
        tool = parsed.get("tool")
        return [tool] if tool else []

    def _extract_calls(self, result: ChatResult) -> list[tuple[str, str, Any, Any]]:
        """Yield (tool_call_id, name, raw_args, parsed_args) for this round."""
        if result.tool_calls:
            return [
                (tc.id, tc.name, tc.arguments, _safe_args(tc)) for tc in result.tool_calls
            ]
        parsed = parse_agent_json(result.content) or {}
        return [("", str(parsed.get("tool") or "unknown"), parsed.get("args"), parsed.get("args"))]

    @staticmethod
    def _extract_final(result: ChatResult) -> str:
        if not result.tool_calls:
            parsed = parse_agent_json(result.content)
            if parsed and isinstance(parsed.get("response"), str):
                return parsed["response"]
            if parsed and isinstance(parsed.get("thought"), str) and not parsed.get("tool"):
                return parsed["thought"]
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
