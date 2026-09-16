"""OpenAI-compatible chat-completions client (httpx, SSE streaming).

One client = one endpoint (``base_url`` + ``api_key`` + ``model``).  See
:mod:`termuxpilot.provider.fallback` for the retry chain across endpoints.

Features:
* SSE streaming with ``on_delta`` callbacks (text and tool-call deltas).
* Native ``tools`` passthrough (function calling) when the endpoint supports
  it; ``json_mode`` adds ``response_format={"type":"json_object"}`` for
  graceful degradation on endpoints without tool support.
* Tolerates servers that (a) refuse ``stream_options.include_usage`` with a
  400, and (b) answer a streaming request with a plain JSON body.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from ..config import ProviderSettings
from .errors import (
    ConnectionFailed,
    HttpError,
    ProtocolError,
    RequestTimeout,
    StreamInterrupted,
)

OnDelta = Callable[[str], None]
OnToolCallDelta = Callable[[dict[str, Any]], None]


@dataclass
class ToolCall:
    """A (possibly complete) function call returned by the model."""

    id: str
    name: str
    arguments: str  # raw JSON string as delivered

    @property
    def arguments_dict(self) -> dict[str, Any]:
        if not self.arguments:
            return {}
        try:
            return json.loads(self.arguments)
        except json.JSONDecodeError as exc:
            raise ProtocolError(
                f"model returned malformed tool-call arguments for "
                f"'{self.name}': {exc}"
            ) from exc


@dataclass
class ChatResult:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    model: str | None = None
    provider: str | None = None  # label of the provider that answered

    @property
    def interrupted(self) -> bool:
        return self.finish_reason in {"interrupted", "error"}


class _StreamAccumulator:
    __slots__ = ("content", "tool_calls", "finish_reason", "usage", "model")

    def __init__(self) -> None:
        self.content = ""
        self.tool_calls: dict[int, dict[str, str]] = {}
        self.finish_reason: str | None = None
        self.usage: dict[str, Any] | None = None
        self.model: str | None = None

    @property
    def has_output(self) -> bool:
        return bool(self.content) or bool(self.tool_calls)

    def apply(self, data: dict[str, Any]) -> tuple[str | None, list[dict[str, Any]]]:
        """Apply one SSE chunk.  Returns (text_delta, tool_call_deltas)."""
        if data.get("usage"):
            self.usage = data["usage"]
        if data.get("model"):
            self.model = data["model"]
        choices = data.get("choices") or []
        text_delta: str | None = None
        tc_deltas: list[dict[str, Any]] = []
        for choice in choices:
            delta = choice.get("delta") or {}
            piece = delta.get("content")
            if piece:
                self.content += piece
                text_delta = piece
            for tc in delta.get("tool_calls") or []:
                idx = int(tc.get("index", 0))
                slot = self.tool_calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] += fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]
                tc_deltas.append(tc)
            reason = choice.get("finish_reason")
            if reason:
                self.finish_reason = reason
        return text_delta, tc_deltas

    def snapshot(self) -> ChatResult:
        calls = [
            ToolCall(id=v.get("id", ""), name=v.get("name", ""), arguments=v.get("arguments", ""))
            for _, v in sorted(self.tool_calls.items())
        ]
        return ChatResult(
            content=self.content,
            tool_calls=calls,
            finish_reason=self.finish_reason or "interrupted",
            usage=self.usage,
            model=self.model,
        )


class OpenAICompatibleClient:
    """Synchronous chat-completions client for one OpenAI-compatible endpoint."""

    def __init__(
        self,
        settings: ProviderSettings,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.settings = settings
        connect = min(10.0, float(settings.timeout))
        self._http = httpx.Client(
            timeout=httpx.Timeout(float(settings.timeout), connect=connect),
            transport=transport,
            follow_redirects=True,
        )

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "OpenAICompatibleClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- public API ---------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = True,
        on_delta: OnDelta | None = None,
        on_tool_call_delta: OnToolCallDelta | None = None,
    ) -> ChatResult:
        payload = self._build_payload(
            messages, tools=tools, tool_choice=tool_choice, json_mode=json_mode,
            temperature=temperature, max_tokens=max_tokens, stream=stream,
        )
        if stream:
            return self._stream_chat(payload, on_delta=on_delta,
                                     on_tool_call_delta=on_tool_call_delta)
        return self._plain_chat(payload)

    # -- payload -------------------------------------------------------------

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        tool_choice: Any | None,
        json_mode: bool,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
    ) -> dict[str, Any]:
        s = self.settings
        if not s.model:
            raise ProtocolError(
                f"provider '{s.label}' has no model configured "
                "(set 'model' in config or pass --model)"
            )
        payload: dict[str, Any] = {"model": s.model, "messages": messages, "stream": stream}
        temp = s.temperature if temperature is None else temperature
        if temp is not None:
            payload["temperature"] = temp
        mt = s.max_tokens if max_tokens is None else max_tokens
        if mt:
            payload["max_tokens"] = mt
        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _url(self, suffix: str) -> str:
        return f"{self.settings.base_url}/{suffix}"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.settings.extra_headers}
        if self.settings.has_api_key():
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        return headers

    # -- non-streaming ---------------------------------------------------------

    def _plain_chat(self, payload: dict[str, Any]) -> ChatResult:
        try:
            response = self._http.post(
                self._url("chat/completions"), json=payload, headers=self._headers()
            )
        except httpx.TimeoutException as exc:
            raise RequestTimeout(
                f"timed out talking to '{self.settings.label}' "
                f"({self.settings.base_url})",
                provider=self.settings.label,
            ) from exc
        except httpx.HTTPError as exc:
            raise ConnectionFailed(
                f"connection to '{self.settings.label}' failed: {exc}",
                provider=self.settings.label,
            ) from exc

        if response.status_code != 200:
            raise self._http_error(response.status_code, response.text)
        return self._parse_plain(response.json(), response)

    def _parse_plain(self, data: Any, response: httpx.Response | None = None) -> ChatResult:
        if not isinstance(data, dict) or "choices" not in data:
            raise ProtocolError(
                f"unexpected non-stream response from '{self.settings.label}' "
                f"(expected a 'choices' array): {str(data)[:200]!r}"
            )
        choices = data["choices"]
        if not choices:
            raise ProtocolError(f"empty 'choices' from '{self.settings.label}'")
        message = choices[0].get("message") or {}
        calls = [
            ToolCall(
                id=str(tc.get("id", "")),
                name=str((tc.get("function") or {}).get("name", "")),
                arguments=str((tc.get("function") or {}).get("arguments", "")),
            )
            for tc in message.get("tool_calls") or []
        ]
        model = data.get("model")
        if model is None and response is not None:
            model = response.headers.get("x-request-model")
        return ChatResult(
            content=str(message.get("content") or ""),
            tool_calls=calls,
            finish_reason=choices[0].get("finish_reason"),
            usage=data.get("usage"),
            model=model,
        )

    # -- streaming ---------------------------------------------------------------

    def _stream_chat(
        self,
        payload: dict[str, Any],
        *,
        on_delta: OnDelta | None,
        on_tool_call_delta: OnToolCallDelta | None,
    ) -> ChatResult:
        acc = _StreamAccumulator()
        try:
            return self._stream_once(payload, acc, include_usage=True,
                                     on_delta=on_delta,
                                     on_tool_call_delta=on_tool_call_delta)
        except HttpError as exc:
            # Some servers (older llama.cpp builds, proxies) reject
            # stream_options — retry once without it.
            if exc.status == 400 and "stream_options" in str(exc):
                return self._stream_once(payload, acc, include_usage=False,
                                         on_delta=on_delta,
                                         on_tool_call_delta=on_tool_call_delta)
            raise

    def _stream_once(
        self,
        payload: dict[str, Any],
        acc: _StreamAccumulator,
        *,
        include_usage: bool,
        on_delta: OnDelta | None,
        on_tool_call_delta: OnToolCallDelta | None,
    ) -> ChatResult:
        body = dict(payload)
        if include_usage:
            body["stream_options"] = {"include_usage": True}
        label = self.settings.label
        try:
            with self._http.stream(
                "POST", self._url("chat/completions"), json=body, headers=self._headers()
            ) as response:
                if response.status_code != 200:
                    detail = response.read().decode("utf-8", errors="replace")
                    raise self._http_error(response.status_code, detail)

                content_type = response.headers.get("content-type", "")
                if "text/event-stream" not in content_type:
                    # Server answered with plain JSON despite stream=true.
                    raw = response.read().decode("utf-8", errors="replace")
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        raise ProtocolError(
                            f"'{label}' returned non-stream body that is not "
                            f"JSON: {raw[:200]!r}"
                        ) from exc
                    result = self._parse_plain(data)
                    self._deliver(acc, result, on_delta, on_tool_call_delta)
                    return result

                for data_line in response.iter_lines():
                    chunk = self._sse_event_to_json(data_line)
                    if chunk is None:
                        continue
                    if not isinstance(chunk, dict):
                        raise ProtocolError(
                            f"'{label}' sent a non-object SSE chunk: {str(chunk)[:200]!r}"
                        )
                    text_delta, tc_deltas = acc.apply(chunk)
                    if text_delta and on_delta:
                        on_delta(text_delta)
                    for tc in tc_deltas:
                        if on_tool_call_delta:
                            on_tool_call_delta(tc)
                return acc.snapshot()
        except httpx.TimeoutException as exc:
            self._raise_stream_http_failure(acc, label, "timeout", exc)
        except httpx.ConnectError as exc:
            self._raise_stream_http_failure(acc, label, f"connection error: {exc}", exc)
        except httpx.RemoteProtocolError as exc:
            self._raise_stream_http_failure(acc, label, f"connection closed mid-stream: {exc}", exc)
        except httpx.HTTPError as exc:
            self._raise_stream_http_failure(acc, label, f"stream error: {exc}", exc)
        raise AssertionError("unreachable")  # pragma: no cover

    def _sse_event_to_json(self, line: str) -> Any | None:
        """One SSE *data* line (iter_lines already split on newlines)."""
        if line is None or line.startswith(":"):
            return None
        if not line.startswith("data:"):
            return None
        part = line[len("data:"):]
        if part.startswith(" "):
            part = part[1:]
        if part == "[DONE]":
            return None
        try:
            return json.loads(part)
        except json.JSONDecodeError as exc:
            raise ProtocolError(
                f"'{self.settings.label}' sent invalid JSON in SSE stream: {part[:200]!r}"
            ) from exc

    def _deliver(
        self,
        acc: _StreamAccumulator,
        result: ChatResult,
        on_delta: OnDelta | None,
        on_tool_call_delta: OnToolCallDelta | None,
    ) -> None:
        """Fire callbacks for a non-stream response so callers see uniform events."""
        if result.content and on_delta:
            on_delta(result.content)
        for call in result.tool_calls:
            if on_tool_call_delta:
                on_tool_call_delta(
                    {"id": call.id, "function": {"name": call.name, "arguments": call.arguments}}
                )

    def _raise_stream_http_failure(
        self, acc: _StreamAccumulator, label: str, message: str, exc: Exception
    ) -> None:
        if acc.has_output:
            raise StreamInterrupted(
                f"stream from '{label}' was interrupted after visible output "
                f"({message}); partial content preserved",
                provider=label,
                partial=acc.snapshot(),
            ) from exc
        if isinstance(exc, httpx.TimeoutException):
            raise RequestTimeout(message, provider=label) from exc
        raise ConnectionFailed(message, provider=label) from exc

    def _http_error(self, status: int, body: str) -> HttpError:
        snippet = self._extract_error_snippet(body)
        if status in (401, 403):
            return HttpError(
                f"'{self.settings.label}' rejected authentication (HTTP {status})"
                + (f": {snippet}" if snippet else ""),
                status=status,
                provider=self.settings.label,
            )
        return HttpError(
            f"'{self.settings.label}' returned HTTP {status}"
            + (f": {snippet}" if snippet else ""),
            status=status,
            provider=self.settings.label,
        )

    @staticmethod
    def _extract_error_snippet(body: str) -> str:
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return body.strip()[:200]
        err = data.get("error")
        if isinstance(err, dict):
            message = err.get("message") or err.get("type") or ""
            return str(message)[:200]
        if isinstance(err, str):
            return err[:200]
        return body.strip()[:200]
