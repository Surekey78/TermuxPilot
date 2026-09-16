from __future__ import annotations

import json
import time

import httpx
import pytest

from termuxpilot.config import ProviderSettings
from termuxpilot.provider import (
    ConnectionFailed,
    HttpError,
    OpenAICompatibleClient,
    ProtocolError,
    RequestTimeout,
)

MESSAGES = [{"role": "user", "content": "hello"}]


def make_settings(**kw) -> ProviderSettings:
    base = dict(label="test", base_url="https://api.test/v1", model="mock-model")
    base.update(kw)
    return ProviderSettings(**base)


def sse_response(*events: str) -> httpx.Response:
    body = "".join(f"data: {e}\n\n" for e in events)
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=body.encode("utf-8"),
    )


def test_non_stream_json(monkeypatch):
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization", "")
        body = json.loads(request.content)
        assert body["stream"] is False
        assert body["model"] == "mock-model"
        assert body["messages"] == MESSAGES
        return httpx.Response(200, json={
            "id": "x", "model": "mock-model-answered",
            "choices": [{"message": {"role": "assistant", "content": "hi there"},
                         "finish_reason": "stop"}],
            "usage": {"total_tokens": 7},
        })

    client = OpenAICompatibleClient(
        make_settings(api_key="sekret"), transport=httpx.MockTransport(handler)
    )
    result = client.chat(MESSAGES, stream=False)
    assert result.content == "hi there"
    assert result.model == "mock-model-answered"
    assert result.finish_reason == "stop"
    assert result.usage == {"total_tokens": 7}
    assert seen["auth"] == "Bearer sekret"


def test_no_auth_header_for_none_key():
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization", "<missing>")
        return httpx.Response(200, json={"choices": [
            {"message": {"content": "ok"}, "finish_reason": "stop"}]})

    client = OpenAICompatibleClient(
        make_settings(api_key="none"), transport=httpx.MockTransport(handler)
    )
    client.chat(MESSAGES, stream=False)
    assert seen["auth"] == "<missing>"


def test_streaming_sse_deltas():
    got: list[str] = []
    events = [
        json.dumps({"choices": [{"delta": {"role": "assistant", "content": ""},
                                 "finish_reason": None}]}),
        json.dumps({"choices": [{"delta": {"content": "Hel"}, "finish_reason": None}]}),
        json.dumps({"choices": [{"delta": {"content": "lo"}, "finish_reason": None}]}),
        json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": {"total_tokens": 9}}),
        "[DONE]",
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return sse_response(*events)

    client = OpenAICompatibleClient(
        make_settings(), transport=httpx.MockTransport(handler)
    )
    result = client.chat(MESSAGES, on_delta=got.append)
    assert result.content == "Hello"
    assert got == ["Hel", "lo"]
    assert result.finish_reason == "stop"
    assert result.usage == {"total_tokens": 9}


def test_streaming_deltas_split_across_chunks():
    body = b'data: {"choices":[{"delta":{"content":"A"}}]}\n\ndata: [DONE]\n\n'
    pieces = [body[:3], body[3:9], body[9:15], body[15:]]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=iter(pieces),
        )

    client = OpenAICompatibleClient(
        make_settings(), transport=httpx.MockTransport(handler)
    )
    got: list[str] = []
    result = client.chat(MESSAGES, on_delta=got.append)
    assert result.content == "A"
    assert got == ["A"]


def test_tool_call_streaming_accumulation():
    def frag(idx_delta: dict) -> str:
        return json.dumps({"choices": [{"delta": idx_delta, "finish_reason": None}]})

    events = [
        frag({"tool_calls": [{"index": 0, "id": "call_1",
                              "function": {"name": "run_shell", "arguments": ""}}]}),
        frag({"tool_calls": [{"index": 0, "function": {"arguments": '{"cmd": "'}}]}),
        frag({"tool_calls": [{"index": 0, "function": {"arguments": 'ls"}'}}]}),
        json.dumps({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
        "[DONE]",
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert b'"tools"' not in request.content
        return sse_response(*events)

    client = OpenAICompatibleClient(
        make_settings(), transport=httpx.MockTransport(handler)
    )
    got_tc: list[dict] = []
    result = client.chat(MESSAGES, on_tool_call_delta=got_tc.append)
    assert result.content == ""
    assert result.finish_reason == "tool_calls"
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.id == "call_1"
    assert call.name == "run_shell"
    assert call.arguments_dict == {"cmd": "ls"}
    assert len(got_tc) == 3


def test_tools_payload_passthrough():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [
            {"message": {"content": ""}, "finish_reason": "stop"}]})

    client = OpenAICompatibleClient(make_settings(), transport=httpx.MockTransport(handler))
    tools = [{"type": "function", "function": {"name": "run_shell", "parameters": {}}}]
    client.chat(MESSAGES, tools=tools, tool_choice="auto", json_mode=True, stream=False)
    assert captured["tools"] == tools
    assert captured["tool_choice"] == "auto"
    assert captured["response_format"] == {"type": "json_object"}


def test_stream_options_400_retry_without_it():
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        attempts.append("stream_options" in body)
        if "stream_options" in body:
            return httpx.Response(400, json={"error": {
                "message": "unknown parameter: stream_options"}})
        return sse_response(
            json.dumps({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}),
            "[DONE]",
        )

    client = OpenAICompatibleClient(
        make_settings(), transport=httpx.MockTransport(handler)
    )
    result = client.chat(MESSAGES)
    assert result.content == "ok"
    assert attempts == [True, False]


def test_server_ignores_stream_flag():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [
            {"message": {"content": "plain"}, "finish_reason": "stop"}]})

    client = OpenAICompatibleClient(
        make_settings(), transport=httpx.MockTransport(handler)
    )
    got: list[str] = []
    result = client.chat(MESSAGES, on_delta=got.append)
    assert result.content == "plain"
    assert got == ["plain"]


def test_http_error_mapping():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    client = OpenAICompatibleClient(
        make_settings(api_key="nope"), transport=httpx.MockTransport(handler)
    )
    with pytest.raises(HttpError) as exc:
        client.chat(MESSAGES, stream=False)
    assert exc.value.status == 401
    assert exc.value.retryable is True
    assert "authentication" in str(exc.value).lower()

    def handler400(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "bad model"}})

    client2 = OpenAICompatibleClient(
        make_settings(), transport=httpx.MockTransport(handler400)
    )
    with pytest.raises(HttpError) as exc2:
        client2.chat(MESSAGES, stream=False)
    assert exc2.value.status == 400
    assert exc2.value.retryable is False


def test_connect_error():
    client = OpenAICompatibleClient(
        make_settings(base_url="http://127.0.0.1:1/v1", timeout=2.0)
    )
    with pytest.raises(ConnectionFailed):
        client.chat(MESSAGES, stream=False)


def test_timeout_error():
    # real socket + slow handler (MockTransport ignores timeouts)
    from mockserver import MockServer

    def slow(messages):
        time.sleep(0.8)
        return "too slow"

    server = MockServer(reply_factory=slow).start()
    try:
        client = OpenAICompatibleClient(
            make_settings(base_url=server.base_url, timeout=0.3)
        )
        with pytest.raises(RequestTimeout):
            client.chat(MESSAGES, stream=False)
    finally:
        server.stop()


def test_missing_model_raises_protocol_error():
    client = OpenAICompatibleClient(make_settings(model=None))
    with pytest.raises(ProtocolError, match="no model"):
        client.chat(MESSAGES, stream=False)


def test_malformed_stream_json_raises_protocol_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response("this is not json")

    client = OpenAICompatibleClient(
        make_settings(), transport=httpx.MockTransport(handler)
    )
    with pytest.raises(ProtocolError):
        client.chat(MESSAGES)
