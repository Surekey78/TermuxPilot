"""A tiny OpenAI-compatible server for tests and manual fallback demos.

Run standalone:
    python tests/mockserver.py --port 8100
    python tests/mockserver.py --port 8101 --fail-next 2 --key sekret

Behaviour knobs:
* ``--fail-next N``   the first N requests get HTTP 503 (then it works)
* ``--fail-status``   status used for the simulated outage
* ``--delay``         seconds between SSE tokens
* ``--key``           require ``Authorization: Bearer <key>``
* ``--drop-after N``  kill the TCP connection mid-stream after N tokens
* ``--json-only``     answer streaming requests with a plain JSON body

Agent-loop scripting (programmatic API, see MockServer):
* ``tool_script=[{"tool": "run_shell", "args": {...}}, {"text": "Done."}]``
  — answers the first chat call with that tool call (native format when the
  request carries ``tools``, JSON-mode object content otherwise) and the next
  call with the final text (repeated if asked again).
* ``reject_tools=True`` — 400 when the request contains a ``tools`` key,
  simulating an endpoint without function-calling support.
"""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

DEFAULT_MODEL = "mock-model-1"

ReplyFactory = Callable[[list[dict[str, Any]]], str]


def default_reply(messages: list[dict[str, Any]]) -> str:
    last_user = ""
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                last_user = content
            break
    return (
        "Mock reply. You said: "
        + (last_user[:120] if last_user else "(no user message)")
    )


class _State:
    """Shared, thread-safe server knobs."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.require_key: str | None = None
        self.fail_next = 0
        self.fail_status = 503
        self.delay = 0.0
        self.drop_after: int | None = None
        self.json_only = False
        self.reply_factory: ReplyFactory = default_reply
        self.request_count = 0
        self.saw_stream_options: bool | None = None
        self.auth_header: str | None = None
        self.last_payload: dict[str, Any] | None = None
        self.tool_script: list[dict[str, Any]] | None = None
        self.tool_step = 0
        self.reject_tools = False
        self.saw_tools: bool | None = None


class _Handler(BaseHTTPRequestHandler):
    server_version = "MockOpenAI/0.1"

    @property
    def state(self) -> _State:
        return self.server.state  # type: ignore[attr-defined]

    def log_message(self, *args: Any) -> None:  # silence
        pass

    # ------------------------------------------------------------- GET

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            self._json(200, {"object": "list", "data": [{"id": DEFAULT_MODEL}]})
        else:
            self._json(404, {"error": {"message": "not found"}})

    # ------------------------------------------------------------- POST

    def do_POST(self) -> None:  # noqa: N802
        state = self.state
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"error": {"message": "invalid JSON"}})
            return

        with state.lock:
            state.request_count += 1
            state.last_payload = payload
            state.auth_header = self.headers.get("Authorization")
            state.saw_tools = "tools" in payload
            if state.require_key:
                expected = f"Bearer {state.require_key}"
                if self.headers.get("Authorization") != expected:
                    self._json(401, {"error": {"message": "invalid api key"}})
                    return
            if state.reject_tools and "tools" in payload:
                self._json(
                    400,
                    {"error": {"message": "unknown parameter: tools"}},
                )
                return
            if state.fail_next > 0:
                state.fail_next -= 1
                self._json(
                    state.fail_status,
                    {"error": {"message": "simulated outage"}},
                )
                return

        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            self._json(404, {"error": {"message": "unknown endpoint"}})
            return

        if state.tool_script is not None:
            self._scripted_response(payload)
            return
        if state.json_only:
            self._json(200, self._completion_payload(payload, stream=False))
            return
        if payload.get("stream"):
            self._stream_response(payload)
        else:
            self._json(200, self._completion_payload(payload, stream=False))

    # ------------------------------------------------ scripted agent loop

    def _next_action(self) -> dict[str, Any]:
        state = self.state
        with state.lock:
            assert state.tool_script is not None
            if state.tool_step < len(state.tool_script):
                action = state.tool_script[state.tool_step]
                state.tool_step += 1
            else:
                # settle on the final entry
                action = state.tool_script[-1]
            return action

    def _scripted_response(self, payload: dict[str, Any]) -> None:
        action = self._next_action()
        native = "tools" in payload
        if "tool" in action:
            if native:
                if payload.get("stream"):
                    self._stream_tool_call(payload, action)
                else:
                    self._json(200, self._tool_call_payload(payload, action))
            else:
                content = json.dumps({
                    "thought": action.get("thought", ""),
                    "tool": action["tool"],
                    "args": action.get("args", {}),
                })
                self._reply_text(payload, content)
        else:
            self._reply_text(payload, action.get("text", "Done."))

    def _reply_text(self, payload: dict[str, Any], text: str) -> None:
        if payload.get("stream"):
            model = payload.get("model") or DEFAULT_MODEL
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def send_chunk(raw: bytes) -> None:
                self.wfile.write(f"{len(raw):X}\r\n".encode("ascii") + raw + b"\r\n")
                self.wfile.flush()

            def send_event(obj: str) -> None:
                send_chunk(f"data: {obj}\n\n".encode("utf-8"))

            for token in [t + " " for t in text.split(" ")]:
                send_event(json.dumps({
                    "id": "chatcmpl-mock",
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [{"index": 0,
                                 "delta": {"content": token},
                                 "finish_reason": None}],
                }))
            send_event(json.dumps({
                "id": "chatcmpl-mock",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }))
            send_event("[DONE]")
            send_chunk(b"")  # terminating 0-chunk
        else:
            self._json(200, {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": payload.get("model") or DEFAULT_MODEL,
                "choices": [{"index": 0,
                             "message": {"role": "assistant", "content": text},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20,
                          "total_tokens": 30},
            })

    def _tool_call_payload(self, payload: dict[str, Any], action: dict) -> dict:
        return {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.get("model") or DEFAULT_MODEL,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_mock_1",
                        "type": "function",
                        "function": {
                            "name": action["tool"],
                            "arguments": json.dumps(action.get("args", {})),
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20,
                      "total_tokens": 30},
        }

    def _stream_tool_call(self, payload: dict[str, Any], action: dict) -> None:
        model = payload.get("model") or DEFAULT_MODEL
        args_json = json.dumps(action.get("args", {}))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def send_chunk(raw: bytes) -> None:
            self.wfile.write(f"{len(raw):X}\r\n".encode("ascii") + raw + b"\r\n")
            self.wfile.flush()

        def send_event(obj: str) -> None:
            send_chunk(f"data: {obj}\n\n".encode("utf-8"))

        def event(delta: dict, finish: str | None) -> str:
            return json.dumps({
                "id": "chatcmpl-mock",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            })

        send_event(event({
            "role": "assistant",
            "tool_calls": [{
                "index": 0,
                "id": "call_mock_1",
                "type": "function",
                "function": {"name": action["tool"], "arguments": ""},
            }],
        }, None))
        send_event(event({
            "tool_calls": [{
                "index": 0,
                "function": {"arguments": args_json},
            }],
        }, None))
        send_event(event({}, "tool_calls"))
        send_event("[DONE]")
        send_chunk(b"")  # terminating 0-chunk

    # ---------------------------------------------------------- helpers

    def _completion_payload(self, payload: dict[str, Any], stream: bool) -> dict[str, Any]:
        messages = payload.get("messages") or []
        reply = self.state.reply_factory(messages)
        return {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.get("model") or DEFAULT_MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

    def _stream_response(self, payload: dict[str, Any]) -> None:
        state = self.state
        with state.lock:
            state.saw_stream_options = "stream_options" in payload
            reply = state.reply_factory(payload.get("messages") or [])
            model = payload.get("model") or DEFAULT_MODEL
            delay = state.delay
            drop_after = state.drop_after

        tokens = [t + " " for t in reply.split(" ")]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def send_chunk(text: str) -> None:
            data = text.encode("utf-8")
            self.wfile.write(f"{len(data):X}\r\n".encode("ascii") + data + b"\r\n")
            self.wfile.flush()

        def sse_event(event: dict[str, Any] | str) -> str:
            if isinstance(event, str):
                return f"data: {event}\n\n"
            return "data: " + json.dumps(event) + "\n\n"

        def chunk(delta: dict[str, Any], finish: str | None = None) -> str:
            return sse_event({
                "id": "chatcmpl-mock",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            })

        send_chunk(chunk({"role": "assistant", "content": ""}, None))
        for i, token in enumerate(tokens):
            if drop_after is not None and i >= drop_after:
                # hard kill mid-stream (no terminating 0-chunk)
                try:
                    self.connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                self.close_connection = True
                return
            if delay:
                time.sleep(delay)
            send_chunk(chunk({"content": token}, None))
        if not (drop_after is not None and len(tokens) >= drop_after):
            send_chunk(chunk({}, "stop"))
            if state.saw_stream_options:
                send_chunk(sse_event({
                    "id": "chatcmpl-mock",
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 20,
                              "total_tokens": 30},
                }))
            send_chunk(sse_event("[DONE]"))
        send_chunk("")  # terminating chunk

    def _json(self, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class MockServer:
    """Threaded wrapper; ``start()`` binds to a free port, ``stop()`` joins."""

    def __init__(
        self,
        *,
        require_key: str | None = None,
        fail_next: int = 0,
        fail_status: int = 503,
        delay: float = 0.0,
        drop_after: int | None = None,
        json_only: bool = False,
        reply_factory: ReplyFactory | None = None,
        tool_script: list[dict[str, Any]] | None = None,
        reject_tools: bool = False,
    ) -> None:
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.state = _State()
        self.httpd.state = self.state  # type: ignore[attr-defined]
        self.state.require_key = require_key
        self.state.fail_next = fail_next
        self.state.fail_status = fail_status
        self.state.delay = delay
        self.state.drop_after = drop_after
        self.state.json_only = json_only
        self.state.reply_factory = reply_factory or default_reply
        self.state.tool_script = tool_script
        self.state.reject_tools = reject_tools
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}/v1"

    @property
    def request_count(self) -> int:
        return self.state.request_count

    def set_reply_factory(self, factory: ReplyFactory) -> None:
        self.state.reply_factory = factory

    def start(self) -> "MockServer":
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--fail-next", type=int, default=0)
    parser.add_argument("--fail-status", type=int, default=503)
    parser.add_argument("--delay", type=float, default=0.02)
    parser.add_argument("--key", default=None)
    parser.add_argument("--drop-after", type=int, default=None)
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args()

    # re-bind on the requested port
    server = MockServer(
        require_key=args.key,
        fail_next=args.fail_next,
        fail_status=args.fail_status,
        delay=args.delay,
        drop_after=args.drop_after,
        json_only=args.json_only,
    )
    server.httpd.server_close()
    server.httpd = ThreadingHTTPServer((args.host, args.port), _Handler)
    server.httpd.state = server.state  # type: ignore[attr-defined]
    print(f"mock OpenAI server on {server.base_url} (Ctrl+C to stop)", flush=True)
    try:
        server.httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
