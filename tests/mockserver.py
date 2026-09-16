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
            if state.require_key:
                expected = f"Bearer {state.require_key}"
                if self.headers.get("Authorization") != expected:
                    self._json(401, {"error": {"message": "invalid api key"}})
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

        if state.json_only:
            self._json(200, self._completion_payload(payload, stream=False))
            return
        if payload.get("stream"):
            self._stream_response(payload)
        else:
            self._json(200, self._completion_payload(payload, stream=False))

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
