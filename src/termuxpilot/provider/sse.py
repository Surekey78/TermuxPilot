"""Minimal, dependency-free SSE (server-sent events) stream parser.

OpenAI-compatible endpoints stream chat completions as::

    data: {"choices":[{"delta":{"content":"Hel"}}]}

    data: {"choices":[{"delta":{"content":"lo"}}]}

    data: [DONE]

This module splits a *line* iterator (as produced by ``httpx``'s
``response.iter_lines()``) into event payloads.  It is deliberately tolerant:
keep-alive comments (``: ping``), empty frames, ``event:``/``id:``/``retry:``
fields, and multiple ``data:`` lines per event are all handled.
"""

from __future__ import annotations

from typing import Iterator, Iterable

DONE = "[DONE]"


def iter_sse_data(lines: Iterable[str]) -> Iterator[str]:
    """Yield the ``data`` payload of each SSE event.

    Multiple ``data:`` lines inside one event are joined with ``\\n``
    (per the SSE spec); OpenAI-compatible servers send a single line.
    """
    data_parts: list[str] = []
    for raw in lines:
        line = raw.rstrip("\r")
        if line.startswith(":"):  # comment / keep-alive
            continue
        if line == "":
            if data_parts:
                yield "\n".join(data_parts)
                data_parts = []
            continue
        if line.startswith("data:"):
            part = line[len("data:"):]
            if part.startswith(" "):
                part = part[1:]
            data_parts.append(part)
        # event: / id: / retry: fields are irrelevant for chat completions
    if data_parts:
        yield "\n".join(data_parts)


def parse_sse_chunks(chunks: Iterable[bytes | str]) -> Iterator[str]:
    """Convenience wrapper: split raw byte *chunks* into lines, then events.

    Handles data split across chunk boundaries (TCP framing is arbitrary).
    """
    buffer = ""
    for chunk in chunks:
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8", errors="replace")
        buffer += chunk
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            yield line
    if buffer:
        yield buffer
