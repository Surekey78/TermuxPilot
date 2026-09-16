"""Bounded output retention for noisy tools.

Shell pipes are decoded and redacted a line at a time *before* head/tail
retention. Even a process producing an unbounded line cannot grow the pending
buffer without limit: that line is omitted, not split through a possible secret.
"""

from __future__ import annotations

import codecs

from ..safety import LineRedactor

DEFAULT_OUTPUT_CHARS = 30_000
MAX_LINE_CHARS = 65_536


class BoundedText:
    """Retain the beginning and end of a stream using O(limit) memory."""

    def __init__(self, limit: int = DEFAULT_OUTPUT_CHARS) -> None:
        if limit < 128:
            raise ValueError("output limit must be at least 128 characters")
        self.limit = limit
        self.total = 0
        self._tail_limit = min(2_000, limit // 4)
        self._head_limit = limit - self._tail_limit
        self._head = ""
        self._tail = ""

    def append(self, text: str) -> None:
        self.total += len(text)
        needed = self._head_limit - len(self._head)
        if needed:
            self._head += text[:needed]
            text = text[needed:]
        if text:
            self._tail = (self._tail + text[-self._tail_limit:])[-self._tail_limit:]

    @property
    def truncated(self) -> bool:
        return self.total > self.limit

    def text(self) -> str:
        if not self.truncated:
            return self._head + self._tail
        note = f"\n… [output truncated; {self.total:,} chars produced] …\n"
        return self._head[:self._head_limit - len(note)] + note + self._tail


class StreamCapture:
    """Incrementally decode UTF-8, redact complete lines, and retain a bound."""

    def __init__(self, limit: int, *, redact: bool = True) -> None:
        self.buffer = BoundedText(limit)
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._redactor = LineRedactor() if redact else None
        self._pending = ""
        self._discarding = False
        self._scan_tail = ""
        self._omitted = False

    def feed(self, data: bytes) -> None:
        self._feed_text(self._decoder.decode(data))

    def _feed_text(self, text: str) -> None:
        pieces = text.split("\n")
        for index, piece in enumerate(pieces):
            complete = index < len(pieces) - 1
            if complete:
                piece += "\n"
            if self._discarding:
                # Still track PEM delimiters in omitted lines, including ones
                # split between chunks. No part of an overlong line is emitted.
                scan = self._scan_tail + piece
                if self._redactor:
                    self._redactor.redact(scan)
                self._scan_tail = scan[-128:]
                if complete:
                    self._discarding = False
                    self._scan_tail = ""
                continue
            self._pending += piece
            if len(self._pending) > MAX_LINE_CHARS:
                if self._redactor:
                    self._redactor.redact(self._pending)
                self.buffer.append("\n[omitted overlong output line]\n")
                self._scan_tail = self._pending[-128:] if not complete else ""
                self._pending = ""
                self._discarding = not complete
                self._omitted = True
            elif complete:
                self._emit_pending()

    def _emit_pending(self) -> None:
        text = self._pending
        self._pending = ""
        self.buffer.append(self._redactor.redact(text) if self._redactor else text)

    def finish(self) -> None:
        self._feed_text(self._decoder.decode(b"", final=True))
        if self._pending:
            self._emit_pending()

    @property
    def truncated(self) -> bool:
        return self._omitted or self.buffer.truncated

    def text(self) -> str:
        return self.buffer.text()
