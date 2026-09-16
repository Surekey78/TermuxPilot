from __future__ import annotations

import tracemalloc

import pytest

from termuxpilot.tools.base import truncate_output
from termuxpilot.tools.output import BoundedText, MAX_LINE_CHARS, StreamCapture


@pytest.mark.parametrize("limit", [128, 512, 30_000])
def test_bounded_text_exact_limit_and_head_tail(limit):
    buffer = BoundedText(limit)
    original = "H" * (limit - 4) + "TAIL"
    for offset in range(0, len(original), 13):
        buffer.append(original[offset:offset + 13])
    assert buffer.text() == original
    assert not buffer.truncated
    buffer.append(" final tail")
    assert buffer.truncated
    assert len(buffer.text()) <= limit
    assert buffer.text().startswith("H")
    assert buffer.text().endswith(" final tail")
    assert "truncated" in buffer.text()
    assert len(truncate_output(original * 4, limit)) <= limit


def test_stream_decodes_split_utf8_and_unterminated_last_line():
    capture = StreamCapture(1024)
    text = "hello 你好 😀\nlast line"
    for byte in text.encode("utf-8"):
        capture.feed(bytes([byte]))
    capture.finish()
    assert capture.text() == text
    assert not capture.truncated


def test_stream_redacts_secret_split_between_chunks():
    capture = StreamCapture(1024)
    secret = "sk-abcdefghijklmnop1234567890ABC"
    for piece in [b"prefix sk-abc", b"defghijklmnop123456", b"7890ABC suffix\n"]:
        capture.feed(piece)
    capture.finish()
    assert secret not in capture.text()
    assert "[REDACTED:openai-key]" in capture.text()
    assert "suffix" in capture.text()


def test_private_key_is_redacted_before_head_tail_truncation():
    capture = StreamCapture(512)
    capture.feed(b"prefix\n" * 1000)
    capture.feed(b"-----BEGIN RSA PRIVATE KEY-----\n")
    for _ in range(1000):
        capture.feed(b"PRIVATE_BODY_MUST_NOT_ESCAPE_1234567890\n")
    capture.feed(b"-----END RSA PRIVATE KEY-----\nsuffix\n")
    capture.finish()
    assert capture.truncated
    assert "PRIVATE_BODY" not in capture.text()
    assert capture.text().endswith("suffix\n")


def test_overlong_line_is_omitted_without_exposing_fragments():
    capture = StreamCapture(1024)
    for _ in range(20):
        capture.feed(b"SENSITIVE_FRAGMENT" * 4096)
    capture.feed(b"\nnormal output\n")
    capture.finish()
    assert capture.truncated
    assert "omitted overlong" in capture.text()
    assert "SENSITIVE_FRAGMENT" not in capture.text()
    assert "normal output" in capture.text()


def test_private_key_delimiter_split_in_omitted_line():
    capture = StreamCapture(1024)
    capture.feed(b"x" * (MAX_LINE_CHARS + 1))
    capture.feed(b"-----BEGIN RSA PRI")
    capture.feed(b"VATE KEY-----\n")
    capture.feed(b"PRIVATE_BODY_MUST_NOT_ESCAPE\n")
    capture.feed(b"-----END RSA PRIVATE KEY-----\nsuffix\n")
    capture.finish()
    assert "PRIVATE_BODY" not in capture.text()
    assert "suffix" in capture.text()


def test_large_stream_has_bounded_python_memory():
    capture = StreamCapture(4096)
    chunk = b"ordinary build output\n" * 700
    tracemalloc.start()
    try:
        for _ in range(1000):  # ~15 MB produced; never retained as one string
            capture.feed(chunk)
        capture.finish()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert capture.truncated
    assert len(capture.text()) <= 4096
    assert peak < 1_000_000


def test_stream_invalid_utf8_and_redaction_opt_out():
    capture = StreamCapture(1024, redact=False)
    capture.feed(b"\xff api_key=syntheticsecret123\n")
    capture.finish()
    assert "\ufffd" in capture.text()
    assert "syntheticsecret123" in capture.text()
