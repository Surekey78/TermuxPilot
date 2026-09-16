from __future__ import annotations

from termuxpilot.provider.sse import DONE, iter_sse_data, parse_sse_chunks


def test_single_event_simple():
    lines = ['data: {"a": 1}', ""]
    assert list(iter_sse_data(lines)) == ['{"a": 1}']


def test_done_sentinel():
    lines = ['data: {"a": 1}', "", "data: [DONE]", ""]
    assert list(iter_sse_data(lines)) == ['{"a": 1}', DONE]


def test_keepalive_comments_ignored():
    lines = [": ping", ": keepalive", 'data: {"a": 2}', ""]
    assert list(iter_sse_data(lines)) == ['{"a": 2}']


def test_multiple_data_lines_joined_into_one_event():
    # per the SSE spec, multiple data: lines in one event join with \n
    lines = ['data: {"a":', 'data: 1}', ""]
    assert list(iter_sse_data(lines)) == ['{"a":\n1}']


def test_event_without_trailing_blank_line():
    lines = ["data: {\"last\": true}"]
    assert list(iter_sse_data(lines)) == ['{"last": true}']


def test_other_fields_ignored():
    lines = ["event: message", "id: 42", "retry: 5000", 'data: {"x": 3}', ""]
    assert list(iter_sse_data(lines)) == ['{"x": 3}']


def test_chunks_split_across_byte_boundaries():
    full = b'data: {"choices":[{"delta":{"content":"He"}}]}\r\n\r\ndata: [DONE]\r\n\r\n'
    # deliberately ugly chunking: mid-line, mid-token
    chunks = [full[:4], full[4:11], full[11:20], full[20:]]
    events = list(iter_sse_data(parse_sse_chunks(chunks)))
    assert events == ['{"choices":[{"delta":{"content":"He"}}]}', DONE]


def test_crlf_line_endings():
    events = list(iter_sse_data(parse_sse_chunks([b"data: {\"a\": 1}\r\n\r\n"])))
    assert events == ['{"a": 1}']
