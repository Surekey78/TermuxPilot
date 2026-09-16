"""Tool layer: registry + router + built-in tools."""

from .base import (
    ExecutionContext,
    EXECUTE,
    READ,
    Tool,
    ToolRequest,
    ToolResult,
    WRITE,
)
from .files import build_diff_tool, build_edit_tool, build_move_tool, build_read_tool, build_write_tool
from .router import MODES, ToolRouter
from .shell import build_shell_tool

__all__ = [
    "ExecutionContext",
    "MODES",
    "Tool",
    "ToolRequest",
    "ToolResult",
    "ToolRouter",
    "build_default_tools",
    "EXECUTE",
    "READ",
    "WRITE",
]


def build_default_tools() -> list[Tool]:
    """The v0.2 toolset: shell + file ops. (Termux:API lands in v0.3.)"""
    return [
        build_shell_tool(),
        build_read_tool(),
        build_write_tool(),
        build_edit_tool(),
        build_move_tool(),
        build_diff_tool(),
    ]
