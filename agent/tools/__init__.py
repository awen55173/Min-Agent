"""内置工具及默认注册表构造。"""
from __future__ import annotations

from .calculator import calculator_tool
from .registry import Tool, ToolContext, ToolError, ToolRegistry, ToolResult
from .search import make_search_tool
from .todo import TodoStore, make_todo_tool
from .weather import make_weather_tool

__all__ = ["Tool", "ToolContext", "ToolError", "ToolRegistry", "ToolResult", "TodoStore", "build_registry"]


def build_registry(
    tavily_api_key: str | None = None,
    todo_store: TodoStore | None = None,
    http_get=None,
    http_post=None,
) -> ToolRegistry:
    """注册 calculator / todo / search / weather；http_* 可注入以便测试。"""
    reg = ToolRegistry()
    reg.register(calculator_tool)
    reg.register(make_todo_tool(todo_store or TodoStore()))
    reg.register(make_search_tool(tavily_api_key, *([http_post] if http_post else [])))
    reg.register(make_weather_tool(*([http_get] if http_get else [])))
    return reg
