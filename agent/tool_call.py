"""仅解析原生 function calling 的工具参数，不解析回答文本。"""
from __future__ import annotations
import json
from typing import Any
from .tools.registry import ToolError, ToolRegistry, validate_args

_MAX_JSON_DEPTH = 32

class ToolCallError(Exception):
    """原生调用的工具名称或参数无效。"""

def _check_depth(root: Any, max_depth: int = _MAX_JSON_DEPTH) -> None:
    """迭代检查 JSON 结构嵌套深度，超过上限抛 ToolCallError，防止深层嵌套引发内部异常。"""
    stack: list[tuple[Any, int]] = [(root, 0)]
    while stack:
        cur, depth = stack.pop()
        if depth > max_depth:
            raise ToolCallError(f"JSON nesting depth exceeds limit ({max_depth})")
        if isinstance(cur, dict):
            for v in cur.values():
                if isinstance(v, (dict, list)):
                    stack.append((v, depth + 1))
        elif isinstance(cur, list):
            for v in cur:
                if isinstance(v, (dict, list)):
                    stack.append((v, depth + 1))

def parse_tool_call_args(name: str, raw_arguments: Any, registry: ToolRegistry) -> dict[str, Any]:
    """解析并校验原生工具调用的工具名与 JSON 参数，不合法时统一抛 ToolCallError。"""
    if not isinstance(name, str) or not name.strip():
        raise ToolCallError("tool call 'name' must be a non-empty string")
    tool = registry.get(name)
    if tool is None:
        raise ToolCallError(f"unknown tool '{name}'. available tools: {', '.join(registry.names())}")
    if isinstance(raw_arguments, dict):
        args = raw_arguments
    elif isinstance(raw_arguments, str):
        raw_str = raw_arguments.strip()
        if not raw_str:
            args = {}
        else:
            try:
                args = json.loads(raw_str)
            except RecursionError:
                raise ToolCallError(f"arguments JSON nesting is too deep for tool '{name}'") from None
            except ValueError as e:
                raise ToolCallError(f"invalid JSON arguments for tool '{name}': {e}") from None
    else:
        raise ToolCallError(f"arguments for tool '{name}' must be a JSON object")
    if not isinstance(args, dict):
        raise ToolCallError(f"arguments for tool '{name}' must be a JSON object")
    _check_depth(args)
    try:
        validate_args(tool.parameters, args)
    except ToolError as e:
        raise ToolCallError(f"invalid arguments for tool '{name}': {e}") from None
    return args
