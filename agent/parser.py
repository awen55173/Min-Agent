"""解析并校验模型输出：仅接受完整的顶层 JSON 对象或完整代码块，提取工具调用或最终答案并强制二者互斥。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .tools.registry import ToolError, ToolRegistry, validate_args

_MAX_JSON_DEPTH = 32


class ParseError(Exception):
    """模型输出不合法；message 会作为反馈回填给模型，让其重新输出。"""


@dataclass
class ParsedOutput:
    """解析结果：kind 为 "action"（工具调用）或 "final"（最终答案）。"""

    kind: str
    thought: str = ""
    action: str | None = None
    action_input: dict[str, Any] = field(default_factory=dict)
    final_answer: str | None = None


def _check_depth(root: Any, max_depth: int = _MAX_JSON_DEPTH) -> None:
    """迭代检查 JSON 结构嵌套深度，超过上限抛 ParseError，防止深层嵌套引发内部异常。"""
    stack: list[tuple[Any, int]] = [(root, 0)]
    while stack:
        cur, depth = stack.pop()
        if depth > max_depth:
            raise ParseError(f"JSON nesting depth exceeds limit ({max_depth})")
        if isinstance(cur, dict):
            for v in cur.values():
                if isinstance(v, (dict, list)):
                    stack.append((v, depth + 1))
        elif isinstance(cur, list):
            for v in cur:
                if isinstance(v, (dict, list)):
                    stack.append((v, depth + 1))


def _unwrap_fence(text: str) -> str:
    """若文本为完整代码块则剥离围栏；存在未闭合或多个代码块时抛 ParseError。"""
    if not text.startswith("```"):
        return text
    if not text.endswith("```") or len(text) < 6:
        raise ParseError("output is not valid JSON: unclosed code block; reply with a single JSON object")
    inner = text[3:-3]
    if "```" in inner:
        raise ParseError(
            "ambiguous output: multiple code blocks are not allowed; reply with a single JSON object"
        )
    if inner[:4].lower() == "json" and (len(inner) == 4 or inner[4] in " \t\r\n{"):
        inner = inner[4:]
    candidate = inner.strip()
    if not candidate:
        raise ParseError("empty code block; reply with a single JSON object")
    return candidate


def extract_json(text: str) -> dict:
    """提取模型输出中的 JSON 对象：仅接受完整的顶层对象或完整代码块，其余情况统一抛 ParseError。"""
    text = text.strip()
    if not text:
        raise ParseError("empty output; reply with a single JSON object")
    candidate = _unwrap_fence(text)
    try:
        obj = json.loads(candidate)
    except RecursionError:
        raise ParseError("JSON nesting is too deep; reply with a flat JSON object") from None
    except ValueError:
        raise ParseError(
            "output is not valid JSON (or contains extra/ambiguous content); reply with a single JSON object"
        ) from None
    if not isinstance(obj, dict):
        raise ParseError("output is not valid JSON object; top-level value must be a JSON object")
    _check_depth(obj)
    return obj


def _present(value: Any) -> bool:
    """字段是否有效出现（None、空字符串或仅含空白的字符串视为未提供）。"""
    if isinstance(value, str):
        return bool(value.strip())
    return value is not None


def parse_output(text: str, registry: ToolRegistry) -> ParsedOutput:
    """解析模型输出并按规则校验：互斥、缺字段、未知工具、参数不合法均抛 ParseError。"""
    obj = extract_json(text)
    if set(obj) - {"thought", "action", "action_input", "final_answer"}:
        raise ParseError("unexpected output fields; Observation/tool results must only come from the runtime")
    thought = obj.get("thought")
    thought = thought if isinstance(thought, str) else ""  # thought 缺失可容忍
    has_action = _present(obj.get("action"))
    has_final = _present(obj.get("final_answer"))

    if has_action and has_final:
        raise ParseError("'action' and 'final_answer' are mutually exclusive; send only one of them")
    if not has_action and not has_final:
        raise ParseError("output must contain either 'action' (+ 'action_input') or a non-blank 'final_answer'")

    if has_final:
        answer = obj["final_answer"]
        if not isinstance(answer, str):
            raise ParseError("'final_answer' must be a string")
        return ParsedOutput("final", thought, final_answer=answer.strip())

    name = obj["action"]
    if not isinstance(name, str):
        raise ParseError("'action' must be a tool name string")
    tool = registry.get(name)
    if tool is None:
        raise ParseError(f"unknown tool '{name}'. available tools: {', '.join(registry.names())}")
    if "action_input" not in obj:
        raise ParseError(f"'action_input' is missing for tool '{name}'")
    args = obj["action_input"]
    try:
        validate_args(tool.parameters, args)
    except ToolError as e:
        raise ParseError(f"invalid action_input for tool '{name}': {e}") from None
    return ParsedOutput("action", thought, action=name, action_input=args)


def parse_tool_call_args(name: str, raw_arguments: Any, registry: ToolRegistry) -> dict[str, Any]:
    """解析并校验原生工具调用的工具名与 JSON 参数，不合法时统一抛 ParseError。"""
    if not isinstance(name, str) or not name.strip():
        raise ParseError("tool call 'name' must be a non-empty string")
    tool = registry.get(name)
    if tool is None:
        raise ParseError(f"unknown tool '{name}'. available tools: {', '.join(registry.names())}")
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
                raise ParseError(f"arguments JSON nesting is too deep for tool '{name}'") from None
            except ValueError as e:
                raise ParseError(f"invalid JSON arguments for tool '{name}': {e}") from None
    else:
        raise ParseError(f"arguments for tool '{name}' must be a JSON object")
    if not isinstance(args, dict):
        raise ParseError(f"arguments for tool '{name}' must be a JSON object")
    _check_depth(args)
    try:
        validate_args(tool.parameters, args)
    except ToolError as e:
        raise ParseError(f"invalid arguments for tool '{name}': {e}") from None
    return args
