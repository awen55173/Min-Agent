"""解析并校验模型输出：提取 thought、工具调用或最终答案，并强制二者互斥。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .tools.registry import ToolError, ToolRegistry, validate_args



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


def _loads_object(text: str) -> dict | None:
    """尝试把文本解析为 JSON 对象，失败返回 None。"""
    try:
        obj = json.loads(text)
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def extract_json(text: str) -> dict:
    """提取输出中唯一的 JSON 对象（整段、代码块或夹在文字中均可）。

    找不到抛 ParseError；出现多个顶层 JSON 对象时视为歧义输出，同样抛 ParseError，不取其中任何一个。
    """
    text = text.strip()
    if not text:
        raise ParseError("empty output; reply with a single JSON object")
    obj = _loads_object(text)
    if obj is not None:
        return obj
    found = _scan_objects(text)
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        raise ParseError(
            f"ambiguous output: found {len(found)} JSON objects; reply with exactly ONE JSON object "
            "(one tool call or one final_answer)"
        )
    raise ParseError("output is not valid JSON; reply with a single JSON object in the required format")


def _scan_objects(text: str) -> list[dict]:
    """扫描文本中所有顶层 JSON 对象（已解析对象的内部不再重复扫描）。"""
    decoder = json.JSONDecoder()
    found: list[dict] = []
    i = 0
    while i < len(text):
        if text[i] == "{":
            try:
                obj, end = decoder.raw_decode(text, i)
            except ValueError:
                i += 1
                continue
            if isinstance(obj, dict):
                found.append(obj)
            i = end
            continue
        i += 1
    return found


def _present(value: Any) -> bool:
    """字段是否有效出现（None、空字符串或仅含空白的字符串视为未提供）。"""
    if isinstance(value, str):
        return bool(value.strip())
    return value is not None


def parse_output(text: str, registry: ToolRegistry) -> ParsedOutput:
    """解析模型输出并按规则校验：互斥、缺字段、未知工具、参数不合法均抛 ParseError。"""
    obj = extract_json(text)
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
