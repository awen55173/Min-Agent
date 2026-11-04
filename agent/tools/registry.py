"""工具接口与注册表：名称、描述、参数 Schema，并在执行前校验参数。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
}


class ToolError(Exception):
    """工具执行中的可预期错误，message 会作为 observation 回填给模型。"""


@dataclass
class ToolContext:
    """工具执行上下文：标识当前用户与会话，用于状态隔离。"""

    user_id: str = "default"
    session_id: str = "default"


@dataclass
class ToolResult:
    ok: bool
    content: str


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema (type=object)
    func: Callable[[dict[str, Any], ToolContext], Any]

    def schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


def validate_args(schema: dict[str, Any], args: Any) -> None:
    """按简化 JSON Schema 校验参数（必填、类型、枚举、未知字段），失败抛 ToolError。"""
    if not isinstance(args, dict):
        raise ToolError("arguments must be a JSON object")
    props = schema.get("properties", {})
    missing = [k for k in schema.get("required", []) if k not in args]
    if missing:
        raise ToolError(f"missing required argument(s): {', '.join(missing)}")
    unknown = [k for k in args if k not in props]
    if unknown:
        raise ToolError(f"unknown argument(s): {', '.join(unknown)}")
    for key, value in args.items():
        spec = props[key]
        types = _TYPES.get(spec.get("type", ""), ())
        # bool 是 int 的子类，数值类型需排除
        if types and (not isinstance(value, types) or (bool not in types and isinstance(value, bool))):
            raise ToolError(f"argument '{key}' must be {spec['type']}")
        if "enum" in spec and value not in spec["enum"]:
            raise ToolError(f"argument '{key}' must be one of {spec['enum']}")


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool
        return tool

    def tool(self, name: str, description: str, parameters: dict[str, Any]):
        """装饰器形式注册工具。"""

        def deco(func):
            self.register(Tool(name, description, parameters, func))
            return func

        return deco

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def execute(self, name: str, args: Any, ctx: ToolContext | None = None) -> ToolResult:
        """校验并执行工具；任何失败都转为 ok=False 的结果，不向外抛异常。"""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(False, f"unknown tool '{name}'. available: {', '.join(self._tools)}")
        try:
            validate_args(tool.parameters, args)
            out = tool.func(args, ctx or ToolContext())
        except ToolError as e:
            return ToolResult(False, str(e))
        except Exception as e:  # 工具内部未预期异常
            return ToolResult(False, f"tool '{name}' failed: {type(e).__name__}: {e}")
        text = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
        return ToolResult(True, text)
