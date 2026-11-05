"""工具接口与注册表：名称、描述、参数 Schema，并在执行前校验参数。"""
from __future__ import annotations

import json
from dataclasses import dataclass
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
    """工具执行上下文：当前用户、会话及对话轮次（turn_id 用于同一轮内的写操作去重）。"""

    user_id: str = "default"
    session_id: str = "default"
    turn_id: int | None = None


@dataclass
class ToolResult:
    """工具执行结果：ok 表示是否成功，content 为回填给模型的文本。"""

    ok: bool
    content: str


@dataclass
class Tool:
    """工具定义：名称、描述、参数 JSON Schema 与执行函数。"""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema (type=object)
    func: Callable[[dict[str, Any], ToolContext], Any]

    def schema(self) -> dict[str, Any]:
        """返回注入 prompt 的工具描述。"""
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


def serialize_result(out: Any) -> str:
    """将工具返回值转为标准 JSON 文本（字符串原样返回）；无法序列化（如 NaN/复数）抛 ToolError。"""
    if isinstance(out, str):
        return out
    try:
        return json.dumps(out, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as e:
        raise ToolError(f"tool result is not JSON serializable: {e}") from None


class ToolRegistry:
    """工具注册表：负责注册、查询、生成 schema 和统一执行。"""

    def __init__(self) -> None:
        """初始化空注册表。"""
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        """注册工具，重名则抛 ValueError。"""
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool
        return tool

    def tool(self, name: str, description: str, parameters: dict[str, Any]):
        """装饰器形式注册工具。"""

        def deco(func):
            """把被装饰函数注册为工具并原样返回。"""
            self.register(Tool(name, description, parameters, func))
            return func

        return deco

    def get(self, name: str) -> Tool | None:
        """按名称获取工具，不存在返回 None。"""
        return self._tools.get(name)

    def names(self) -> list[str]:
        """返回所有已注册工具名。"""
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        """返回所有工具的 schema 列表。"""
        return [t.schema() for t in self._tools.values()]

    def execute(self, name: str, args: Any, ctx: ToolContext | None = None) -> ToolResult:
        """校验并执行工具；校验、执行、结果序列化的任何失败都转为 ok=False，不向外抛异常。"""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(False, f"unknown tool '{name}'. available: {', '.join(self._tools)}")
        try:
            validate_args(tool.parameters, args)
            return ToolResult(True, serialize_result(tool.func(args, ctx or ToolContext())))
        except ToolError as e:
            return ToolResult(False, str(e))
        except Exception as e:  # 工具内部未预期异常
            return ToolResult(False, f"tool '{name}' failed: {type(e).__name__}: {e}")
