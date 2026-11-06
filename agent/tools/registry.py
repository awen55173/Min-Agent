"""工具接口与注册表：名称、描述、参数 Schema，支持同步与异步（含超时与取消）执行。"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import inspect
import json
import threading
from typing import Any, Awaitable, Callable
import uuid

TurnId = str | int  # 对话轮次标识；同一会话内必须唯一


def new_turn_id() -> str:
    """生成全局唯一的轮次 ID（不依赖实例计数，重启、多实例、复用注册表都不会冲突）。"""
    return f"turn-{uuid.uuid4().hex[:16]}"


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
    turn_id: TurnId | None = None


@dataclass
class ToolResult:
    """工具执行结果：ok 表示是否成功，content 为回填给模型的文本。"""

    ok: bool
    content: str
    timed_out: bool = False
    execution_state: str = "completed"


@dataclass
class Tool:
    """工具定义：名称、描述、参数 JSON Schema、同步执行函数与可选异步执行函数。"""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema (type=object)
    func: Callable[[dict[str, Any], ToolContext], Any]
    afunc: Callable[[dict[str, Any], ToolContext], Awaitable[Any]] | None = None
    read_only: bool = True

    def schema(self) -> dict[str, Any]:
        """返回注入 prompt 的工具描述。"""
        return {"name": self.name, "description": self.description, "parameters": self.parameters}

    def openai_schema(self) -> dict[str, Any]:
        """返回 OpenAI Chat Completions tools 参数所需的函数工具定义。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


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
    """工具注册表：负责注册、查询、生成 schema 以及同步/异步统一执行。"""

    def __init__(self) -> None:
        """初始化空注册表。"""
        self._tools: dict[str, Tool] = {}
        self._write_lock = threading.Lock()

    def register(self, tool: Tool) -> Tool:
        """注册工具，重名则抛 ValueError。"""
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool
        return tool

    def tool(self, name: str, description: str, parameters: dict[str, Any], *, read_only: bool = True):
        """装饰器形式注册同步或异步工具。"""

        def deco(func):
            """把被装饰函数注册为工具并原样返回。"""
            afn = func if inspect.iscoroutinefunction(func) else None
            self.register(Tool(name, description, parameters, func, afunc=afn, read_only=read_only))
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

    def openai_tools(self) -> list[dict[str, Any]]:
        """导出所有已注册工具的 OpenAI 函数调用 Schema 列表。"""
        return [t.openai_schema() for t in self._tools.values()]

    def execute(self, name: str, args: Any, ctx: ToolContext | None = None) -> ToolResult:
        """同步校验并执行工具；校验、执行、结果序列化的任何失败都转为 ok=False，不向外抛异常。"""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(False, f"unknown tool '{name}'. available: {', '.join(self._tools)}")
        try:
            validate_args(tool.parameters, args)
            out = tool.func(args, ctx or ToolContext())
            if inspect.isawaitable(out):
                out = asyncio.run(out)
            return ToolResult(True, serialize_result(out))
        except ToolError as e:
            return ToolResult(False, str(e))
        except Exception as e:  # 工具内部未预期异常
            return ToolResult(False, f"tool '{name}' failed: {type(e).__name__}: {e}")

    async def aexecute(
        self,
        name: str,
        args: Any,
        ctx: ToolContext | None = None,
        timeout: float | None = None,
    ) -> ToolResult:
        """在线程中运行同步工具；只读超时报告未确定状态，写操作等完成后报告结果。"""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(False, f"unknown tool '{name}'. available: {', '.join(self._tools)}")
        try:
            validate_args(tool.parameters, args)
            eff_ctx = ctx or ToolContext()

            async def _invoke() -> Any:
                """统一封装异步函数、协程返回值与普通同步函数的调用。"""
                if tool.afunc is not None:
                    return await tool.afunc(args, eff_ctx)
                if inspect.iscoroutinefunction(tool.func):
                    return await tool.func(args, eff_ctx)
                res = await asyncio.to_thread(tool.func, args, eff_ctx)
                if inspect.isawaitable(res):
                    return await res
                return res

            if tool.read_only:
                coro = _invoke()
                out = await (asyncio.wait_for(coro, timeout=timeout) if timeout is not None else coro)
                return ToolResult(True, serialize_result(out))

            # 写操作只允许串行执行；超时后仍等待线程结束，避免未知副作用与后续写入竞态。
            while not self._write_lock.acquire(blocking=False):
                await asyncio.sleep(0.005)
            try:
                task = asyncio.create_task(_invoke())
                try:
                    out = await (asyncio.wait_for(asyncio.shield(task), timeout=timeout)
                                 if timeout is not None else asyncio.shield(task))
                    return ToolResult(True, serialize_result(out))
                except asyncio.CancelledError:
                    # 调用方取消后仍等写入结束，再释放串行锁并传递取消信号。
                    try:
                        await asyncio.shield(task)
                    except Exception:
                        pass
                    raise
                except asyncio.TimeoutError:
                    try:
                        out = await task
                        return ToolResult(True, serialize_result(out), True, "completed_after_timeout")
                    except ToolError as e:
                        return ToolResult(False, str(e), True, "completed_after_timeout")
                    except Exception as e:
                        return ToolResult(False, f"tool '{name}' failed: {type(e).__name__}: {e}",
                                          True, "completed_after_timeout")
            finally:
                self._write_lock.release()
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            return ToolResult(False, f"tool '{name}' timed out after {timeout}s; "
                              "the background execution may still be running", True, "unknown_after_timeout")
        except ToolError as e:
            return ToolResult(False, str(e))
        except Exception as e:
            return ToolResult(False, f"tool '{name}' failed: {type(e).__name__}: {e}")
