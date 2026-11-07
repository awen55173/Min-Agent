"""原生 Agent 的结果、预算异常和调用记录。"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from .llm import Message
from .tools.registry import TurnId

OK = "ok"
MAX_STEPS = "max_steps"
OUTPUT_EXHAUSTED = "output_retries_exhausted"
LLM_FAILED = "llm_error"
INTERNAL_ERROR = "internal_error"
INTERRUPTED = "interrupted"
TOOL_NOT_EXECUTED = "tool_not_executed"
CONTEXT_LIMIT = "context_limit"
PERSIST_FAILED = "persist_error"

class ModelOutputError(Exception):
    """携带模型预算耗尽或回答无效时的失败状态。"""

    def __init__(self, message: str, status: str = OUTPUT_EXHAUSTED):
        """保存可读原因和失败状态。"""
        super().__init__(message)
        self.status = status

@dataclass
class ActionRecord:
    """一次已执行的工具调用记录：区分内部幂等 call_id 与模型提供的 api_call_id。"""

    step: int
    thought: str
    tool: str
    args: dict[str, Any]
    ok: bool
    result: str
    call_id: str = ""
    duration_ms: float = 0.0
    reused: bool = False
    api_call_id: str = ""

@dataclass
class AgentResult:
    """一次请求的结果：status 为终止原因，answer 为返回给用户的文本。"""

    status: str
    answer: str
    actions: list[ActionRecord] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    turn_id: TurnId | None = None
    trace_events: list[dict[str, Any]] = field(default_factory=list)
    attempt_id: int | str = 1

def _summarize(actions: list[ActionRecord], unfinished: str = "") -> str:
    """把已完成的工具操作与未完成说明整理为简短文本，用于各类非正常退出提示。"""
    if not actions:
        base = "本轮尚未执行任何工具操作。"
    else:
        lines = [f"- {a.tool}({a.args}) -> {'成功' if a.ok else '失败'}" for a in actions]
        base = "本轮已执行的操作：\n" + "\n".join(lines)
    if unfinished:
        return f"{base}\n未完成说明：{unfinished}"
    return base

def _find_call_record(records: list[dict[str, Any]], call_id: str) -> dict[str, Any] | None:
    """在调用记录列表中按内部幂等 call_id 查找已有记录。"""
    for rec in records:
        if rec.get("call_id") == call_id:
            return rec
    return None

def _upsert_call_record(records: list[dict[str, Any]], new_rec: dict[str, Any]) -> dict[str, Any]:
    """按 call_id 插入或原地更新一条调用记录，并返回列表中的字典引用。"""
    existing = _find_call_record(records, str(new_rec.get("call_id", "")))
    if existing is not None:
        existing.update(new_rec)
        return existing
    records.append(new_rec)
    return new_rec
