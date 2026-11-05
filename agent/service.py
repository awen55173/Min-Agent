"""带会话持久化、受限上下文管理、写操作即时落盘与 Trace 日志的对话服务。"""
from __future__ import annotations

import copy
import json
from typing import Callable

from .context import (
    DEFAULT_CONTEXT_TOKENS,
    DEFAULT_RESERVED_TOKENS,
    ContextBudgetError,
    build_context,
    refresh_summary_todos,
)
from .llm import LLMClient, Message
from .loop import CONTEXT_LIMIT, INTERRUPTED, OK, PERSIST_FAILED, Agent, AgentResult
from .prompt import build_system_prompt
from .session import SessionError, SessionManager
from .tools.registry import ToolContext, ToolRegistry
from .tools.todo import TodoStore
from .trace import TraceLogger

INTERRUPTED_NOTE = "（本轮被意外中断，未能给出回复；已完成的操作已保存。）"

RegistryFactory = Callable[[TodoStore], ToolRegistry]


class ChatService:
    """把 Agent、Context、TraceLogger 与 SessionManager 串起来：一次 send 对应会话内的一轮对话。"""

    def __init__(
        self,
        llm: LLMClient,
        manager: SessionManager,
        registry_factory: RegistryFactory,
        max_steps: int = 8,
        max_parse_retries: int = 3,
        max_history_turns: int = 6,
        max_context_tokens: int = DEFAULT_CONTEXT_TOKENS,
        reserved_tokens: int = DEFAULT_RESERVED_TOKENS,
        trace_logger: TraceLogger | None = None,
    ):
        """保存依赖及循环/上下文预算参数。"""
        self._llm = llm
        self._manager = manager
        self._factory = registry_factory
        self._max_steps = max_steps
        self._max_parse_retries = max_parse_retries
        self._max_history_turns = max_history_turns
        self._max_context_tokens = max_context_tokens
        self._reserved_tokens = reserved_tokens
        self._trace_logger = trace_logger or TraceLogger(self._manager._root / "traces")

    @property
    def trace_logger(self) -> TraceLogger:
        """返回关联的 TraceLogger 实例。"""
        return self._trace_logger

    def send(self, user_id: str, session_id: str, text: str, create: bool = False) -> AgentResult:
        """在会话内处理一轮输入并保存。

        - 若上一轮因进程崩溃遗留未闭合的 Observation，先补齐中断说明以保证历史结构完整；
        - 若存在未完成轮次（unfinished_turn），恢复/重试时复用原 turn_id 与 call_id，实现写操作幂等；
        - 待办写操作（add/complete/delete）成功后立即将待办状态与成功调用记录在同一原子保存中落盘；
        - 全程记录 JSONL Trace（含正常结束、模型失败、预算耗尽、保存失败及中断）。
        """
        with self._manager.open(user_id, session_id, create=create) as session:
            data = session.data
            if data.history and data.history[-1].get("role") == "user":
                data.history.append(_assistant_final(INTERRUPTED_NOTE))

            if data.unfinished_turn is not None and isinstance(data.unfinished_turn.get("turn_id"), (int, str)):
                turn_id = data.unfinished_turn["turn_id"]
            else:
                turn_id = session.next_turn_id()

            ctx = ToolContext(user_id, session_id, turn_id)
            store = TodoStore()
            store.load_state(ctx, data.todos)
            registry = self._factory(store)
            try:
                ctx_plan = build_context(
                    system_prompt=build_system_prompt(registry),
                    history=data.history,
                    user_input=text,
                    existing_summary=data.summary,
                    todos=data.todos,
                    max_history_turns=self._max_history_turns,
                    max_context_tokens=self._max_context_tokens,
                    reserved_tokens=self._reserved_tokens,
                )
            except ContextBudgetError as e:
                ans = f"{e}\n本轮尚未执行任何工具操作。\n未完成说明：输入或固定上下文超出预算限制，未调用模型。"
                ev_start = self._trace_logger.log_turn_start(user_id, session_id, turn_id, text)
                ev_end = self._trace_logger.log_turn_end(
                    user_id, session_id, turn_id, CONTEXT_LIMIT, ans, 0, 0.0, error=ans,
                )
                messages = [{"role": "user", "content": text}, _assistant_final(ans)]
                data.history += messages
                data.unfinished_turn = {"turn_id": turn_id, "prompt": text, "status": CONTEXT_LIMIT}
                session.save()
                return AgentResult(
                    CONTEXT_LIMIT, ans, messages=messages, turn_id=turn_id, trace_events=[ev_start, ev_end],
                )

            data.summary = ctx_plan.summary
            base_history = list(data.history)
            prefix_len = len(ctx_plan.messages) - 1  # 本轮 user_input 之前的上下文消息数
            data.unfinished_turn = {"turn_id": turn_id, "prompt": text, "status": "in_progress"}

            def on_step_change(cur_messages: list[Message], rec: dict, persist_now: bool) -> None:
                """在工具状态变更时同步会话数据；待办写操作成功后立即原子保存待办状态与调用记录。"""
                prev_history = list(data.history)
                prev_todos = copy.deepcopy(data.todos)
                prev_summary = data.summary
                data.todos = store.export_state(ctx)
                data.history = base_history + cur_messages[prefix_len:]
                data.summary = refresh_summary_todos(data.summary, data.todos)
                data.unfinished_turn = {"turn_id": turn_id, "prompt": text, "status": "in_progress"}
                if persist_now:
                    try:
                        session.save()
                    except (OSError, SessionError):
                        data.history = prev_history
                        data.todos = prev_todos
                        data.summary = prev_summary
                        raise

            agent = Agent(
                self._llm,
                registry,
                self._max_steps,
                self._max_parse_retries,
                max_context_tokens=self._max_context_tokens,
                trace_logger=self._trace_logger,
            )
            try:
                result = agent.run(
                    text,
                    user_id,
                    session_id,
                    turn_id,
                    context_messages=ctx_plan.messages,
                    call_records=data.tool_calls,
                    todo_store=store,
                    on_step_change=on_step_change,
                    trace_logger=self._trace_logger,
                )
            except BaseException as e:
                partial: AgentResult | None = getattr(e, "agent_result", None)
                if partial is not None:
                    new_messages = partial.messages[prefix_len:]
                else:  # 尚未进入循环就被中断
                    new_messages = [{"role": "user", "content": text}]
                new_messages.append(_assistant_final(INTERRUPTED_NOTE))
                data.history = base_history + new_messages
                data.todos = store.export_state(ctx)
                data.summary = refresh_summary_todos(data.summary, data.todos)
                data.unfinished_turn = {"turn_id": turn_id, "prompt": text, "status": INTERRUPTED}
                session.save()
                raise

            new_messages = result.messages[prefix_len:]  # 只追加本轮新增的原始消息，保留完整历史
            if result.status != OK:
                # 失败提示作为 assistant 回复入历史，保持对话轮次完整，模型后续追问能知道发生了什么
                new_messages.append(_assistant_final(result.answer))
                data.unfinished_turn = {"turn_id": turn_id, "prompt": text, "status": result.status}
            else:
                data.unfinished_turn = None
            data.history = base_history + new_messages
            data.todos = store.export_state(ctx)
            data.summary = refresh_summary_todos(data.summary, data.todos)
            if result.status == PERSIST_FAILED:
                try:
                    session.save()
                except (OSError, SessionError):
                    pass
            else:
                session.save()
            return result


def _assistant_final(text: str) -> dict[str, str]:
    """构造一条以 final_answer 形式记录的 assistant 消息。"""
    return {"role": "assistant", "content": json.dumps({"final_answer": text}, ensure_ascii=False)}
