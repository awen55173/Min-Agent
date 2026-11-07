"""带会话持久化、受限上下文管理、写操作即时落盘与 Trace 日志的对话服务。"""
from __future__ import annotations

import copy
import re
from typing import Any, Callable

from .context import (
    DEFAULT_CONTEXT_TOKENS,
    DEFAULT_RESERVED_TOKENS,
    ContextBudgetError,
    _ensure_tool_Pairs,
    build_context,
    refresh_summary_todos,
)
from .llm import LLMClient, Message
from .memory import apply_user_memory, select_memories, wants_memory_recall
from .loop import CONTEXT_LIMIT, INTERRUPTED, OK, PERSIST_FAILED, Agent, AgentResult
from .prompt import build_system_prompt
from .session import SessionError, SessionManager
from .tools.registry import ToolContext, ToolRegistry
from .tools.todo import TodoStore
from .trace import TraceLogger

INTERRUPTED_NOTE = "（本轮被意外中断，未能给出回复；已完成的操作已保存。）"

RegistryFactory = Callable[[TodoStore], ToolRegistry]

_NEW_INTENT_RE = re.compile(
    r"(另外|额外|再记|再加|再创建|再添加|再建|新增一条|多记|多加|第二条|新的一条|另一个|另一条|同名|重新创建一条|another|one\s+more|additional)",
    re.I,
)
_RETRY_INTENT_RE = re.compile(
    r"(继续|重试|接着|刚才|恢复|上次|未完成|retry|continue|resume)",
    re.I,
)


def _is_retry_request(unfinished: dict[str, Any] | None, text: str, retry: bool | None = None) -> bool:
    """判断本次输入是否为对上一轮未完成请求的恢复重试（与新建请求区分）。"""
    if not isinstance(unfinished, dict) or not isinstance(unfinished.get("turn_id"), (int, str)):
        return False
    if retry is not None:
        return bool(retry)
    norm_text = " ".join(text.strip().split())
    prev_prompt = " ".join(str(unfinished.get("prompt", "")).strip().split())
    if norm_text and norm_text == prev_prompt:
        return True
    if _NEW_INTENT_RE.search(norm_text):
        return False
    if _RETRY_INTENT_RE.search(norm_text):
        return True
    return False


class ChatService:
    """把 Agent、Context、TraceLogger 与 SessionManager 串起来：一次 send 对应会话内的一轮对话。"""

    def __init__(
        self,
        llm: LLMClient,
        manager: SessionManager,
        registry_factory: RegistryFactory,
        max_steps: int = 8,
        max_output_retries: int = 3,
        max_history_turns: int = 6,
        max_context_tokens: int = DEFAULT_CONTEXT_TOKENS,
        reserved_tokens: int = DEFAULT_RESERVED_TOKENS,
        trace_logger: TraceLogger | None = None,
        tool_timeout: float | None = None,
        parallel_tool_calls: bool = True,
        max_concurrency: int = 4,
    ):
        """保存依赖及循环/上下文预算参数。"""
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
        self._llm = llm
        self._manager = manager
        self._factory = registry_factory
        self._max_steps = max_steps
        self._max_output_retries = max_output_retries
        self._max_history_turns = max_history_turns
        self._max_context_tokens = max_context_tokens
        self._reserved_tokens = reserved_tokens
        self._trace_logger = trace_logger or TraceLogger(self._manager._root / "traces")
        self._tool_timeout = tool_timeout
        self._parallel_tool_calls = parallel_tool_calls
        self._max_concurrency = max_concurrency

    @property
    def trace_logger(self) -> TraceLogger:
        """返回关联的 TraceLogger 实例。"""
        return self._trace_logger

    def send(
        self,
        user_id: str,
        session_id: str,
        text: str,
        create: bool = False,
        retry: bool | None = None,
    ) -> AgentResult:
        """在会话内处理一轮输入并保存。

        - 明确区分恢复重试与新建请求：仅当恢复重试未完成轮次时复用原 turn_id 与 call_id，新请求分配新 turn_id；
        - 待办写操作（add/complete/delete）成功后立即将待办状态与成功调用记录在同一原子保存中落盘；
        - 收尾保存先于 turn_end 日志执行，若最终落盘失败则统一返回 persist_error 并记录真实失败状态。
        """
        with self._manager.open(user_id, session_id, create=create) as session:
            data = session.data
            if data.history:
                data.history = _ensure_tool_Pairs(data.history)
                if data.history[-1].get("role") in ("user", "tool"):
                    data.history.append(_assistant_final(INTERRUPTED_NOTE))

            if _is_retry_request(data.unfinished_turn, text, retry=retry):
                assert data.unfinished_turn is not None
                turn_id = data.unfinished_turn["turn_id"]
                prev_attempt = data.unfinished_turn.get("attempt", 1)
                next_from_state = (prev_attempt + 1) if isinstance(prev_attempt, int) else 2
                attempt_id = max(next_from_state, self._trace_logger.next_attempt_id(user_id, session_id, turn_id))
                tracked_prompt = str(data.unfinished_turn.get("prompt") or text)
            else:
                turn_id = session.next_turn_id()
                attempt_id = self._trace_logger.next_attempt_id(user_id, session_id, turn_id)
                tracked_prompt = text

            ctx = ToolContext(user_id, session_id, turn_id)
            store = TodoStore()
            store.load_state(ctx, data.todos)
            registry = self._factory(store)
            data.memories = apply_user_memory(data.memories, text)
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
                    tools_schema=registry.openai_tools(),
                    recalled_memories=select_memories(data.memories, text),
                    memory_question=wants_memory_recall(text),
                    session_memories=data.memories,
                )
            except ContextBudgetError as e:
                ans = f"{e}\n本轮尚未执行任何工具操作。\n未完成说明：输入或固定上下文超出预算限制，未调用模型。"
                ev_start = self._trace_logger.log_turn_start(
                    user_id, session_id, turn_id, text, attempt_id=attempt_id,
                )
                prev_history = list(data.history)
                prev_unfinished = copy.deepcopy(data.unfinished_turn)
                messages = [{"role": "user", "content": text}, _assistant_final(ans)]
                data.history += messages
                data.unfinished_turn = {
                    "turn_id": turn_id,
                    "attempt": attempt_id,
                    "prompt": tracked_prompt,
                    "status": CONTEXT_LIMIT,
                }
                final_status = CONTEXT_LIMIT
                try:
                    session.save()
                except (OSError, SessionError) as save_err:
                    data.history = prev_history
                    data.unfinished_turn = prev_unfinished
                    final_status = PERSIST_FAILED
                    ans = f"保存写操作失败：{save_err}\n本轮尚未执行任何工具操作。\n未完成说明：{e}；且会话状态未能落盘。"
                ev_end = self._trace_logger.log_turn_end(
                    user_id, session_id, turn_id, final_status, ans, 0, 0.0,
                    error=ans, attempt_id=attempt_id,
                )
                return AgentResult(
                    final_status, ans, messages=messages, turn_id=turn_id,
                    trace_events=[ev_start, ev_end], attempt_id=attempt_id,
                )

            data.summary = ctx_plan.summary
            base_history = list(data.history)
            prefix_len = len(ctx_plan.messages) - 1  # 本轮 user_input 之前的上下文消息数
            data.unfinished_turn = {
                "turn_id": turn_id,
                "attempt": attempt_id,
                "prompt": tracked_prompt,
                "status": "in_progress",
            }

            def on_step_change(cur_messages: list[Message], rec: dict, persist_now: bool) -> None:
                """在工具状态变更时同步会话数据；待办写操作成功后立即原子保存待办状态与调用记录。"""
                prev_history = list(data.history)
                prev_todos = copy.deepcopy(data.todos)
                prev_summary = data.summary
                prev_unfinished = copy.deepcopy(data.unfinished_turn)
                data.todos = store.export_state(ctx)
                data.history = base_history + cur_messages[prefix_len:]
                data.summary = refresh_summary_todos(data.summary, data.todos)
                data.unfinished_turn = {
                    "turn_id": turn_id,
                    "attempt": attempt_id,
                    "prompt": tracked_prompt,
                    "status": "in_progress",
                }
                if persist_now:
                    try:
                        session.save()
                    except (OSError, SessionError):
                        data.history = prev_history
                        data.todos = prev_todos
                        data.summary = prev_summary
                        data.unfinished_turn = prev_unfinished
                        raise

            def on_turn_end(status: str, answer: str, cur_messages: list[Message]) -> None:
                """在记录 turn_end 日志前执行最终会话落盘；失败时恢复内存状态并向上抛出以转为 persist_error。"""
                prev_history = list(data.history)
                prev_todos = copy.deepcopy(data.todos)
                prev_summary = data.summary
                prev_unfinished = copy.deepcopy(data.unfinished_turn)
                new_messages = list(cur_messages[prefix_len:])
                if status != OK:
                    new_messages.append(_assistant_final(answer))
                    data.unfinished_turn = {
                        "turn_id": turn_id,
                        "attempt": attempt_id,
                        "prompt": tracked_prompt,
                        "status": status,
                    }
                else:
                    data.unfinished_turn = None
                data.history = base_history + new_messages
                data.todos = store.export_state(ctx)
                data.summary = refresh_summary_todos(data.summary, data.todos)
                try:
                    session.save()
                except (OSError, SessionError):
                    data.history = prev_history
                    data.todos = prev_todos
                    data.summary = prev_summary
                    data.unfinished_turn = prev_unfinished
                    raise

            agent = Agent(
                self._llm,
                registry,
                self._max_steps,
                self._max_output_retries,
                max_context_tokens=self._max_context_tokens,
                trace_logger=self._trace_logger,
                parallel_tool_calls=self._parallel_tool_calls,
                tool_timeout=self._tool_timeout,
                max_concurrency=self._max_concurrency,
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
                    on_turn_end=on_turn_end,
                    trace_logger=self._trace_logger,
                    attempt_id=attempt_id,
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
                data.unfinished_turn = {
                    "turn_id": turn_id,
                    "attempt": attempt_id,
                    "prompt": tracked_prompt,
                    "status": INTERRUPTED,
                }
                try:
                    session.save()
                except (OSError, SessionError):
                    pass
                raise

            return result

    async def asend(
        self,
        user_id: str,
        session_id: str,
        text: str,
        create: bool = False,
        retry: bool | None = None,
    ) -> AgentResult:
        """异步在会话内处理一轮输入并保存，使用非阻塞会话锁与异步保存避免阻塞其他并发会话。"""
        async with self._manager.aopen(user_id, session_id, create=create) as session:
            data = session.data
            if data.history:
                data.history = _ensure_tool_Pairs(data.history)
                if data.history[-1].get("role") in ("user", "tool"):
                    data.history.append(_assistant_final(INTERRUPTED_NOTE))

            if _is_retry_request(data.unfinished_turn, text, retry=retry):
                assert data.unfinished_turn is not None
                turn_id = data.unfinished_turn["turn_id"]
                prev_attempt = data.unfinished_turn.get("attempt", 1)
                next_from_state = (prev_attempt + 1) if isinstance(prev_attempt, int) else 2
                attempt_id = max(next_from_state, self._trace_logger.next_attempt_id(user_id, session_id, turn_id))
                tracked_prompt = str(data.unfinished_turn.get("prompt") or text)
            else:
                turn_id = session.next_turn_id()
                attempt_id = self._trace_logger.next_attempt_id(user_id, session_id, turn_id)
                tracked_prompt = text

            ctx = ToolContext(user_id, session_id, turn_id)
            store = TodoStore()
            store.load_state(ctx, data.todos)
            registry = self._factory(store)
            data.memories = apply_user_memory(data.memories, text)
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
                    tools_schema=registry.openai_tools(),
                    recalled_memories=select_memories(data.memories, text),
                    memory_question=wants_memory_recall(text),
                    session_memories=data.memories,
                )
            except ContextBudgetError as e:
                ans = f"{e}\n本轮尚未执行任何工具操作。\n未完成说明：输入或固定上下文超出预算限制，未调用模型。"
                ev_start = self._trace_logger.log_turn_start(
                    user_id, session_id, turn_id, text, attempt_id=attempt_id,
                )
                prev_history = list(data.history)
                prev_unfinished = copy.deepcopy(data.unfinished_turn)
                messages = [{"role": "user", "content": text}, _assistant_final(ans)]
                data.history += messages
                data.unfinished_turn = {
                    "turn_id": turn_id,
                    "attempt": attempt_id,
                    "prompt": tracked_prompt,
                    "status": CONTEXT_LIMIT,
                }
                final_status = CONTEXT_LIMIT
                try:
                    await session.asave()
                except (OSError, SessionError) as save_err:
                    data.history = prev_history
                    data.unfinished_turn = prev_unfinished
                    final_status = PERSIST_FAILED
                    ans = f"保存写操作失败：{save_err}\n本轮尚未执行任何工具操作。\n未完成说明：{e}；且会话状态未能落盘。"
                ev_end = self._trace_logger.log_turn_end(
                    user_id, session_id, turn_id, final_status, ans, 0, 0.0,
                    error=ans, attempt_id=attempt_id,
                )
                return AgentResult(
                    final_status, ans, messages=messages, turn_id=turn_id,
                    trace_events=[ev_start, ev_end], attempt_id=attempt_id,
                )

            data.summary = ctx_plan.summary
            base_history = list(data.history)
            prefix_len = len(ctx_plan.messages) - 1
            data.unfinished_turn = {
                "turn_id": turn_id,
                "attempt": attempt_id,
                "prompt": tracked_prompt,
                "status": "in_progress",
            }

            async def on_step_change(cur_messages: list[Message], rec: dict, persist_now: bool) -> None:
                """异步在工具状态变更时同步会话数据；待办写操作成功后立即异步原子保存。"""
                prev_history = list(data.history)
                prev_todos = copy.deepcopy(data.todos)
                prev_summary = data.summary
                prev_unfinished = copy.deepcopy(data.unfinished_turn)
                data.todos = store.export_state(ctx)
                data.history = base_history + cur_messages[prefix_len:]
                data.summary = refresh_summary_todos(data.summary, data.todos)
                data.unfinished_turn = {
                    "turn_id": turn_id,
                    "attempt": attempt_id,
                    "prompt": tracked_prompt,
                    "status": "in_progress",
                }
                if persist_now:
                    try:
                        await session.asave()
                    except (OSError, SessionError):
                        data.history = prev_history
                        data.todos = prev_todos
                        data.summary = prev_summary
                        data.unfinished_turn = prev_unfinished
                        raise

            async def on_turn_end(status: str, answer: str, cur_messages: list[Message]) -> None:
                """异步在记录 turn_end 日志前执行最终会话落盘；失败时恢复内存状态并向上抛出。"""
                prev_history = list(data.history)
                prev_todos = copy.deepcopy(data.todos)
                prev_summary = data.summary
                prev_unfinished = copy.deepcopy(data.unfinished_turn)
                new_messages = list(cur_messages[prefix_len:])
                if status != OK:
                    new_messages.append(_assistant_final(answer))
                    data.unfinished_turn = {
                        "turn_id": turn_id,
                        "attempt": attempt_id,
                        "prompt": tracked_prompt,
                        "status": status,
                    }
                else:
                    data.unfinished_turn = None
                data.history = base_history + new_messages
                data.todos = store.export_state(ctx)
                data.summary = refresh_summary_todos(data.summary, data.todos)
                try:
                    await session.asave()
                except (OSError, SessionError):
                    data.history = prev_history
                    data.todos = prev_todos
                    data.summary = prev_summary
                    data.unfinished_turn = prev_unfinished
                    raise

            agent = Agent(
                self._llm,
                registry,
                self._max_steps,
                self._max_output_retries,
                max_context_tokens=self._max_context_tokens,
                trace_logger=self._trace_logger,
                parallel_tool_calls=self._parallel_tool_calls,
                tool_timeout=self._tool_timeout,
                max_concurrency=self._max_concurrency,
            )
            try:
                result = await agent.arun(
                    text,
                    user_id,
                    session_id,
                    turn_id,
                    context_messages=ctx_plan.messages,
                    call_records=data.tool_calls,
                    todo_store=store,
                    on_step_change=on_step_change,
                    on_turn_end=on_turn_end,
                    trace_logger=self._trace_logger,
                    attempt_id=attempt_id,
                )
            except BaseException as e:
                partial: AgentResult | None = getattr(e, "agent_result", None)
                if partial is not None:
                    new_messages = partial.messages[prefix_len:]
                else:
                    new_messages = [{"role": "user", "content": text}]
                new_messages.append(_assistant_final(INTERRUPTED_NOTE))
                data.history = base_history + new_messages
                data.todos = store.export_state(ctx)
                data.summary = refresh_summary_todos(data.summary, data.todos)
                data.unfinished_turn = {
                    "turn_id": turn_id,
                    "attempt": attempt_id,
                    "prompt": tracked_prompt,
                    "status": INTERRUPTED,
                }
                try:
                    await session.asave()
                except (OSError, SessionError):
                    pass
                raise

            return result


def _assistant_final(text: str) -> dict[str, str]:
    """构造直接文本形式的 assistant 结束回复。"""
    return {"role": "assistant", "content": text}
