"""原生 Agent 循环：预算、思考、回答校验、执行、观察与收尾分别处理。"""
from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field
import json
import time
from typing import Any, Callable

from .context import DEFAULT_CONTEXT_TOKENS, ContextBudgetError, check_context_budget
from .execution import NativeToolExecutor, _BatchCallItem, _maybe_await
from .llm import LLMClient, LLMError, LLMResponse, Message
from .prompt import build_system_prompt
from .runtime import (
    ActionRecord, AgentResult, ModelOutputError, _summarize,
    OK, MAX_STEPS, OUTPUT_EXHAUSTED, LLM_FAILED, INTERNAL_ERROR,
    INTERRUPTED, TOOL_NOT_EXECUTED, CONTEXT_LIMIT, PERSIST_FAILED,
)
from .session import SessionError
from .tool_call import ToolCallError
from .tools.registry import ToolContext, ToolRegistry, TurnId, new_turn_id
from .tools.todo import TodoStore
from .trace import TraceLogger


@dataclass
class Reply:
    """一次模型回复：直接回答或已校验、已分组的原生工具调用。"""

    is_answer: bool
    content: str = ""
    actions: list[list[_BatchCallItem]] = field(default_factory=list)


async def _drive(agent: _Turn) -> AgentResult:
    """按预算推进思考、工具执行与观察，或接受有效回答。"""
    while agent.may_think():
        reply = await agent.think()
        if reply.is_answer:
            if await agent.disputed(reply):
                continue
            return await agent.accept(reply, actions=agent.actions)
        for action in reply.actions:
            await agent.observe(await agent.perform(action))
    raise ModelOutputError("模型调用预算内未生成有效回答；" + agent.stop_reason, agent.stop_status)


class Agent:
    """配置原生工具运行时，每轮使用独立状态并复用异步主循环。"""

    def __init__(self, llm: LLMClient, registry: ToolRegistry, max_steps: int = 8,
                 max_output_retries: int = 3, max_context_tokens: int = DEFAULT_CONTEXT_TOKENS,
                 trace_logger: TraceLogger | None = None, parallel_tool_calls: bool = True,
                 tool_timeout: float | None = None, max_concurrency: int = 4):
        """保存模型、工具和预算配置，仅接受原生调用客户端。"""
        if max_steps < 0 or max_output_retries < 0 or max_context_tokens <= 0 or max_concurrency < 1:
            raise ValueError("budgets must be nonnegative; context and concurrency must be positive")
        if not callable(getattr(llm, "acomplete", None)) and not callable(getattr(llm, "complete", None)):
            raise TypeError("model client must implement native complete or acomplete")
        self._llm, self._registry = llm, registry
        self._max_steps, self._max_output_retries = max_steps, max_output_retries
        self._max_context_tokens, self._trace_logger = max_context_tokens, trace_logger
        self._parallel_tool_calls, self._tool_timeout = parallel_tool_calls, tool_timeout
        self._max_concurrency = max_concurrency

    def run(
        self,
        user_input: str,
        user_id: str = "default",
        session_id: str = "default",
        turn_id: TurnId | None = None,
        history: list[Message] | None = None,
        summary: str | None = None,
        context_messages: list[Message] | None = None,
        call_records: list[dict[str, Any]] | None = None,
        todo_store: TodoStore | None = None,
        on_step_change: Callable[[list[Message], dict[str, Any], bool], Any] | None = None,
        on_turn_end: Callable[[str, str, list[Message]], Any] | None = None,
        trace_logger: TraceLogger | None = None,
        attempt_id: int | str | None = None,
    ) -> AgentResult:
        """在同步入口复用异步循环，模型使用同步连接以支持多轮会话。"""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arun(
                user_input, user_id, session_id, turn_id, history, summary, context_messages,
                call_records, todo_store, on_step_change, on_turn_end, trace_logger, attempt_id,
                _sync_model=True))
        raise RuntimeError("事件循环正在运行，请使用 await agent.arun(...)")

    async def arun(
        self,
        user_input: str,
        user_id: str = "default",
        session_id: str = "default",
        turn_id: TurnId | None = None,
        history: list[Message] | None = None,
        summary: str | None = None,
        context_messages: list[Message] | None = None,
        call_records: list[dict[str, Any]] | None = None,
        todo_store: TodoStore | None = None,
        on_step_change: Callable[[list[Message], dict[str, Any], bool], Any] | None = None,
        on_turn_end: Callable[[str, str, list[Message]], Any] | None = None,
        trace_logger: TraceLogger | None = None,
        attempt_id: int | str | None = None,
        _sync_model: bool = False,
    ) -> AgentResult:
        """建立独立轮次状态，统一处理失败、中断及最终持久化。"""
        ctx = ToolContext(user_id, session_id, new_turn_id() if turn_id is None else turn_id)
        agent = _Turn(self, user_input, ctx,
                      history, summary, context_messages, call_records, todo_store,
                      on_step_change, on_turn_end, trace_logger, attempt_id, _sync_model)
        agent.start()
        try:
            return await _drive(agent)
        except ModelOutputError as error:
            return await agent.fail(error.status, str(error))
        except ContextBudgetError as error:
            return await agent.fail(CONTEXT_LIMIT, str(error))
        except LLMError as error:
            return await agent.fail(LLM_FAILED, f"模型服务暂时不可用，请稍后重试。（{error}）")
        except (OSError, SessionError) as error:
            agent.repair_interrupted()
            return await agent.fail(PERSIST_FAILED, f"保存写操作失败：{error}")
        except Exception as error:
            agent.repair_interrupted()
            return await agent.fail(INTERNAL_ERROR, f"内部错误：{type(error).__name__}: {error}")
        except BaseException as error:
            agent.repair_interrupted()
            error.agent_result = await agent.fail(INTERRUPTED, "本轮被中断，未生成最终回复。")
            raise


class _Turn(NativeToolExecutor):
    """封装单轮对话的预算、消息、工具状态和保存钩子。"""

    def __init__(self, owner, user_input, ctx, history, summary, context_messages, records,
                 store, on_step_change, on_turn_end, logger, attempt, sync_model):
        """初始化独立轮次状态并准备本轮模型上下文。"""
        self._llm, self._registry = owner._llm, owner._registry
        self._max_steps, self._max_output_retries = owner._max_steps, owner._max_output_retries
        self._max_context_tokens = owner._max_context_tokens
        self._parallel_tool_calls, self._tool_timeout = owner._parallel_tool_calls, owner._tool_timeout
        self._max_concurrency, self.sync_model = owner._max_concurrency, sync_model
        tool = self._registry.get("todo")
        candidate = getattr(tool, "store", None)
        self.store = store if store is not None else (candidate if isinstance(candidate, TodoStore) else None)
        self.ctx, self.user_input = ctx, user_input
        self.records = records if records is not None else (self.store.get_call_records(ctx) if self.store else [])
        self.logger = logger if logger is not None else owner._trace_logger
        self.attempt = attempt if attempt is not None else (self.logger.next_attempt_id(ctx.user_id, ctx.session_id, ctx.turn_id) if self.logger else 1)
        self.on_step_change, self.on_turn_end = on_step_change, on_turn_end
        if context_messages is not None:
            self.messages = copy.deepcopy(context_messages)
        else:
            self.messages = [{"role": "system", "content": build_system_prompt(self._registry)}]
            if summary and summary.strip():
                self.messages.append({"role": "system", "content": summary})
            self.messages.extend(copy.deepcopy(history or []))
            self.messages.append({"role": "user", "content": user_input})
        self.actions, self.events = [], []
        self.failures, self.model_calls = 0, 0
        self.unexecuted_tool_output = False
        self.budget_exceeded = False
        self._active_batch = None
        self.stop_status, self.stop_reason = OUTPUT_EXHAUSTED, "未得到有效回答"
        self.started = time.perf_counter()

    def start(self):
        """记录本轮开始事件。"""
        event = self.logger.log_turn_start(self.ctx.user_id, self.ctx.session_id, self.ctx.turn_id,
                                          self.user_input, attempt_id=self.attempt) if self.logger else {
            "event": "turn_start", "user_id": self.ctx.user_id, "session_id": self.ctx.session_id,
            "turn_id": self.ctx.turn_id, "attempt_id": self.attempt, "step": 0, "user_input": self.user_input}
        self.events.append(event)

    def may_think(self) -> bool:
        """检查工具步数、累计纠错次数及模型请求次数是否允许继续。"""
        if self.budget_exceeded:
            self.stop_status, self.stop_reason = MAX_STEPS, f"已达最大步数（{self._max_steps}），任务可能未完成。"
            return False
        return self.failures <= self._max_output_retries and self.model_calls < self._max_steps + self._max_output_retries + 1

    async def think(self) -> Reply:
        """检查消息与工具定义总预算，调用模型并准备原生回复。"""
        schema = self._registry.openai_tools()
        check_context_budget(self.messages, self._max_context_tokens, schema)
        self.model_calls += 1
        if not self.sync_model and callable(getattr(self._llm, "acomplete", None)):
            response = await self._llm.acomplete(self.messages, tools=schema, parallel_tool_calls=self._parallel_tool_calls)
        elif callable(getattr(self._llm, "complete", None)):
            response = await asyncio.to_thread(self._llm.complete, self.messages, tools=schema, parallel_tool_calls=self._parallel_tool_calls)
        else:
            response = await self._llm.acomplete(self.messages, tools=schema, parallel_tool_calls=self._parallel_tool_calls)
        return self.prepare_reply(response)

    def prepare_reply(self, response: LLMResponse) -> Reply:
        """区分文本回答和原生请求，校验工具并按依赖顺序分组。"""
        if not isinstance(response, LLMResponse):
            self.feedback("client must return LLMResponse")
            return Reply(False)
        if not response.has_tool_calls():
            return Reply(True, response.content)
        try:
            items, self.budget_exceeded = self._prepare_batch_items(
                response.tool_calls, response.content or "", self.actions, self.ctx, self.ctx.turn_id)
        except ToolCallError as error:
            self.unexecuted_tool_output = True
            self.feedback(str(error))
            return Reply(False)
        if not items:
            self.budget_exceeded = True
            return Reply(False)
        self._active_batch = items
        self.messages.append({"role": "assistant", "content": response.content or "", "tool_calls": [
            {"id": it.api_call_id, "type": "function", "function": {
                "name": it.tool_name, "arguments": json.dumps(it.args, ensure_ascii=False)}} for it in items]})
        return Reply(False, actions=[batch for _, batch in self._partition_batch_segments(items)])

    def feedback(self, reason: str, status: str = OUTPUT_EXHAUSTED):
        """记录校验原因并回填纠错提示，消耗一次累计重试预算。"""
        self.failures += 1
        self.stop_status, self.stop_reason = status, reason
        event = self.logger.log_validation_error(self.ctx.user_id, self.ctx.session_id, self.ctx.turn_id,
                    len(self.actions) + 1, reason, attempt_id=self.attempt) if self.logger else {
            "event": "validation_error", "user_id": self.ctx.user_id, "session_id": self.ctx.session_id,
            "turn_id": self.ctx.turn_id, "attempt_id": self.attempt, "step": len(self.actions) + 1, "error": reason}
        self.events.append(event)
        self.messages.append({"role": "user", "content": f"Validation error: {reason}. "
            "请返回非空回答或修正原生工具调用。本次工具未执行（the tool was not executed），请勿虚构成功结果。"})

    async def disputed(self, reply: Reply) -> bool:
        """拒绝空回答及工具未执行后的完成声称，不校验回答 JSON 格式。"""
        if not isinstance(reply.content, str) or not reply.content.strip():
            self.feedback("answer must be non-empty text")
            return True
        if self.unexecuted_tool_output:
            self.feedback("工具调用未执行，无法确认请求已完成；请先实际执行工具。", TOOL_NOT_EXECUTED)
            return True
        return False

    async def accept(self, reply: Reply, *, actions: list[ActionRecord]) -> AgentResult:
        """保存直接文本回答并完成本轮收尾。"""
        answer = reply.content.strip()
        self.messages.append({"role": "assistant", "content": answer})
        return await self.finish(OK, answer, actions)

    def repair_interrupted(self):
        """补齐执行中工具的消息与 Trace，供失败或取消时保存。"""
        self._prepare_interrupted_state(self.messages, self.actions, self.records, self.ctx,
                                        self.logger, self.events, self.attempt)

    async def fail(self, status: str, reason: str) -> AgentResult:
        """按失败状态返回原因与已完成操作，保留恢复所需记录。"""
        prefix = {MAX_STEPS: "已达最大步数，任务可能未完成。", OUTPUT_EXHAUSTED: "模型回答或工具调用多次无效，已停止。",
                  CONTEXT_LIMIT: "上下文预算不足，未继续调用模型。"}.get(status, "")
        return await self.finish(status, f"{prefix}{reason}\n{_summarize(self.actions, reason)}", self.actions)

    async def finish(self, status: str, answer: str, actions: list[ActionRecord]) -> AgentResult:
        """先执行最终保存，再记录终止事件和实际结果状态。"""
        if self.on_turn_end is not None and status != INTERRUPTED:
            try:
                await _maybe_await(self.on_turn_end(status, answer, self.messages))
            except (OSError, SessionError) as error:
                status = PERSIST_FAILED
                answer = f"保存写操作失败：{error}\n{_summarize(actions, '最终状态未落盘，此前保存的写操作已保留。')}"
        duration = round((time.perf_counter() - self.started) * 1000, 3)
        error = None if status == OK else answer
        event = self.logger.log_turn_end(self.ctx.user_id, self.ctx.session_id, self.ctx.turn_id,
                    status, answer, len(actions), duration, error=error, attempt_id=self.attempt) if self.logger else {
            "event": "turn_end", "user_id": self.ctx.user_id, "session_id": self.ctx.session_id,
            "turn_id": self.ctx.turn_id, "attempt_id": self.attempt, "status": status, "answer": answer,
            "steps": len(actions), "duration_ms": duration, "error": error}
        self.events.append(event)
        return AgentResult(status, answer, actions, self.messages, self.ctx.turn_id, self.events, self.attempt)
