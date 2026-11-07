"""原生工具调度、观察回填、即时保存与中断恢复。"""
from __future__ import annotations
import asyncio
import copy
from dataclasses import dataclass, field
import inspect
import json
import time
from typing import Any
from .context import estimate_tokens, truncate_to_tokens
from .llm import Message, ToolCallItem
from .runtime import ActionRecord, _find_call_record, _upsert_call_record
from .session import SessionError
from .tool_call import ToolCallError, parse_tool_call_args
from .tools.registry import ToolContext, TurnId
from .tools.todo import TodoStore
from .trace import ToolCallRecord, TraceLogger, _normalize_args, is_write_tool, make_call_id, reconcile_todo_call

_OBS_MAX_TOKENS = 2000

async def _maybe_await(val: Any) -> Any:
    """若返回值为可等待对象则 await，否则原样返回。"""
    if inspect.isawaitable(val):
        return await val
    return val

@dataclass
class _BatchCallItem:
    """单批原生工具调用中的一项及其运行时状态。"""

    step: int
    thought: str
    tool_name: str
    args: dict[str, Any]
    api_call_id: str
    call_id: str
    write_op: bool
    parallel_safe: bool
    rec: dict[str, Any] = field(default_factory=dict)
    reused: bool = False
    reconciled_from_uncertain: bool = False
    result_ok: bool = False
    result_content: str = ""
    duration_ms: float = 0.0
    request_logged: bool = False
    result_logged: bool = False
    completed: bool = False
    message_appended: bool = False
    action_appended: bool = False

class NativeToolExecutor:
    """管理单轮原生工具调用及其副作用记录。"""

    def _is_parallel_safe(self, tool_name: str, args: dict[str, Any], batch_has_todo_write: bool) -> bool:
        """判断单个工具调用是否可放入受控并行段：仅无依赖只读工具可并发，待办写及同批相关读写保持串行。"""
        if is_write_tool(tool_name, args):
            return False
        if tool_name == "todo":
            return not batch_has_todo_write
        tool = self._registry.get(tool_name)
        return bool(tool is not None and getattr(tool, "read_only", True))

    def _prepare_batch_items(
        self,
        raw_calls: list[ToolCallItem],
        thought: str,
        actions: list[ActionRecord],
        ctx: ToolContext,
        turn_id: TurnId,
    ) -> tuple[list[_BatchCallItem], bool]:
        """校验整批工具调用参数，按剩余步数预算裁剪并预分配确定性的 step、api_call_id 与幂等 call_id。"""
        if any(not isinstance(tc, ToolCallItem) for tc in raw_calls):
            raise ToolCallError("tool_calls must contain native ToolCallItem values")
        ids = [tc.id for tc in raw_calls]
        if any(not isinstance(cid, str) or not cid.strip() for cid in ids) or len(ids) != len(set(ids)):
            raise ToolCallError("tool_call_id must be non-empty and unique within a response")
        parsed_calls: list[tuple[str, str, dict[str, Any]]] = []
        for tc in raw_calls:
            args = parse_tool_call_args(tc.name, tc.arguments, self._registry)
            parsed_calls.append((tc.id, tc.name, args))

        remaining = self._max_steps - len(actions)
        if remaining <= 0:
            return [], True
        budget_exceeded = len(parsed_calls) > remaining
        allowed_calls = parsed_calls[:remaining]
        batch_has_todo_write = any(is_write_tool(name, args) for _, name, args in allowed_calls)

        prior_pairs: list[tuple[str, dict[str, Any]]] = [
            (a.tool, _normalize_args(a.tool, a.args)) for a in actions
        ]
        items: list[_BatchCallItem] = []
        for idx, (raw_id, tool_name, args) in enumerate(allowed_calls, start=1):
            step = len(actions) + idx
            api_call_id = raw_id or f"call_{step}"
            norm_target = _normalize_args(tool_name, args)
            occurrence = sum(1 for t_name, n_args in prior_pairs if t_name == tool_name and n_args == norm_target) + 1
            prior_pairs.append((tool_name, norm_target))
            call_id = make_call_id(ctx.user_id, ctx.session_id, turn_id, tool_name, args, occurrence)
            write_op = is_write_tool(tool_name, args)
            parallel_safe = self._parallel_tool_calls and self._is_parallel_safe(tool_name, args, batch_has_todo_write)
            items.append(
                _BatchCallItem(
                    step=step,
                    thought=thought,
                    tool_name=tool_name,
                    args=args,
                    api_call_id=api_call_id,
                    call_id=call_id,
                    write_op=write_op,
                    parallel_safe=parallel_safe,
                )
            )
        return items, budget_exceeded

    @staticmethod
    def _partition_batch_segments(batch_items: list[_BatchCallItem]) -> list[tuple[str, list[_BatchCallItem]]]:
        """将一批工具调用切分为受控执行段：连续只读无依赖项合并为 parallel 段，写操作或相关读写项各自为 serial 段。"""
        segments: list[tuple[str, list[_BatchCallItem]]] = []
        for it in batch_items:
            if not it.parallel_safe:
                segments.append(("serial", [it]))
            elif segments and segments[-1][0] == "parallel":
                segments[-1][1].append(it)
            else:
                segments.append(("parallel", [it]))
        return segments

    def _emit_batch_item_request(
        self,
        it: _BatchCallItem,
        ctx: ToolContext,
        turn_id: TurnId,
        attempt: int | str,
        logger: TraceLogger | None,
        trace_events: list[dict[str, Any]],
    ) -> None:
        """为批量调用中的单项记录 tool_request Trace 事件。"""
        if it.request_logged:
            return
        if logger is not None:
            req_ev = logger.log_tool_request(
                ctx.user_id, ctx.session_id, turn_id, it.step, it.call_id,
                it.thought, it.tool_name, it.args, attempt_id=attempt, api_call_id=it.api_call_id,
            )
        else:
            req_ev = {
                "event": "tool_request",
                "user_id": ctx.user_id,
                "session_id": ctx.session_id,
                "turn_id": turn_id,
                "attempt_id": attempt,
                "step": it.step,
                "call_id": it.call_id,
                "api_call_id": it.api_call_id,
                "thought": it.thought,
                "tool": it.tool_name,
                "args": it.args,
                "status": "started",
            }
        trace_events.append(req_ev)
        it.request_logged = True

    def _emit_batch_item_result(
        self,
        it: _BatchCallItem,
        ctx: ToolContext,
        turn_id: TurnId,
        attempt: int | str,
        logger: TraceLogger | None,
        trace_events: list[dict[str, Any]],
        error_override: str | None = None,
    ) -> None:
        """为批量调用中的单项记录 tool_result Trace 事件。"""
        if it.result_logged:
            return
        ok = False if error_override is not None else it.result_ok
        status_str = "error" if not ok else "ok"
        err_str = error_override if error_override is not None else (None if ok else it.result_content)
        if logger is not None:
            res_ev = logger.log_tool_result(
                ctx.user_id, ctx.session_id, turn_id, it.step, it.call_id,
                it.thought, it.tool_name, it.args,
                ok, status_str, it.result_content, it.duration_ms,
                error=err_str, reused=it.reused, attempt_id=attempt, api_call_id=it.api_call_id,
            )
        else:
            res_ev = {
                "event": "tool_result",
                "user_id": ctx.user_id,
                "session_id": ctx.session_id,
                "turn_id": turn_id,
                "attempt_id": attempt,
                "step": it.step,
                "call_id": it.call_id,
                "api_call_id": it.api_call_id,
                "thought": it.thought,
                "tool": it.tool_name,
                "args": it.args,
                "ok": ok,
                "status": status_str,
                "result": it.result_content,
                "error": err_str,
                "duration_ms": it.duration_ms,
                "reused": it.reused,
            }
        trace_events.append(res_ev)
        it.result_logged = True

    def _try_reuse_batch_item(
        self,
        it: _BatchCallItem,
        records: list[dict[str, Any]],
        todo_store: TodoStore | None,
        ctx: ToolContext,
        turn_id: TurnId,
    ) -> bool:
        """按内部幂等 call_id 检查批量项是否可复用已完成结果或从不确定状态核对恢复。"""
        existing = _find_call_record(records, it.call_id)
        is_todo_list = it.tool_name == "todo" and it.args.get("action") == "list"
        if existing is None or is_todo_list:
            return False
        prev_status = existing.get("status")
        reused = False
        reconciled_from_uncertain = False
        result_ok = False
        result_content = ""
        if prev_status == "ok":
            todos_snapshot = todo_store.export_state(ctx) if todo_store is not None else None
            reconciled = reconcile_todo_call(existing, todos_snapshot, turn_id)
            if reconciled is not None:
                reused = True
                result_ok, result_content = reconciled
            elif existing.get("result"):
                reused = True
                result_ok, result_content = True, str(existing["result"])
        elif prev_status in ("pending", "interrupted"):
            todos_snapshot = todo_store.export_state(ctx) if todo_store is not None else None
            reconciled = reconcile_todo_call(existing, todos_snapshot, turn_id)
            if reconciled is not None:
                reused = True
                reconciled_from_uncertain = True
                result_ok, result_content = reconciled
                existing["status"] = "ok"
                existing["result"] = result_content
                existing["error"] = None
                existing["step"] = it.step
        if not reused:
            return False
        if it.api_call_id:
            existing["api_call_id"] = it.api_call_id
        it.rec = existing
        it.reused = True
        it.reconciled_from_uncertain = reconciled_from_uncertain
        it.result_ok = result_ok
        it.result_content = result_content
        it.completed = True
        return True

    def _handle_batch_persist_failure(
        self,
        failed_item: _BatchCallItem,
        batch_items: list[_BatchCallItem],
        prev_todos: dict[str, Any] | None,
        todo_store: TodoStore | None,
        ctx: ToolContext,
        turn_id: TurnId,
        attempt: int | str,
        messages: list[Message],
        logger: TraceLogger | None,
        trace_events: list[dict[str, Any]],
        err: Exception,
    ) -> None:
        """处理批量执行中待办写操作即时落盘失败：回滚未落盘待办并保持消息配对完整。"""
        if prev_todos is not None and todo_store is not None:
            todo_store.load_state(ctx, prev_todos)
        err_msg = f"persist failed: {err}"
        failed_item.rec["status"] = "error"
        failed_item.rec["error"] = err_msg
        self._emit_batch_item_result(
            failed_item, ctx, turn_id, attempt, logger, trace_events, error_override=err_msg,
        )
        other_completed = any(x.completed and x is not failed_item for x in batch_items)
        if not other_completed:
            # 本批尚无其他已完成项：直接弹出刚追加的 tool 消息与本批 assistant 消息
            if failed_item.message_appended and messages and messages[-1].get("role") == "tool":
                messages.pop()
                failed_item.message_appended = False
            if messages and messages[-1].get("role") == "assistant":
                messages.pop()
        else:
            # 本批已有先前的完成项：保留 assistant 消息，把失败项及后续未执行项补齐 tool 错误消息以保证配对
            if failed_item.message_appended and messages and messages[-1].get("role") == "tool":
                messages[-1] = _build_native_tool_message(
                    failed_item.api_call_id, failed_item.tool_name, False, err_msg,
                )
            for rem in batch_items:
                if not rem.message_appended:
                    messages.append(
                        _build_native_tool_message(
                            rem.api_call_id, rem.tool_name, False, "not executed due to earlier persist failure",
                        )
                    )
                    rem.message_appended = True
        self._active_batch = None

    def _prepare_interrupted_state(
        self,
        messages: list[Message],
        actions: list[ActionRecord],
        records: list[dict[str, Any]],
        ctx: ToolContext,
        logger: TraceLogger | None,
        trace_events: list[dict[str, Any]],
        attempt: int | str,
    ) -> None:
        """中断时补齐原生工具结果和 Trace，保留已完成操作。"""
        active_batch = self._active_batch
        self._active_batch = None
        if active_batch is not None:
            turn_id = ctx.turn_id if ctx.turn_id is not None else "default"
            for it in active_batch:
                if it.completed:
                    if not it.result_logged:
                        self._emit_batch_item_result(it, ctx, turn_id, attempt, logger, trace_events)
                    if not it.message_appended:
                        messages.append(
                            _build_native_tool_message(it.api_call_id, it.tool_name, it.result_ok, it.result_content)
                        )
                        it.message_appended = True
                    if not it.action_appended:
                        actions.append(
                            ActionRecord(
                                it.step, it.thought, it.tool_name, it.args,
                                it.result_ok, it.result_content,
                                call_id=it.call_id, duration_ms=it.duration_ms,
                                reused=it.reused, api_call_id=it.api_call_id,
                            )
                        )
                        it.action_appended = True
                else:
                    if not it.request_logged:
                        self._emit_batch_item_request(it, ctx, turn_id, attempt, logger, trace_events)
                    if it.rec:
                        it.rec["status"] = "interrupted"
                        it.rec["error"] = "interrupted during tool execution"
                    else:
                        it.rec = _upsert_call_record(
                            records,
                            ToolCallRecord(
                                call_id=it.call_id,
                                user_id=ctx.user_id,
                                session_id=ctx.session_id,
                                turn_id=turn_id,
                                step=it.step,
                                tool=it.tool_name,
                                args=it.args,
                                status="interrupted",
                                error="interrupted during tool execution",
                                api_call_id=it.api_call_id,
                            ).to_dict(),
                        )
                    if not it.result_logged:
                        if logger is not None:
                            ev = logger.log_tool_result(
                                ctx.user_id, ctx.session_id, turn_id, it.step, it.call_id,
                                it.thought, it.tool_name, it.args, False, "interrupted", "", 0.0,
                                error="interrupted during tool execution",
                                attempt_id=attempt, api_call_id=it.api_call_id,
                            )
                        else:
                            ev = {
                                "event": "tool_result",
                                "user_id": ctx.user_id,
                                "session_id": ctx.session_id,
                                "turn_id": turn_id,
                                "attempt_id": attempt,
                                "step": it.step,
                                "call_id": it.call_id,
                                "api_call_id": it.api_call_id,
                                "thought": it.thought,
                                "tool": it.tool_name,
                                "args": it.args,
                                "ok": False,
                                "status": "interrupted",
                                "result": "",
                                "error": "interrupted during tool execution",
                                "duration_ms": 0.0,
                                "reused": False,
                            }
                        trace_events.append(ev)
                        it.result_logged = True
                    if not it.message_appended:
                        interrupted_text = (
                            f"Observation (tool={it.tool_name}, status=interrupted): the run was interrupted while "
                            "this tool was executing; it may or may not have taken effect. Check the current state "
                            "(e.g. list todos) before retrying."
                        )
                        messages.append({
                            "role": "tool",
                            "tool_call_id": it.api_call_id,
                            "name": it.tool_name,
                            "content": interrupted_text,
                        })
                        it.message_appended = True
            return



    async def perform(self, batch):
        """执行一个只读并行组或单项写操作，返回待回填结果。"""
        previous = copy.deepcopy(self.store.export_state(self.ctx)) if (self.store and any(it.write_op for it in batch)) else None
        pending = []
        for it in batch:
            self._emit_batch_item_request(it, self.ctx, self.ctx.turn_id, self.attempt, self.logger, self.events)
            if not self._try_reuse_batch_item(it, self.records, self.store, self.ctx, self.ctx.turn_id):
                it.rec = _upsert_call_record(self.records, ToolCallRecord(
                    call_id=it.call_id, user_id=self.ctx.user_id, session_id=self.ctx.session_id,
                    turn_id=self.ctx.turn_id, step=it.step, tool=it.tool_name, args=it.args,
                    status="pending", api_call_id=it.api_call_id,
                ).to_dict())
                pending.append(it)
            elif not it.write_op:
                self._emit_batch_item_result(it, self.ctx, self.ctx.turn_id, self.attempt, self.logger, self.events)
        if pending and self.on_step_change:
            await _maybe_await(self.on_step_change(self.messages, pending[-1].rec, False))
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def execute(it):
            """受并发上限约束执行工具并记录真实完成结果。"""
            async with semaphore:
                started = time.perf_counter()
                result = await self._registry.aexecute(it.tool_name, it.args, self.ctx, timeout=self._tool_timeout)
                it.duration_ms = round((time.perf_counter() - started) * 1000, 3)
                it.result_ok, it.result_content = result.ok, result.content
                it.rec.update(status="ok" if result.ok else "error", result=result.content,
                              error=None if result.ok else result.content, duration_ms=it.duration_ms)
                it.completed = True
                if not it.write_op:
                    self._emit_batch_item_result(it, self.ctx, self.ctx.turn_id, self.attempt, self.logger, self.events)

        if len(pending) == 1:
            await execute(pending[0])
        elif pending:
            tasks = [asyncio.create_task(execute(it)) for it in pending]
            try:
                await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
        return batch, previous

    async def observe(self, observation):
        """按请求顺序回填工具结果，写操作保存成功后再继续调度。"""
        batch, previous = observation
        for it in batch:
            self.messages.append(_build_native_tool_message(it.api_call_id, it.tool_name, it.result_ok, it.result_content))
            it.message_appended = True
            if self.on_step_change:
                persist = it.write_op and it.result_ok and (not it.reused or it.reconciled_from_uncertain)
                try:
                    await _maybe_await(self.on_step_change(self.messages, it.rec, persist))
                except (OSError, SessionError) as error:
                    self._handle_batch_persist_failure(
                        it, self._active_batch or batch, previous, self.store, self.ctx, self.ctx.turn_id,
                        self.attempt, self.messages, self.logger, self.events, error)
                    raise
            self.actions.append(ActionRecord(
                it.step, it.thought, it.tool_name, it.args, it.result_ok, it.result_content,
                call_id=it.call_id, duration_ms=it.duration_ms, reused=it.reused, api_call_id=it.api_call_id))
            it.action_appended = True
            self._emit_batch_item_result(it, self.ctx, self.ctx.turn_id, self.attempt, self.logger, self.events)
        if any(it.result_ok for it in batch):
            self.unexecuted_tool_output = False
        if self._active_batch and all(it.action_appended for it in self._active_batch):
            self._active_batch = None


def _build_native_tool_message(api_call_id: str, tool: str, ok: bool, result_content: str) -> Message:
    """构造原生工具调用协议的 role='tool' 回填消息（带 tool_call_id 与截断保护）。"""
    prefix = "" if ok else "Error: "
    content = prefix + result_content
    if estimate_tokens(content) > _OBS_MAX_TOKENS:
        marker = " ...(truncated) 结果已截断，可缩小查询范围或分批读取。"
        available = _OBS_MAX_TOKENS - estimate_tokens(marker)
        content = truncate_to_tokens(content, available) + marker
    return {
        "role": "tool",
        "tool_call_id": api_call_id,
        "name": tool,
        "content": content,
    }
