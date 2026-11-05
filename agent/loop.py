"""Agent 主循环：调用 LLM → 解析 → 执行工具（含 call_id 幂等与写操作立即落盘）→ 回填 observation，并记录 Trace。"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .context import (
    DEFAULT_CONTEXT_TOKENS,
    ContextBudgetError,
    check_context_budget,
    estimate_tokens,
    truncate_to_tokens,
)
from .llm import LLMClient, LLMError, Message
from .parser import ParseError, parse_output
from .prompt import build_system_prompt
from .session import SessionError
from .tools.registry import ToolContext, ToolRegistry, TurnId, new_turn_id
from .tools.todo import TodoStore
from .trace import (
    ToolCallRecord,
    TraceLogger,
    _normalize_args,
    is_write_tool,
    make_call_id,
    reconcile_todo_call,
)

OK = "ok"
MAX_STEPS = "max_steps"
PARSE_EXHAUSTED = "parse_retries_exhausted"
LLM_FAILED = "llm_error"
INTERNAL_ERROR = "internal_error"
INTERRUPTED = "interrupted"
TOOL_NOT_EXECUTED = "tool_not_executed"
CONTEXT_LIMIT = "context_limit"
PERSIST_FAILED = "persist_error"

_OBS_MAX_TOKENS = 2000  # 单条 observation 的最大估算 token 数


@dataclass
class ActionRecord:
    """一次已执行的工具调用记录。"""

    step: int
    thought: str
    tool: str
    args: dict[str, Any]
    ok: bool
    result: str
    call_id: str = ""
    duration_ms: float = 0.0
    reused: bool = False


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
    """在调用记录列表中按 call_id 查找已有记录。"""
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


class Agent:
    """最小 Agent：单次请求内的 ReAct 循环（支持工具幂等、写操作立即落盘与 Trace 记录）。"""

    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        max_steps: int = 8,
        max_parse_retries: int = 3,
        max_context_tokens: int = DEFAULT_CONTEXT_TOKENS,
        trace_logger: TraceLogger | None = None,
    ):
        """保存依赖与预算：限制工具步数、解析重试次数和每次模型请求的上下文大小。"""
        self._llm = llm
        self._registry = registry
        self._max_steps = max_steps
        self._max_parse_retries = max_parse_retries
        self._max_context_tokens = max_context_tokens
        self._trace_logger = trace_logger
        self._pending: str | None = None  # 正在执行的工具名
        self._pending_record: dict[str, Any] | None = None  # 正在执行的工具调用记录

    def _discover_todo_store(self) -> TodoStore | None:
        """从已注册的 todo 工具上自动提取绑定的 TodoStore（若存在）。"""
        todo_tool = self._registry.get("todo")
        store = getattr(todo_tool, "store", None) if todo_tool is not None else None
        return store if isinstance(store, TodoStore) else None

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
        on_step_change: Callable[[list[Message], dict[str, Any], bool], None] | None = None,
        on_turn_end: Callable[[str, str, list[Message]], None] | None = None,
        trace_logger: TraceLogger | None = None,
        attempt_id: int | str | None = None,
    ) -> AgentResult:
        """处理一次用户输入并返回结果；支持按 call_id 幂等复用、写操作即时落盘与带 attempt_id 的 Trace 日志。"""
        if turn_id is None:
            turn_id = new_turn_id()
        ctx = ToolContext(user_id, session_id, turn_id)
        store = todo_store if todo_store is not None else self._discover_todo_store()
        if call_records is not None:
            records = call_records
        elif store is not None:
            records = store.get_call_records(ctx)
        else:
            records = []
        logger = trace_logger if trace_logger is not None else self._trace_logger
        if attempt_id is not None:
            attempt = attempt_id
        elif logger is not None:
            attempt = logger.next_attempt_id(user_id, session_id, turn_id)
        else:
            attempt = 1

        if context_messages is not None:
            messages: list[Message] = [dict(m) for m in context_messages]
        else:
            messages = [{"role": "system", "content": build_system_prompt(self._registry)}]
            if summary and summary.strip():
                messages.append({"role": "system", "content": summary})
            messages += history or []
            messages.append({"role": "user", "content": user_input})

        actions: list[ActionRecord] = []
        trace_events: list[dict[str, Any]] = []
        turn_t0 = time.perf_counter()

        def emit_start() -> None:
            """记录单轮开始 Trace 事件。"""
            if logger is not None:
                ev = logger.log_turn_start(user_id, session_id, turn_id, user_input, attempt_id=attempt)
            else:
                ev = {
                    "event": "turn_start",
                    "user_id": user_id,
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "attempt_id": attempt,
                    "step": 0,
                    "user_input": user_input,
                }
            trace_events.append(ev)

        def finish(status: str, answer: str) -> AgentResult:
            """先执行收尾持久化钩子，再记录单轮终止 Trace 事件并组装 AgentResult。"""
            if on_turn_end is not None and status != INTERRUPTED:
                try:
                    on_turn_end(status, answer, messages)
                except (OSError, SessionError) as e:
                    if status != PERSIST_FAILED:
                        status = PERSIST_FAILED
                        answer = (
                            f"保存写操作失败：{e}\n"
                            f"{_summarize(actions, '最终会话状态未能落盘，此前已落盘的写操作已保留。')}"
                        )
            dur_ms = round((time.perf_counter() - turn_t0) * 1000, 3)
            err = None if status == OK else answer
            if logger is not None:
                ev = logger.log_turn_end(
                    user_id, session_id, turn_id, status, answer, len(actions), dur_ms,
                    error=err, attempt_id=attempt,
                )
            else:
                ev = {
                    "event": "turn_end",
                    "user_id": user_id,
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "attempt_id": attempt,
                    "status": status,
                    "answer": answer,
                    "steps": len(actions),
                    "duration_ms": dur_ms,
                    "error": err,
                }
            trace_events.append(ev)
            return AgentResult(status, answer, actions, messages, turn_id, trace_events, attempt)

        emit_start()
        try:
            return self._loop(
                messages, ctx, actions, records, store, on_step_change, logger, trace_events, attempt, finish,
            )
        except Exception as e:  # 未预期异常：保留已完成的操作与消息，交给调用方持久化
            return finish(
                INTERNAL_ERROR,
                f"内部错误：{type(e).__name__}: {e}\n{_summarize(actions, '运行期发生未预期异常，后续步骤未完成。')}",
            )
        except BaseException as e:  # Ctrl+C 等中断：整理出一致的部分结果挂到异常上，再继续抛出
            e.agent_result = self._interrupted(finish, messages, actions, ctx, logger, trace_events, attempt)  # type: ignore[attr-defined]
            raise

    def _interrupted(
        self,
        finish,
        messages: list[Message],
        actions: list[ActionRecord],
        ctx: ToolContext,
        logger: TraceLogger | None,
        trace_events: list[dict[str, Any]],
        attempt: int | str,
    ) -> AgentResult:
        """把被中断的消息与调用记录整理成一致状态：执行中的工具补写中断 observation 与 Trace 结果。"""
        pending = self._pending
        pending_rec = self._pending_record
        self._pending = None
        self._pending_record = None
        if pending is not None:
            if pending_rec is not None:
                pending_rec["status"] = "interrupted"
                pending_rec["error"] = "interrupted during tool execution"
                step = int(pending_rec.get("step", len(actions) + 1))
                cid = str(pending_rec.get("call_id", ""))
                args = pending_rec.get("args") if isinstance(pending_rec.get("args"), dict) else {}
                if logger is not None:
                    ev = logger.log_tool_result(
                        ctx.user_id, ctx.session_id, ctx.turn_id or "", step, cid,
                        "", pending, args, False, "interrupted", "", 0.0,
                        error="interrupted during tool execution", attempt_id=attempt,
                    )
                else:
                    ev = {
                        "event": "tool_result",
                        "user_id": ctx.user_id,
                        "session_id": ctx.session_id,
                        "turn_id": ctx.turn_id,
                        "attempt_id": attempt,
                        "step": step,
                        "call_id": cid,
                        "thought": "",
                        "tool": pending,
                        "args": args,
                        "ok": False,
                        "status": "interrupted",
                        "result": "",
                        "error": "interrupted during tool execution",
                        "duration_ms": 0.0,
                        "reused": False,
                    }
                trace_events.append(ev)
            messages.append({
                "role": "user",
                "content": f"Observation (tool={pending}, status=interrupted): the run was interrupted while this "
                           "tool was executing; it may or may not have taken effect. Check the current state "
                           "(e.g. list todos) before retrying.",
            })
        elif len(messages) > 1 and messages[-1]["role"] == "assistant":
            messages.pop()  # 尚未执行的模型输出：丢弃，避免恢复后被误认为已执行
        return finish(INTERRUPTED, f"本轮被中断。\n{_summarize(actions, '请求在执行期间被中断，未生成最终回复。')}")

    def _loop(
        self,
        messages: list[Message],
        ctx: ToolContext,
        actions: list[ActionRecord],
        records: list[dict[str, Any]],
        todo_store: TodoStore | None,
        on_step_change: Callable[[list[Message], dict[str, Any], bool], None] | None,
        logger: TraceLogger | None,
        trace_events: list[dict[str, Any]],
        attempt: int | str,
        finish,
    ) -> AgentResult:
        """核心循环：调用 LLM、解析、执行工具（或幂等复用）并回填，直到得到最终答案或预算耗尽。"""
        parse_failures = 0
        unexecuted_tool_output = False
        turn_id = ctx.turn_id if ctx.turn_id is not None else "default"

        while True:
            self._pending = None
            self._pending_record = None
            try:
                check_context_budget(messages, self._max_context_tokens)
                raw = self._llm.chat(messages)
            except ContextBudgetError as e:
                return finish(
                    CONTEXT_LIMIT,
                    f"{e}\n本轮任务可能未完成，已执行操作已保留。\n{_summarize(actions, '上下文超出预算限制，未继续调用模型。')}",
                )
            except LLMError as e:
                return finish(
                    LLM_FAILED,
                    f"模型服务暂时不可用，请稍后重试。（{e}）\n{_summarize(actions, '模型请求失败，尚未生成最终回复。')}",
                )
            try:
                parsed = parse_output(raw, self._registry)
            except ParseError as e:
                # 非法输出只进入纠错提示，不作为 assistant 历史保存，避免伪造 observation 污染上下文。
                unexecuted_tool_output = unexecuted_tool_output or (
                    '"action"' in raw or "observation" in raw.lower()
                )
                parse_failures += 1
                if logger is not None:
                    ev = logger.log_parse_error(
                        ctx.user_id, ctx.session_id, turn_id, len(actions) + 1, str(e), attempt_id=attempt,
                    )
                else:
                    ev = {
                        "event": "parse_error",
                        "user_id": ctx.user_id,
                        "session_id": ctx.session_id,
                        "turn_id": turn_id,
                        "attempt_id": attempt,
                        "step": len(actions) + 1,
                        "error": str(e),
                    }
                trace_events.append(ev)
                if parse_failures > self._max_parse_retries:
                    return finish(
                        PARSE_EXHAUSTED,
                        f"模型输出格式多次不合法，已停止。最近一次原因：{e}\n{_summarize(actions, '模型连续返回非法格式，未能完成后续步骤。')}",
                    )
                messages.append({
                    "role": "user",
                    "content": (
                        f"Format error: {e}. Reply again with a single valid JSON object. "
                        "本次工具未执行（the tool was not executed）；请勿声称工具已经成功。"
                    ),
                })
                continue
            parse_failures = 0

            if parsed.kind == "final":
                if unexecuted_tool_output:
                    return finish(
                        TOOL_NOT_EXECUTED,
                        "工具调用未执行，无法确认请求已完成。请重新发送请求，让我先实际执行工具。\n"
                        + _summarize(actions, "模型在工具未成功执行的情况下直接给出了最终回复。"),
                    )
                messages.append({"role": "assistant", "content": raw})
                return finish(OK, parsed.final_answer or "")

            if len(actions) >= self._max_steps:
                return finish(
                    MAX_STEPS,
                    f"已达最大步数（{self._max_steps}），任务可能未完成。\n{_summarize(actions, '达到工具调用步数上限，尚未得出最终回复。')}",
                )

            step = len(actions) + 1
            norm_target = _normalize_args(parsed.action, parsed.action_input)
            occurrence = (
                sum(
                    1
                    for a in actions
                    if a.tool == parsed.action and _normalize_args(a.tool, a.args) == norm_target
                )
                + 1
            )
            call_id = make_call_id(ctx.user_id, ctx.session_id, turn_id, parsed.action, parsed.action_input, occurrence)

            if logger is not None:
                req_ev = logger.log_tool_request(
                    ctx.user_id, ctx.session_id, turn_id, step, call_id,
                    parsed.thought, parsed.action, parsed.action_input, attempt_id=attempt,
                )
            else:
                req_ev = {
                    "event": "tool_request",
                    "user_id": ctx.user_id,
                    "session_id": ctx.session_id,
                    "turn_id": turn_id,
                    "attempt_id": attempt,
                    "step": step,
                    "call_id": call_id,
                    "thought": parsed.thought,
                    "tool": parsed.action,
                    "args": parsed.action_input,
                    "status": "started",
                }
            trace_events.append(req_ev)

            messages.append({"role": "assistant", "content": raw})
            existing = _find_call_record(records, call_id)
            reused = False
            reconciled_from_uncertain = False
            result_ok = False
            result_content = ""
            duration_ms = 0.0

            is_todo_list = parsed.action == "todo" and parsed.action_input.get("action") == "list"
            if existing is not None and not is_todo_list:
                prev_status = existing.get("status")
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
                        existing["step"] = step

            if reused:
                unexecuted_tool_output = False
                rec = existing if existing is not None else {}
                obs_msg = _build_observation_message(parsed.action, True, result_content)
                messages.append(obs_msg)
                if on_step_change is not None and rec:
                    on_step_change(messages, rec, reconciled_from_uncertain and is_write_tool(parsed.action, parsed.action_input))
            else:
                rec = _upsert_call_record(
                    records,
                    ToolCallRecord(
                        call_id=call_id,
                        user_id=ctx.user_id,
                        session_id=ctx.session_id,
                        turn_id=turn_id,
                        step=step,
                        tool=parsed.action,
                        args=parsed.action_input,
                        status="pending",
                    ).to_dict(),
                )
                self._pending = parsed.action
                self._pending_record = rec
                if on_step_change is not None:
                    on_step_change(messages, rec, False)

                write_op = is_write_tool(parsed.action, parsed.action_input)
                prev_todos = (
                    copy.deepcopy(todo_store.export_state(ctx))
                    if (write_op and todo_store is not None)
                    else None
                )
                t0 = time.perf_counter()
                result = self._registry.execute(parsed.action, parsed.action_input, ctx)
                duration_ms = round((time.perf_counter() - t0) * 1000, 3)
                self._pending = None
                self._pending_record = None

                result_ok = result.ok
                result_content = result.content
                rec["status"] = "ok" if result_ok else "error"
                rec["result"] = result_content
                rec["error"] = None if result_ok else result_content
                rec["duration_ms"] = duration_ms
                if result_ok:
                    unexecuted_tool_output = False

                obs_msg = _build_observation_message(parsed.action, result_ok, result_content)
                messages.append(obs_msg)

                if on_step_change is not None:
                    try:
                        on_step_change(messages, rec, write_op and result_ok)
                    except (OSError, SessionError) as e:
                        if prev_todos is not None and todo_store is not None:
                            todo_store.load_state(ctx, prev_todos)
                        messages.pop()  # 移除未落盘的 observation
                        messages.pop()  # 移除未落盘的 assistant action
                        rec["status"] = "error"
                        rec["error"] = f"persist failed: {e}"
                        err_msg = f"persist failed: {e}"
                        if logger is not None:
                            res_ev = logger.log_tool_result(
                                ctx.user_id, ctx.session_id, turn_id, step, call_id,
                                parsed.thought, parsed.action, parsed.action_input,
                                False, "error", result_content, duration_ms,
                                error=err_msg, attempt_id=attempt,
                            )
                        else:
                            res_ev = {
                                "event": "tool_result",
                                "user_id": ctx.user_id,
                                "session_id": ctx.session_id,
                                "turn_id": turn_id,
                                "attempt_id": attempt,
                                "step": step,
                                "call_id": call_id,
                                "thought": parsed.thought,
                                "tool": parsed.action,
                                "args": parsed.action_input,
                                "ok": False,
                                "status": "error",
                                "result": result_content,
                                "error": err_msg,
                                "duration_ms": duration_ms,
                                "reused": False,
                            }
                        trace_events.append(res_ev)
                        return finish(
                            PERSIST_FAILED,
                            f"保存写操作失败：{e}\n{_summarize(actions, '写操作未能落盘，已回滚本次未保存的变更。')}",
                        )

            actions.append(
                ActionRecord(
                    step, parsed.thought, parsed.action, parsed.action_input,
                    result_ok, result_content,
                    call_id=call_id, duration_ms=duration_ms, reused=reused,
                )
            )
            status_str = "ok" if result_ok else "error"
            err_str = None if result_ok else result_content
            if logger is not None:
                res_ev = logger.log_tool_result(
                    ctx.user_id, ctx.session_id, turn_id, step, call_id,
                    parsed.thought, parsed.action, parsed.action_input,
                    result_ok, status_str, result_content, duration_ms,
                    error=err_str, reused=reused, attempt_id=attempt,
                )
            else:
                res_ev = {
                    "event": "tool_result",
                    "user_id": ctx.user_id,
                    "session_id": ctx.session_id,
                    "turn_id": turn_id,
                    "attempt_id": attempt,
                    "step": step,
                    "call_id": call_id,
                    "thought": parsed.thought,
                    "tool": parsed.action,
                    "args": parsed.action_input,
                    "ok": result_ok,
                    "status": status_str,
                    "result": result_content,
                    "error": err_str,
                    "duration_ms": duration_ms,
                    "reused": reused,
                }
            trace_events.append(res_ev)


def _build_observation_message(tool: str, ok: bool, result_content: str) -> Message:
    """构造单条回填给模型的 Observation 消息，超长时按 token 上限截断。"""
    status = "ok" if ok else "error"
    prefix = f"Observation (tool={tool}, status={status}): "
    content = result_content
    if estimate_tokens(prefix + content) > _OBS_MAX_TOKENS:
        marker = " ...(truncated) 结果已截断，可缩小查询范围或分批读取。"
        available = _OBS_MAX_TOKENS - estimate_tokens(prefix) - estimate_tokens(marker)
        content = truncate_to_tokens(content, available) + marker
    return {"role": "user", "content": prefix + content}
