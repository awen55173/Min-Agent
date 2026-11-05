"""工具调用标识、幂等状态核对与 JSONL Trace 日志记录。"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .tools.registry import TurnId

_WRITE_TODO_ACTIONS = {"add", "complete", "delete"}


def _now() -> str:
    """返回当前 UTC 时间的 ISO 字符串（精确到毫秒）。"""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def is_write_tool(tool: str, args: Any) -> bool:
    """判断工具调用是否为会修改持久化状态的写操作（todo 的 add/complete/delete）。"""
    return tool == "todo" and isinstance(args, dict) and args.get("action") in _WRITE_TODO_ACTIONS


def _normalize_args(tool: str, args: Any) -> Any:
    """规范化工具参数以便生成稳定的调用签名（如去除待办文本首尾空白）。"""
    if not isinstance(args, dict):
        return args
    norm = dict(args)
    if tool == "todo" and isinstance(norm.get("text"), str):
        norm["text"] = norm["text"].strip()
    return norm


def make_call_id(
    user_id: str,
    session_id: str,
    turn_id: TurnId,
    tool: str,
    args: dict[str, Any],
    occurrence: int = 1,
) -> str:
    """生成关联用户、会话、轮次与工具参数的确定性 call_id；同轮重试自动复用相同标识。"""
    norm_args = _normalize_args(tool, args)
    canon = json.dumps(norm_args, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    raw = f"{user_id}:{session_id}:{turn_id}:{tool}:{canon}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    base = f"call-{turn_id}-{tool}-{digest}"
    if is_write_tool(tool, args) or occurrence <= 1:
        return base
    return f"{base}-{occurrence}"


@dataclass
class ToolCallRecord:
    """持久化的单次工具调用记录：关联用户、会话、轮次、call_id、参数、执行状态与结果。"""

    call_id: str
    user_id: str
    session_id: str
    turn_id: TurnId
    step: int
    tool: str
    args: dict[str, Any]
    status: str  # "pending" | "ok" | "error" | "interrupted"
    result: str = ""
    error: str | None = None
    duration_ms: float = 0.0
    timestamp: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        """转为可持久化为 JSON 的字典。"""
        return {
            "call_id": self.call_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "step": self.step,
            "tool": self.tool,
            "args": self.args,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
        }


def reconcile_todo_call(
    record: dict[str, Any],
    todos_state: dict[str, Any] | None,
    turn_id: TurnId,
) -> tuple[bool, str] | None:
    """在恢复或重试待办写操作时核对实际存储状态。

    - 若待办状态中已反映该写操作，返回 (True, 结果JSON)，避免重复执行；
    - 若状态未定（pending/interrupted）且存储中确实未生效，返回 None 由调用方复用 call_id 重新执行。
    """
    if record.get("tool") != "todo" or not isinstance(todos_state, dict):
        if record.get("status") == "ok" and record.get("result"):
            return True, str(record["result"])
        return None

    args = record.get("args") if isinstance(record.get("args"), dict) else {}
    action = args.get("action")
    items = todos_state.get("items") if isinstance(todos_state.get("items"), list) else []
    rec_turn = record.get("turn_id", turn_id)

    if action == "add":
        text = str(args.get("text", "")).strip()
        for it in items:
            if not isinstance(it, dict):
                continue
            if it.get("text") == text and (it.get("turn") == rec_turn or it.get("turn") == turn_id):
                public_item = {"id": it["id"], "text": it["text"], "done": bool(it.get("done", False))}
                return True, json.dumps({"created": False, "item": public_item}, ensure_ascii=False)
        if record.get("status") == "ok" and record.get("result"):
            try:
                parsed_res = json.loads(str(record["result"]))
                if isinstance(parsed_res, dict) and "item" in parsed_res:
                    parsed_res["created"] = False
                    return True, json.dumps(parsed_res, ensure_ascii=False)
            except ValueError:
                pass
            return True, str(record["result"])
        return None

    if action == "complete":
        tid = args.get("id")
        for it in items:
            if isinstance(it, dict) and it.get("id") == tid:
                if it.get("done") is True:
                    if record.get("status") == "ok" and record.get("result"):
                        return True, str(record["result"])
                    public_item = {"id": it["id"], "text": it["text"], "done": True}
                    return True, json.dumps({"complete": public_item}, ensure_ascii=False)
                return None
        if record.get("status") == "ok" and record.get("result"):
            return True, str(record["result"])
        return None

    if action == "delete":
        tid = args.get("id")
        still_exists = any(isinstance(it, dict) and it.get("id") == tid for it in items)
        if still_exists:
            return None
        if record.get("status") == "ok" and record.get("result"):
            return True, str(record["result"])
        next_id = todos_state.get("next", 1)
        if isinstance(tid, int) and isinstance(next_id, int) and 1 <= tid < next_id:
            return True, json.dumps({"delete": {"id": tid, "text": "", "done": False}}, ensure_ascii=False)
        return None

    return None


class TraceLogger:
    """JSONL 格式的 Trace 执行日志记录器，支持按会话写入、查询与完整性核查。"""

    def __init__(self, root: str | os.PathLike = "logs"):
        """初始化日志根目录（或单个 .jsonl 文件路径）。"""
        self._root = Path(root)

    def path_for(self, user_id: str = "default", session_id: str = "default") -> Path:
        """返回指定用户与会话的 JSONL 日志文件路径。"""
        if self._root.suffix == ".jsonl":
            return self._root
        return self._root / user_id / f"{session_id}.jsonl"

    def append(self, event: dict[str, Any]) -> None:
        """向对应会话的 JSONL 文件追加一条事件记录并立即刷盘。"""
        path = self.path_for(str(event.get("user_id", "default")), str(event.get("session_id", "default")))
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"timestamp": _now(), **event}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    def next_attempt_id(self, user_id: str, session_id: str, turn_id: TurnId) -> int:
        """计算某轮次（turn_id）的下一次尝试序号（attempt_id，从 1 开始递增）。"""
        count = sum(
            1
            for ev in self.read_events(user_id, session_id)
            if ev.get("event") == "turn_start" and ev.get("turn_id") == turn_id
        )
        return count + 1

    def log_turn_start(
        self,
        user_id: str,
        session_id: str,
        turn_id: TurnId,
        user_input: str,
        attempt_id: int | str = 1,
    ) -> dict[str, Any]:
        """记录单轮请求开始事件（含 attempt_id 区分同轮重试）。"""
        ev = {
            "event": "turn_start",
            "user_id": user_id,
            "session_id": session_id,
            "turn_id": turn_id,
            "attempt_id": attempt_id,
            "step": 0,
            "user_input": user_input,
        }
        self.append(ev)
        return ev

    def log_tool_request(
        self,
        user_id: str,
        session_id: str,
        turn_id: TurnId,
        step: int,
        call_id: str,
        thought: str,
        tool: str,
        args: dict[str, Any],
        attempt_id: int | str = 1,
    ) -> dict[str, Any]:
        """在工具执行前记录工具调用请求事件。"""
        ev = {
            "event": "tool_request",
            "user_id": user_id,
            "session_id": session_id,
            "turn_id": turn_id,
            "attempt_id": attempt_id,
            "step": step,
            "call_id": call_id,
            "thought": thought,
            "tool": tool,
            "args": args,
            "status": "started",
        }
        self.append(ev)
        return ev

    def log_tool_result(
        self,
        user_id: str,
        session_id: str,
        turn_id: TurnId,
        step: int,
        call_id: str,
        thought: str,
        tool: str,
        args: dict[str, Any],
        ok: bool,
        status: str,
        result: str,
        duration_ms: float,
        error: str | None = None,
        reused: bool = False,
        attempt_id: int | str = 1,
    ) -> dict[str, Any]:
        """在工具执行后（或复用/中断时）记录工具结果事件，与同次尝试的 tool_request 成对。"""
        ev = {
            "event": "tool_result",
            "user_id": user_id,
            "session_id": session_id,
            "turn_id": turn_id,
            "attempt_id": attempt_id,
            "step": step,
            "call_id": call_id,
            "thought": thought,
            "tool": tool,
            "args": args,
            "ok": ok,
            "status": status,
            "result": result,
            "error": error,
            "duration_ms": duration_ms,
            "reused": reused,
        }
        self.append(ev)
        return ev

    def log_parse_error(
        self,
        user_id: str,
        session_id: str,
        turn_id: TurnId,
        step: int,
        error: str,
        attempt_id: int | str = 1,
    ) -> dict[str, Any]:
        """记录模型输出解析失败事件。"""
        ev = {
            "event": "parse_error",
            "user_id": user_id,
            "session_id": session_id,
            "turn_id": turn_id,
            "attempt_id": attempt_id,
            "step": step,
            "error": error,
        }
        self.append(ev)
        return ev

    def log_turn_end(
        self,
        user_id: str,
        session_id: str,
        turn_id: TurnId,
        status: str,
        answer: str,
        steps: int,
        duration_ms: float,
        error: str | None = None,
        attempt_id: int | str = 1,
    ) -> dict[str, Any]:
        """记录单轮请求终止事件（含正常结束与各类异常终止原因）。"""
        ev = {
            "event": "turn_end",
            "user_id": user_id,
            "session_id": session_id,
            "turn_id": turn_id,
            "attempt_id": attempt_id,
            "status": status,
            "answer": answer,
            "steps": steps,
            "duration_ms": duration_ms,
            "error": error,
        }
        self.append(ev)
        return ev

    def read_events(self, user_id: str = "default", session_id: str = "default") -> list[dict[str, Any]]:
        """读取指定会话的全部 Trace 事件。"""
        path = self.path_for(user_id, session_id)
        if not path.exists():
            return []
        events: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                events.append(json.loads(line))
        return events

    def find_unfinished_calls(self, user_id: str = "default", session_id: str = "default") -> list[dict[str, Any]]:
        """识别会话中因中断或崩溃留下的未完成工具调用。"""
        return find_unfinished_in_events(self.read_events(user_id, session_id))

    def format_trace(self, user_id: str = "default", session_id: str = "default") -> str:
        """将指定会话的 Trace 事件格式化为可读文本，供 CLI 展示。"""
        events = self.read_events(user_id, session_id)
        if not events:
            return f"No trace records for user={user_id} session={session_id}"
        lines: list[str] = []
        for ev in events:
            kind = ev.get("event")
            turn_id = ev.get("turn_id")
            attempt_id = ev.get("attempt_id", 1)
            if kind == "turn_start":
                lines.append(f"[turn {turn_id}] START attempt={attempt_id} user_input={ev.get('user_input')!r}")
            elif kind == "tool_request":
                lines.append(
                    f"[turn {turn_id}][step {ev.get('step')}] REQUEST call_id={ev.get('call_id')} "
                    f"attempt={attempt_id} {ev.get('tool')}({ev.get('args')}) thought={ev.get('thought')!r}"
                )
            elif kind == "tool_result":
                reused_tag = " reused=True" if ev.get("reused") else ""
                err_tag = f" error={ev.get('error')!r}" if ev.get("error") else ""
                lines.append(
                    f"[turn {turn_id}][step {ev.get('step')}] RESULT  call_id={ev.get('call_id')} "
                    f"attempt={attempt_id} {ev.get('tool')} status={ev.get('status')} ok={ev.get('ok')} "
                    f"duration={ev.get('duration_ms')}ms{reused_tag}{err_tag} -> {ev.get('result')}"
                )
            elif kind == "parse_error":
                lines.append(
                    f"[turn {turn_id}][step {ev.get('step')}] PARSE_ERROR attempt={attempt_id} {ev.get('error')}"
                )
            elif kind == "turn_end":
                err_tag = f" error={ev.get('error')!r}" if ev.get("error") else ""
                lines.append(
                    f"[turn {turn_id}] END status={ev.get('status')} steps={ev.get('steps')} "
                    f"attempt={attempt_id} duration={ev.get('duration_ms')}ms{err_tag}"
                )
        unfinished = find_unfinished_in_events(events)
        if unfinished:
            ids = ", ".join(str(u.get("call_id")) for u in unfinished)
            lines.append(f"[unfinished_calls] count={len(unfinished)} call_ids={ids}")
        return "\n".join(lines)


def find_unfinished_in_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """从事件序列中找出未完成的工具调用（缺少配对结果或结果状态为 interrupted）。"""
    pending: dict[str, dict[str, Any]] = {}
    for ev in events:
        kind = ev.get("event")
        cid = ev.get("call_id")
        if not isinstance(cid, str) or not cid:
            continue
        if kind == "tool_request":
            pending[cid] = ev
        elif kind == "tool_result":
            if ev.get("status") in ("ok", "error"):
                pending.pop(cid, None)
            elif ev.get("status") == "interrupted":
                pending[cid] = ev
    return list(pending.values())


def verify_trace_integrity(events: list[dict[str, Any]]) -> dict[str, Any]:
    """按每次尝试 (turn_id, attempt_id) 校验 Trace 完整性：请求/结果成对、步数从 1 连续递增、终止原因有记录。"""
    requests: list[tuple[Any, Any, int, str]] = []
    results_by_key: dict[tuple[Any, Any, int, str], dict[str, Any]] = {}
    steps_by_attempt: dict[tuple[Any, Any], list[int]] = {}
    turn_starts: list[tuple[Any, Any]] = []
    turn_ends: dict[tuple[Any, Any], dict[str, Any]] = {}
    implicit_attempts: dict[Any, int] = {}

    for ev in events:
        kind = ev.get("event")
        tid = ev.get("turn_id")
        if kind == "turn_start":
            aid = ev.get("attempt_id")
            if aid is None:
                aid = implicit_attempts.get(tid, 0) + 1
            implicit_attempts[tid] = aid
            turn_starts.append((tid, aid))
        elif kind == "tool_request":
            aid = ev.get("attempt_id", implicit_attempts.get(tid, 1))
            step = int(ev.get("step", 0))
            cid = str(ev.get("call_id", ""))
            requests.append((tid, aid, step, cid))
            steps_by_attempt.setdefault((tid, aid), []).append(step)
        elif kind == "tool_result":
            aid = ev.get("attempt_id", implicit_attempts.get(tid, 1))
            key = (tid, aid, int(ev.get("step", 0)), str(ev.get("call_id", "")))
            results_by_key[key] = ev
        elif kind == "turn_end":
            aid = ev.get("attempt_id", implicit_attempts.get(tid, 1))
            turn_ends[(tid, aid)] = ev

    paired = all(req_key in results_by_key for req_key in requests)
    contiguous_steps = all(
        steps == list(range(1, len(steps) + 1))
        for steps in steps_by_attempt.values()
    )
    valid_ends = (
        len(turn_ends) == len(turn_starts)
        and all(
            k in turn_ends
            and bool(turn_ends[k].get("status"))
            and (turn_ends[k].get("status") == "ok" or bool(turn_ends[k].get("error")))
            for k in turn_starts
        )
    )
    unfinished = find_unfinished_in_events(events)
    return {
        "valid": paired and contiguous_steps and valid_ends,
        "paired": paired,
        "contiguous_steps": contiguous_steps,
        "valid_ends": valid_ends,
        "unfinished_calls": unfinished,
    }
