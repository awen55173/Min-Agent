"""上下文管理：从会话历史选取最近轮次、剥离历史思考过程，并在超出轮数或 token 预算时生成结构化摘要。"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .llm import Message
from .parser import ParseError, extract_json

_OBS_RE = re.compile(r"^Observation \(tool=([^,]+), status=(ok|error|interrupted)\): (.*)$", re.S)
_TODO_LINE_RE = re.compile(r"^- #(\d+)\s+\[(pending|done|deleted)[^\]]*\]\s+(.*)$")
DEFAULT_CONTEXT_TOKENS = 16000
DEFAULT_RESERVED_TOKENS = 4000
DEFAULT_SUMMARY_TOKENS = 2000
_ABNORMAL_PREFIXES = (
    "已达最大步数",
    "模型输出格式多次不合法",
    "模型服务暂时不可用",
    "内部错误：",
    "本轮被中断",
    "（本轮被意外中断",
    "上下文预算不足",
    "工具调用未执行",
)


class ContextBudgetError(Exception):
    """固定上下文无法放入预算，调用方应结束请求并保存已完成操作。"""


def estimate_tokens(text: str) -> int:
    """估算文本的 token 数：中文字符按 1 token 计，其余字符每 4 个按 1 token 计。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return cjk + (other + 3) // 4


def estimate_messages_tokens(messages: list[Message]) -> int:
    """估算消息列表的总 token 数（含每条消息的角色与格式开销）。"""
    return sum(4 + estimate_tokens(m.get("content", "")) for m in messages)


def _clip_preserve_edges(text: str, max_chars: int) -> str:
    """截断文本时同时保留开头和结尾，避免末尾的金额、地点或偏好丢失。"""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 12:
        return text[:max_chars]
    head = max_chars // 2
    marker = " …(已压缩)… "
    tail = max_chars - head - len(marker)
    return f"{text[:head]}{marker}{text[-tail:]}" if tail > 0 else text[:max_chars]


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """把单条消息压缩到 token 上限，并优先保留两端内容。"""
    if max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    low, high = 0, len(text)
    best = ""
    while low <= high:
        mid = (low + high) // 2
        candidate = _clip_preserve_edges(text, mid)
        if estimate_tokens(candidate) <= max_tokens:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    return best


def check_context_budget(messages: list[Message], max_tokens: int) -> None:
    """校验每次模型请求的硬预算，超限时明确拒绝发送。"""
    used = estimate_messages_tokens(messages)
    if used > max_tokens:
        raise ContextBudgetError(f"上下文预算不足：估算 {used} tokens，上限 {max_tokens}；请缩短输入或分批处理。")


def _is_turn_start(msg: Message) -> bool:
    """判断消息是否为新一轮真实用户输入（排除工具 Observation 与格式错误反馈）。"""
    if msg.get("role") != "user":
        return False
    content = msg.get("content", "")
    return not content.startswith("Observation (tool=") and not content.startswith("Format error: ")


def split_into_turns(history: list[Message]) -> list[list[Message]]:
    """按真实用户输入将会话历史切分为对话轮次列表。"""
    turns: list[list[Message]] = []
    current: list[Message] = []
    for msg in history:
        if _is_turn_start(msg) and current:
            turns.append(current)
            current = [msg]
        else:
            current.append(msg)
    if current:
        turns.append(current)
    return turns


def _strip_assistant_thought(content: str) -> str:
    """剥离历史 assistant 消息中的 thought 字段，仅保留工具调用或最终答案。"""
    try:
        obj = extract_json(content)
    except ParseError:
        return content
    cleaned = {k: v for k, v in obj.items() if k != "thought"}
    return json.dumps(cleaned, ensure_ascii=False)


def sanitize_turn(turn: list[Message]) -> list[Message]:
    """清理单轮历史消息：移除格式错误重试噪音，并剥离 assistant 的历史 thought。"""
    out: list[Message] = []
    i = 0
    while i < len(turn):
        msg = turn[i]
        role, content = msg.get("role", ""), msg.get("content", "")
        if (
            role == "assistant"
            and i + 1 < len(turn)
            and turn[i + 1].get("role") == "user"
            and turn[i + 1].get("content", "").startswith("Format error: ")
        ):
            i += 2
            continue
        if role == "user" and content.startswith("Format error: "):
            i += 1
            continue
        if role == "assistant":
            out.append({"role": "assistant", "content": _strip_assistant_thought(content)})
        else:
            out.append({"role": role, "content": content})
        i += 1
    return out


def _add_unique(items: list[str], value: str) -> None:
    """向列表中追加去重后的非空字符串。"""
    v = " ".join(value.strip().split())
    if v and v not in items:
        items.append(v)


def _select_prioritized(items: list[str], limit: int) -> list[str]:
    """超出条数上限时保留最早的条目与最近的条目，防止初始任务或近期事实丢失。"""
    if len(items) <= limit or limit <= 0:
        return list(items[:limit]) if limit > 0 else []
    if limit == 1:
        return [items[-1]]
    head = max(1, limit // 3)
    tail = limit - head
    return items[:head] + items[-tail:]


_COMMON_CITIES = (
    "北京|上海|广州|深圳|苏州|杭州|成都|重庆|武汉|西安|南京|天津|青岛|厦门|长沙|郑州|济南|合肥|福州|昆明|大连|宁波|无锡|哈尔滨|沈阳|石家庄"
)


def _extract_user_facts(text: str) -> list[str]:
    """从用户输入提取金额、地点和偏好等短事实，避免摘要只保留前缀。"""
    facts: list[str] = []
    for match in re.finditer(r"(?:预算|金额|价格|费用|花费)\s*[:：]?\s*([\d,.]+)\s*(元|人民币|块)?", text, re.I):
        _add_unique(facts, f"金额/预算: {match.group(1)}{match.group(2) or ''}")
    for match in re.finditer(rf"(?:{_COMMON_CITIES})(?:市)?", text):
        _add_unique(facts, f"地点: {match.group(0)}")
    for match in re.finditer(r"(?:城市|目的地|地点|居住地)\s*[:：为是]?\s*([^，。；,;、\s]{1,16})", text):
        _add_unique(facts, f"地点: {match.group(1)}")
    for match in re.finditer(r"\d+(?:\.\d+)?\s*(?:万元|元|人|天|小时|分钟|度|岁|%|％|公里|千米|kg|km|°C)", text):
        _add_unique(facts, f"数值: {match.group(0).strip()}")
    for match in re.finditer(r"(?:素食|清真|无糖|低脂|不吃[^，。；,;\s]{1,8}|偏好\s*[:：]?\s*[^，。；,;\s]{1,12})", text, re.I):
        _add_unique(facts, f"偏好: {match.group(0).strip()}")
    return facts


@dataclass
class StructuredSummary:
    """结构化摘要数据：保留用户目标、关键事实、待办 ID 与状态、已执行操作和未完成事项。"""

    user_goals: list[str] = field(default_factory=list)
    key_facts: list[str] = field(default_factory=list)
    todos: dict[int, dict[str, Any]] = field(default_factory=dict)
    completed_actions: list[str] = field(default_factory=list)
    unfinished_notes: list[str] = field(default_factory=list)

    def apply_todo_store(self, todos_state: dict[str, Any] | None) -> None:
        """与待办存储状态对齐：更新已有待办状态，并将存储中不存在的历史待办标记为已删除。"""
        if not isinstance(todos_state, dict) or not isinstance(todos_state.get("items"), list):
            return
        live_ids: set[int] = set()
        for it in todos_state["items"]:
            if isinstance(it, dict) and isinstance(it.get("id"), int):
                tid = it["id"]
                live_ids.add(tid)
                self.todos[tid] = {
                    "id": tid,
                    "text": str(it.get("text", "")),
                    "done": bool(it.get("done", False)),
                    "deleted": False,
                }
        for tid, info in self.todos.items():
            if tid not in live_ids:
                info["deleted"] = True

    def _todo_lines(self) -> list[str]:
        """生成待办 ID 与状态行列表。"""
        lines: list[str] = []
        for tid in sorted(self.todos):
            info = self.todos[tid]
            if info.get("deleted"):
                status = "deleted/已删除"
            elif info.get("done"):
                status = "done/已完成"
            else:
                status = "pending/未完成"
            lines.append(f"- #{tid} [{status}] {info.get('text', '')}")
        return lines

    def _unfinished_lines(self) -> list[str]:
        """生成未完成事项行列表（含所有未完成待办与中断/超步数记录）。"""
        lines: list[str] = []
        for tid in sorted(self.todos):
            info = self.todos[tid]
            if not info.get("done") and not info.get("deleted"):
                lines.append(f"- 待办 #{tid} (pending/未完成): {info.get('text', '')}")
        for note in self.unfinished_notes:
            lines.append(f"- {note}")
        return lines

    def _render(self, max_items: int, max_chars: int) -> str:
        """按指定的条目上限与单条字符上限渲染摘要文本。"""
        goals = [f"- {_clip_preserve_edges(g, max_chars)}" for g in _select_prioritized(self.user_goals, max_items)]
        tool_facts = [f for f in self.key_facts if not f.startswith("对话:")]
        chat_facts = [f for f in self.key_facts if f.startswith("对话:")]
        kept_facts = _select_prioritized(tool_facts, max_items)
        if len(kept_facts) < max_items:
            kept_facts += _select_prioritized(chat_facts, max_items - len(kept_facts))
        facts = [f"- {_clip_preserve_edges(f, max_chars)}" for f in kept_facts]
        todo_lines = self._todo_lines()
        todos = [_clip_preserve_edges(t, max_chars + 36) for t in _select_prioritized(todo_lines, max_items)]
        if len(todo_lines) > max_items:
            todos.append(f"- 另有 {len(todo_lines) - max_items} 条待办未展示；请调用 todo.list 获取完整 ID 与实时状态。")
        actions = [f"- {_clip_preserve_edges(a, max_chars)}" for a in _select_prioritized(self.completed_actions, max_items)]
        unfinished = [_clip_preserve_edges(u, max_chars + 36) for u in _select_prioritized(self._unfinished_lines(), max_items)]

        sections = [
            "[历史对话结构化摘要 / Conversation Summary]",
            "（注：待办事项的实时状态以 todo 工具存储为准）",
            "1. 用户目标 (User Goals):",
            *(goals or ["- 无 (None)"]),
            "2. 关键事实 (Key Facts):",
            *(facts or ["- 无 (None)"]),
            "3. 待办 ID 与状态 (Todo IDs & Status):",
            *(todos or ["- 无 (None)"]),
            "4. 已执行操作 (Completed Actions):",
            *(actions or ["- 无 (None)"]),
            "5. 未完成事项 (Unfinished Items):",
            *(unfinished or ["- 无 (None)"]),
        ]
        return "\n".join(sections)

    def format(self, max_tokens: int | None = None) -> str:
        """渲染摘要并限制条目与待办大小，预算不足时拒绝生成残缺摘要。"""
        for max_items, max_chars in ((12, 160), (8, 120), (5, 90), (3, 60), (2, 45), (1, 35)):
            text = self._render(max_items, max_chars)
            if max_tokens is None or estimate_tokens(text) <= max_tokens:
                return text
        raise ContextBudgetError("上下文预算不足：结构化摘要无法放入预算；请增加预算或分批处理。")


def parse_summary(summary_text: str) -> StructuredSummary:
    """解析已持久化的结构化摘要文本，以便与新压缩轮次增量合并。"""
    summary = StructuredSummary()
    if not summary_text or not summary_text.strip():
        return summary
    if "[历史对话结构化摘要" not in summary_text:
        _add_unique(summary.key_facts, summary_text.strip())
        return summary

    section = 0
    for raw_line in summary_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("[历史对话") or line.startswith("（注："):
            continue
        if line.startswith("1. 用户目标"):
            section = 1
            continue
        if line.startswith("2. 关键事实"):
            section = 2
            continue
        if line.startswith("3. 待办 ID"):
            section = 3
            continue
        if line.startswith("4. 已执行操作"):
            section = 4
            continue
        if line.startswith("5. 未完成事项"):
            section = 5
            continue
        if not line.startswith("- ") or line == "- 无 (None)":
            continue
        item = line[2:].strip()
        if section == 1:
            _add_unique(summary.user_goals, item)
        elif section == 2:
            _add_unique(summary.key_facts, item)
        elif section == 3:
            m = _TODO_LINE_RE.match(line)
            if m:
                tid = int(m.group(1))
                state = m.group(2)
                summary.todos[tid] = {
                    "id": tid,
                    "text": m.group(3).strip(),
                    "done": state == "done",
                    "deleted": state == "deleted",
                }
        elif section == 4:
            _add_unique(summary.completed_actions, item)
        elif section == 5 and not item.startswith("待办 #"):
            _add_unique(summary.unfinished_notes, item)
    return summary


def _extract_observation(tool: str, status: str, payload: str, pending_args: dict, summary: StructuredSummary) -> None:
    """从单条工具 Observation 中提取关键事实、待办 ID 与状态、已执行操作或未完成说明。"""
    if status == "interrupted":
        _add_unique(summary.unfinished_notes, f"工具 {tool} 执行中断")
        return
    if status != "ok":
        return
    try:
        data = json.loads(payload)
    except ValueError:
        data = None

    if tool == "todo" and isinstance(data, dict):
        if isinstance(data.get("item"), dict):
            it = data["item"]
            tid = it.get("id")
            if isinstance(tid, int):
                summary.todos[tid] = {
                    "id": tid,
                    "text": str(it.get("text", "")),
                    "done": bool(it.get("done", False)),
                    "deleted": False,
                }
                _add_unique(summary.completed_actions, f"todo.add -> #{tid} {it.get('text', '')}")
        elif isinstance(data.get("complete"), dict):
            it = data["complete"]
            tid = it.get("id")
            if isinstance(tid, int):
                summary.todos[tid] = {
                    "id": tid,
                    "text": str(it.get("text", "")),
                    "done": True,
                    "deleted": False,
                }
                _add_unique(summary.completed_actions, f"todo.complete -> #{tid} {it.get('text', '')} (done)")
        elif isinstance(data.get("delete"), dict):
            it = data["delete"]
            tid = it.get("id")
            if isinstance(tid, int):
                summary.todos[tid] = {
                    "id": tid,
                    "text": str(it.get("text", "")),
                    "done": bool(it.get("done", False)),
                    "deleted": True,
                }
                _add_unique(summary.completed_actions, f"todo.delete -> #{tid} {it.get('text', '')} (deleted)")
        elif isinstance(data.get("items"), list):
            for it in data["items"]:
                if isinstance(it, dict) and isinstance(it.get("id"), int):
                    tid = it["id"]
                    summary.todos[tid] = {
                        "id": tid,
                        "text": str(it.get("text", "")),
                        "done": bool(it.get("done", False)),
                        "deleted": False,
                    }
            _add_unique(summary.completed_actions, f"todo.list -> {len(data['items'])} items")
        return

    if tool == "weather" and isinstance(data, dict):
        city = data.get("city", pending_args.get("city", ""))
        temp = data.get("temp_c", "")
        desc = data.get("description", "")
        hum = data.get("humidity", "")
        _add_unique(summary.key_facts, f"天气 {city}: {temp}°C, {desc}, 湿度 {hum}%")
        _add_unique(summary.completed_actions, f"weather({city}) -> {temp}°C {desc}")
        return

    if tool == "calculator" and isinstance(data, dict):
        expr = data.get("expression", pending_args.get("expression", ""))
        res = data.get("result", "")
        _add_unique(summary.key_facts, f"计算 {expr} = {res}")
        _add_unique(summary.completed_actions, f"calculator({expr}) -> {res}")
        return

    if tool == "search" and isinstance(data, dict):
        query = data.get("query", pending_args.get("query", ""))
        results = data.get("results") if isinstance(data.get("results"), list) else []
        parts = [f"{r.get('title', '')}: {str(r.get('snippet', ''))[:60]}" for r in results[:2] if isinstance(r, dict)]
        detail = " | ".join(p for p in parts if p.strip(": "))
        _add_unique(summary.key_facts, f"搜索 {query}: {detail or '无结果'}")
        _add_unique(summary.completed_actions, f"search({query}) -> {len(results)} results")
        return

    _add_unique(summary.key_facts, f"{tool}: {payload[:100]}")
    _add_unique(summary.completed_actions, f"{tool}({pending_args}) -> ok")


def summarize_turns(
    turns: list[list[Message]],
    existing_summary: str = "",
    todos: dict[str, Any] | None = None,
) -> StructuredSummary:
    """从旧对话轮次和工具结果中提取结构化摘要（不保留历史 thought），并与已有摘要及待办存储对齐。"""
    summary = parse_summary(existing_summary)
    for turn in turns:
        user_text = ""
        final_answer = ""
        pending_args: dict[str, Any] = {}
        had_tool_fact = False

        for msg in turn:
            role, content = msg.get("role", ""), msg.get("content", "")
            if role == "user":
                m = _OBS_RE.match(content)
                if m:
                    tool, status, payload = m.group(1), m.group(2), m.group(3)
                    _extract_observation(tool, status, payload, pending_args, summary)
                    if status == "ok" and tool in ("weather", "calculator", "search"):
                        had_tool_fact = True
                    continue
                if content.startswith("Format error: "):
                    continue
                if not user_text:
                    user_text = content.strip()
                    _add_unique(summary.user_goals, user_text)
                    for fact in _extract_user_facts(user_text):
                        _add_unique(summary.key_facts, fact)
            elif role == "assistant":
                try:
                    obj = extract_json(content)
                except ParseError:
                    continue
                # 历史 thought 不作为长期记忆，此处绝不读取 obj["thought"]
                if isinstance(obj.get("action"), str):
                    args = obj.get("action_input")
                    pending_args = args if isinstance(args, dict) else {}
                elif isinstance(obj.get("final_answer"), str):
                    final_answer = obj["final_answer"].strip()

        if final_answer:
            if any(final_answer.startswith(prefix) for prefix in _ABNORMAL_PREFIXES):
                first_line = final_answer.splitlines()[0]
                _add_unique(summary.unfinished_notes, f"未完成请求「{user_text[:40]}」: {first_line[:60]}")
            elif user_text:
                for fact in _extract_user_facts(final_answer):
                    _add_unique(summary.key_facts, f"回答中的{fact}")
                prefix = "对话: " if had_tool_fact else "事实: "
                _add_unique(summary.key_facts, f"{prefix}{user_text[:60]} -> {final_answer[:80]}")

    summary.apply_todo_store(todos)
    return summary


def refresh_summary_todos(
    summary_text: str, todos: dict[str, Any] | None, max_tokens: int | None = None,
) -> str:
    """用待办实时状态刷新已有摘要，并按需限制发送长度。"""
    if not summary_text or not summary_text.strip():
        return ""
    summary = parse_summary(summary_text)
    summary.apply_todo_store(todos)
    return summary.format(max_tokens=max_tokens)


@dataclass
class ContextPlan:
    """构建完成的受限上下文及元信息。"""

    messages: list[Message]
    recent_messages: list[Message]
    summary: str
    compressed: bool
    compressed_turn_count: int
    kept_turn_count: int
    estimated_tokens: int


def build_context(
    system_prompt: str,
    history: list[Message],
    user_input: str,
    existing_summary: str = "",
    todos: dict[str, Any] | None = None,
    max_history_turns: int = 6,
    max_context_tokens: int = DEFAULT_CONTEXT_TOKENS,
    reserved_tokens: int = DEFAULT_RESERVED_TOKENS,
    max_summary_tokens: int = DEFAULT_SUMMARY_TOKENS,
) -> ContextPlan:
    """构建发送给模型的受限上下文：系统提示 +（可选）结构化摘要 + 最近对话原文（剥离 thought）+ 本轮输入。

    按历史轮数上限和预留工具空间后的目标预算触发压缩，并保证最终消息不超过
    max_context_tokens；不修改传入的 history 列表。
    """
    turns = split_into_turns(history)
    sanitized_turns = [sanitize_turn(t) for t in turns]
    all_recent = [m for t in sanitized_turns for m in t]

    sys_msg: Message = {"role": "system", "content": system_prompt}
    user_msg: Message = {"role": "user", "content": user_input}
    hard_budget = max_context_tokens
    fixed_tokens = estimate_messages_tokens([sys_msg, user_msg])
    check_context_budget([sys_msg, user_msg], hard_budget)
    budget = max(hard_budget - max(reserved_tokens, 0), fixed_tokens)

    refreshed_existing = refresh_summary_todos(existing_summary, todos, max_summary_tokens) if existing_summary.strip() else ""
    base_msgs: list[Message] = [sys_msg]
    if refreshed_existing:
        base_msgs.append({"role": "system", "content": refreshed_existing})
    base_msgs.extend(all_recent)
    base_msgs.append(user_msg)

    need_compress = len(turns) > max_history_turns or estimate_messages_tokens(base_msgs) > budget
    if not need_compress:
        return ContextPlan(
            messages=base_msgs,
            recent_messages=all_recent,
            summary=refreshed_existing,
            compressed=bool(refreshed_existing),
            compressed_turn_count=0,
            kept_turn_count=len(turns),
            estimated_tokens=estimate_messages_tokens(base_msgs),
        )

    keep_count = min(len(turns), max(max_history_turns, 0))
    if keep_count == len(turns):
        keep_count = max(len(turns) - 1, 0)

    while True:
        split_idx = len(turns) - keep_count
        compressed_turns = turns[:split_idx]
        kept_turns = sanitized_turns[split_idx:]
        recent_messages = [m for t in kept_turns for m in t]

        summary_obj = summarize_turns(compressed_turns, existing_summary=existing_summary, todos=todos)
        # 保留整轮历史，并按实际剩余空间分配摘要预算；不截断系统提示和本轮输入。
        remaining = budget - fixed_tokens - estimate_messages_tokens(recent_messages) - 4
        try:
            summary_text = summary_obj.format(max_tokens=min(max_summary_tokens, remaining))
        except ContextBudgetError:
            if keep_count > 0:
                keep_count -= 1
                continue
            if budget < hard_budget:
                budget = hard_budget
                continue
            raise
        candidate_msgs: list[Message] = [
            sys_msg,
            {"role": "system", "content": summary_text},
            *recent_messages,
            user_msg,
        ]
        total_tokens = estimate_messages_tokens(candidate_msgs)
        check_context_budget(candidate_msgs, hard_budget)
        if total_tokens <= budget:
            return ContextPlan(
                messages=candidate_msgs,
                recent_messages=recent_messages,
                summary=summary_text,
                compressed=True,
                compressed_turn_count=len(compressed_turns),
                kept_turn_count=keep_count,
                estimated_tokens=total_tokens,
            )
        keep_count -= 1
