"""上下文管理：从会话历史选取最近轮次、剥离历史思考过程，并在超出轮数或 token 预算时生成结构化摘要。"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any

from .llm import Message
from .memory import filter_historical_user_input, strip_memory_declarations

_OBS_RE = re.compile(r"^Observation \(tool=([^,]+), status=(ok|error|interrupted)\): (.*)$", re.S)
_TODO_LINE_RE = re.compile(r"^- #(\d+)\s+\[(pending|done|deleted)[^\]]*\]\s+(.*)$")
DEFAULT_CONTEXT_TOKENS = 16000
DEFAULT_RESERVED_TOKENS = 4000
DEFAULT_SUMMARY_TOKENS = 2000
_ABNORMAL_PREFIXES = (
    "已达最大步数",
    "模型回答或工具调用多次无效",
    "模型服务暂时不可用",
    "内部错误：",
    "本轮被中断",
    "（本轮被意外中断",
    "上下文预算不足",
    "工具调用未执行",
    "保存写操作失败",
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
    """估算消息列表的总 token 数（含每条消息的角色、内容及原生 tool_calls / tool_call_id 开销）。"""
    total = 0
    for m in messages:
        total += 4 + estimate_tokens(str(m.get("content", "")))
        tcs = m.get("tool_calls")
        if isinstance(tcs, list):
            for tc in tcs:
                if isinstance(tc, dict):
                    fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                    total += 4 + estimate_tokens(str(tc.get("id", "")))
                    total += estimate_tokens(str(fn.get("name", "")))
                    total += estimate_tokens(str(fn.get("arguments", "")))
        if m.get("role") == "tool":
            total += 2 + estimate_tokens(str(m.get("tool_call_id", ""))) + estimate_tokens(str(m.get("name", "")))
    return total


def estimate_tool_schema_tokens(tools: list[dict[str, Any]] | None) -> int:
    """估算原生工具定义的请求开销。"""
    if not tools:
        return 0
    return estimate_tokens(json.dumps(tools, ensure_ascii=False, separators=(",", ":"))) + 4 * len(tools)


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


def check_context_budget(
    messages: list[Message], max_tokens: int, tools: list[dict[str, Any]] | None = None,
) -> None:
    """检查消息与原生工具定义的总预算，超限时拒绝发送。"""
    used = estimate_messages_tokens(messages) + estimate_tool_schema_tokens(tools)
    if used > max_tokens:
        raise ContextBudgetError(f"上下文预算不足：估算 {used} tokens，上限 {max_tokens}；请缩短输入或分批处理。")


def _is_turn_start(msg: Message) -> bool:
    """判断消息是否为新一轮真实用户输入（排除运行时校验反馈）。"""
    if msg.get("role") != "user":
        return False
    content = msg.get("content", "")
    return not content.startswith("Validation error: ")


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


def sanitize_turn(turn: list[Message], memories: dict[str, str] | None = None) -> list[Message]:
    """清理历史噪音与过期记忆声明，保留有效消息和工具配对。"""
    if turn and _is_turn_start(turn[0]):
        original = turn[0].get("content", "")
        if original and not filter_historical_user_input(original, memories) and all(
            m.get("role") == "assistant" and not m.get("tool_calls") for m in turn[1:]
        ):
            return []
    out: list[Message] = []
    i = 0
    while i < len(turn):
        msg = turn[i]
        role, content = msg.get("role", ""), msg.get("content", "")
        if (
            role == "assistant"
            and not msg.get("tool_calls")
            and i + 1 < len(turn)
            and turn[i + 1].get("role") == "user"
            and turn[i + 1].get("content", "").startswith("Validation error: ")
        ):
            i += 2
            continue
        if role == "user" and content.startswith("Validation error: "):
            i += 1
            continue
        if role == "assistant":
            tcs = msg.get("tool_calls")
            if isinstance(tcs, list) and tcs:
                # 原生工具调用：剥离伴随调用的思考文本 content，完整保留 tool_calls
                out.append({"role": "assistant", "content": "", "tool_calls": copy.deepcopy(tcs)})
            else:
                out.append({"role": "assistant", "content": content})
        elif role == "tool":
            tool_msg: Message = {
                "role": "tool",
                "tool_call_id": str(msg.get("tool_call_id", "")),
                "content": content,
            }
            if msg.get("name"):
                tool_msg["name"] = str(msg["name"])
            out.append(tool_msg)
        else:
            out.append({"role": role, "content": filter_historical_user_input(content, memories)
                        if role == "user" and _is_turn_start(msg) else content})
        i += 1
    return _ensure_tool_Pairs(out)


def _ensure_tool_Pairs(messages: list[Message]) -> list[Message]:
    """确保单轮内所有 assistant.tool_calls 都有配对的 role='tool' 结果消息，防止残缺配对导致 API 报错。"""
    existing_tool_ids = {
        str(m.get("tool_call_id", ""))
        for m in messages
        if m.get("role") == "tool" and m.get("tool_call_id")
    }
    healed: list[Message] = []
    for m in messages:
        healed.append(m)
        tcs = m.get("tool_calls") if m.get("role") == "assistant" else None
        if isinstance(tcs, list):
            for tc in tcs:
                if not isinstance(tc, dict):
                    continue
                tc_id = str(tc.get("id", ""))
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                t_name = str(fn.get("name", ""))
                if tc_id and tc_id not in existing_tool_ids:
                    existing_tool_ids.add(tc_id)
                    healed.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "name": t_name,
                        "content": (
                            f"Observation (tool={t_name}, status=interrupted): the run was interrupted while this "
                            "tool was executing; it may or may not have taken effect."
                        ),
                    })
    return healed


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
    """从旧对话轮次和工具结果（原生 tool_calls/tool）中提取结构化摘要，不保留历史 thought。"""
    summary = parse_summary(existing_summary)
    for turn in turns:
        user_text = ""
        final_answer = ""
        native_calls: dict[str, tuple[str, dict[str, Any]]] = {}
        had_tool_fact = False

        for msg in turn:
            role, content = msg.get("role", ""), msg.get("content", "")
            if role == "user":
                if content.startswith("Validation error: "):
                    continue
                if not user_text:
                    user_text = strip_memory_declarations(content).strip()
                    _add_unique(summary.user_goals, user_text)
                    for fact in _extract_user_facts(user_text):
                        _add_unique(summary.key_facts, fact)
            elif role == "tool":
                tc_id = str(msg.get("tool_call_id", ""))
                tool_name, tool_args = native_calls.get(tc_id, (str(msg.get("name", "")), {}))
                m = _OBS_RE.match(content)
                if m:
                    status, payload = m.group(2), m.group(3)
                elif content.startswith("Error: "):
                    status, payload = "error", content[7:]
                else:
                    status, payload = "ok", content
                _extract_observation(tool_name, status, payload, tool_args, summary)
                if status == "ok" and tool_name in ("weather", "calculator", "search"):
                    had_tool_fact = True
            elif role == "assistant":
                tcs = msg.get("tool_calls")
                if isinstance(tcs, list) and tcs:
                    # 原生工具调用消息：伴随的 content 属于当轮思考过程，绝不读取为长期记忆
                    for tc in tcs:
                        if not isinstance(tc, dict):
                            continue
                        tc_id = str(tc.get("id", ""))
                        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                        t_name = str(fn.get("name", ""))
                        raw_args = fn.get("arguments", "{}")
                        try:
                            parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                        except ValueError:
                            parsed_args = {}
                        native_calls[tc_id] = (t_name, parsed_args if isinstance(parsed_args, dict) else {})
                    continue
                final_answer = content.strip()

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
    tools_schema: list[dict[str, Any]] | None = None,
    recalled_memories: dict[str, str] | None = None,
    memory_question: bool = False,
    session_memories: dict[str, str] | None = None,
) -> ContextPlan:
    """按消息、用户记忆及工具定义预算构建受限上下文，不修改历史原文。"""
    turns = split_into_turns(history)
    sanitized_turns = [sanitize_turn(t, session_memories) for t in turns]
    all_recent = [m for t in sanitized_turns for m in t]

    sys_msg: Message = {"role": "system", "content": system_prompt}
    user_msg: Message = {"role": "user", "content": user_input}
    memory_msg: Message | None = None
    if recalled_memories:
        lines = ["当前会话中用户明确提供的相关信息（以此为准，勿从历史回答或工具结果补全）："]
        for key, value in recalled_memories.items():
            line = f"- {key}: {value}"
            if estimate_tokens("\n".join([*lines, line])) > 800:
                break
            lines.append(line)
        if len(lines) > 1:
            memory_msg = {"role": "system", "content": "\n".join(lines)}
    elif memory_question:
        memory_msg = {"role": "system", "content": (
            "当前会话没有已保存的相关用户记忆；不要从历史回答或工具结果推断个人信息。"
        )}
    fixed_msgs = [sys_msg, *([memory_msg] if memory_msg else []), user_msg]
    hard_budget = max_context_tokens
    schema_tokens = estimate_tool_schema_tokens(tools_schema)
    fixed_tokens = estimate_messages_tokens(fixed_msgs) + schema_tokens
    check_context_budget(fixed_msgs, hard_budget, tools_schema)
    budget = max(hard_budget - max(reserved_tokens, 0), fixed_tokens)

    refreshed_existing = refresh_summary_todos(existing_summary, todos, max_summary_tokens) if existing_summary.strip() else ""
    base_msgs: list[Message] = [sys_msg]
    if memory_msg:
        base_msgs.append(memory_msg)
    if refreshed_existing:
        base_msgs.append({"role": "system", "content": refreshed_existing})
    base_msgs.extend(all_recent)
    base_msgs.append(user_msg)

    need_compress = len(turns) > max_history_turns or estimate_messages_tokens(base_msgs) + schema_tokens > budget
    if not need_compress:
        return ContextPlan(
            messages=base_msgs,
            recent_messages=all_recent,
            summary=refreshed_existing,
            compressed=bool(refreshed_existing),
            compressed_turn_count=0,
            kept_turn_count=len(turns),
            estimated_tokens=estimate_messages_tokens(base_msgs) + schema_tokens,
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
            *([memory_msg] if memory_msg else []),
            {"role": "system", "content": summary_text},
            *recent_messages,
            user_msg,
        ]
        total_tokens = estimate_messages_tokens(candidate_msgs) + schema_tokens
        check_context_budget(candidate_msgs, hard_budget, tools_schema)
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
