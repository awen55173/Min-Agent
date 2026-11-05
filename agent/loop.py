"""Agent 主循环：调用 LLM → 解析 → 执行工具 → 回填 observation，直到得到最终答案或预算耗尽。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .llm import LLMClient, LLMError, Message
from .parser import ParseError, parse_output
from .prompt import build_system_prompt
from .tools.registry import ToolContext, ToolRegistry, TurnId, new_turn_id

OK = "ok"
MAX_STEPS = "max_steps"
PARSE_EXHAUSTED = "parse_retries_exhausted"
LLM_FAILED = "llm_error"

_OBS_MAX = 2000  # 单条 observation 的最大字符数


@dataclass
class ActionRecord:
    """一次已执行的工具调用记录。"""

    step: int
    thought: str
    tool: str
    args: dict[str, Any]
    ok: bool
    result: str


@dataclass
class AgentResult:
    """一次请求的结果：status 为终止原因，answer 为返回给用户的文本。"""

    status: str
    answer: str
    actions: list[ActionRecord] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    turn_id: TurnId | None = None


def _summarize(actions: list[ActionRecord]) -> str:
    """把已完成的工具操作整理为简短文本，用于失败提示。"""
    if not actions:
        return "本轮尚未执行任何工具操作。"
    lines = [f"- {a.tool}({a.args}) -> {'成功' if a.ok else '失败'}" for a in actions]
    return "本轮已执行的操作：\n" + "\n".join(lines)


class Agent:
    """最小 Agent：单次请求内的 ReAct 循环（同步工具调用）。"""

    def __init__(self, llm: LLMClient, registry: ToolRegistry, max_steps: int = 8, max_parse_retries: int = 3):
        """保存依赖与预算：max_steps 为单轮最大工具调用数，max_parse_retries 为连续解析失败上限。"""
        self._llm = llm
        self._registry = registry
        self._max_steps = max_steps
        self._max_parse_retries = max_parse_retries

    def run(
        self,
        user_input: str,
        user_id: str = "default",
        session_id: str = "default",
        turn_id: TurnId | None = None,
        history: list[Message] | None = None,
    ) -> AgentResult:
        """处理一次用户输入并返回结果；整轮使用同一个 turn_id 供待办去重，未传入时自动生成全局唯一 ID。"""
        if turn_id is None:
            turn_id = new_turn_id()
        ctx = ToolContext(user_id, session_id, turn_id)
        messages: list[Message] = [{"role": "system", "content": build_system_prompt(self._registry)}]
        messages += history or []
        messages.append({"role": "user", "content": user_input})

        actions: list[ActionRecord] = []

        def finish(status: str, answer: str) -> AgentResult:
            """组装终止结果。"""
            return AgentResult(status, answer, actions, messages, turn_id)

        parse_failures = 0
        while True:
            try:
                raw = self._llm.chat(messages)
            except LLMError as e:
                return finish(LLM_FAILED, f"模型服务暂时不可用，请稍后重试。（{e}）\n{_summarize(actions)}")
            messages.append({"role": "assistant", "content": raw})

            try:
                parsed = parse_output(raw, self._registry)
            except ParseError as e:
                parse_failures += 1
                if parse_failures > self._max_parse_retries:
                    return finish(PARSE_EXHAUSTED, f"模型输出格式多次不合法，已停止。最近一次原因：{e}\n{_summarize(actions)}")
                messages.append({"role": "user", "content": f"Format error: {e}. Reply again with a single valid JSON object."})
                continue
            parse_failures = 0

            if parsed.kind == "final":
                return finish(OK, parsed.final_answer or "")

            if len(actions) >= self._max_steps:
                return finish(MAX_STEPS, f"已达最大步数（{self._max_steps}），任务可能未完成。\n{_summarize(actions)}")
            result = self._registry.execute(parsed.action, parsed.action_input, ctx)
            actions.append(
                ActionRecord(len(actions) + 1, parsed.thought, parsed.action, parsed.action_input, result.ok, result.content)
            )
            status = "ok" if result.ok else "error"
            content = result.content if len(result.content) <= _OBS_MAX else result.content[:_OBS_MAX] + "...(truncated)"
            messages.append({"role": "user", "content": f"Observation (tool={parsed.action}, status={status}): {content}"})
