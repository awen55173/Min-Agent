"""带会话持久化与受限上下文管理的对话服务：加载历史与待办 → 构建上下文/压缩摘要 → 运行 Agent → 原子保存。"""
from __future__ import annotations

import json
from typing import Callable

from .context import build_context, refresh_summary_todos
from .llm import LLMClient
from .loop import OK, Agent, AgentResult
from .prompt import build_system_prompt
from .session import SessionManager
from .tools.registry import ToolContext, ToolRegistry
from .tools.todo import TodoStore

INTERRUPTED_NOTE = "（本轮被意外中断，未能给出回复；已完成的操作已保存。）"

RegistryFactory = Callable[[TodoStore], ToolRegistry]


class ChatService:
    """把 Agent、Context 与 SessionManager 串起来：一次 send 对应会话内的一轮对话。"""

    def __init__(
        self,
        llm: LLMClient,
        manager: SessionManager,
        registry_factory: RegistryFactory,
        max_steps: int = 8,
        max_parse_retries: int = 3,
        max_history_turns: int = 6,
        max_context_tokens: int = 4000,
        reserved_tokens: int = 1200,
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

    def send(self, user_id: str, session_id: str, text: str, create: bool = False) -> AgentResult:
        """在会话内处理一轮输入并保存。

        - 调用 Agent.run() 前通过 build_context 构建受限上下文（系统提示 + 摘要 + 最近 N 轮对话 + 本轮输入）；
        - 区分完整持久化历史（data.history，用于会话回放）与发送给模型的有限上下文；
        - 摘要随会话持久化（data.summary），并在工具执行后与待办存储实时对齐；
        - 失败或中断退出时同样保存历史、摘要与已完成的待办操作。
        """
        with self._manager.open(user_id, session_id, create=create) as session:
            data = session.data
            ctx = ToolContext(user_id, session_id)
            store = TodoStore()
            store.load_state(ctx, data.todos)
            registry = self._factory(store)
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
            data.summary = ctx_plan.summary
            prefix_len = len(ctx_plan.messages) - 1  # 本轮 user_input 之前的上下文消息数
            agent = Agent(self._llm, registry, self._max_steps, self._max_parse_retries)
            turn_id = session.next_turn_id()
            try:
                result = agent.run(text, user_id, session_id, turn_id, context_messages=ctx_plan.messages)
            except BaseException as e:
                partial: AgentResult | None = getattr(e, "agent_result", None)
                if partial is not None:
                    new_messages = partial.messages[prefix_len:]
                else:  # 尚未进入循环就被中断
                    new_messages = [{"role": "user", "content": text}]
                new_messages.append(_assistant_final(INTERRUPTED_NOTE))
                data.history += new_messages
                data.todos = store.export_state(ctx)
                data.summary = refresh_summary_todos(data.summary, data.todos)
                session.save()
                raise
            new_messages = result.messages[prefix_len:]  # 只追加本轮新增的原始消息，保留完整历史
            if result.status != OK:
                # 失败提示作为 assistant 回复入历史，保持对话轮次完整，模型后续追问能知道发生了什么
                new_messages.append(_assistant_final(result.answer))
            data.history += new_messages
            data.todos = store.export_state(ctx)
            data.summary = refresh_summary_todos(data.summary, data.todos)
            session.save()
            return result


def _assistant_final(text: str) -> dict[str, str]:
    """构造一条以 final_answer 形式记录的 assistant 消息。"""
    return {"role": "assistant", "content": json.dumps({"final_answer": text}, ensure_ascii=False)}
