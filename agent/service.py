"""带会话持久化的对话服务：加载历史与待办 → 运行 Agent → 原子保存，全程持有会话锁。"""
from __future__ import annotations

import json
from typing import Callable

from .llm import LLMClient
from .loop import OK, Agent, AgentResult
from .session import SessionManager
from .tools.registry import ToolContext, ToolRegistry
from .tools.todo import TodoStore

INTERRUPTED_NOTE = "（本轮被意外中断，未能给出回复；已完成的操作已保存。）"

RegistryFactory = Callable[[TodoStore], ToolRegistry]


class ChatService:
    """把 Agent 与 SessionManager 串起来：一次 send 对应会话内的一轮对话。"""

    def __init__(
        self,
        llm: LLMClient,
        manager: SessionManager,
        registry_factory: RegistryFactory,
        max_steps: int = 8,
        max_parse_retries: int = 3,
    ):
        """保存依赖；registry_factory 用会话恢复出的待办存储构建工具注册表。"""
        self._llm = llm
        self._manager = manager
        self._factory = registry_factory
        self._max_steps = max_steps
        self._max_parse_retries = max_parse_retries

    def send(self, user_id: str, session_id: str, text: str, create: bool = False) -> AgentResult:
        """在会话内处理一轮输入并保存。

        锁覆盖加载—执行—保存；Agent 失败（预算耗尽、解析失败、API 失败、内部错误）时同样保存历史与
        已完成的待办操作；进程被中断时保存待办及本轮已发生的消息（含工具调用与 observation），再把异常继续抛出。
        """
        with self._manager.open(user_id, session_id, create=create) as session:
            data = session.data
            ctx = ToolContext(user_id, session_id)
            store = TodoStore()
            store.load_state(ctx, data.todos)
            agent = Agent(self._llm, self._factory(store), self._max_steps, self._max_parse_retries)
            history = list(data.history)  # 不含 system 提示词，由 Agent 每轮重新生成
            turn_id = session.next_turn_id()
            try:
                result = agent.run(text, user_id, session_id, turn_id, history)
            except BaseException as e:
                partial: AgentResult | None = getattr(e, "agent_result", None)
                if partial is not None:
                    new_messages = partial.messages[1 + len(history):]
                else:  # 尚未进入循环就被中断
                    new_messages = [{"role": "user", "content": text}]
                new_messages.append(_assistant_final(INTERRUPTED_NOTE))
                data.history += new_messages
                data.todos = store.export_state(ctx)
                session.save()
                raise
            new_messages = result.messages[1 + len(history):]  # 去掉 system 与已有历史，只追加本轮新增
            if result.status != OK:
                # 失败提示作为 assistant 回复入历史，保持对话轮次完整，模型后续追问能知道发生了什么
                new_messages.append(_assistant_final(result.answer))
            data.history += new_messages
            data.todos = store.export_state(ctx)
            session.save()
            return result


def _assistant_final(text: str) -> dict[str, str]:
    """构造一条以 final_answer 形式记录的 assistant 消息。"""
    return {"role": "assistant", "content": json.dumps({"final_answer": text}, ensure_ascii=False)}
