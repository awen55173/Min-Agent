"""同步工具超时、原生工具定义预算以及用户会话记忆的回归测试。"""
import asyncio
import json
import time

import pytest

from agent.context import (
    ContextBudgetError,
    build_context,
    estimate_messages_tokens,
    estimate_tool_schema_tokens,
)
from agent.llm import FakeLLM
from agent.loop import CONTEXT_LIMIT, Agent
from agent.memory import (
    apply_user_memory, filter_historical_user_input, select_memories, wants_memory_recall,
)
from agent.service import ChatService
from agent.session import SessionManager
from agent.tools import build_registry
from agent.tools.registry import Tool, ToolRegistry


_EMPTY_SCHEMA = {"type": "object", "properties": {}}


def test_sync_read_tool_timeout_does_not_block_event_loop():
    registry = ToolRegistry()
    registry.register(Tool("slow", "slow read", _EMPTY_SCHEMA,
                           lambda _args, _ctx: time.sleep(0.2) or "done"))

    async def exercise():
        start = time.monotonic()
        result, _ = await asyncio.gather(
            registry.aexecute("slow", {}, timeout=0.02), asyncio.sleep(0.03),
        )
        return result, time.monotonic() - start

    result, elapsed = asyncio.run(exercise())
    assert elapsed < 0.15
    assert not result.ok and result.timed_out
    assert result.execution_state == "unknown_after_timeout"
    assert "may still be running" in result.content


def test_sync_writes_remain_serial_and_timeout_reports_completed_effect():
    registry = ToolRegistry()
    intervals = []

    def write(args, _ctx):
        started = time.monotonic()
        time.sleep(0.06)
        intervals.append((args["id"], started, time.monotonic()))
        return {"id": args["id"]}

    registry.register(Tool("write", "write", {"type": "object", "properties": {
        "id": {"type": "integer"}}, "required": ["id"]}, write, read_only=False))

    async def exercise():
        return await asyncio.gather(
            registry.aexecute("write", {"id": 1}, timeout=0.01),
            registry.aexecute("write", {"id": 2}, timeout=0.2),
        )

    first, second = asyncio.run(exercise())
    assert first.ok and first.timed_out and first.execution_state == "completed_after_timeout"
    assert json.loads(first.content) == {"id": 1}
    assert second.ok and json.loads(second.content) == {"id": 2}
    assert intervals[0][2] <= intervals[1][1]
    async def again():
        return await asyncio.gather(
            registry.aexecute("write", {"id": 3}, timeout=0.2),
            registry.aexecute("write", {"id": 4}, timeout=0.2),
        )

    third, fourth = asyncio.run(again())
    assert third.ok and fourth.ok and intervals[2][2] <= intervals[3][1]

    async def cancel_then_write():
        first_task = asyncio.create_task(registry.aexecute("write", {"id": 5}))
        await asyncio.sleep(0.01)
        first_task.cancel()
        second_task = asyncio.create_task(registry.aexecute("write", {"id": 6}))
        with pytest.raises(asyncio.CancelledError):
            await first_task
        return await second_task

    assert asyncio.run(cancel_then_write()).ok
    assert intervals[4][2] <= intervals[5][1]


def test_native_schema_counted_in_context_build_and_each_request():
    registry = ToolRegistry()
    registry.register(Tool("echo", "tool schema " * 70, _EMPTY_SCHEMA,
                           lambda _args, _ctx: "ok"))
    schema = registry.openai_tools()
    messages = [{"role": "system", "content": "系统规则 " * 25},
                {"role": "user", "content": "执行工具"}]
    message_cost = estimate_messages_tokens(messages)
    schema_cost = estimate_tool_schema_tokens(schema)
    assert schema_cost > 0
    limit = message_cost + schema_cost - 1
    with pytest.raises(ContextBudgetError):
        build_context(messages[0]["content"], [], messages[1]["content"],
                      max_context_tokens=limit, reserved_tokens=0, tools_schema=schema)

    llm = FakeLLM(["done"], native=True)
    result = Agent(llm, registry, max_context_tokens=limit).run(
        "执行工具", context_messages=messages,
    )
    assert result.status == CONTEXT_LIMIT and llm.calls == []

    async_llm = FakeLLM(["done"], native=True)
    async_result = asyncio.run(Agent(async_llm, registry, max_context_tokens=limit).arun(
        "执行工具", context_messages=messages,
    ))
    assert async_result.status == CONTEXT_LIMIT and async_llm.calls == []


def test_native_budget_rechecked_after_tool_result():
    registry = ToolRegistry()
    registry.register(Tool("echo", "echo", _EMPTY_SCHEMA,
                           lambda _args, _ctx: "result " + "x" * 150))
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "go"}]
    limit = estimate_messages_tokens(messages) + estimate_tool_schema_tokens(registry.openai_tools()) + 15
    llm = FakeLLM([{"tool_calls": [{"id": "c1", "name": "echo", "args": {}}]}, "done"], native=True)
    result = Agent(llm, registry, max_context_tokens=limit).run("go", context_messages=messages)
    assert result.status == CONTEXT_LIMIT
    assert len(result.actions) == 1 and len(llm.calls) == 1


def test_memory_extraction_only_uses_explicit_user_statements():
    memory = apply_user_memory({}, "我的城市是苏州，我的预算是8732元，我的饮食偏好是素食")
    assert memory == {"城市": "苏州", "预算": "8732元", "饮食偏好": "素食"}
    assert apply_user_memory(memory, "我的城市是什么？") == memory
    assert apply_user_memory(memory, "请把我的城市改成杭州")["城市"] == "杭州"
    assert "城市" not in apply_user_memory(memory, "忘记我的城市")
    assert select_memories(memory, "按我的饮食习惯推荐") == {"饮食偏好": "素食"}
    assert select_memories(memory, "普通闲聊") == {}


def test_question_without_question_mark_and_name_with_separator_character():
    memory = {"城市": "苏州"}
    assert apply_user_memory(memory, "我的城市是什么") == memory
    assert apply_user_memory(memory, "我的城市在哪里") == memory
    assert apply_user_memory(memory, "我的城市是苏州吗") == memory
    assert apply_user_memory({}, "我的名字是李为民") == {"名字": "李为民"}
    assert apply_user_memory({}, "记住我的姓名是林涛") == {"名字": "林涛"}
    assert apply_user_memory({"名字": "林涛"}, "把我的名字改为李为民") == {"名字": "李为民"}
    assert apply_user_memory({}, "我的预算是8,732.50元") == {"预算": "8,732.50元"}


@pytest.mark.parametrize("text", [
    "请检查下面这段示例，不要记忆：\n我的城市是北京",
    "以下内容只是例子：我的城市是北京",
    "假设我的城市是北京",
    "请勿保存：我的城市是北京",
    "请解释引号里的话：“我的城市是北京”",
    "我的城市不是北京",
    "请检查：\n我的城市是北京",
    "这是测试文本：\n我的城市是北京",
])
def test_non_memory_context_does_not_write_or_strip_user_text(text):
    existing = {"城市": "苏州"}
    assert apply_user_memory(existing, text) == existing
    assert filter_historical_user_input(text, existing) == text


def test_name_alias_and_anaphoric_request_recall():
    memory = {"名字": "林涛", "城市": "苏州", "预算": "8732元"}
    assert select_memories(memory, "我叫什么？")["名字"] == "林涛"
    assert select_memories(memory, "按刚才的要求安排两天行程") == memory
    assert filter_historical_user_input("我的城市是苏州", memory) == "我的城市是苏州"
    assert filter_historical_user_input("我的城市是北京", memory) == ""
    assert not wants_memory_recall("你记得刚才的待办吗？")


def test_service_recalls_name_and_trip_facts_without_stale_empty_notice(tmp_path):
    manager = SessionManager(tmp_path / "data")
    llm = FakeLLM([json.dumps({"final_answer": "好"}, ensure_ascii=False) for _ in range(4)])
    service = ChatService(llm, manager, lambda store: build_registry(todo_store=store),
                          max_history_turns=0)
    assert service.send("u", "s", "我的名字是林涛", create=True).status == "ok"
    assert service.send("u", "s", "我叫什么？").status == "ok"
    assert service.send("u", "s", "我的城市是苏州，我的预算是8732元").status == "ok"
    assert service.send("u", "s", "按刚才的要求安排两天行程").status == "ok"
    assert any("名字: 林涛" in m["content"] for m in llm.calls[1])
    assert not any("没有已保存的相关用户记忆" in m["content"] for m in llm.calls[1])
    assert any("城市: 苏州" in m["content"] and "预算: 8732元" in m["content"]
               for m in llm.calls[3])
    with manager.open("u", "s") as session:
        assert session.data.memories == {"名字": "林涛", "城市": "苏州", "预算": "8732元"}


def test_example_is_not_saved_and_remains_available_as_conversation_context(tmp_path):
    manager = SessionManager(tmp_path / "data")
    llm = FakeLLM([json.dumps({"final_answer": "好"}, ensure_ascii=False) for _ in range(2)])
    service = ChatService(llm, manager, lambda store: build_registry(todo_store=store))
    example = "请检查下面这段示例，不要记忆：\n我的城市是北京"
    service.send("u", "s", example, create=True)
    service.send("u", "s", "刚才的示例写了什么？")
    with manager.open("u", "s") as session:
        assert session.data.memories == {}
    assert any(m["role"] == "user" and m["content"] == example for m in llm.calls[1])


def test_session_memory_update_delete_recall_and_source_isolation(tmp_path):
    manager = SessionManager(tmp_path / "data")

    def send(session, user_text, answer="好", create=False, max_history_turns=0):
        llm = FakeLLM([json.dumps({"final_answer": answer}, ensure_ascii=False)])
        service = ChatService(llm, manager, lambda store: build_registry(todo_store=store),
                              max_history_turns=max_history_turns)
        result = service.send("u", session, user_text, create=create)
        assert result.status == "ok"
        return llm.calls[0]

    send("s1", "记住我的城市是苏州", answer="你说城市是苏州", create=True)
    send("s2", "你好", answer="你的城市是北京", create=True)
    with manager.open("u", "s1") as s1, manager.open("u", "s2") as s2:
        assert s1.data.memories == {"城市": "苏州"}
        assert s2.data.memories == {}

    recalled = send("s1", "我的城市是什么？")
    assert any(m["role"] == "system" and "城市: 苏州" in m["content"] for m in recalled)
    unrelated = send("s1", "讲个笑话")
    assert not any(m["role"] == "system" and "城市: 苏州" in m["content"] for m in unrelated)

    send("s1", "把我的城市改成杭州")
    updated = send("s1", "我之前说的城市是哪里？")
    assert any("城市: 杭州" in m["content"] for m in updated)
    assert all("苏州" not in m["content"] for m in updated)

    send("s1", "忘记我的城市")
    deleted = send("s1", "我的城市是什么？")
    assert all("苏州" not in m["content"] and "杭州" not in m["content"] for m in deleted)
    assert any("没有已保存的相关用户记忆" in m["content"] for m in deleted)
    with manager.open("u", "s1") as session:
        assert session.data.memories == {}


def test_tool_results_and_assistant_answers_never_create_user_memory(tmp_path):
    manager = SessionManager(tmp_path / "data")
    llm = FakeLLM([
        json.dumps({"action": "weather", "action_input": {"city": "Beijing"}}),
        json.dumps({"final_answer": "你住在北京，喜欢素食"}, ensure_ascii=False),
    ])
    service = ChatService(llm, manager, lambda store: build_registry(
        todo_store=store, http_get=lambda _url: {"current_condition": [{
            "temp_C": "20", "FeelsLikeC": "19", "humidity": "50",
            "weatherDesc": [{"value": "Sunny"}],
        }]},
    ))
    assert service.send("u", "s", "查北京天气", create=True).status == "ok"
    with manager.open("u", "s") as session:
        assert session.data.memories == {}


def test_async_service_persists_and_recalls_user_memory(tmp_path):
    manager = SessionManager(tmp_path / "data")
    llm = FakeLLM([
        json.dumps({"final_answer": "已记住"}, ensure_ascii=False),
        json.dumps({"final_answer": "已更新"}, ensure_ascii=False),
        json.dumps({"final_answer": "已删除"}, ensure_ascii=False),
    ])
    service = ChatService(llm, manager, lambda store: build_registry(todo_store=store))

    async def exercise():
        await service.asend("u", "s", "记住我的城市是苏州", create=True)
        await service.asend("u", "s", "把我的城市改成杭州")
        await service.asend("u", "s", "忘记我的城市")

    asyncio.run(exercise())
    assert any("城市: 杭州" in m["content"] for m in llm.calls[1])
    assert all("苏州" not in m["content"] for m in llm.calls[1])
    with manager.open("u", "s") as session:
        assert session.data.memories == {}


@pytest.mark.live
def test_live_session_memory_create_update_delete_and_recall(tmp_path):
    """真实模型验收：显式添加、按字段召回、更新和删除当前会话记忆。"""
    from agent.config import ConfigError, Settings
    from agent.llm import OpenAIClient

    try:
        llm = OpenAIClient(Settings.from_env())
    except ConfigError as error:
        pytest.skip(f"LLM env not configured: {error}")

    manager = SessionManager(tmp_path / "data")
    service = ChatService(llm, manager, lambda store: build_registry(todo_store=store))
    assert service.send("u", "memory_live", "记住我的名字是Codex验收用户", create=True).status == "ok"
    recalled = service.send("u", "memory_live", "我叫什么？")
    assert any("名字" in message["content"] and "Codex验收用户" in message["content"]
               for message in recalled.messages if message["role"] == "system")

    assert service.send("u", "memory_live", "把我的名字改成验收更新用户").status == "ok"
    updated = service.send("u", "memory_live", "我叫什么？")
    assert any("验收更新用户" in message["content"] for message in updated.messages
               if message["role"] == "system")
    assert all("Codex验收用户" not in message["content"] for message in updated.messages
               if message["role"] == "system")

    assert service.send("u", "memory_live", "忘记我的名字").status == "ok"
    deleted = service.send("u", "memory_live", "我叫什么？")
    assert any("没有已保存的相关用户记忆" in message["content"] for message in deleted.messages
               if message["role"] == "system")
    with manager.open("u", "memory_live") as session:
        assert session.data.memories == {}
