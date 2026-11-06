"""原生工具调用（Function Calling）测试：阶段 1 导出工具 Schema、统一模型响应结构与真实 API 支持验证。"""
from __future__ import annotations

import json
import types
import pytest

from agent.config import ConfigError, Settings
from agent.llm import FakeLLM, LLMError, LLMResponse, OpenAIClient, ToolCallItem, normalize_llm_response
from agent.loop import MAX_STEPS, OK, PARSE_EXHAUSTED, TOOL_NOT_EXECUTED, Agent
from agent.prompt import build_native_system_prompt, build_system_prompt
from agent.tools import TodoStore, build_registry


def test_registry_exports_openai_tools_schema():
    """注册表生成的 OpenAI tools Schema 包含 type=function 与完整的 name/description/parameters。"""
    reg = build_registry()
    tools = reg.openai_tools()
    assert len(tools) == 4
    by_name = {t["function"]["name"]: t for t in tools}
    assert set(by_name) == {"calculator", "todo", "search", "weather"}
    for name, spec in by_name.items():
        assert spec["type"] == "function"
        fn = spec["function"]
        assert fn["name"] == name
        assert isinstance(fn["description"], str) and fn["description"]
        assert fn["parameters"]["type"] == "object"
        assert "properties" in fn["parameters"]


def test_openai_client_extract_response_unifies_text_and_tool_calls(monkeypatch: pytest.MonkeyPatch):
    """OpenAIClient.complete 将模型响应统一为文本回答 + 工具调用列表；支持仅含 tool_calls（content=None）的响应。"""
    settings = Settings(api_key="sk-test", model="test-model", base_url="https://example.com/v1")
    client = OpenAIClient(settings)

    fake_tc = types.SimpleNamespace(
        id="call_abc_1",
        type="function",
        function=types.SimpleNamespace(name="weather", arguments='{"city": "Beijing"}'),
    )
    fake_resp = types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=None, tool_calls=[fake_tc]))]
    )
    captured_kwargs: dict = {}

    def fake_create(**kwargs):
        """记录传入参数并返回模拟的原生 tool_calls 响应。"""
        captured_kwargs.update(kwargs)
        return fake_resp

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    reg = build_registry()
    out = client.complete(
        [{"role": "user", "content": "北京天气"}],
        tools=reg.openai_tools(),
        parallel_tool_calls=False,
    )
    assert isinstance(out, LLMResponse)
    assert out.content == ""
    assert out.has_tool_calls() is True
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0] == ToolCallItem(id="call_abc_1", name="weather", arguments='{"city": "Beijing"}')
    assert out.tool_calls[0].to_openai_dict() == {
        "id": "call_abc_1",
        "type": "function",
        "function": {"name": "weather", "arguments": '{"city": "Beijing"}'},
    }
    assert captured_kwargs["tool_choice"] == "auto"
    assert captured_kwargs["parallel_tool_calls"] is False
    assert len(captured_kwargs["tools"]) == 4


def test_openai_client_complete_rejects_empty_text_and_no_tool_calls(monkeypatch: pytest.MonkeyPatch):
    """当响应既无有效文本也无 tool_calls 时，complete 抛出 LLMError。"""
    settings = Settings(api_key="sk-test", model="test-model", base_url="https://example.com/v1")
    client = OpenAIClient(settings)
    empty_resp = types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="   ", tool_calls=None))]
    )
    monkeypatch.setattr(client._client.chat.completions, "create", lambda **_: empty_resp)
    with pytest.raises(LLMError, match="empty content and no tool_calls"):
        client.complete([{"role": "user", "content": "hi"}])


def test_fake_llm_and_prompt_modes():
    """FakeLLM.complete 支持返回结构化工具调用与文本回复；prompt 区分 native 与 text 模式。"""
    reg = build_registry()
    native_p = build_system_prompt(reg, tool_mode="native")
    text_p = build_system_prompt(reg, tool_mode="text")
    assert native_p == build_native_system_prompt(reg)
    assert '"action_input"' not in native_p
    assert '"action_input"' in text_p

    fake = FakeLLM([
        {"content": "先查天气", "tool_calls": [{"id": "c1", "name": "weather", "args": {"city": "Beijing"}}]},
        "北京今天晴，20°C",
    ])
    r1 = fake.complete([{"role": "user", "content": "查北京天气"}], tools=reg.openai_tools())
    assert r1.content == "先查天气"
    assert len(r1.tool_calls) == 1
    assert r1.tool_calls[0].id == "c1"
    assert json.loads(r1.tool_calls[0].arguments) == {"city": "Beijing"}

    r2 = fake.complete([{"role": "user", "content": "继续"}])
    assert r2.content == "北京今天晴，20°C"
    assert r2.has_tool_calls() is False
    assert normalize_llm_response(r2) is r2


@pytest.mark.live
def test_live_model_supports_native_tool_calling():
    """真实 API 验证：当前配置的模型与服务支持 OpenAI 原生 tools / tool_calls 返回。"""
    try:
        settings = Settings.from_env()
    except ConfigError as e:
        pytest.skip(f"LLM env not configured: {e}")
    client = OpenAIClient(settings)
    reg = build_registry()
    resp = client.complete(
        [
            {"role": "system", "content": build_native_system_prompt(reg)},
            {"role": "user", "content": "请调用 calculator 工具计算 123 * 456"},
        ],
        tools=reg.openai_tools(),
        parallel_tool_calls=False,
    )
    assert resp.has_tool_calls() is True
    assert resp.tool_calls[0].name == "calculator"
    args = json.loads(resp.tool_calls[0].arguments)
    assert "123" in str(args.get("expression", ""))


# ---------------------------------------------------------------------------
# 阶段 2：原生工具调用串行闭环（读取工具名/参数/调用 ID -> 执行 -> 协议回填）
# ---------------------------------------------------------------------------


def test_native_serial_loop_executes_tools_and_backfills_tool_role_messages():
    """原生串行闭环：依次执行 weather 与 todo，按 OpenAI role='tool' + tool_call_id 协议回填并得出最终回答。"""
    store = TodoStore()
    reg = build_registry(
        todo_store=store,
        http_get=lambda _url: {
            "current_condition": [
                {"temp_C": "22", "FeelsLikeC": "21", "humidity": "45", "weatherDesc": [{"value": "Sunny"}]}
            ]
        },
    )
    fake = FakeLLM([
        {
            "content": "先查北京天气",
            "tool_calls": [{"id": "call_w_1", "name": "weather", "args": {"city": "Beijing"}}],
        },
        {
            "content": "把查到的天气记为待办",
            "tool_calls": [{"id": "call_t_2", "name": "todo", "args": {"action": "add", "text": "北京晴 22°C"}}],
        },
        "北京今天晴（22°C），已为您添加待办 #1。",
    ])
    agent = Agent(fake, reg, tool_mode="native")
    res = agent.run("查北京天气并记成待办", user_id="u1", session_id="s1", turn_id=1)

    assert res.status == OK
    assert "22°C" in res.answer
    assert len(res.actions) == 2
    assert res.actions[0].tool == "weather"
    assert res.actions[0].api_call_id == "call_w_1"
    assert res.actions[0].call_id.startswith("call-1-weather-")
    assert res.actions[1].tool == "todo"
    assert res.actions[1].api_call_id == "call_t_2"
    assert res.actions[1].call_id.startswith("call-1-todo-")

    # 校验消息序列符合 OpenAI function calling 协议（assistant.tool_calls -> role=tool + tool_call_id）
    tool_msgs = [m for m in res.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 2
    assert tool_msgs[0]["tool_call_id"] == "call_w_1"
    assert tool_msgs[0]["name"] == "weather"
    assert '"temp_c": "22"' in tool_msgs[0]["content"]
    assert tool_msgs[1]["tool_call_id"] == "call_t_2"
    assert tool_msgs[1]["name"] == "todo"
    assert '"id": 1' in tool_msgs[1]["content"]


def test_native_serial_loop_parameter_validation_and_retry():
    """原生模式下 Schema 校验失败或 JSON 损坏时进入纠错重试，不执行工具；若未执行直接声称成功则返回 tool_not_executed。"""
    reg = build_registry()
    fake_recover = FakeLLM([
        {
            "content": "",
            "tool_calls": [{"id": "bad_1", "name": "todo", "arguments": '{"text": "缺必填 action"}'}],
        },
        {
            "content": "",
            "tool_calls": [{"id": "ok_1", "name": "todo", "arguments": '{"action": "add", "text": "买咖啡"}'}],
        },
        "已添加买咖啡待办。",
    ])
    res = Agent(fake_recover, reg, tool_mode="native").run("记买咖啡")
    assert res.status == OK
    assert len(res.actions) == 1
    assert res.actions[0].api_call_id == "ok_1"

    # 若非法调用后未重新执行工具就直接声称成功，应被拦截为 TOOL_NOT_EXECUTED
    fake_unexec = FakeLLM([
        {
            "content": "",
            "tool_calls": [{"id": "bad_2", "name": "unknown_tool", "arguments": "{}"}],
        },
        "我已经帮你执行好了！",
    ])
    res_unexec = Agent(fake_unexec, reg, tool_mode="native").run("执行未知工具")
    assert res_unexec.status == TOOL_NOT_EXECUTED


def test_native_serial_loop_rejects_multiple_calls_when_parallel_disabled_and_enforces_max_steps():
    """串行闭环下若模型一次返回多个工具调用，回填错误要求逐个调用；超出 max_steps 返回 max_steps。"""
    reg = build_registry()
    fake_multi = FakeLLM([
        {
            "content": "",
            "tool_calls": [
                {"id": "c1", "name": "calculator", "args": {"expression": "1+1"}},
                {"id": "c2", "name": "calculator", "args": {"expression": "2+2"}},
            ],
        },
        {
            "content": "",
            "tool_calls": [{"id": "c1_retry", "name": "calculator", "args": {"expression": "1+1"}}],
        },
        "答案是 2",
    ])
    res = Agent(fake_multi, reg, tool_mode="native", parallel_tool_calls=False).run("算一下")
    assert res.status == OK
    assert len(res.actions) == 1
    assert res.actions[0].api_call_id == "c1_retry"

    fake_steps = FakeLLM([
        {"content": "", "tool_calls": [{"id": "s1", "name": "calculator", "args": {"expression": "1+1"}}]},
        {"content": "", "tool_calls": [{"id": "s2", "name": "calculator", "args": {"expression": "2+2"}}]},
    ])
    res_max = Agent(fake_steps, reg, max_steps=1, tool_mode="native").run("算两步")
    assert res_max.status == MAX_STEPS
    assert len(res_max.actions) == 1

    fake_exhaust = FakeLLM(["   ", "   "], native=True)
    res_ex = Agent(fake_exhaust, reg, max_parse_retries=1, tool_mode="native").run("空回复")
    assert res_ex.status == PARSE_EXHAUSTED


def test_explicit_text_mode_compatibility_option():
    """显式传入 tool_mode='text' 时仍按旧版 JSON 协议运行。"""
    reg = build_registry()
    fake = FakeLLM(
        [
            '{"thought": "calc", "action": "calculator", "action_input": {"expression": "3 * 9"}}',
            '{"thought": "done", "final_answer": "结果是 27"}',
        ],
        native=True,  # 即便客户端支持 native，显式指定 tool_mode='text' 也强制走文本模式
    )
    res = Agent(fake, reg, tool_mode="text").run("3*9")
    assert res.status == OK
    assert res.answer == "结果是 27"
    assert res.actions[0].api_call_id == ""


@pytest.mark.live
def test_live_native_serial_closed_loop():
    """真实 API 验收：原生工具调用串行闭环（计算器 + 待办添加 -> 最终回复）。"""
    try:
        settings = Settings.from_env()
    except ConfigError as e:
        pytest.skip(f"LLM env not configured: {e}")
    client = OpenAIClient(settings)
    store = TodoStore()
    reg = build_registry(todo_store=store)
    agent = Agent(client, reg, tool_mode="native", parallel_tool_calls=False)
    res = agent.run("请先用 calculator 计算 15 * 16，然后把计算结果添加为一条待办（如：核对数据 240）")
    assert res.status == OK
    assert any(a.tool == "calculator" and a.ok and a.api_call_id for a in res.actions)
    assert any(a.tool == "todo" and a.ok and a.api_call_id for a in res.actions)
    assert any(m.get("role") == "tool" for m in res.messages)
