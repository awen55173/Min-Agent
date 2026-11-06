"""原生工具调用（Function Calling）测试：阶段 1 导出工具 Schema、统一模型响应结构与真实 API 支持验证。"""
from __future__ import annotations

import json
import types
import pytest

from agent.config import ConfigError, Settings
from agent.llm import FakeLLM, LLMError, LLMResponse, OpenAIClient, ToolCallItem, normalize_llm_response
from agent.prompt import build_native_system_prompt, build_system_prompt
from agent.tools import build_registry


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
