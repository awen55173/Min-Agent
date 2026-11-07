"""原生工具调用（Function Calling）测试：阶段 1 导出工具 Schema、统一模型响应结构与真实 API 支持验证。"""
from __future__ import annotations

from pathlib import Path
import asyncio
import json
import time
import types
import pytest

from agent.config import ConfigError, Settings
from agent.context import build_context
from agent.llm import FakeLLM, LLMError, LLMResponse, OpenAIClient, ToolCallItem, normalize_llm_response
from agent.loop import INTERRUPTED, LLM_FAILED, MAX_STEPS, OK, OUTPUT_EXHAUSTED, TOOL_NOT_EXECUTED, Agent
from agent.prompt import build_system_prompt
from agent.service import ChatService
from agent.session import SessionCorruptError, SessionManager
from agent.tools import TodoStore, build_registry
from agent.tools.search import make_search_tool
from agent.tools.weather import make_weather_tool
from agent.trace import verify_trace_integrity
import cli


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


def test_openai_client_returns_empty_text_for_runtime_correction(monkeypatch: pytest.MonkeyPatch):
    """空原生响应交由运行时回答校验和重试处理。"""
    settings = Settings(api_key="sk-test", model="test-model", base_url="https://example.com/v1")
    client = OpenAIClient(settings)
    empty_resp = types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="   ", tool_calls=None))]
    )
    monkeypatch.setattr(client._client.chat.completions, "create", lambda **_: empty_resp)
    assert client.complete([{"role": "user", "content": "hi"}]).content == "   "


def test_fake_llm_and_native_prompt():
    """模拟客户端返回原生调用或直接文本，提示词不包含自定义 JSON 输出协议。"""
    reg = build_registry()
    native_p = build_system_prompt(reg)
    assert '"action_input"' not in native_p
    assert "Reply with exactly ONE JSON object" not in native_p

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
            {"role": "system", "content": build_system_prompt(reg)},
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
    agent = Agent(fake, reg)
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
    res = Agent(fake_recover, reg).run("记买咖啡")
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
    res_unexec = Agent(fake_unexec, reg, max_output_retries=1).run("执行未知工具")
    assert res_unexec.status == TOOL_NOT_EXECUTED


def test_native_serial_loop_schedules_multiple_calls_serially_and_enforces_max_steps():
    """关闭并行时仍接受原生多工具请求并串行执行，超出工具步数时返回失败。"""
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
    res = Agent(fake_multi, reg, parallel_tool_calls=False).run("算一下")
    assert res.status == OK
    assert [a.api_call_id for a in res.actions] == ["c1", "c2", "c1_retry"]

    fake_steps = FakeLLM([
        {"content": "", "tool_calls": [{"id": "s1", "name": "calculator", "args": {"expression": "1+1"}}]},
        {"content": "", "tool_calls": [{"id": "s2", "name": "calculator", "args": {"expression": "2+2"}}]},
    ])
    res_max = Agent(fake_steps, reg, max_steps=1).run("算两步")
    assert res_max.status == MAX_STEPS
    assert len(res_max.actions) == 1

    fake_exhaust = FakeLLM(["   ", "   "])
    res_ex = Agent(fake_exhaust, reg, max_output_retries=1).run("空回复")
    assert res_ex.status == OUTPUT_EXHAUSTED


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
    agent = Agent(client, reg, parallel_tool_calls=False)
    res = agent.run("请先用 calculator 计算 15 * 16，然后把计算结果添加为一条待办（如：核对数据 240）")
    assert res.status == OK
    assert any(a.tool == "calculator" and a.ok and a.api_call_id for a in res.actions)
    assert any(a.tool == "todo" and a.ok and a.api_call_id for a in res.actions)
    assert any(m.get("role") == "tool" for m in res.messages)


# ---------------------------------------------------------------------------
# 阶段 3：适配会话与上下文（持久化、历史回放、摘要压缩、api_call_id vs call_id、旧会话兼容）
# ---------------------------------------------------------------------------


def test_native_session_persistence_replay_and_call_id_distinction(tmp_path: Path, capsys):
    """原生工具调用会话完整持久化 tool_calls 与 role=tool 消息，区分 api_call_id 与内部幂等 call_id，并支持 CLI 回放。"""
    mgr = SessionManager(tmp_path / "sessions")
    fake = FakeLLM([
        {
            "content": "内部思考：先添加待办",
            "tool_calls": [{"id": "call_model_xyz", "name": "todo", "args": {"action": "add", "text": "买牛奶"}}],
        },
        "已为您添加待办：买牛奶（#1）。",
    ])
    svc = ChatService(fake, mgr, lambda s: build_registry(todo_store=s))
    res = svc.send("alice", "s_native", "帮我记买牛奶", create=True)
    assert res.status == OK
    assert res.actions[0].api_call_id == "call_model_xyz"
    assert res.actions[0].call_id.startswith("call-1-todo-")
    assert res.actions[0].api_call_id != res.actions[0].call_id

    with mgr.open("alice", "s_native") as s:
        hist = s.data.history
        assert len(hist) == 4  # user -> assistant(tool_calls) -> tool -> assistant(final)
        assert hist[1]["role"] == "assistant"
        assert hist[1]["tool_calls"][0]["id"] == "call_model_xyz"
        assert hist[2]["role"] == "tool"
        assert hist[2]["tool_call_id"] == "call_model_xyz"
        assert hist[3] == {"role": "assistant", "content": "已为您添加待办：买牛奶（#1）。"}
        rec = s.data.tool_calls[0]
        assert rec["api_call_id"] == "call_model_xyz"
        assert rec["call_id"] == res.actions[0].call_id

        # 验证 CLI 历史回放（含 verbose 工具步骤输出）
        cli._print_history(hist, verbose=True)
        captured = capsys.readouterr()
        assert "you> 帮我记买牛奶" in captured.out
        assert "已为您添加待办：买牛奶（#1）。" in captured.out
        assert "[step 1] todo({'action': 'add', 'text': '买牛奶'}) ok=True ->" in captured.err

    # 损坏的 role=tool（缺 tool_call_id）应被识别为 SessionCorruptError
    session_file = tmp_path / "sessions" / "alice" / "s_native.json"
    raw = json.loads(session_file.read_text(encoding="utf-8"))
    raw["history"][2] = {"role": "tool", "content": "{}"}
    session_file.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SessionCorruptError):
        with mgr.open("alice", "s_native"):
            pass


def test_native_context_compression_strips_thought_keeps_pairs_and_completes_todo(tmp_path: Path):
    """原生工具调用历史触发压缩后：剥离历史思考文本，保留最近轮次的 tool_calls/tool 配对，并能在压缩后完成早期创建的待办。"""
    mgr = SessionManager(tmp_path / "sessions")
    fake = FakeLLM([
        # Turn 1：查北京天气并添加待办 #1
        {
            "content": "隐藏思考过程_不应进入长期记忆_SECRET_THOUGHT_1",
            "tool_calls": [{"id": "tc_w1", "name": "weather", "args": {"city": "Beijing"}}],
        },
        {
            "content": "隐藏思考过程_SECRET_THOUGHT_2",
            "tool_calls": [{"id": "tc_t1", "name": "todo", "args": {"action": "add", "text": "带伞去北京出差"}}],
        },
        "北京小雨 18°C，已添加待办 #1：带伞去北京出差。",
        # Turn 2：纯对话（预算 5000 元）
        "好的，已记下您的出差预算为 5000 元。",
        # Turn 3：计算器
        {
            "content": "思考计算",
            "tool_calls": [{"id": "tc_c3", "name": "calculator", "args": {"expression": "5000 - 1200"}}],
        },
        "扣除机票后还剩 3800 元。",
        # Turn 4（此时 max_history_turns=2 触发压缩，Turn 1 被压缩进结构化摘要）：完成刚才那个待办
        {
            "content": "根据摘要中的待办 #1 标记完成",
            "tool_calls": [{"id": "tc_t4", "name": "todo", "args": {"action": "complete", "id": 1}}],
        },
        "已将待办 #1（带伞去北京出差）标记为完成！",
    ])
    svc = ChatService(
        fake,
        mgr,
        lambda s: build_registry(
            todo_store=s,
            http_get=lambda _u: {
                "current_condition": [
                    {"temp_C": "18", "FeelsLikeC": "17", "humidity": "80", "weatherDesc": [{"value": "Light rain"}]}
                ]
            },
        ),
        max_history_turns=2,
    )
    svc.send("u1", "s_comp", "查北京天气并记待办：带伞去北京出差", create=True)
    svc.send("u1", "s_comp", "这次出差预算 5000 元")
    svc.send("u1", "s_comp", "机票花了 1200，帮我算还剩多少")
    r4 = svc.send("u1", "s_comp", "把刚才那个出差待办完成掉")
    assert r4.status == OK

    with mgr.open("u1", "s_comp") as s:
        assert s.data.todos["items"][0]["done"] is True
        assert "SECRET_THOUGHT" not in s.data.summary
        assert "#1 [done/已完成] 带伞去北京出差" in s.data.summary
        assert "天气 Beijing: 18°C" in s.data.summary

    # 检查第 4 轮发给模型的上下文：最近轮次中的 assistant.tool_calls 思考文本已被清空，且与 role=tool 成对
    sent_msgs = fake.calls[-1]
    for idx, m in enumerate(sent_msgs):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            assert "SECRET_THOUGHT" not in str(m.get("content", ""))
            tc_id = m["tool_calls"][0]["id"]
            assert sent_msgs[idx + 1]["role"] == "tool"
            assert sent_msgs[idx + 1]["tool_call_id"] == tc_id


# ---------------------------------------------------------------------------
# 阶段 4：改造异步执行链（异步 LLM、搜索/天气、Agent.arun、超时/取消/重试、非阻塞会话锁与 asend）
# ---------------------------------------------------------------------------


def test_async_llm_timeout_retry_and_cancellation(monkeypatch: pytest.MonkeyPatch):
    """验证 OpenAIClient.acomplete 的异步重试、超时转换（LLMError）以及 CancelledError 直接向上传播。"""

    async def _run():
        """执行异步 LLM 重试、超时与取消测试。"""
        settings = Settings(
            api_key="sk-test",
            model="test-model",
            base_url="https://example.com/v1",
            timeout=0.05,
            max_retries=1,
        )
        client = OpenAIClient(settings, backoff=0.01)
        calls = 0

        async def flaky_create(**kwargs):
            """第一次抛超时错误，第二次返回正常结果。"""
            nonlocal calls
            calls += 1
            if calls == 1:
                raise asyncio.TimeoutError("temporary timeout")
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="重试成功", tool_calls=None))]
            )

        monkeypatch.setattr(client._async_client.chat.completions, "create", flaky_create)
        resp = await client.acomplete([{"role": "user", "content": "hi"}])
        assert resp.content == "重试成功"
        assert calls == 2

        # 超时应重试后转为 LLMError
        async def slow_create(**kwargs):
            """模拟超时请求。"""
            await asyncio.sleep(0.3)

        monkeypatch.setattr(client._async_client.chat.completions, "create", slow_create)
        with pytest.raises(LLMError, match="LLM unavailable after 2 attempts"):
            await client.acomplete([{"role": "user", "content": "hi"}])

        # CancelledError 必须直接传播，不得被吞掉或重试
        cancel_calls = 0

        async def cancel_create(**kwargs):
            """模拟被取消的请求。"""
            nonlocal cancel_calls
            cancel_calls += 1
            raise asyncio.CancelledError()

        monkeypatch.setattr(client._async_client.chat.completions, "create", cancel_create)
        with pytest.raises(asyncio.CancelledError):
            await client.acomplete([{"role": "user", "content": "hi"}])
        assert cancel_calls == 1

    asyncio.run(_run())


def test_async_search_weather_and_agent_arun_with_tool_timeout():
    """验证 search/weather 异步执行链、Agent.arun 主循环及工具超时（tool_timeout）处理。"""

    async def _run():
        """执行异步工具调用与超时测试。"""
        async def fake_apost(url: str, payload: dict, api_key: str) -> dict:
            """模拟异步 Tavily 搜索响应。"""
            await asyncio.sleep(0.01)
            return {
                "answer": "Python 3.12 released",
                "results": [{"title": "Py", "url": "https://python.org", "content": "Details"}],
            }

        async def fake_aget(url: str) -> dict:
            """模拟异步天气响应（SlowCity 故意慢以触发超时）。"""
            if "SlowCity" in url:
                await asyncio.sleep(0.25)
            else:
                await asyncio.sleep(0.01)
            return {
                "current_condition": [
                    {"temp_C": "22", "FeelsLikeC": "21", "humidity": "50", "weatherDesc": [{"value": "Clear"}]}
                ]
            }

        reg = build_registry(tavily_api_key="tvly-test", http_post=fake_apost, http_get=fake_aget)

        fake_llm = FakeLLM([
            {"content": "", "tool_calls": [{"id": "tc_s1", "name": "search", "args": {"query": "python"}}]},
            {"content": "", "tool_calls": [{"id": "tc_w1", "name": "weather", "args": {"city": "Beijing"}}]},
            {"content": "", "tool_calls": [{"id": "tc_w2", "name": "weather", "args": {"city": "SlowCity"}}]},
            "搜索与北京天气成功，SlowCity 查询超时。",
        ])
        agent = Agent(fake_llm, reg, tool_timeout=0.06)
        res = await agent.arun("测试异步工具链")
        assert res.status == OK
        assert len(res.actions) == 3
        assert res.actions[0].tool == "search" and res.actions[0].ok is True
        assert res.actions[1].tool == "weather" and res.actions[1].ok is True
        assert res.actions[2].tool == "weather" and res.actions[2].ok is False
        assert "timed out" in str(res.actions[2].result)

    asyncio.run(_run())


def test_async_session_lock_and_asend_do_not_block_other_sessions(tmp_path: Path):
    """验证异步会话锁与 ChatService.asend：慢会话持锁或等待锁时，不会阻塞其他会话并发完成。"""

    async def _run():
        """并发运行慢会话 s_slow 与快会话 s_fast，并测试异步等待同一会话锁时不阻塞事件循环。"""
        mgr = SessionManager(tmp_path / "sessions", lock_timeout=2.0)

        async def slow_aget(url: str) -> dict:
            """对 SlowCity 延迟 0.2s，其余立即返回。"""
            if "SlowCity" in url:
                await asyncio.sleep(0.20)
            return {
                "current_condition": [
                    {"temp_C": "20", "FeelsLikeC": "20", "humidity": "40", "weatherDesc": [{"value": "Sunny"}]}
                ]
            }

        def make_reg(store: TodoStore):
            """构建包含异步 weather 工具的注册表。"""
            return build_registry(todo_store=store, http_get=slow_aget)

        # s_slow 调用耗时 0.2s 的工具，s_fast 直接完成待办添加
        llm_slow = FakeLLM([
            {"content": "", "tool_calls": [{"id": "c_slow", "name": "weather", "args": {"city": "SlowCity"}}]},
            "慢会话完成",
            "慢会话第二次完成",
        ])
        llm_fast = FakeLLM([
            {"content": "", "tool_calls": [{"id": "c_fast", "name": "todo", "args": {"action": "add", "text": "快会话待办"}}]},
            "快会话完成",
            "快会话第二次完成",
        ])
        svc_slow = ChatService(llm_slow, mgr, make_reg)
        svc_fast = ChatService(llm_fast, mgr, make_reg)

        t0 = time.monotonic()
        task_slow = asyncio.create_task(svc_slow.asend("u1", "s_slow", "查慢城市", create=True))
        await asyncio.sleep(0.02)  # 确保 s_slow 已持锁进入工具调用
        res_fast = await svc_fast.asend("u1", "s_fast", "记快待办", create=True)
        fast_elapsed = time.monotonic() - t0
        res_slow = await task_slow

        assert res_fast.status == OK
        assert res_slow.status == OK
        assert fast_elapsed < 0.15  # s_fast 未被 0.20s 的 s_slow 阻塞

        # 进一步验证：当同一会话 s_slow 被锁住 0.12s 时，等待 s_slow 的协程通过 asyncio.sleep 轮询，不阻塞无关协程
        async def hold_session_briefly():
            """持锁 0.12s 后释放。"""
            async with mgr.aopen("u1", "s_slow"):
                await asyncio.sleep(0.12)

        holder = asyncio.create_task(hold_session_briefly())
        await asyncio.sleep(0.02)

        waiter = asyncio.create_task(svc_slow.asend("u1", "s_slow", "第二次请求"))
        t1 = time.monotonic()
        res_other = await svc_fast.asend("u1", "s_fast", "再记一条")
        other_elapsed = time.monotonic() - t1
        assert res_other.status == OK
        assert other_elapsed < 0.08  # 即使 waiter 正在等待 s_slow 的锁，s_fast 仍立即完成
        await holder
        res_waiter = await waiter
        assert res_waiter.status == OK

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 阶段 5：受控并行调度（只读并发、待办写与相关读写串行、并发上限、单项超时不连坐）
# ---------------------------------------------------------------------------


def test_parallel_weather_queries_execute_concurrently():
    """核心场景：“同时查询北京、上海天气”，验证独立只读调用确实并发执行。"""

    async def _run():
        """验证两个各耗时 0.14s 的天气查询在一次批量调用中并发完成（总耗时显著小于串行 0.28s）。"""
        active = 0
        peak_active = 0

        async def concurrent_aget(url: str) -> dict:
            """模拟耗时 0.14s 的异步天气查询并统计并发峰值。"""
            nonlocal active, peak_active
            active += 1
            peak_active = max(peak_active, active)
            try:
                await asyncio.sleep(0.14)
                temp = "25" if "Beijing" in url else "28"
                return {
                    "current_condition": [
                        {"temp_C": temp, "FeelsLikeC": temp, "humidity": "50", "weatherDesc": [{"value": "Sunny"}]}
                    ]
                }
            finally:
                active -= 1

        reg = build_registry(http_get=concurrent_aget)
        fake_llm = FakeLLM([
            {
                "content": "",
                "tool_calls": [
                    {"id": "call_bj", "name": "weather", "args": {"city": "Beijing"}},
                    {"id": "call_sh", "name": "weather", "args": {"city": "Shanghai"}},
                ],
            },
            "北京 25°C，上海 28°C。",
        ])
        agent = Agent(fake_llm, reg, parallel_tool_calls=True)
        t0 = time.monotonic()
        res = await agent.arun("同时查询北京、上海天气")
        elapsed = time.monotonic() - t0

        assert res.status == OK
        assert peak_active == 2
        assert elapsed < 0.25  # 串行至少 0.28s，并发约 0.14s
        assert len(res.actions) == 2
        assert res.actions[0].api_call_id == "call_bj" and '"temp_c": "25"' in res.actions[0].result
        assert res.actions[1].api_call_id == "call_sh" and '"temp_c": "28"' in res.actions[1].result

    asyncio.run(_run())


def test_dependent_weather_then_todo_and_same_session_todo_writes_stay_serial(tmp_path: Path):
    """核心场景：“查北京天气，再把结果记成待办”保持串行；同一批次内的待办写操作及相关读写也严格串行。"""

    async def _run():
        """验证跨步依赖（先查天气后写待办）与同批混合调用（并发读天气 + 串行写/读待办）均严格按序执行。"""
        mgr = SessionManager(tmp_path / "sessions")
        events_order: list[str] = []

        async def tracked_aget(url: str) -> dict:
            """记录天气请求顺序并返回不同城市的温度。"""
            city = "Beijing" if "Beijing" in url else "Shanghai"
            if city == "Beijing":
                await asyncio.sleep(0.06)
            else:
                await asyncio.sleep(0.01)
            events_order.append(f"weather:{city}")
            return {
                "current_condition": [
                    {"temp_C": "21", "FeelsLikeC": "20", "humidity": "45", "weatherDesc": [{"value": "Cloudy"}]}
                ]
            }

        # 1) 查北京天气 -> 根据结果记成待办（两步串行依赖）
        llm_dep = FakeLLM([
            {"content": "", "tool_calls": [{"id": "w_bj", "name": "weather", "args": {"city": "Beijing"}}]},
            {"content": "", "tool_calls": [{"id": "t_add", "name": "todo", "args": {"action": "add", "text": "北京多云 21°C 记得带外套"}}]},
            "已查询北京天气（21°C 多云）并记入待办 #1。",
        ])
        svc_dep = ChatService(llm_dep, mgr, lambda s: build_registry(todo_store=s, http_get=tracked_aget))
        r_dep = await svc_dep.asend("u1", "s_dep", "查北京天气，再把结果记成待办", create=True)
        assert r_dep.status == OK
        assert [a.tool for a in r_dep.actions] == ["weather", "todo"]
        with mgr.open("u1", "s_dep") as s:
            assert s.data.todos["items"][0]["text"] == "北京多云 21°C 记得带外套"

        # 2) 同一批次内包含 2 个只读 weather + 3 个相关 todo 操作（add #1 -> add #2 -> complete #1 -> list）
        llm_mix = FakeLLM([
            {
                "content": "",
                "tool_calls": [
                    {"id": "m_w1", "name": "weather", "args": {"city": "Beijing"}},
                    {"id": "m_w2", "name": "weather", "args": {"city": "Shanghai"}},
                    {"id": "m_t1", "name": "todo", "args": {"action": "add", "text": "待办A"}},
                    {"id": "m_t2", "name": "todo", "args": {"action": "add", "text": "待办B"}},
                    {"id": "m_t3", "name": "todo", "args": {"action": "complete", "id": 1}},
                    {"id": "m_t4", "name": "todo", "args": {"action": "list", "status": "all"}},
                ],
            },
            "全部完成。",
        ])
        svc_mix = ChatService(llm_mix, mgr, lambda s: build_registry(todo_store=s, http_get=tracked_aget))
        r_mix = await svc_mix.asend("u1", "s_mix", "查两城天气并依次操作待办", create=True)
        assert r_mix.status == OK
        assert len(r_mix.actions) == 6
        # 验证最后一步 list 看到了前两步 add 和第三步 complete 的串行结果
        list_res = json.loads(r_mix.actions[5].result)
        assert len(list_res["items"]) == 2
        assert list_res["items"][0] == {"id": 1, "text": "待办A", "done": True}
        assert list_res["items"][1] == {"id": 2, "text": "待办B", "done": False}

    asyncio.run(_run())


def test_parallel_concurrency_limit_and_partial_timeout_preserves_other_results():
    """核心场景：设置并发上限（max_concurrency=2），且一项查询超时时其他成功结果完整保留、每个调用都有明确结果。"""

    async def _run():
        """验证 4 个并发只读调用受 max_concurrency=2 限制，且 SlowCity 超时不影响其余 3 个城市成功返回。"""
        active = 0
        max_active = 0

        async def bounded_aget(url: str) -> dict:
            """统计并发度；SlowCity 延迟 0.25s 触发超时，其余城市 0.03s 正常返回。"""
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            try:
                if "SlowCity" in url:
                    await asyncio.sleep(0.25)
                else:
                    await asyncio.sleep(0.03)
                return {
                    "current_condition": [
                        {"temp_C": "26", "FeelsLikeC": "26", "humidity": "60", "weatherDesc": [{"value": "Fine"}]}
                    ]
                }
            finally:
                active -= 1

        reg = build_registry(http_get=bounded_aget)
        fake_llm = FakeLLM([
            {
                "content": "",
                "tool_calls": [
                    {"id": "c_bj", "name": "weather", "args": {"city": "Beijing"}},
                    {"id": "c_slow", "name": "weather", "args": {"city": "SlowCity"}},
                    {"id": "c_gz", "name": "weather", "args": {"city": "Guangzhou"}},
                    {"id": "c_sz", "name": "weather", "args": {"city": "Shenzhen"}},
                ],
            },
            "北京、广州、深圳均为 26°C，SlowCity 查询超时。",
        ])
        agent = Agent(
            fake_llm,
            reg,

            parallel_tool_calls=True,
            tool_timeout=0.07,
            max_concurrency=2,
        )
        res = await agent.arun("查四个城市天气")
        assert res.status == OK
        assert max_active == 2  # 严格遵守并发上限 2
        assert len(res.actions) == 4

        by_id = {a.api_call_id: a for a in res.actions}
        assert by_id["c_bj"].ok is True and '"city": "Beijing"' in by_id["c_bj"].result
        assert by_id["c_gz"].ok is True and '"city": "Guangzhou"' in by_id["c_gz"].result
        assert by_id["c_sz"].ok is True and '"city": "Shenzhen"' in by_id["c_sz"].result
        assert by_id["c_slow"].ok is False and "timed out" in by_id["c_slow"].result

        # 验证回填的 role=tool 消息与每个 tool_call_id 一一对应
        tool_msgs = {m["tool_call_id"]: m for m in res.messages if m.get("role") == "tool"}
        assert set(tool_msgs) == {"c_bj", "c_slow", "c_gz", "c_sz"}
        assert '"city": "Beijing"' in tool_msgs["c_bj"]["content"]
        assert "Error:" in tool_msgs["c_slow"]["content"] and "timed out" in tool_msgs["c_slow"]["content"]

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 阶段 6：完善恢复、Trace 与测试（乱序完成精准映射、逐个计入预算、写入后模型失败恢复、部分取消与重启恢复、真实 API 验收）
# ---------------------------------------------------------------------------


def test_out_of_order_parallel_completion_maps_accurately_and_counts_per_call_budget():
    """并行完成顺序不同（上海最快、广州次之、北京最慢），结果仍准确对应各自调用；批量调用逐个计入 max_steps 预算。"""

    async def _run():
        """验证乱序完成的并行调用结果映射准确、Trace 校验通过，且超预算批次按剩余步数裁剪并触发 max_steps。"""
        delays = {"Beijing": 0.08, "Shanghai": 0.01, "Guangzhou": 0.04}
        temps = {"Beijing": "19", "Shanghai": "24", "Guangzhou": "30"}

        async def out_of_order_aget(url: str) -> dict:
            """按城市设定不同延迟使完成顺序与请求顺序相反。"""
            for city, delay in delays.items():
                if city in url:
                    await asyncio.sleep(delay)
                    t = temps[city]
                    return {
                        "current_condition": [
                            {"temp_C": t, "FeelsLikeC": t, "humidity": "55", "weatherDesc": [{"value": "Clear"}]}
                        ]
                    }
            raise AssertionError(f"unexpected url: {url}")

        reg = build_registry(http_get=out_of_order_aget)
        fake_llm = FakeLLM([
            {
                "content": "",
                "tool_calls": [
                    {"id": "api_bj", "name": "weather", "args": {"city": "Beijing"}},
                    {"id": "api_sh", "name": "weather", "args": {"city": "Shanghai"}},
                    {"id": "api_gz", "name": "weather", "args": {"city": "Guangzhou"}},
                ],
            },
            "三城天气查询完毕。",
        ])
        res = await Agent(fake_llm, reg).arun("查北上广天气")
        assert res.status == OK

        # Trace 中 tool_result 按实际完成顺序记录（step 2 上海 -> step 3 广州 -> step 1 北京）
        result_steps = [ev["step"] for ev in res.trace_events if ev.get("event") == "tool_result"]
        assert result_steps == [2, 3, 1]

        # 但 actions 与回填给模型的 role=tool 消息严格保持请求顺序且结果与各城市精确对应
        assert [a.api_call_id for a in res.actions] == ["api_bj", "api_sh", "api_gz"]
        assert '"temp_c": "19"' in res.actions[0].result
        assert '"temp_c": "24"' in res.actions[1].result
        assert '"temp_c": "30"' in res.actions[2].result

        tool_msgs = [m for m in res.messages if m.get("role") == "tool"]
        assert [m["tool_call_id"] for m in tool_msgs] == ["api_bj", "api_sh", "api_gz"]
        assert '"city": "Beijing"' in tool_msgs[0]["content"]
        assert '"city": "Shanghai"' in tool_msgs[1]["content"]
        assert '"city": "Guangzhou"' in tool_msgs[2]["content"]

        # Trace 完整性校验依然通过
        report = verify_trace_integrity(res.trace_events)
        assert report["valid"] is True

        # 验证批量调用逐个计入工具预算：max_steps=2 时请求 3 个工具，仅执行前 2 个并返回 max_steps
        fake_budget = FakeLLM([
            {
                "content": "",
                "tool_calls": [
                    {"id": "b1", "name": "weather", "args": {"city": "Beijing"}},
                    {"id": "b2", "name": "weather", "args": {"city": "Shanghai"}},
                    {"id": "b3", "name": "weather", "args": {"city": "Guangzhou"}},
                ],
            },
        ])
        res_budget = await Agent(fake_budget, reg, max_steps=2).arun("超预算批量查询")
        assert res_budget.status == MAX_STEPS
        assert len(res_budget.actions) == 2
        assert [a.api_call_id for a in res_budget.actions] == ["b1", "b2"]
        assert verify_trace_integrity(res_budget.trace_events)["valid"] is True

    asyncio.run(_run())


def test_write_succeeds_then_llm_fails_recovery_reuses_call_id_and_trace_is_valid(tmp_path: Path):
    """核心场景：写入成功后模型失败，恢复重试时不重复写入，且 Trace 跨 attempt 关联完整有效。"""

    async def _run():
        """第 1 次尝试写待办成功但随后 LLM 报错；重启恢复后复用 turn_id 与 call_id，不重复创建待办。"""
        session_root = tmp_path / "sessions"
        mgr1 = SessionManager(session_root)
        llm_fail = FakeLLM([
            {
                "content": "",
                "tool_calls": [{"id": "call_add_1", "name": "todo", "args": {"action": "add", "text": "提交季度预算"}}],
            },
            LLMError("upstream gateway timeout"),
        ])
        svc1 = ChatService(llm_fail, mgr1, lambda s: build_registry(todo_store=s))
        r1 = await svc1.asend("u1", "s_rec", "帮我记待办：提交季度预算", create=True)
        assert r1.status == LLM_FAILED
        assert r1.turn_id == 1
        assert r1.attempt_id == 1
        assert len(r1.actions) == 1 and r1.actions[0].ok is True

        # 写操作已即时落盘
        with mgr1.open("u1", "s_rec") as s:
            assert len(s.data.todos["items"]) == 1
            assert s.data.todos["items"][0]["text"] == "提交季度预算"
            assert s.data.unfinished_turn is not None
            assert s.data.unfinished_turn["turn_id"] == 1

        # 模拟进程重启后使用新管理器恢复并发送“继续”
        mgr2 = SessionManager(session_root)
        llm_recover = FakeLLM([
            {
                "content": "",
                # 模型在新一次尝试中生成了全新的 api_call_id，但内部幂等 call_id 保持稳定
                "tool_calls": [{"id": "call_add_retry_new_id", "name": "todo", "args": {"action": "add", "text": "提交季度预算"}}],
            },
            "已为您恢复并确认待办 #1（提交季度预算）创建成功。",
        ])
        svc2 = ChatService(llm_recover, mgr2, lambda s: build_registry(todo_store=s))
        r2 = await svc2.asend("u1", "s_rec", "继续")
        assert r2.status == OK
        assert r2.turn_id == 1
        assert r2.attempt_id == 2
        assert len(r2.actions) == 1
        assert r2.actions[0].reused is True
        assert r2.actions[0].api_call_id == "call_add_retry_new_id"
        assert r2.actions[0].call_id == r1.actions[0].call_id

        with mgr2.open("u1", "s_rec") as s:
            assert len(s.data.todos["items"]) == 1  # 绝不重复创建
            assert s.data.unfinished_turn is None

        report = svc2.trace_logger.verify_integrity("u1", "s_rec")
        assert report["valid"] is True
        assert report["unfinished_calls"] == []

    asyncio.run(_run())


def test_mid_batch_cancellation_and_restart_recovery(tmp_path: Path):
    """批量调用中部分完成、部分被取消（CancelledError）：已完成写操作落盘保留，未完成调用记为 interrupted，重启恢复后可继续完成。"""

    async def _run():
        """批次含 [todo.add, weather(SlowCity)]，在 weather 执行期间取消任务，随后重启恢复。"""
        session_root = tmp_path / "sessions"
        mgr1 = SessionManager(session_root)
        weather_started = asyncio.Event()

        async def cancellable_aget(url: str) -> dict:
            """首次调用时通知已进入并挂起等待取消。"""
            weather_started.set()
            await asyncio.sleep(5.0)
            return {}

        llm1 = FakeLLM([
            {
                "content": "",
                "tool_calls": [
                    {"id": "b_todo", "name": "todo", "args": {"action": "add", "text": "买高铁票"}},
                    {"id": "b_wea", "name": "weather", "args": {"city": "Beijing"}},
                ],
            },
            "不会到达这里",
        ])
        svc1 = ChatService(llm1, mgr1, lambda s: build_registry(todo_store=s, http_get=cancellable_aget))
        task = asyncio.create_task(svc1.asend("u1", "s_cancel", "记买高铁票并查北京天气", create=True))
        await asyncio.wait_for(weather_started.wait(), timeout=2.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # 验证取消后：todo.add 已持久化，weather 被记录为 interrupted，Trace 完整且能识别未完成调用
        with mgr1.open("u1", "s_cancel") as s:
            assert len(s.data.todos["items"]) == 1
            assert s.data.todos["items"][0]["text"] == "买高铁票"
            assert s.data.unfinished_turn is not None
            assert s.data.unfinished_turn["status"] == INTERRUPTED
            tool_msgs = [m for m in s.data.history if m.get("role") == "tool"]
            assert len(tool_msgs) == 2
            assert tool_msgs[0]["tool_call_id"] == "b_todo" and '"id": 1' in tool_msgs[0]["content"]
            assert tool_msgs[1]["tool_call_id"] == "b_wea" and "status=interrupted" in tool_msgs[1]["content"]

        unfinished = svc1.trace_logger.find_unfinished_calls("u1", "s_cancel")
        assert len(unfinished) == 1
        assert unfinished[0]["tool"] == "weather"
        assert svc1.trace_logger.verify_integrity("u1", "s_cancel")["valid"] is True

        # 重启恢复：todo.add 走幂等复用（不重复写入），weather 重新执行成功
        async def fast_aget(url: str) -> dict:
            """恢复后正常返回北京天气。"""
            return {
                "current_condition": [
                    {"temp_C": "23", "FeelsLikeC": "22", "humidity": "40", "weatherDesc": [{"value": "Sunny"}]}
                ]
            }

        mgr2 = SessionManager(session_root)
        llm2 = FakeLLM([
            {
                "content": "",
                "tool_calls": [
                    {"id": "b_todo_2", "name": "todo", "args": {"action": "add", "text": "买高铁票"}},
                    {"id": "b_wea_2", "name": "weather", "args": {"city": "Beijing"}},
                ],
            },
            "已恢复：待办 #1（买高铁票）已存在，北京今天晴 23°C。",
        ])
        svc2 = ChatService(llm2, mgr2, lambda s: build_registry(todo_store=s, http_get=fast_aget))
        r2 = await svc2.asend("u1", "s_cancel", "继续")
        assert r2.status == OK
        assert r2.attempt_id == 2
        assert r2.actions[0].tool == "todo" and r2.actions[0].reused is True
        assert r2.actions[1].tool == "weather" and r2.actions[1].ok is True

        with mgr2.open("u1", "s_cancel") as s:
            assert len(s.data.todos["items"]) == 1  # 仍只有 1 条待办
            assert s.data.unfinished_turn is None

        report2 = svc2.trace_logger.verify_integrity("u1", "s_cancel")
        assert report2["valid"] is True
        assert report2["unfinished_calls"] == []

    asyncio.run(_run())


@pytest.mark.live
def test_live_parallel_weather_and_serial_dependent_todo(tmp_path: Path):
    """真实 API 核心流程验收：1) 同时查询北京、上海天气；2) 查北京天气再把结果记成待办。"""

    async def _run():
        """使用真实模型与 ChatService.asend 验证并发只读与串行依赖闭环（外部 wttr.in 抖动时注入稳定天气源以验证真实模型调度）。"""
        try:
            settings = Settings.from_env()
        except ConfigError as e:
            pytest.skip(f"LLM env not configured: {e}")

        async def reliable_weather_aget(url: str) -> dict:
            """为真实 LLM 调用提供稳定的异步天气响应并保留 0.1s I/O 延迟以检验并发。"""
            await asyncio.sleep(0.10)
            temp = "21" if "Beijing" in url else "26"
            return {
                "current_condition": [
                    {"temp_C": temp, "FeelsLikeC": temp, "humidity": "50", "weatherDesc": [{"value": "Sunny"}]}
                ]
            }

        client = OpenAIClient(settings)
        mgr = SessionManager(tmp_path / "sessions")
        svc = ChatService(client, mgr, lambda s: build_registry(todo_store=s, http_get=reliable_weather_aget))

        # 场景 1：同时查询北京、上海天气
        r1 = await svc.asend("live_u", "live_parallel", "请同时查询北京和上海现在的天气", create=True)
        assert r1.status == OK
        weather_calls = [a for a in r1.actions if a.tool == "weather" and a.ok]
        assert len(weather_calls) >= 2
        assert "21" in r1.answer and "26" in r1.answer

        # 场景 2：查北京天气，再把结果记成待办
        r2 = await svc.asend(
            "live_u",
            "live_dep",
            "请先查询北京的天气，然后再把北京的天气结果添加为一条待办事项",
            create=True,
        )
        assert r2.status == OK
        tools_used = [a.tool for a in r2.actions if a.ok]
        assert "weather" in tools_used and "todo" in tools_used
        with mgr.open("live_u", "live_dep") as s:
            assert len(s.data.todos["items"]) >= 1
        assert svc.trace_logger.verify_integrity("live_u", "live_dep")["valid"] is True

    asyncio.run(_run())


