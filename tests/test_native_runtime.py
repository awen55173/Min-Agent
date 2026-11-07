"""仅原生运行时的阶段边界、纠错预算及文本透明性回归。"""
import asyncio
import json

import pytest

import cli
from agent.context import sanitize_turn, summarize_turns
from agent.llm import FakeLLM, LLMResponse, ToolCallItem
from agent.loop import Agent, OK, OUTPUT_EXHAUSTED, TOOL_NOT_EXECUTED
from agent.prompt import build_system_prompt
from agent.service import ChatService
from agent.session import SessionManager
from agent.tool_call import ToolCallError, parse_tool_call_args
from agent.tools import build_registry
from agent.tools.registry import Tool, ToolRegistry
from agent.trace import verify_trace_integrity


@pytest.mark.parametrize("answer", [
    "正常中文回答，不需要 JSON",
    '{"thought":"这是用户需要的示例","action":"todo","action_input":{"action":"add","text":"不要执行"},"final_answer":"示例"}',
    '```json\n{"final_answer":"这只是文本"}\n```',
])
@pytest.mark.parametrize("use_async", [False, True])
def test_answer_text_is_never_decoded_or_executed(tmp_path, capsys, answer, use_async):
    """正文直接显示和保存，即使长得像旧协议也不会触发工具。"""
    llm = FakeLLM([answer])
    manager = SessionManager(tmp_path)
    service = ChatService(llm, manager, lambda store: build_registry(todo_store=store))
    result = (asyncio.run(service.asend("u", "s", "展示示例", create=True)) if use_async
              else service.send("u", "s", "展示示例", create=True))
    assert result.status == OK and result.answer == answer and result.actions == []
    assert len(llm.calls) == 1 and llm.tools_passed[0]
    assert "Reply with exactly ONE JSON object" not in build_system_prompt(build_registry())
    with manager.open("u", "s") as session:
        history = session.data.history
        assert history[-1]["content"] == answer
        assert session.data.todos["items"] == []
    assert sanitize_turn(history)[-1]["content"] == answer
    assert summarize_turns([history]).key_facts
    cli._print_history(history, verbose=True)
    assert answer in capsys.readouterr().out


@pytest.mark.parametrize("raw", [
    "{broken", "[]", '{"expression":1}', '{}',
    '{"expression":"1+1","extra":true}', '{"a":' * 60 + "1" + "}" * 60,
    '{"a":' * 2000 + "1" + "}" * 2000,
])
def test_native_arguments_remain_validated(raw):
    """只校验原生工具参数，拒绝损坏、类型错误和过深的参数对象。"""
    with pytest.raises(ToolCallError):
        parse_tool_call_args("calculator", raw, build_registry())


@pytest.mark.parametrize("ids", [("same", "same"), ("", "valid")])
def test_invalid_call_ids_rejected_before_side_effects(ids):
    """整批 ID 不合法时不执行任何工具。"""
    calls = [ToolCallItem(cid, "todo", json.dumps({"action": "add", "text": cid})) for cid in ids]
    result = Agent(FakeLLM([LLMResponse(tool_calls=calls)]), build_registry(),
                   max_output_retries=0).run("添加")
    assert result.status == OUTPUT_EXHAUSTED and result.actions == []
    assert verify_trace_integrity(result.trace_events)["valid"]


def test_disputed_answer_can_recover_by_executing_real_tool():
    """未执行后的回答会回填错误，模型可修正调用后再结束。"""
    llm = FakeLLM([
        {"tool_calls": [{"id": "bad", "name": "calculator", "args": {}}]},
        "计算成功",
        {"tool_calls": [{"id": "good", "name": "calculator", "args": {"expression": "2+3"}}]},
        "结果是 5",
    ])
    result = Agent(llm, build_registry(), max_output_retries=2).run("计算")
    assert result.status == OK and len(result.actions) == 1
    assert "工具调用未执行" in llm.calls[2][-1]["content"]
    assert not any(m["content"] == "计算成功" for m in result.messages)
    assert len([e for e in result.trace_events if e["event"] == "validation_error"]) == 2


def test_disputed_answers_exhaust_retry_budget():
    """无效工具之后持续声称完成时按预算失败并保留真实操作记录。"""
    llm = FakeLLM([
        {"tool_calls": [{"id": "bad", "name": "missing", "args": {}}]},
        "已经完成", "已经完成",
    ])
    result = Agent(llm, build_registry(), max_output_retries=2).run("执行")
    assert result.status == TOOL_NOT_EXECUTED and result.actions == []
    assert len(llm.calls) == 3 and "未完成说明" in result.answer


@pytest.mark.parametrize("parallel,expected_peak", [(False, 1), (True, 2)])
def test_parallel_switch_controls_execution_not_response_format(parallel, expected_peak):
    """相同原生多调用响应按开关串行或并发执行，并保持结果顺序。"""
    async def exercise():
        active = peak = 0

        async def read(args, ctx):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            active -= 1
            return args["value"]

        registry = ToolRegistry()
        registry.register(Tool("read", "读取", {"type": "object", "properties": {
            "value": {"type": "integer"}}, "required": ["value"]}, read, afunc=read))
        llm = FakeLLM([{"tool_calls": [
            {"id": str(i), "name": "read", "args": {"value": i}} for i in range(3)]}, "完成"])
        result = await Agent(llm, registry, parallel_tool_calls=parallel, max_concurrency=2).arun("读取")
        assert result.status == OK and peak == expected_peak
        assert [m["tool_call_id"] for m in result.messages if m["role"] == "tool"] == ["0", "1", "2"]

    asyncio.run(exercise())


def test_native_tool_output_serialization_error_is_observed():
    """不能序列化的工具返回值作为工具错误反馈，不中断模型循环。"""
    registry = ToolRegistry()
    registry.register(Tool("bad", "不合法结果", {"type": "object", "properties": {}},
                           lambda args, ctx: object()))
    llm = FakeLLM([{"tool_calls": [{"id": "call", "name": "bad", "args": {}}]}, "工具返回错误"])
    result = Agent(llm, registry).run("读取")
    assert result.status == OK and not result.actions[0].ok
    assert llm.calls[1][-1]["role"] == "tool"
    assert "not JSON serializable" in llm.calls[1][-1]["content"]


def test_text_only_client_and_cli_option_are_removed():
    """文本客户端不能自动降级，CLI 也不再提供文本工具模式。"""
    class TextOnly:
        def chat(self, messages):
            return '{"final_answer":"旧协议"}'

    with pytest.raises(TypeError, match="native"):
        Agent(TextOnly(), build_registry())
    with pytest.raises(SystemExit):
        cli._build_parser().parse_args(["--tool-mode", "text", "你好"])


def test_sync_native_client_fallback_does_not_block_event_loop():
    """只有同步原生接口的客户端在线程执行，等待时其他协程可推进。"""
    import threading

    entered = threading.Event()
    released = threading.Event()

    class SyncOnly:
        def complete(self, messages, **kwargs):
            entered.set()
            assert released.wait(2)
            assert kwargs["tools"]
            return LLMResponse("完成")

    async def exercise():
        task = asyncio.create_task(Agent(SyncOnly(), build_registry()).arun("你好"))
        for _ in range(100):
            if entered.is_set():
                break
            await asyncio.sleep(0.01)
        released.set()
        assert (await task).status == OK

    asyncio.run(exercise())
