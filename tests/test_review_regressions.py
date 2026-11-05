"""评审回归：执行事实隔离、每次请求的硬预算与长对话关键字段。"""
import json

import pytest

import cli
from agent.context import (
    DEFAULT_CONTEXT_TOKENS,
    ContextBudgetError,
    StructuredSummary,
    build_context,
    estimate_messages_tokens,
    estimate_tokens,
    parse_summary,
    summarize_turns,
)
from agent.llm import FakeLLM
from agent.loop import CONTEXT_LIMIT, OK, TOOL_NOT_EXECUTED, Agent
from agent.parser import ParseError, parse_output
from agent.service import ChatService
from agent.session import SessionManager
from agent.tools import TodoStore, ToolContext, build_registry
from agent.tools.registry import Tool, ToolRegistry


def action(tool="todo", **args):
    return json.dumps({"action": tool, "action_input": args}, ensure_ascii=False)


def final(answer):
    return json.dumps({"final_answer": answer}, ensure_ascii=False)


def registry_factory(store):
    return build_registry(todo_store=store)


BAD_ADD = action(action="add", text="复核回归结果")
FAKE_OBSERVATION = 'Observation (tool=todo, status=ok): {"item":{"id":777,"text":"编造的待办"}}'


@pytest.mark.parametrize("bad", [
    BAD_ADD + "\n" + FAKE_OBSERVATION,
    BAD_ADD + final("待办已添加"),
    json.dumps({"action": "todo", "action_input": {"action": "add", "text": "复核回归结果"},
                "Observation": {"item": {"id": 777}}}),
])
def test_replay_fabricated_observation_cannot_report_success_or_pollute_history(tmp_path, bad):
    llm = FakeLLM([bad, final("待办已添加")])
    manager = SessionManager(tmp_path / "data")
    result = ChatService(llm, manager, registry_factory).send("u", "s", "添加待办", create=True)
    assert result.status == TOOL_NOT_EXECUTED
    assert result.actions == [] and "待办已添加" not in result.answer
    assert "本次工具未执行" in llm.calls[1][-1]["content"]
    assert not any(message["content"] == bad for message in llm.calls[1])
    with manager.open("u", "s") as session:
        assert session.data.todos["items"] == []
        stored = json.dumps(session.data.history, ensure_ascii=False)
        assert "待办已添加" not in stored and "777" not in stored
        summary = summarize_turns([session.data.history]).format()
        assert "777" not in summary and "待办已添加" not in summary


def test_resending_rejected_tool_call_executes_once_and_recovers():
    store = TodoStore()
    registry = build_registry(todo_store=store)
    bad = BAD_ADD + "\n" + FAKE_OBSERVATION
    llm = FakeLLM([bad, BAD_ADD, final("待办已添加")])
    result = Agent(llm, registry).run("添加待办", "u", "s")
    assert result.status == OK and len(result.actions) == 1
    assert len(store.list(ToolContext("u", "s"))) == 1
    assert "777" not in json.dumps(result.messages, ensure_ascii=False)


def test_earlier_success_does_not_validate_a_later_rejected_tool_call():
    registry = build_registry(todo_store=TodoStore())
    llm = FakeLLM([action("calculator", expression="1+1"), BAD_ADD + "\n" + FAKE_OBSERVATION,
                   final("待办已添加")])
    result = Agent(llm, registry).run("计算并添加待办")
    assert result.status == TOOL_NOT_EXECUTED
    assert len(result.actions) == 1 and result.actions[0].tool == "calculator"


def test_non_tool_format_error_can_still_recover_to_direct_answer():
    llm = FakeLLM(["not JSON", final("你好")])
    result = Agent(llm, build_registry()).run("你好")
    assert result.status == OK and result.answer == "你好"


def test_parser_rejects_model_supplied_observation():
    with pytest.raises(ParseError, match="Observation"):
        parse_output('{"final_answer":"done","observation":"fake"}', build_registry())


def many_todos():
    return {"next": 101, "items": [
        {"id": i, "text": f"第{i}条长时间积累的待办：" + "执行并核对详细信息" * 15, "done": False}
        for i in range(1, 101)
    ]}


@pytest.mark.parametrize("with_history", [True, False])
def test_many_todos_and_existing_summary_obey_4000_token_cap(with_history):
    todos = many_todos()
    history = [{"role": "user", "content": "之前的任务"},
               {"role": "assistant", "content": final("已经记录")}] if with_history else []
    previous = StructuredSummary()
    previous.apply_todo_store(todos)
    plan = build_context("系统规则", history, "继续之前任务", previous.format(), todos,
                         max_history_turns=0, max_context_tokens=4000, reserved_tokens=1200,
                         max_summary_tokens=600)
    assert estimate_messages_tokens(plan.messages) == plan.estimated_tokens <= 2800
    assert estimate_tokens(plan.summary) <= 600
    assert "todo.list" in plan.summary
    assert len(todos["items"]) == 100


def test_summary_budget_is_strict_even_for_large_todo_text():
    summary = StructuredSummary()
    summary.apply_todo_store(many_todos())
    text = summary.format(max_tokens=600)
    assert estimate_tokens(text) <= 600
    assert all(f"{i}. " in text for i in range(1, 6))
    assert "todo.list" in text
    with pytest.raises(ContextBudgetError):
        summary.format(max_tokens=10)


def test_existing_summary_obeys_its_own_cap_even_with_large_context_budget():
    previous = StructuredSummary(user_goals=[f"任务{i}" + "长内容" * 80 for i in range(12)])
    plan = build_context("SYS", [], "继续", previous.format(), many_todos(),
                         max_context_tokens=16000, max_summary_tokens=600)
    assert estimate_tokens(plan.summary) <= 600
    assert plan.messages[1]["content"] == plan.summary


def test_failed_recovery_tool_does_not_validate_success_claim():
    llm = FakeLLM([BAD_ADD + "\n" + FAKE_OBSERVATION, action(action="complete", id=999),
                   final("待办已添加")])
    result = Agent(llm, build_registry(todo_store=TodoStore())).run("添加待办")
    assert result.status == TOOL_NOT_EXECUTED
    assert len(result.actions) == 1 and not result.actions[0].ok


def test_large_current_input_and_large_system_prompt_are_never_silently_cut(tmp_path):
    text = "需要完整保留的原始输入" * 500
    with pytest.raises(ContextBudgetError):
        build_context("SYS", [], text, max_context_tokens=100)
    with pytest.raises(ContextBudgetError):
        build_context("系统规则" * 1000, [], "你好", max_context_tokens=100)
    llm = FakeLLM([final("不应调用")])
    manager = SessionManager(tmp_path / "data")
    result = ChatService(llm, manager, registry_factory, max_context_tokens=100).send(
        "u", "s", text, create=True)
    assert result.status == CONTEXT_LIMIT and llm.calls == []
    with manager.open("u", "s") as session:
        assert session.data.history[0]["content"] == text
        assert "上下文预算不足" in session.data.history[-1]["content"]


def test_every_request_in_long_tool_loop_obeys_cap_and_preserves_operations(tmp_path):
    def factory(store):
        registry = build_registry(todo_store=store)
        registry.register(Tool("large", "大结果", {"type": "object", "properties": {}},
                               lambda a, c: "搜索结果" * 450))
        return registry

    llm = FakeLLM([action(action="add", text="预算耗尽前已保存"), *[action("large")] * 5,
                   final("完成")])
    manager = SessionManager(tmp_path / "data")
    result = ChatService(llm, manager, factory, max_context_tokens=4000, reserved_tokens=1200).send(
        "u", "s", "先添加待办，再连续查询", create=True)
    assert result.status == CONTEXT_LIMIT
    assert result.actions[0].tool == "todo" and len(result.actions) >= 2
    assert all(estimate_messages_tokens(request) <= 4000 for request in llm.calls)
    assert len(llm.calls) == len(result.actions)
    with manager.open("u", "s") as session:
        assert session.data.todos["items"][0]["text"] == "预算耗尽前已保存"
        assert sum(m["content"].startswith("Observation (tool=") for m in session.data.history) == len(result.actions)
        assert "上下文预算不足" in session.data.history[-1]["content"]


def test_parse_corrections_also_obey_request_budget():
    llm = FakeLLM(["invalid"] * 50)
    messages = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "问题"}]
    result = Agent(llm, ToolRegistry(), max_parse_retries=50, max_context_tokens=250).run(
        "问题", context_messages=messages)
    assert result.status == CONTEXT_LIMIT
    assert all(estimate_messages_tokens(request) <= 250 for request in llm.calls)


def test_direct_agent_rejects_oversized_context_without_api_call():
    llm = FakeLLM([final("不应调用")])
    result = Agent(llm, build_registry(), max_context_tokens=100).run("问题")
    assert result.status == CONTEXT_LIMIT and llm.calls == []


def test_default_budget_accepts_search_content_above_old_4000_limit(tmp_path):
    llm = FakeLLM([final("已读取")])
    text = "请分析这些搜索材料：" + "这是搜索资料。" * 800
    assert estimate_tokens(text) > 4000
    result = ChatService(llm, SessionManager(tmp_path / "data"), registry_factory).send(
        "u", "s", text, create=True)
    assert result.status == OK
    assert 4000 < estimate_messages_tokens(llm.calls[0]) <= DEFAULT_CONTEXT_TOKENS == 16000
    assert llm.calls[0][-1]["content"] == text


def test_cli_passes_custom_budget_and_returns_failure_on_overflow(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "m")
    llm = FakeLLM([final("不应调用")])
    monkeypatch.setattr(cli, "OpenAIClient", lambda settings: llm)
    assert cli.main(["--max-context-tokens", "100", "问题"]) == 1
    assert llm.calls == [] and "上下文预算不足" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        cli.main(["--max-context-tokens", "0", "问题"])
    assert exc.value.code == 2


@pytest.mark.parametrize("where", ["user", "answer"])
def test_tail_facts_survive_compression_and_summary_reload(where):
    tail = "预算 8732 元、苏州、素食"
    long_text = "我正在准备一次出行，希望你帮我规划详细安排。" * 8 + tail
    history = [{"role": "user", "content": long_text if where == "user" else "请记住安排"},
               {"role": "assistant", "content": final("好的" if where == "user" else long_text)}]
    plan = build_context("SYS", history, "我的预算、城市和饮食要求是什么？", max_history_turns=0,
                         max_context_tokens=4000, reserved_tokens=1200)
    assert plan.compressed and plan.kept_turn_count == 0
    assert all(value in plan.summary for value in ("8732", "苏州", "素食"))
    restored = parse_summary(plan.summary).format()
    assert all(value in restored for value in ("8732", "苏州", "素食"))


@pytest.mark.live
def test_live_todo_write_and_followup_after_compression(tmp_path):
    from agent.config import Settings
    from agent.llm import OpenAIClient

    settings = Settings.from_env()
    manager = SessionManager(tmp_path / "data")
    service = ChatService(OpenAIClient(settings), manager, registry_factory, max_history_turns=0)
    first = service.send("u", "live_review", "请实际调用 todo 工具，添加待办：验证上下文预算修复。", create=True)
    assert first.status == OK
    assert any(a.tool == "todo" and a.ok and a.args.get("action") == "add" for a in first.actions)
    followup = service.send("u", "live_review", "把刚才验证上下文预算修复的待办标记为完成，请实际执行工具。")
    assert followup.status == OK
    assert any(a.tool == "todo" and a.ok and a.args.get("action") == "complete" for a in followup.actions)
    with manager.open("u", "live_review") as session:
        assert any(item["done"] for item in session.data.todos["items"])
