import json

import pytest

import cli
from agent.llm import FakeLLM, LLMError
from agent.loop import LLM_FAILED, MAX_STEPS, OK, OUTPUT_EXHAUSTED, Agent
from agent.tools import TodoStore, build_registry

WTTR = {"current_condition": [{"temp_C": "20", "FeelsLikeC": "19", "humidity": "50",
                               "weatherDesc": [{"value": "Sunny"}]}]}


def act(tool, **args):
    """构造原生工具调用测试响应。"""
    return {"content": "", "tool_calls": [{"id": "fixture_call", "type": "function", "function": {"name": tool, "arguments": json.dumps(args, ensure_ascii=False)}}]}


def final(text):
    """构造直接文本回答。"""
    return text


@pytest.fixture
def reg():
    return build_registry("key", todo_store=TodoStore(), http_get=lambda u: WTTR,
                          http_post=lambda u, b, k: {"results": []})


# ---------------- loop ----------------
def test_direct_answer(reg):
    llm = FakeLLM([final("你好")])
    r = Agent(llm, reg).run("hi")
    assert (r.status, r.answer, r.actions) == (OK, "你好", [])
    assert len(llm.calls) == 1


def test_multi_tool_chain_weather_todo_reply(reg):
    llm = FakeLLM([
        act("weather", city="Beijing"),
        act("todo", action="add", text="带伞，北京 20 度晴"),
        act("todo", action="list"),
        final("北京 20℃ 晴，已记待办"),
    ])
    r = Agent(llm, reg).run("查北京天气并记待办", "u", "s")
    assert r.status == OK and r.answer.startswith("北京")
    assert [a.tool for a in r.actions] == ["weather", "todo", "todo"]
    assert all(a.ok for a in r.actions)
    # observation 依次回填
    obs = [m["content"] for m in llm.calls[-1] if m["role"] == "tool"]
    assert len(obs) == 3 and "Sunny" in obs[0] and '"created": true' in obs[1]
    # 最后一次调用能看到完整历史：system, user, (assistant, observation)*3
    assert [m["role"] for m in llm.calls[-1]] == ["system", "user"] + ["assistant", "tool"] * 3


def test_tool_error_is_fed_back_and_model_recovers(reg):
    llm = FakeLLM([
        act("calculator", expression="1/0"),
        act("calculator", expression="1/2"),
        final("0.5"),
    ])
    r = Agent(llm, reg).run("calc")
    assert r.status == OK
    assert [a.ok for a in r.actions] == [False, True]
    assert "Error: " in llm.calls[1][-1]["content"] and "division by zero" in llm.calls[1][-1]["content"]


def test_invalid_output_retried_then_succeeds(reg):
    llm = FakeLLM([
        "", {"tool_calls": [{"id": "bad", "name": "calculator", "args": {}}]},
        act("calculator", expression="1+1"), final("2"),
    ])
    r = Agent(llm, reg).run("q")
    assert r.status == OK and r.answer == "2" and len(r.actions) == 1
    assert llm.calls[1][-1]["content"].startswith("Validation error")
    assert "missing required" in llm.calls[2][-1]["content"]


def test_output_retries_exhausted(reg):
    llm = FakeLLM([""] * 4)
    r = Agent(llm, reg, max_output_retries=3).run("q")
    assert r.status == OUTPUT_EXHAUSTED and "无效" in r.answer
    assert len(llm.calls) == 4


def test_output_retry_budget_is_cumulative_after_valid_tool(reg):
    llm = FakeLLM(["", "", act("calculator", expression="1+1"), "", "", final("2")])
    r = Agent(llm, reg, max_output_retries=2).run("q")
    assert r.status == OUTPUT_EXHAUSTED


def test_max_steps_returns_clear_result_and_actions(reg):
    llm = FakeLLM([act("calculator", expression="1+1")] * 5)
    r = Agent(llm, reg, max_steps=2).run("loop")
    assert r.status == MAX_STEPS and "最大步数" in r.answer
    assert len(r.actions) == 2 and "calculator" in r.answer
    assert len(llm.calls) == 3  # 第 3 次请求工具时被拒绝执行


def test_model_can_answer_after_last_allowed_step(reg):
    llm = FakeLLM([act("calculator", expression="1+1"), final("2")])
    assert Agent(llm, reg, max_steps=1).run("q").status == OK


def test_llm_failure_keeps_completed_actions(reg):
    class Flaky(FakeLLM):
        def complete(self, messages, **kwargs):
            if len(self.calls) == 1:
                self.calls.append(messages)
                raise LLMError("down")
            return super().complete(messages, **kwargs)

    llm = Flaky([act("todo", action="add", text="x")])
    r = Agent(llm, reg).run("q")
    assert r.status == LLM_FAILED and "暂时不可用" in r.answer and "todo" in r.answer
    assert len(r.actions) == 1


def test_same_turn_id_for_all_calls_dedupes_todo():
    store = TodoStore()
    reg = build_registry(todo_store=store)
    llm = FakeLLM([act("todo", action="add", text="x"), act("todo", action="add", text="x"), final("ok")])
    r = Agent(llm, reg).run("q", "u", "s")
    assert '"created": false' in r.actions[1].result
    # 下一轮（新 turn_id）允许再次新增同名待办
    llm2 = FakeLLM([act("todo", action="add", text="x"), final("ok")])
    r2 = Agent(llm2, reg).run("again", "u", "s", turn_id="next-turn")
    assert '"created": true' in r2.actions[0].result


def test_turn_ids_unique_across_runs_and_instances(reg):
    a1, a2 = Agent(FakeLLM([final("a"), final("b")]), reg), Agent(FakeLLM([final("c")]), reg)
    ids = {a1.run("1").turn_id, a1.run("2").turn_id, a2.run("3").turn_id}
    assert len(ids) == 3


def test_explicit_turn_id_does_not_collide_with_auto():
    reg = build_registry(todo_store=TodoStore())
    llm = FakeLLM([act("todo", action="add", text="x"), final("ok")] * 2)
    Agent(llm, reg).run("a", "u", "s", turn_id=1)
    r = Agent(llm, reg).run("b", "u", "s")  # 自动生成的 ID 不会等于显式的 1
    assert '"created": true' in r.actions[0].result


def test_two_run_once_sharing_registry_do_not_merge_todos():
    reg = build_registry(todo_store=TodoStore())
    for _ in range(2):
        r = cli.run_once(FakeLLM([act("todo", action="add", text="same"), final("ok")]), "add same", reg)
        assert '"created": true' in r.actions[0].result
    from agent.tools import ToolContext
    assert len(reg.get("todo").func({"action": "list"}, ToolContext())["items"]) == 2


@pytest.mark.parametrize("blank", ["", " ", "\n\t  \n"])
def test_blank_final_answer_goes_through_format_correction(reg, blank):
    llm = FakeLLM([blank, final("真正的答案")])
    r = Agent(llm, reg).run("q")
    assert r.status == OK and r.answer == "真正的答案"
    assert llm.calls[1][-1]["content"].startswith("Validation error")


def test_blank_final_answer_retries_exhausted(reg):
    r = Agent(FakeLLM(["  "] * 4), reg, max_output_retries=3).run("q")
    assert r.status == OUTPUT_EXHAUSTED


def test_observation_truncated():
    from agent.tools.registry import Tool
    reg = build_registry()
    reg.register(Tool("big", "d", {"type": "object", "properties": {}}, lambda a, c: "x" * 20000))
    llm = FakeLLM([act("big"), final("ok")])
    Agent(llm, reg).run("q")
    assert "truncated" in llm.calls[1][-1]["content"]
    from agent.context import estimate_tokens
    assert estimate_tokens(llm.calls[1][-1]["content"]) < 2100


def test_history_is_inserted_before_user_input(reg):
    llm = FakeLLM([final("ok")])
    hist = [{"role": "user", "content": "old"}, {"role": "assistant", "content": final("old a")}]
    Agent(llm, reg).run("new", history=hist)
    assert [m["content"] for m in llm.calls[0][1:3]] == ["old", final("old a")]
    assert llm.calls[0][-1]["content"] == "new"


# ---------------- cli ----------------
def test_cli_main_with_fake(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "m")
    fake = FakeLLM([act("calculator", expression="2*3"), final("结果是 6")])
    monkeypatch.setattr(cli, "OpenAIClient", lambda s: fake)
    assert cli.main(["-v", "2*3?"]) == 0
    out = capsys.readouterr()
    assert "结果是 6" in out.out and "calculator" in out.err


def test_cli_nonzero_exit_on_failure(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setattr(cli, "OpenAIClient", lambda s: FakeLLM([""] * 10))
    assert cli.main(["q"]) == 1


# ---------------- live ----------------
@pytest.mark.live
def test_live_weather_then_todo():
    from agent.config import Settings
    from agent.llm import OpenAIClient
    s = Settings.from_env()
    store = TodoStore()
    reg = build_registry(s.tavily_api_key, todo_store=store)
    r = Agent(OpenAIClient(s), reg).run("查一下北京现在的天气，然后把天气情况记成一条待办，最后告诉我。", "u", "s")
    assert r.status == OK, r.answer
    tools = [a.tool for a in r.actions]
    assert "weather" in tools and "todo" in tools
    from agent.tools import ToolContext
    assert len(store.list(ToolContext("u", "s"))) >= 1
