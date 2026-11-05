import json

import pytest

import cli
from agent.llm import FakeLLM, LLMError
from agent.loop import LLM_FAILED, MAX_STEPS, OK, PARSE_EXHAUSTED, Agent
from agent.parser import ParseError, parse_output
from agent.prompt import build_system_prompt
from agent.tools import TodoStore, build_registry

WTTR = {"current_condition": [{"temp_C": "20", "FeelsLikeC": "19", "humidity": "50",
                               "weatherDesc": [{"value": "Sunny"}]}]}


def J(**kw):
    return json.dumps(kw, ensure_ascii=False)


def act(tool, **args):
    return J(thought="t", action=tool, action_input=args)


def final(text):
    return J(thought="t", final_answer=text)


@pytest.fixture
def reg():
    return build_registry("key", todo_store=TodoStore(), http_get=lambda u: WTTR,
                          http_post=lambda u, b, k: {"results": []})


# ---------------- parser ----------------
def test_parse_plain_json_action(reg):
    p = parse_output(act("calculator", expression="1+1"), reg)
    assert (p.kind, p.action, p.action_input, p.thought) == ("action", "calculator", {"expression": "1+1"}, "t")


def test_parse_final(reg):
    p = parse_output(final(" hi "), reg)
    assert p.kind == "final" and p.final_answer == "hi"


def test_parse_code_block_and_surrounding_text(reg):
    text = "Sure:\n```json\n" + final("ok") + "\n```\nbye"
    assert parse_output(text, reg).final_answer == "ok"
    assert parse_output("I will answer " + final("x") + " done", reg).final_answer == "x"


def test_parse_missing_thought_tolerated(reg):
    assert parse_output('{"final_answer": "a"}', reg).thought == ""


def test_parse_null_fields_treated_as_absent(reg):
    p = parse_output('{"action": null, "final_answer": "a"}', reg)
    assert p.kind == "final"


@pytest.mark.parametrize("text,frag", [
    (J(action="calculator", action_input={"expression": "1"}, final_answer="x"), "mutually exclusive"),
    (J(thought="t"), "either"),
    (J(action="calculator"), "action_input"),
    (J(action="calculator", action_input={}), "missing required"),
    (J(action="calculator", action_input={"expression": 1}), "must be string"),
    (J(action="calculator", action_input="1+1"), "JSON object"),
    (J(action="nope", action_input={}), "available tools"),
    (J(action=5, action_input={}), "tool name"),
    (J(final_answer=5), "string"),
    ("just some text", "not valid JSON"),
    ("", "empty"),
    ("[1, 2]", "not valid JSON"),
    ("{bad json", "not valid JSON"),
])
def test_parse_errors(reg, text, frag):
    with pytest.raises(ParseError, match=frag):
        parse_output(text, reg)


# ---------------- prompt ----------------
def test_prompt_contains_schema_and_format(reg):
    p = build_system_prompt(reg)
    for name in reg.names():
        assert f'"name": "{name}"' in p
    assert "final_answer" in p and "action_input" in p and "mutually exclusive" in p
    assert '"required"' in p  # 参数 schema 已导出


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
    obs = [m["content"] for m in llm.calls[-1] if m["content"].startswith("Observation")]
    assert len(obs) == 3 and "Sunny" in obs[0] and '"created": true' in obs[1]
    # 最后一次调用能看到完整历史：system, user, (assistant, observation)*3
    assert [m["role"] for m in llm.calls[-1]] == ["system", "user"] + ["assistant", "user"] * 3


def test_tool_error_is_fed_back_and_model_recovers(reg):
    llm = FakeLLM([
        act("calculator", expression="1/0"),
        act("calculator", expression="1/2"),
        final("0.5"),
    ])
    r = Agent(llm, reg).run("calc")
    assert r.status == OK
    assert [a.ok for a in r.actions] == [False, True]
    assert "status=error" in llm.calls[1][-1]["content"] and "division by zero" in llm.calls[1][-1]["content"]


def test_invalid_output_retried_then_succeeds(reg):
    llm = FakeLLM(["not json", J(action="calculator", action_input={}, final_answer="x"), final("ok")])
    r = Agent(llm, reg).run("q")
    assert r.status == OK and r.answer == "ok" and r.actions == []
    assert llm.calls[1][-1]["content"].startswith("Format error")
    assert "mutually exclusive" in llm.calls[2][-1]["content"]


def test_conflicting_output_never_executes_tool():
    store = TodoStore()
    reg = build_registry(todo_store=store)
    bad = J(action="todo", action_input={"action": "add", "text": "x"}, final_answer="done")
    llm = FakeLLM([bad, final("ok")])
    Agent(llm, reg).run("q", "u", "s")
    from agent.tools import ToolContext
    assert store.list(ToolContext("u", "s")) == []


def test_parse_retries_exhausted(reg):
    llm = FakeLLM(["bad"] * 4)
    r = Agent(llm, reg, max_parse_retries=3).run("q")
    assert r.status == PARSE_EXHAUSTED and "格式" in r.answer
    assert len(llm.calls) == 4


def test_parse_failure_counter_resets_after_valid_output(reg):
    llm = FakeLLM(["bad", "bad", act("calculator", expression="1+1"), "bad", "bad", final("2")])
    r = Agent(llm, reg, max_parse_retries=2).run("q")
    assert r.status == OK


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
        def chat(self, messages):
            if len(self.calls) == 1:
                self.calls.append(messages)
                raise LLMError("down")
            return super().chat(messages)

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
    llm = FakeLLM([J(thought="t", final_answer=blank), final("真正的答案")])
    r = Agent(llm, reg).run("q")
    assert r.status == OK and r.answer == "真正的答案"
    assert llm.calls[1][-1]["content"].startswith("Format error")


def test_blank_final_answer_retries_exhausted(reg):
    r = Agent(FakeLLM([J(final_answer="  ")] * 4), reg, max_parse_retries=3).run("q")
    assert r.status == PARSE_EXHAUSTED


def test_parse_blank_final_answer_error(reg):
    with pytest.raises(ParseError, match="non-blank"):
        parse_output(J(final_answer=" \n "), reg)


def test_observation_truncated():
    from agent.tools.registry import Tool
    reg = build_registry()
    reg.register(Tool("big", "d", {"type": "object", "properties": {}}, lambda a, c: "x" * 5000))
    llm = FakeLLM([act("big"), final("ok")])
    Agent(llm, reg).run("q")
    assert "(truncated)" in llm.calls[1][-1]["content"]
    assert len(llm.calls[1][-1]["content"]) < 2200


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
    monkeypatch.setattr(cli, "OpenAIClient", lambda s: FakeLLM(["bad"] * 10))
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
