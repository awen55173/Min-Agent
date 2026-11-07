import json

import pytest

import cli
from agent.llm import FakeLLM, LLMError
from agent.loop import INTERNAL_ERROR, OK
from agent.service import INTERRUPTED_NOTE, ChatService
from agent.session import SessionBusyError, SessionManager, SessionNotFoundError
from agent.tools import build_registry

WTTR = {"current_condition": [{"temp_C": "20", "FeelsLikeC": "19", "humidity": "50",
                               "weatherDesc": [{"value": "Sunny"}]}]}


def act(tool, **args):
    """构造原生工具调用测试响应。"""
    return {"content": "", "tool_calls": [{"id": "fixture_call", "type": "function", "function": {"name": tool, "arguments": json.dumps(args, ensure_ascii=False)}}]}


def final(text):
    """构造直接文本回答。"""
    return text


def factory(store):
    return build_registry("k", todo_store=store, http_get=lambda u: WTTR)


def service(tmp_path, replies, **kw):
    llm = FakeLLM(replies)
    return ChatService(llm, SessionManager(tmp_path / "data", lock_timeout=0.3), factory, **kw), llm


def state(tmp_path, user, sid):
    with SessionManager(tmp_path / "data").open(user, sid) as s:
        return s.data


def todo_texts(data):
    return [i["text"] for i in data.todos["items"]]


# ---------- 核心验收：两个窗口互不影响，关闭后分别恢复 ----------
def test_two_windows_independent_and_recoverable(tmp_path):
    # 窗口 1：查天气并记待办
    svc, _ = service(tmp_path, [act("weather", city="Beijing"),
                                act("todo", action="add", text="带伞(北京20度)"), final("已记录")])
    assert svc.send("alice", "w1", "查北京天气并记待办", create=True).status == OK
    # 窗口 2：写周报并记待办
    svc, _ = service(tmp_path, [act("todo", action="add", text="写周报"), final("已记录周报")])
    assert svc.send("alice", "w2", "帮我写周报并记待办", create=True).status == OK

    d1, d2 = state(tmp_path, "alice", "w1"), state(tmp_path, "alice", "w2")
    assert todo_texts(d1) == ["带伞(北京20度)"] and todo_texts(d2) == ["写周报"]
    assert d1.todos["items"][0]["id"] == 1 and d2.todos["items"][0]["id"] == 1  # 各自独立编号
    assert "天气" in d1.history[0]["content"] and "周报" in d2.history[0]["content"]
    assert not any("周报" in m["content"] for m in d1.history)
    assert not any("Sunny" in m["content"] for m in d2.history)

    # 关闭后（全新 service/manager）分别恢复并继续
    svc, llm = service(tmp_path, [act("todo", action="list"), final("窗口1有1条待办")])
    r = svc.send("alice", "w1", "我有哪些待办？")
    assert "带伞" in r.actions[0].result and "周报" not in r.actions[0].result
    # 恢复后模型看到了窗口 1 的历史，且只有一个 system 提示词
    first_call = llm.calls[0]
    assert [m["role"] for m in first_call].count("system") == 1 and first_call[0]["role"] == "system"
    assert any("北京" in m["content"] for m in first_call) and not any("周报" in m["content"] for m in first_call[1:])

    svc, _ = service(tmp_path, [act("todo", action="complete", id=1), final("完成")])
    svc.send("alice", "w2", "周报那条完成了")
    assert state(tmp_path, "alice", "w2").todos["items"][0]["done"] is True
    assert state(tmp_path, "alice", "w1").todos["items"][0]["done"] is False


def test_same_session_id_different_users_independent(tmp_path):
    svc, _ = service(tmp_path, [act("todo", action="add", text="A"), final("ok")])
    svc.send("alice", "s", "x", create=True)
    svc, _ = service(tmp_path, [act("todo", action="add", text="B"), final("ok")])
    svc.send("bob", "s", "x", create=True)
    assert todo_texts(state(tmp_path, "alice", "s")) == ["A"]
    assert todo_texts(state(tmp_path, "bob", "s")) == ["B"]


# ---------- 历史不重复拼接 ----------
def test_history_has_no_system_and_no_duplication(tmp_path):
    svc, llm = service(tmp_path, [final("a1"), final("a2"), final("a3")])
    for i in range(3):
        svc.send("u", "s", f"q{i}", create=(i == 0))
    h = state(tmp_path, "u", "s").history
    assert [m["role"] for m in h] == ["user", "assistant"] * 3
    assert [m["content"] for m in h if m["role"] == "user"] == ["q0", "q1", "q2"]
    # 第 3 次调用 = system + 前两轮(4条) + 新输入
    assert len(llm.calls[2]) == 1 + 4 + 1
    assert [m["role"] for m in llm.calls[2]].count("system") == 1


def test_tool_followup_sees_previous_tool_results(tmp_path):
    svc, _ = service(tmp_path, [act("weather", city="Beijing"), final("20度")])
    svc.send("u", "s", "北京天气", create=True)
    svc, llm = service(tmp_path, [final("还是 20 度")])
    svc.send("u", "s", "刚才几度？")
    assert any(m["role"] == "tool" and m.get("name") == "weather" and "Sunny" in m["content"] for m in llm.calls[0])


# ---------- 轮次与去重 ----------
def test_turn_ids_persist_across_restarts_and_dedupe_is_per_turn(tmp_path):
    svc, _ = service(tmp_path, [act("todo", action="add", text="x"), act("todo", action="add", text="x"), final("ok")])
    r1 = svc.send("u", "s", "add x", create=True)
    assert '"created": false' in r1.actions[1].result  # 同轮去重
    svc, _ = service(tmp_path, [act("todo", action="add", text="x"), final("ok")])
    r2 = svc.send("u", "s", "再加一条 x")  # 新进程新服务：下一轮允许新增
    assert '"created": true' in r2.actions[0].result
    assert r2.turn_id == r1.turn_id + 1
    d = state(tmp_path, "u", "s")
    assert len(d.todos["items"]) == 2 and d.next_turn == 3


# ---------- 失败退出也保留已完成操作 ----------
def test_failure_results_still_persist_history_and_todos(tmp_path):
    svc, _ = service(tmp_path, [act("todo", action="add", text="k")] * 3, max_steps=1)
    r = svc.send("u", "s", "loop", create=True)
    assert r.status == "max_steps"
    d = state(tmp_path, "u", "s")
    assert todo_texts(d) == ["k"]
    assert d.history[0] == {"role": "user", "content": "loop"}
    assert d.history[-1]["content"] == r.answer  # 失败提示入历史
    # 之后可接着追问
    svc, llm = service(tmp_path, [final("还剩 1 条 k")])
    svc.send("u", "s", "现在有什么待办")
    assert any("loop" == m["content"] for m in llm.calls[0])


def test_parse_exhausted_and_llm_error_persist(tmp_path):
    svc, _ = service(tmp_path, [""] * 4, max_output_retries=3)
    assert svc.send("u", "p", "q", create=True).status == "output_retries_exhausted"
    assert state(tmp_path, "u", "p").version == 1

    class Down(FakeLLM):
        def complete(self, messages, **kwargs):
            if self.calls:
                raise LLMError("down")
            return super().complete(messages, **kwargs)

    llm = Down([act("todo", action="add", text="done-before-crash")])
    svc = ChatService(llm, SessionManager(tmp_path / "data"), factory)
    r = svc.send("u", "l", "q", create=True)
    llm.calls.append([])  # 之后的调用全部失败
    assert r.status == "llm_error"
    assert todo_texts(state(tmp_path, "u", "l")) == ["done-before-crash"]


def test_internal_error_is_contained_and_saved(tmp_path):
    class Buggy(FakeLLM):
        def complete(self, messages, **kwargs):
            if self.calls:
                raise RuntimeError("bug")
            return super().complete(messages, **kwargs)

    svc = ChatService(Buggy([act("todo", action="add", text="kept")]), SessionManager(tmp_path / "data"), factory)
    r = svc.send("u", "s", "q", create=True)
    assert r.status == INTERNAL_ERROR and "RuntimeError" in r.answer and "todo" in r.answer
    assert todo_texts(state(tmp_path, "u", "s")) == ["kept"]


def test_keyboard_interrupt_saves_progress_and_releases_lock(tmp_path):
    class Interrupt(FakeLLM):
        def complete(self, messages, **kwargs):
            if self.calls:
                raise KeyboardInterrupt
            return super().complete(messages, **kwargs)

    mgr = SessionManager(tmp_path / "data", lock_timeout=0.3)
    svc = ChatService(Interrupt([act("todo", action="add", text="kept")]), mgr, factory)
    with pytest.raises(KeyboardInterrupt):
        svc.send("u", "s", "记一下", create=True)
    d = state(tmp_path, "u", "s")
    assert todo_texts(d) == ["kept"]
    assert d.history[0]["content"] == "记一下"
    assert INTERRUPTED_NOTE in d.history[-1]["content"]
    with mgr.open("u", "s"):  # 锁已释放
        pass


def test_no_partial_save_when_lock_unavailable(tmp_path):
    mgr = SessionManager(tmp_path / "data", lock_timeout=0.2)
    svc = ChatService(FakeLLM([final("x")]), mgr, factory)
    svc.send("u", "s", "q", create=True)
    llm = FakeLLM([final("y")])
    svc2 = ChatService(llm, SessionManager(tmp_path / "data", lock_timeout=0.2), factory)
    with mgr.open("u", "s"):
        with pytest.raises(SessionBusyError):
            svc2.send("u", "s", "blocked")
    assert llm.calls == []  # 拿不到锁时根本不会调用模型
    assert len(state(tmp_path, "u", "s").history) == 2


def test_resume_missing_session_raises(tmp_path):
    svc, _ = service(tmp_path, [final("x")])
    with pytest.raises(SessionNotFoundError):
        svc.send("u", "ghost", "q")


# ---------- CLI ----------
@pytest.fixture
def cli_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "m")
    return tmp_path


def test_cli_default_new_session_shows_id_and_can_resume(cli_env, monkeypatch, capsys):
    monkeypatch.setattr(cli, "OpenAIClient", lambda s: FakeLLM([act("todo", action="add", text="t1"), final("ok1")]))
    assert cli.main(["-u", "alice", "add t1"]) == 0
    err = capsys.readouterr().err
    sid = err.split("id=")[1].split()[0]
    assert sid in SessionManager(cli_env / "data").list_sessions("alice")

    fake = FakeLLM([act("todo", action="list"), final("有 t1")])
    monkeypatch.setattr(cli, "OpenAIClient", lambda s: fake)
    assert cli.main(["-u", "alice", "-s", sid, "list"]) == 0
    assert any("t1" in m["content"] for m in fake.calls[0])  # 恢复了待办与历史


def test_cli_two_default_sessions_are_independent(cli_env, monkeypatch, capsys):
    ids = []
    for text in ("A", "B"):
        monkeypatch.setattr(cli, "OpenAIClient", lambda s, t=text: FakeLLM([act("todo", action="add", text=t), final("ok")]))
        cli.main(["-u", "alice", f"add {text}"])
        ids.append(capsys.readouterr().err.split("id=")[1].split()[0])
    assert ids[0] != ids[1]
    assert todo_texts(state(cli_env, "alice", ids[0])) == ["A"]
    assert todo_texts(state(cli_env, "alice", ids[1])) == ["B"]


def test_cli_resume_unknown_session_errors(cli_env, monkeypatch, capsys):
    monkeypatch.setattr(cli, "OpenAIClient", lambda s: FakeLLM([]))
    assert cli.main(["-s", "nope", "hi"]) == 1
    assert "does not exist" in capsys.readouterr().err


def test_cli_invalid_session_id_and_busy(cli_env, monkeypatch, capsys):
    monkeypatch.setattr(cli, "OpenAIClient", lambda s: FakeLLM([final("x")]))
    assert cli.main(["-s", "../evil", "hi"]) == 1
    assert "invalid session_id" in capsys.readouterr().err
    cli.main(["-u", "bob", "hi"])
    sid = capsys.readouterr().err.split("id=")[1].split()[0]
    monkeypatch.setattr(cli, "SessionManager", lambda d: SessionManager(d, lock_timeout=0.2))
    with SessionManager(cli_env / "data").open("bob", sid):
        assert cli.main(["-u", "bob", "-s", sid, "again"]) == 1
    assert "in use" in capsys.readouterr().err


def test_cli_list_and_flag_conflict(cli_env, monkeypatch, capsys):
    monkeypatch.setattr(cli, "OpenAIClient", lambda s: FakeLLM([final("x")]))
    cli.main(["-u", "carol", "hi"])
    sid = capsys.readouterr().err.split("id=")[1].split()[0]
    assert cli.main(["-u", "carol", "--list"]) == 0
    assert capsys.readouterr().out.split() == [sid]
    with pytest.raises(SystemExit):
        cli.main(["--new", "-s", "x", "hi"])


def test_cli_interactive_mode(cli_env, monkeypatch, capsys):
    monkeypatch.setattr(cli, "OpenAIClient", lambda s: FakeLLM([final("r1"), final("r2")]))
    lines = iter(["q1", "q2", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))
    assert cli.main(["-u", "dave"]) == 0
    out = capsys.readouterr()
    assert "r1" in out.out and "r2" in out.out and "resume with" in out.err
    sid = SessionManager(cli_env / "data").list_sessions("dave")[0]
    assert len(state(cli_env, "dave", sid).history) == 4


def test_cli_interactive_resume_replays_history_and_verbose_steps(cli_env, monkeypatch, capsys):
    # 第一轮（含工具调用）创建会话
    monkeypatch.setattr(cli, "OpenAIClient", lambda s: FakeLLM([act("todo", action="add", text="buy milk"), final("已添加待办")]))
    assert cli.main(["-u", "eve", "帮我记买牛奶"]) == 0
    sid = capsys.readouterr().err.split("id=")[1].split()[0]

    # 单次命令模式恢复会话：不回放旧历史，保持输出干净
    monkeypatch.setattr(cli, "OpenAIClient", lambda s: FakeLLM([final("单次回复")]))
    assert cli.main(["-u", "eve", "-s", sid, "第二问"]) == 0
    single_out = capsys.readouterr()
    assert "you> 帮我记买牛奶" not in single_out.out
    assert "已添加待办" not in single_out.out
    assert single_out.out.strip() == "单次回复"

    # 交互模式恢复会话（不带 -v）：回放用户提问与最终回答，不显示工具步骤和 Observation
    monkeypatch.setattr(cli, "OpenAIClient", lambda s: FakeLLM([final("交互新回复")]))
    lines = iter(["第三问", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))
    assert cli.main(["-u", "eve", "-s", sid]) == 0
    resumed = capsys.readouterr()
    assert "you> 帮我记买牛奶\n已添加待办\nyou> 第二问\n单次回复\n交互新回复\n" == resumed.out
    assert "Observation" not in resumed.out
    assert "[step 1]" not in resumed.err

    # 交互模式恢复会话（带 -v）：额外在 stderr 回放历史工具步骤
    monkeypatch.setattr(cli, "OpenAIClient", lambda s: FakeLLM([]))
    monkeypatch.setattr("builtins.input", lambda prompt="": "exit")
    assert cli.main(["-u", "eve", "-s", sid, "-v"]) == 0
    resumed_v = capsys.readouterr()
    assert "you> 帮我记买牛奶\n已添加待办" in resumed_v.out
    assert "[step 1] todo({'action': 'add', 'text': 'buy milk'}) ok=True ->" in resumed_v.err


# ---------- live ----------
@pytest.mark.live
def test_live_two_windows_recoverable(tmp_path):
    from agent.config import Settings
    from agent.llm import OpenAIClient

    s = Settings.from_env()
    mgr = SessionManager(tmp_path / "data")
    fac = lambda store: build_registry(s.tavily_api_key, todo_store=store)  # noqa: E731

    def chat(sid, text, create=False):
        return ChatService(OpenAIClient(s), mgr, fac).send("alice", sid, text, create=create)

    r1 = chat("w1", "查一下北京现在的天气，并把天气情况记成一条待办，然后告诉我。", True)
    r2 = chat("w2", "帮我写一份简短的周报提纲，并记一条待办：提交周报。然后告诉我。", True)
    assert r1.status == OK and r2.status == OK, (r1.answer, r2.answer)
    d1, d2 = state(tmp_path, "alice", "w1"), state(tmp_path, "alice", "w2")
    assert len(d1.todos["items"]) >= 1 and len(d2.todos["items"]) >= 1
    assert not any("周报" in i["text"] for i in d1.todos["items"])
    assert not any("天气" in i["text"] or "°" in i["text"] for i in d2.todos["items"])
    # 重新恢复窗口 1，追问带工具
    r3 = chat("w1", "列出我现在的待办事项。")
    assert r3.status == OK
    assert "周报" not in r3.answer
