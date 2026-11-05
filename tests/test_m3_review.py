"""M3 评审修复的回归测试：活跃锁不被误回收、中断后历史完整、歧义多 JSON 输出被拒绝。"""
import json
import threading
import time
from pathlib import Path

import pytest

from agent.llm import FakeLLM
from agent.loop import OK, Agent
from agent.parser import ParseError, parse_output
from agent.service import INTERRUPTED_NOTE, ChatService
from agent.session import LockLostError, SessionBusyError, SessionManager
from agent.tools import ToolContext, TodoStore, build_registry
from agent.tools.registry import Tool


def J(**kw):
    return json.dumps(kw, ensure_ascii=False)


def act(tool, **args):
    return J(thought="t", action=tool, action_input=args)


def final(text):
    return J(thought="t", final_answer=text)


def lock_path(mgr, user="u", sid="s"):
    return Path(str(mgr._path(user, sid)) + ".lock")


def seed(mgr, user="u", sid="s"):
    with mgr.open(user, sid, create=True) as s:
        s.save()


# ---------------- 活跃锁不被误回收 ----------------
def test_long_running_holder_keeps_lock_beyond_stale_after(tmp_path):
    """持有时间远超 stale_after 时，心跳保持锁新鲜，其他窗口始终拿不到锁，持有者最终正常保存。"""
    root = tmp_path / "data"
    holder = SessionManager(root, stale_after=0.3)
    other = SessionManager(root, lock_timeout=1.5, stale_after=0.3)
    seed(holder)
    with holder.open("u", "s") as s:
        t0 = time.monotonic()
        with pytest.raises(SessionBusyError):
            with other.open("u", "s"):
                pass
        assert time.monotonic() - t0 >= 1.4  # 等满超时也没能接管（>4 倍 stale_after）
        s.data.history.append({"role": "user", "content": "long"})
        s.save()  # 原持有者保存成功
    assert not lock_path(holder).exists()  # 释放后无残留锁
    with other.open("u", "s") as s:
        assert s.data.history[-1]["content"] == "long"


def test_crashed_holder_lock_is_reclaimed(tmp_path):
    """心跳停止（进程崩溃）的锁在 stale_after 后可被回收。"""
    mgr = SessionManager(tmp_path / "data", lock_timeout=2, stale_after=0.3)
    seed(mgr)
    lock_path(mgr).write_text(json.dumps({"token": "dead"}), encoding="utf-8")  # 无心跳的遗留锁
    t0 = time.monotonic()
    with mgr.open("u", "s"):
        pass
    assert 0.25 <= time.monotonic() - t0 < 2


def test_lock_taken_over_holder_refuses_to_save(tmp_path):
    """锁被接管后原持有者拒绝保存，不覆盖他人数据，也不删除他人的锁。"""
    mgr = SessionManager(tmp_path / "data")
    seed(mgr)
    p = mgr._path("u", "s")
    before = p.read_bytes()
    with mgr.open("u", "s") as s:
        lock_path(mgr).write_text(json.dumps({"token": "other"}), encoding="utf-8")
        s.data.history.append({"role": "user", "content": "x"})
        with pytest.raises(LockLostError):
            s.save()
    assert p.read_bytes() == before
    assert json.loads(lock_path(mgr).read_text(encoding="utf-8"))["token"] == "other"


def test_heartbeat_detects_loss(tmp_path):
    mgr = SessionManager(tmp_path / "data", stale_after=0.2)
    seed(mgr)
    with mgr.open("u", "s") as s:
        lock_path(mgr).unlink()
        time.sleep(0.2)
        assert s._lock.lost
        with pytest.raises(LockLostError):
            s.save()


def test_concurrent_reclaimers_do_not_both_enter(tmp_path):
    """多个等待者同时发现过期锁时，回收经守卫串行化，同一时刻最多一个持有者。"""
    root = tmp_path / "data"
    seed(SessionManager(root))
    lp = lock_path(SessionManager(root))
    lp.write_text("{}", encoding="utf-8")
    import os
    old = time.time() - 100
    os.utime(lp, (old, old))

    inside, peak, lock = [0], [0], threading.Lock()

    def work():
        m = SessionManager(root, lock_timeout=10, stale_after=0.5)
        with m.open("u", "s") as s:
            with lock:
                inside[0] += 1
                peak[0] = max(peak[0], inside[0])
            time.sleep(0.05)
            s.data.history.append({"role": "user", "content": "x"})
            s.save()
            with lock:
                inside[0] -= 1

    threads = [threading.Thread(target=work) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert peak[0] == 1
    with SessionManager(root).open("u", "s") as s:
        assert len(s.data.history) == 6


def test_leftover_reclaim_guard_is_cleaned(tmp_path):
    import os
    mgr = SessionManager(tmp_path / "data", lock_timeout=2, stale_after=0.2)
    seed(mgr)
    lp = lock_path(mgr)
    guard = Path(str(lp) + ".reclaim")
    for f in (lp, guard):
        f.write_text("{}", encoding="utf-8")
        os.utime(f, (time.time() - 100, time.time() - 100))
    with mgr.open("u", "s"):
        pass
    assert not guard.exists()


# ---------------- 中断后历史完整 ----------------
def factory_with(extra=None):
    def factory(store):
        reg = build_registry(todo_store=store)
        if extra:
            reg.register(extra)
        return reg
    return factory


class InterruptOnCall(FakeLLM):
    """第 n 次调用（从 0 计）时抛 KeyboardInterrupt。"""

    def __init__(self, replies, at):
        super().__init__(replies)
        self.at = at

    def chat(self, messages):
        if len(self.calls) == self.at:
            self.calls.append(messages)
            raise KeyboardInterrupt
        return super().chat(messages)


def history(tmp_path):
    with SessionManager(tmp_path / "data").open("u", "s") as s:
        return s.data.history, s.data.todos


def test_interrupt_after_tool_keeps_tool_call_and_observation(tmp_path):
    mgr = SessionManager(tmp_path / "data")
    llm = InterruptOnCall([act("todo", action="add", text="buy milk")], at=1)
    with pytest.raises(KeyboardInterrupt):
        ChatService(llm, mgr, factory_with()).send("u", "s", "记一下买牛奶", create=True)
    h, todos = history(tmp_path)
    roles = [m["role"] for m in h]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert json.loads(h[1]["content"])["action"] == "todo"
    assert h[2]["content"].startswith("Observation (tool=todo, status=ok)") and "buy milk" in h[2]["content"]
    assert INTERRUPTED_NOTE in json.loads(h[3]["content"])["final_answer"]
    assert todos["items"][0]["text"] == "buy milk"

    # 恢复后追问：模型能看到之前的工具调用与 observation
    llm2 = FakeLLM([final("你刚才记了 buy milk")])
    r = ChatService(llm2, mgr, factory_with()).send("u", "s", "我刚才记了什么？")
    assert r.status == OK
    seen = llm2.calls[0]
    assert any(m["content"].startswith("Observation (tool=todo, status=ok)") for m in seen)
    assert [m["role"] for m in seen].count("system") == 1


def test_interrupt_during_tool_execution_records_unknown_outcome(tmp_path):
    def boom(args, ctx):
        raise KeyboardInterrupt

    slow = Tool("slow", "d", {"type": "object", "properties": {}}, boom)
    mgr = SessionManager(tmp_path / "data")
    llm = FakeLLM([act("todo", action="add", text="a"), act("slow")])
    with pytest.raises(KeyboardInterrupt):
        ChatService(llm, mgr, factory_with(slow)).send("u", "s", "go", create=True)
    h, todos = history(tmp_path)
    assert [m["role"] for m in h] == ["user", "assistant", "user", "assistant", "user", "assistant"]
    assert "status=interrupted" in h[4]["content"] and "tool=slow" in h[4]["content"]
    assert len(todos["items"]) == 1
    assert not lock_path(mgr).exists()


def test_interrupt_before_tool_runs_drops_unexecuted_output(tmp_path):
    """模型输出已返回、尚未执行就中断：不把未执行的调用写进历史。"""
    mgr = SessionManager(tmp_path / "data")

    class InterruptInParse(FakeLLM):
        def chat(self, messages):
            out = super().chat(messages)
            if len(self.calls) == 2:
                import agent.loop as loop_mod
                orig = loop_mod.parse_output

                def raise_once(*a, **k):
                    loop_mod.parse_output = orig
                    raise KeyboardInterrupt

                loop_mod.parse_output = raise_once
            return out

    llm = InterruptInParse([act("todo", action="add", text="a"), act("todo", action="add", text="b")])
    with pytest.raises(KeyboardInterrupt):
        ChatService(llm, mgr, factory_with()).send("u", "s", "go", create=True)
    h, todos = history(tmp_path)
    assert [i["text"] for i in todos["items"]] == ["a"]
    assert not any('"b"' in m["content"] for m in h)
    assert [m["role"] for m in h] == ["user", "assistant", "user", "assistant"]


def test_agent_attaches_partial_result_to_interrupt():
    reg = build_registry(todo_store=TodoStore())
    llm = InterruptOnCall([act("todo", action="add", text="x")], at=1)
    with pytest.raises(KeyboardInterrupt) as ei:
        Agent(llm, reg).run("q")
    r = ei.value.agent_result
    assert r.status == "interrupted" and len(r.actions) == 1 and "todo" in r.answer


# ---------------- 歧义输出 ----------------
@pytest.fixture
def reg():
    return build_registry(todo_store=TodoStore())


@pytest.mark.parametrize("text", [
    act("todo", action="add", text="x") + final("done"),
    act("todo", action="add", text="x") + "\n" + act("todo", action="add", text="y"),
    "```json\n" + act("todo", action="add", text="x") + "\n```\n```json\n" + final("ok") + "\n```",
    "first " + final("a") + " then " + final("b"),
])
def test_multiple_json_objects_rejected(reg, text):
    with pytest.raises(ParseError, match="ambiguous"):
        parse_output(text, reg)


def test_single_object_in_fence_with_braces_in_string_still_ok(reg):
    text = "```json\n" + act("todo", action="add", text="a {b} c") + "\n```"
    p = parse_output(text, reg)
    assert p.action == "todo" and p.action_input["text"] == "a {b} c"


@pytest.mark.parametrize("bad_output", [
    "[" + act("todo", action="add", text="in-array") + "]",
    "```json\n[" + act("todo", action="add", text="in-fenced-array") + "]\n```",
    '{"outer": ' + act("todo", action="add", text="in-broken-outer"),
    '{"outer": ' + act("todo", action="add", text="in-wrapped-outer") + "}",
    "Sure, here it is: " + act("todo", action="add", text="in-prose") + " -- thanks",
    "```json\n" + act("todo", action="add", text="unclosed-fence"),
])
def test_wrapped_or_partial_outputs_rejected_and_never_execute_tool(bad_output):
    store = TodoStore()
    reg = build_registry(todo_store=store)
    with pytest.raises(ParseError):
        parse_output(bad_output, reg)
    llm = FakeLLM([bad_output, final("ok")])
    r = Agent(llm, reg).run("q", "u", "s")
    assert r.status == "tool_not_executed" and r.actions == []
    assert store.list(ToolContext("u", "s")) == []
    assert llm.calls[1][-1]["content"].startswith("Format error:")


@pytest.mark.parametrize("deep", [
    '{"a":' * 2000 + "1" + "}" * 2000,
    "[" * 2000 + "]" * 2000,
    '{"action": "todo", "action_input": ' + '{"a":' * 60 + "1" + "}" * 61,
])
def test_deeply_nested_input_triggers_format_correction_not_internal_error(deep):
    store = TodoStore()
    reg = build_registry(todo_store=store)
    with pytest.raises(ParseError):
        parse_output(deep, reg)
    llm = FakeLLM([deep, final("recovered")])
    r = Agent(llm, reg).run("q", "u", "s")
    assert r.status == ("tool_not_executed" if '"action"' in deep else OK)
    if r.status == OK:
        assert r.answer == "recovered"
    assert llm.calls[1][-1]["content"].startswith("Format error:")
    r_exhausted = Agent(FakeLLM([deep] * 4), reg, max_parse_retries=3).run("q", "u", "s")
    assert r_exhausted.status == "parse_retries_exhausted"


def test_ambiguous_output_never_executes_tool():
    store = TodoStore()
    reg = build_registry(todo_store=store)
    llm = FakeLLM([act("todo", action="add", text="x") + final("done"), final("ok")])
    r = Agent(llm, reg).run("q", "u", "s")
    assert r.status == "tool_not_executed" and r.actions == []
    assert store.list(ToolContext("u", "s")) == []
    assert "ambiguous" in llm.calls[1][-1]["content"]
