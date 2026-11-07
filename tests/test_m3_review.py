"""M3 评审修复的回归测试：活跃锁不被误回收、中断后历史完整、歧义多 JSON 输出被拒绝。"""
import json
import threading
import time
from pathlib import Path

import pytest

from agent.llm import FakeLLM
from agent.loop import OK, Agent
from agent.service import INTERRUPTED_NOTE, ChatService
from agent.session import LockLostError, SessionBusyError, SessionManager
from agent.tools import ToolContext, TodoStore, build_registry
from agent.tools.registry import Tool


def act(tool, **args):
    """构造原生工具调用测试响应。"""
    return {"content": "", "tool_calls": [{"id": "fixture_call", "type": "function", "function": {"name": tool, "arguments": json.dumps(args, ensure_ascii=False)}}]}


def final(text):
    """构造直接文本回答。"""
    return text


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

    def complete(self, messages, **kwargs):
        if len(self.calls) == self.at:
            self.calls.append(messages)
            raise KeyboardInterrupt
        return super().complete(messages, **kwargs)


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
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert h[1]["tool_calls"][0]["function"]["name"] == "todo"
    assert h[2]["role"] == "tool" and "buy milk" in h[2]["content"]
    assert INTERRUPTED_NOTE in h[3]["content"]
    assert todos["items"][0]["text"] == "buy milk"

    # 恢复后追问：模型能看到之前的工具调用与 observation
    llm2 = FakeLLM([final("你刚才记了 buy milk")])
    r = ChatService(llm2, mgr, factory_with()).send("u", "s", "我刚才记了什么？")
    assert r.status == OK
    seen = llm2.calls[0]
    assert any(m["role"] == "tool" and m.get("name") == "todo" for m in seen)
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
    assert [m["role"] for m in h] == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert "status=interrupted" in h[4]["content"] and "tool=slow" in h[4]["content"]
    assert len(todos["items"]) == 1
    assert not lock_path(mgr).exists()


def test_interrupt_before_tool_runs_drops_unexecuted_output(tmp_path):
    """模型输出已返回、尚未执行就中断：不把未执行的调用写进历史。"""
    mgr = SessionManager(tmp_path / "data")

    class InterruptInParse(FakeLLM):
        def complete(self, messages, **kwargs):
            out = super().complete(messages, **kwargs)
            if len(self.calls) == 2:
                import agent.loop as loop_mod
                orig = loop_mod._Turn.prepare_reply

                def raise_once(*a, **k):
                    loop_mod._Turn.prepare_reply = orig
                    raise KeyboardInterrupt

                loop_mod._Turn.prepare_reply = raise_once
            return out

    llm = InterruptInParse([act("todo", action="add", text="a"), act("todo", action="add", text="b")])
    with pytest.raises(KeyboardInterrupt):
        ChatService(llm, mgr, factory_with()).send("u", "s", "go", create=True)
    h, todos = history(tmp_path)
    assert [i["text"] for i in todos["items"]] == ["a"]
    assert not any('"b"' in m["content"] for m in h)
    assert [m["role"] for m in h] == ["user", "assistant", "tool", "assistant"]


def test_agent_attaches_partial_result_to_interrupt():
    reg = build_registry(todo_store=TodoStore())
    llm = InterruptOnCall([act("todo", action="add", text="x")], at=1)
    with pytest.raises(KeyboardInterrupt) as ei:
        Agent(llm, reg).run("q")
    r = ei.value.agent_result
    assert r.status == "interrupted" and len(r.actions) == 1 and "todo" in r.answer
