import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from agent import session as session_mod
from agent.session import (
    InvalidIdError, SessionBusyError, SessionCorruptError, SessionManager, SessionNotFoundError,
    VersionConflictError,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def mgr(tmp_path):
    return SessionManager(tmp_path / "data", lock_timeout=0.3)


def seed(mgr, user="alice", sid="s1", text="hello"):
    with mgr.open(user, sid, create=True) as s:
        s.data.history.append({"role": "user", "content": text})
        s.save()


# ---------- ids & validation ----------
@pytest.mark.parametrize("bad", ["", "../x", "a/b", "a\\b", "a b", "x" * 65, ".", "..", "a.b", None, 5])
def test_invalid_ids_rejected(mgr, bad):
    with pytest.raises(InvalidIdError):
        mgr.exists("alice", bad)
    with pytest.raises(InvalidIdError):
        mgr.exists(bad, "s1")


def test_not_found_without_create(mgr):
    with pytest.raises(SessionNotFoundError):
        with mgr.open("alice", "nope"):
            pass
    assert not mgr.exists("alice", "nope")  # 失败的打开不会落盘


def test_create_is_lazy_until_save(mgr):
    with mgr.open("alice", "s1", create=True) as s:
        assert s.data.version == 0
        assert not mgr.exists("alice", "s1")
        s.save()
    assert mgr.exists("alice", "s1")


def _file(mgr, user="alice", sid="s1"):
    return mgr._path(user, sid)


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(user_id="bob"),
    lambda d: d.update(session_id="zzz"),
    lambda d: d.update(version=-1),
    lambda d: d.update(version="1"),
    lambda d: d.update(version=True),
    lambda d: d.update(next_turn=0),
    lambda d: d.update(history="x"),
    lambda d: d.update(history=[{"role": "system", "content": "x"}]),
    lambda d: d.update(history=[{"role": "user"}]),
    lambda d: d.update(todos={"next": 1}),
    lambda d: d.update(todos={"next": 1, "items": [{"id": 1, "text": "a", "done": False}]}),  # next 必须 > id
    lambda d: d.update(todos={"next": 3, "items": [{"id": 1, "text": "a", "done": False},
                                                  {"id": 1, "text": "b", "done": False}]}),
    lambda d: d.update(todos={"next": 3, "items": [{"id": 1, "text": "a", "done": "no"}]}),
    lambda d: d.pop("created_at"),
])
def test_corrupt_content_detected_and_never_overwritten(mgr, mutate):
    seed(mgr)
    p = _file(mgr)
    d = json.loads(p.read_text(encoding="utf-8"))
    mutate(d)
    p.write_text(json.dumps(d), encoding="utf-8")
    before = p.read_bytes()
    with pytest.raises(SessionCorruptError):
        with mgr.open("alice", "s1", create=True):
            pass
    assert p.read_bytes() == before
    assert not Path(str(p) + ".lock").exists()  # 失败后锁也已释放


@pytest.mark.parametrize("content", ["", "{not json", "[1,2]", "null"])
def test_unparseable_file_is_corrupt(mgr, content):
    seed(mgr)
    _file(mgr).write_text(content, encoding="utf-8")
    with pytest.raises(SessionCorruptError):
        with mgr.open("alice", "s1"):
            pass


def test_invalid_utf8_is_corrupt(mgr):
    seed(mgr)
    _file(mgr).write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(SessionCorruptError):
        with mgr.open("alice", "s1"):
            pass


# ---------- isolation & restart ----------
def test_sessions_isolated_per_user_and_session(mgr):
    seed(mgr, "alice", "s1", "a1")
    seed(mgr, "alice", "s2", "a2")
    seed(mgr, "bob", "s1", "b1")  # 不同用户相同 session ID
    for user, sid, text in (("alice", "s1", "a1"), ("alice", "s2", "a2"), ("bob", "s1", "b1")):
        with mgr.open(user, sid) as s:
            assert [m["content"] for m in s.data.history] == [text]
    assert mgr.list_sessions("alice") == ["s1", "s2"] and mgr.list_sessions("bob") == ["s1"]
    assert mgr.list_sessions("nobody") == []


def test_restart_recovers_all_fields(mgr, tmp_path):
    with mgr.open("alice", "s1", create=True) as s:
        s.data.history.append({"role": "user", "content": "你好"})
        s.data.todos = {"next": 3, "items": [{"id": 2, "text": "写周报", "done": True, "turn": 7}]}
        assert s.next_turn_id() == 1 and s.next_turn_id() == 2
        s.save()
    fresh = SessionManager(tmp_path / "data")
    with fresh.open("alice", "s1") as s:
        assert s.data.version == 1 and s.data.next_turn == 3
        assert s.data.history[0]["content"] == "你好"
        assert s.data.todos["items"][0]["text"] == "写周报" and s.data.todos["next"] == 3


def test_new_session_ids_unique(mgr):
    ids = {mgr.new_session_id("alice") for _ in range(20)}
    assert len(ids) == 20
    sid = mgr.new_session_id("alice")
    assert sid.isalnum()


# ---------- atomic write ----------
def test_write_interruption_keeps_original_and_cleans_tmp(mgr, monkeypatch):
    seed(mgr, text="original")
    p = _file(mgr)
    before = p.read_bytes()

    def boom(src, dst):
        raise OSError("disk crashed")

    monkeypatch.setattr(session_mod.os, "replace", boom)
    with pytest.raises(OSError):
        with mgr.open("alice", "s1") as s:
            s.data.history.append({"role": "user", "content": "new"})
            s.save()
    monkeypatch.undo()
    assert p.read_bytes() == before
    assert [f.name for f in p.parent.iterdir()] == ["s1.json"]  # 无临时文件、无残留锁
    with mgr.open("alice", "s1") as s:  # 之后仍可正常打开
        assert s.data.version == 1


def test_fsync_failure_keeps_original(mgr, monkeypatch):
    seed(mgr)
    before = _file(mgr).read_bytes()
    monkeypatch.setattr(session_mod.os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("fsync")))
    with pytest.raises(OSError):
        with mgr.open("alice", "s1") as s:
            s.save()
    monkeypatch.undo()
    assert _file(mgr).read_bytes() == before


def test_partial_tmp_file_does_not_affect_load(mgr):
    seed(mgr, text="ok")
    p = _file(mgr)
    (p.parent / f".{p.name}.999.deadbeef.tmp").write_text("{trunc", encoding="utf-8")  # 模拟崩溃遗留
    with mgr.open("alice", "s1") as s:
        assert s.data.history[0]["content"] == "ok"
    assert mgr.list_sessions("alice") == ["s1"]


def test_save_increments_version_and_writes_utf8(mgr):
    with mgr.open("alice", "s1", create=True) as s:
        s.data.history.append({"role": "user", "content": "中文"})
        s.save()
        s.save()
        assert s.data.version == 2
    assert "中文" in _file(mgr).read_text(encoding="utf-8")


# ---------- version conflict ----------
def test_version_conflict_detected(mgr):
    seed(mgr)
    with mgr.open("alice", "s1") as s:
        # 模拟锁失效后另一方写入了新版本
        p = _file(mgr)
        d = json.loads(p.read_text(encoding="utf-8"))
        d["version"] += 1
        p.write_text(json.dumps(d), encoding="utf-8")
        s.data.history.append({"role": "user", "content": "lost?"})
        with pytest.raises(VersionConflictError):
            s.save()
    assert json.loads(p.read_text(encoding="utf-8"))["version"] == 2  # 对方的写入未被覆盖


def test_version_conflict_when_file_deleted_and_recreated(mgr):
    with mgr.open("alice", "s1", create=True) as s:
        s.save()
        _file(mgr).unlink()
        with pytest.raises(VersionConflictError):
            s.save()


# ---------- locking ----------
def test_second_open_busy_then_released(mgr):
    seed(mgr)
    with mgr.open("alice", "s1"):
        t0 = time.monotonic()
        with pytest.raises(SessionBusyError):
            with mgr.open("alice", "s1"):
                pass
        assert time.monotonic() - t0 < 2
        with mgr.open("alice", "s2", create=True):  # 其他会话不受影响
            pass
    with mgr.open("alice", "s1"):  # 释放后可再次打开
        pass


def test_lock_released_when_body_raises(mgr):
    seed(mgr)
    with pytest.raises(RuntimeError):
        with mgr.open("alice", "s1"):
            raise RuntimeError("x")
    assert not Path(str(_file(mgr)) + ".lock").exists()
    with mgr.open("alice", "s1"):
        pass


def test_other_manager_instance_blocked_by_file_lock(mgr, tmp_path):
    seed(mgr)
    other = SessionManager(tmp_path / "data", lock_timeout=0.2)  # 不同实例 → 只有文件锁能挡住
    with mgr.open("alice", "s1"):
        with pytest.raises(SessionBusyError):
            with other.open("alice", "s1"):
                pass


def test_stale_lock_is_reclaimed(mgr, tmp_path):
    seed(mgr)
    lock = Path(str(_file(mgr)) + ".lock")
    lock.write_text("{}", encoding="utf-8")
    old = time.time() - 10_000
    os.utime(lock, (old, old))
    with mgr.open("alice", "s1"):
        pass
    fresh = SessionManager(tmp_path / "data", lock_timeout=0.2, stale_after=600)
    lock.write_text("{}", encoding="utf-8")  # 新鲜的锁不会被回收
    with pytest.raises(SessionBusyError):
        with fresh.open("alice", "s1"):
            pass


def test_release_does_not_delete_foreign_lock(mgr):
    seed(mgr)
    lock = Path(str(_file(mgr)) + ".lock")
    with mgr.open("alice", "s1"):
        lock.write_text(json.dumps({"token": "someone-else"}), encoding="utf-8")  # 被他人接管
    assert lock.exists()


def test_waiter_gets_lock_after_holder_finishes(mgr):
    seed(mgr)
    m2 = SessionManager(mgr._root, lock_timeout=3)
    order = []

    def holder():
        with mgr.open("alice", "s1"):
            order.append("h-in")
            time.sleep(0.4)
            order.append("h-out")

    t = threading.Thread(target=holder)
    t.start()
    time.sleep(0.1)
    with m2.open("alice", "s1"):
        order.append("w-in")
    t.join()
    assert order == ["h-in", "h-out", "w-in"]


def test_concurrent_increments_are_serialized(tmp_path):
    root = tmp_path / "data"
    with SessionManager(root).open("u", "s", create=True) as s0:
        s0.save()

    def work():
        m = SessionManager(root, lock_timeout=10)
        for _ in range(5):
            with m.open("u", "s") as s:
                s.data.history.append({"role": "user", "content": "x"})
                s.save()

    threads = [threading.Thread(target=work) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    with SessionManager(root).open("u", "s") as s:
        assert len(s.data.history) == 20 and s.data.version == 21


def test_cross_process_lock(mgr, tmp_path):
    seed(mgr)
    code = (
        "import sys,time;from agent.session import SessionManager;"
        "m=SessionManager(sys.argv[1]);"
        "cm=m.open('alice','s1');cm.__enter__();print('locked',flush=True);time.sleep(3);cm.__exit__(None,None,None)"
    )
    proc = subprocess.Popen([sys.executable, "-c", code, str(tmp_path / "data")], cwd=ROOT,
                            stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "locked"
        with pytest.raises(SessionBusyError):
            with mgr.open("alice", "s1"):
                pass
    finally:
        proc.wait(timeout=15)
    with mgr.open("alice", "s1"):  # 子进程退出后释放
        pass
