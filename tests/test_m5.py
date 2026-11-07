"""M5 阶段测试：工具调用记录与 call_id、写操作立即原子落盘、幂等与中断恢复、异常退出与 JSONL Trace。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import pytest

from agent.config import ConfigError, Settings
from agent.llm import FakeLLM, LLMError, Message, OpenAIClient
from agent.loop import (
    CONTEXT_LIMIT,
     INTERNAL_ERROR,
    INTERRUPTED,
    LLM_FAILED,
    MAX_STEPS,
    OK,
    OUTPUT_EXHAUSTED,
    PERSIST_FAILED,
    TOOL_NOT_EXECUTED,
    Agent,
)
from agent.service import ChatService
from agent.session import SessionCorruptError, SessionManager
from agent.tools import ToolContext, TodoStore, build_registry
from agent.trace import (
    TraceLogger,
    find_unfinished_in_events,
    is_write_tool,
    make_call_id,
    reconcile_todo_call,
    verify_trace_integrity,
)
import cli


class ScriptedLLM(FakeLLM):
    """使用原生响应和可调用步骤的测试模型。"""



def _act(tool: str, thought: str = "t", **kwargs) -> str:
    """构造原生工具调用测试响应。"""
    return {"content": thought, "tool_calls": [{"id": "fixture_call", "type": "function", "function": {"name": tool, "arguments": json.dumps(kwargs, ensure_ascii=False)}}]}


def _fin(text: str, thought: str = "done") -> str:
    """构造直接文本回答。"""
    return text


def _make_service(tmp_path: Path, llm, **kwargs) -> tuple[SessionManager, ChatService, TraceLogger]:
    """创建绑定到临时目录的 SessionManager、TraceLogger 与 ChatService。"""
    mgr = SessionManager(tmp_path / "sessions")
    trace = TraceLogger(tmp_path / "traces")
    svc = ChatService(
        llm,
        mgr,
        lambda store: build_registry(todo_store=store),
        trace_logger=trace,
        **kwargs,
    )
    return mgr, svc, trace


# ---------------------------------------------------------------------------
# 1. 调用记录与标识（call_id 稳定性、持久化字段与校验）
# ---------------------------------------------------------------------------


def test_call_id_deterministic_and_scoped():
    """同一 (user, session, turn_id, tool, args) 生成相同 call_id；首尾空白规范化；不同轮次生成不同 call_id。"""
    c1 = make_call_id("u1", "s1", 1, "todo", {"action": "add", "text": " 买牛奶 "})
    c2 = make_call_id("u1", "s1", 1, "todo", {"action": "add", "text": "买牛奶"})
    c_other_turn = make_call_id("u1", "s1", 2, "todo", {"action": "add", "text": "买牛奶"})
    c_other_session = make_call_id("u1", "s2", 1, "todo", {"action": "add", "text": "买牛奶"})
    assert c1 == c2
    assert c1 != c_other_turn
    assert c1 != c_other_session
    assert is_write_tool("todo", {"action": "add", "text": "x"}) is True
    assert is_write_tool("todo", {"action": "complete", "id": 1}) is True
    assert is_write_tool("todo", {"action": "delete", "id": 1}) is True
    assert is_write_tool("todo", {"action": "list"}) is False
    assert is_write_tool("calculator", {"expression": "1+1"}) is False


def test_tool_call_records_persisted_with_session(tmp_path: Path):
    """会话持久化文件保存工具名、参数、call_id、user_id、session_id、turn_id、状态、结果与耗时。"""
    llm = ScriptedLLM([
        _act("todo", action="add", text="买牛奶"),
        _act("calculator", expression="6 * 7"),
        _fin("已添加待办并算完 42"),
    ])
    mgr, svc, _ = _make_service(tmp_path, llm)
    res = svc.send("alice", "s1", "帮我记买牛奶并算 6*7", create=True)
    assert res.status == OK
    assert len(res.actions) == 2
    assert all(a.call_id for a in res.actions)

    with mgr.open("alice", "s1") as s:
        calls = s.data.tool_calls
        assert len(calls) == 2
        assert calls[0]["call_id"] == res.actions[0].call_id
        assert calls[0]["user_id"] == "alice"
        assert calls[0]["session_id"] == "s1"
        assert calls[0]["turn_id"] == 1
        assert calls[0]["tool"] == "todo"
        assert calls[0]["args"] == {"action": "add", "text": "买牛奶"}
        assert calls[0]["status"] == "ok"
        assert '"id": 1' in calls[0]["result"]
        assert isinstance(calls[0]["duration_ms"], (int, float))
        assert calls[1]["tool"] == "calculator"
        assert calls[1]["status"] == "ok"
        assert s.data.unfinished_turn is None


def test_corrupt_tool_calls_or_unfinished_turn_rejected(tmp_path: Path):
    """损坏的 tool_calls 或 unfinished_turn 字段在加载时被识别为 SessionCorruptError。"""
    llm = ScriptedLLM([_fin("ok")])
    mgr, svc, _ = _make_service(tmp_path, llm)
    svc.send("u", "s1", "hi", create=True)
    path = tmp_path / "sessions" / "u" / "s1.json"
    raw = json.loads(path.read_text(encoding="utf-8"))

    bad_calls = {**raw, "tool_calls": [{"call_id": "", "tool": "todo", "args": {}, "status": "unknown"}]}
    path.write_text(json.dumps(bad_calls), encoding="utf-8")
    with pytest.raises(SessionCorruptError):
        with mgr.open("u", "s1"):
            pass

    bad_unfinished = {**raw, "unfinished_turn": {"turn_id": True, "status": "in_progress"}}
    path.write_text(json.dumps(bad_unfinished), encoding="utf-8")
    with pytest.raises(SessionCorruptError):
        with mgr.open("u", "s1"):
            pass


# ---------------------------------------------------------------------------
# 2. 写操作立即落盘（add / complete / delete 与调用记录同一次原子保存）
# ---------------------------------------------------------------------------


def test_write_operations_persist_immediately_before_turn_ends(tmp_path: Path):
    """待办 add/complete/delete 执行后立即落盘，不等整轮结束；待办状态与成功调用记录在同一原子保存中写入。"""
    session_file = tmp_path / "sessions" / "u1" / "imm.json"
    snapshots_on_disk: list[dict] = []

    def step2_after_add(_messages):
        """在第二次 LLM 调用时检查磁盘：此时本轮尚未结束，但 add 必须已落盘。"""
        snapshots_on_disk.append(json.loads(session_file.read_text(encoding="utf-8")))
        return _act("todo", action="complete", id=1)

    def step3_after_complete(_messages):
        """在第三次 LLM 调用时检查磁盘：complete 必须已落盘。"""
        snapshots_on_disk.append(json.loads(session_file.read_text(encoding="utf-8")))
        return _act("todo", action="delete", id=1)

    def step4_after_delete(_messages):
        """在第四次 LLM 调用时检查磁盘：delete 必须已落盘。"""
        snapshots_on_disk.append(json.loads(session_file.read_text(encoding="utf-8")))
        return _fin("全部完成")

    llm = ScriptedLLM([
        _act("todo", action="add", text="立即落盘测试"),
        step2_after_add,
        step3_after_complete,
        step4_after_delete,
    ])
    _, svc, _ = _make_service(tmp_path, llm)
    res = svc.send("u1", "imm", "测试写操作立即保存", create=True)
    assert res.status == OK
    assert len(snapshots_on_disk) == 3

    # 1) add 之后、整轮未结束时的磁盘快照：version=1，todos 已有 #1，tool_calls 已有 1 条 ok 记录
    s_add = snapshots_on_disk[0]
    assert s_add["version"] == 1
    assert s_add["todos"]["items"] == [{"id": 1, "text": "立即落盘测试", "done": False, "turn": 1}]
    assert len(s_add["tool_calls"]) == 1
    assert s_add["tool_calls"][0]["status"] == "ok"
    assert s_add["tool_calls"][0]["args"] == {"action": "add", "text": "立即落盘测试"}
    assert s_add["unfinished_turn"]["status"] == "in_progress"

    # 2) complete 之后的磁盘快照：version=2，todos #1 为 done=True，tool_calls 有 2 条 ok 记录
    s_comp = snapshots_on_disk[1]
    assert s_comp["version"] == 2
    assert s_comp["todos"]["items"][0]["done"] is True
    assert len(s_comp["tool_calls"]) == 2
    assert s_comp["tool_calls"][1]["status"] == "ok"
    assert s_comp["tool_calls"][1]["args"] == {"action": "complete", "id": 1}

    # 3) delete 之后的磁盘快照：version=3，todos items 为空，tool_calls 有 3 条 ok 记录
    s_del = snapshots_on_disk[2]
    assert s_del["version"] == 3
    assert s_del["todos"]["items"] == []
    assert len(s_del["tool_calls"]) == 3
    assert s_del["tool_calls"][2]["status"] == "ok"
    assert s_del["tool_calls"][2]["args"] == {"action": "delete", "id": 1}

    # 整轮正常结束保存后 version=4，unfinished_turn 清空
    final_disk = json.loads(session_file.read_text(encoding="utf-8"))
    assert final_disk["version"] == 4
    assert final_disk["unfinished_turn"] is None


# ---------------------------------------------------------------------------
# 3. 幂等与中断恢复（重点验收场景）
# ---------------------------------------------------------------------------


def test_add_todo_then_llm_fails_resume_does_not_duplicate_and_next_turn_can_add_same_name(tmp_path: Path):
    """重点验收 1：添加待办后模型请求失败，恢复重试时复用 call_id 不重复添加；完成后下一轮仍可创建同名待办。"""
    llm = ScriptedLLM([
        # 第 1 次尝试：add 成功后，第二次调模型抛出 LLMError
        _act("todo", action="add", text="买牛奶"),
        LLMError("503 upstream timeout"),
        # 恢复重试：模型再次发出相同的 add("买牛奶")，应直接复用原 call_id 与结果，不重复创建
        _act("todo", action="add", text="买牛奶"),
        _fin("已为您添加待办：买牛奶（#1）"),
        # 第 2 轮（全新轮次）：用户再次要求添加同名待办“买牛奶”，应正常创建 #2
        _act("todo", action="add", text="买牛奶"),
        _fin("已再次添加同名待办：买牛奶（#2）"),
    ])
    mgr, svc, _ = _make_service(tmp_path, llm)

    r1 = svc.send("u1", "s1", "帮我记个待办：买牛奶", create=True)
    assert r1.status == LLM_FAILED
    assert "本轮已执行的操作" in r1.answer
    assert "未完成说明" in r1.answer

    # 恢复会话重试
    r_retry = svc.send("u1", "s1", "继续刚才的请求：帮我记个待办买牛奶")
    assert r_retry.status == OK
    assert r_retry.turn_id == r1.turn_id  # 复用未完成轮次 ID
    assert r_retry.actions[0].call_id == r1.actions[0].call_id  # 复用原 call_id
    assert r_retry.actions[0].reused is True
    assert '"created": false' in r_retry.actions[0].result

    with mgr.open("u1", "s1") as s:
        assert len(s.data.todos["items"]) == 1
        assert s.data.todos["items"][0]["id"] == 1
        assert s.data.todos["items"][0]["text"] == "买牛奶"
        assert len(s.data.tool_calls) == 1  # 同一个 call_id 只存一条记录

    # 下一轮正常对话：应分配新 turn_id，允许创建同名待办 #2
    r_next = svc.send("u1", "s1", "再记一条买牛奶")
    assert r_next.status == OK
    assert r_next.turn_id != r1.turn_id
    assert r_next.actions[0].reused is False
    assert '"created": true' in r_next.actions[0].result

    with mgr.open("u1", "s1") as s:
        items = s.data.todos["items"]
        assert len(items) == 2
        assert [it["id"] for it in items] == [1, 2]
        assert [it["text"] for it in items] == ["买牛奶", "买牛奶"]


def test_new_request_after_failed_turn_is_not_mistaken_for_retry(tmp_path: Path):
    """回归 P1-1：添加待办后模型失败，若下一次输入是新建请求（如“另外再创建一条同名待办”），不得误判为重试去重。"""
    llm = ScriptedLLM([
        # 第 1 轮：添加“买牛奶”成功（#1），随后第二次模型调用失败，留下 unfinished_turn
        _act("todo", action="add", text="买牛奶"),
        LLMError("503 upstream timeout"),
        # 第 2 轮：用户未选择重试旧请求，而是明确要求“另外再创建一条同名待办：买牛奶”
        _act("todo", action="add", text="买牛奶"),
        _fin("已为您另外创建第二条同名待办：买牛奶（#2）"),
    ])
    mgr, svc, trace = _make_service(tmp_path, llm)

    r1 = svc.send("u1", "s_new_after_fail", "帮我记个待办：买牛奶", create=True)
    assert r1.status == LLM_FAILED

    r2 = svc.send("u1", "s_new_after_fail", "另外再创建一条同名待办：买牛奶")
    assert r2.status == OK
    assert r2.turn_id != r1.turn_id  # 识别为新请求，分配新 turn_id
    assert r2.actions[0].call_id != r1.actions[0].call_id
    assert r2.actions[0].reused is False
    assert '"created": true' in r2.actions[0].result

    with mgr.open("u1", "s_new_after_fail") as s:
        items = s.data.todos["items"]
        assert len(items) == 2
        assert [it["id"] for it in items] == [1, 2]
        assert [it["text"] for it in items] == ["买牛奶", "买牛奶"]
    assert verify_trace_integrity(trace.read_events("u1", "s_new_after_fail"))["valid"] is True


def test_final_save_failure_returns_persist_error_and_logs_accurate_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """回归 P1-2：写操作即时保存成功后，若整轮结束时最终保存失败，统一返回 persist_error，且 Trace 记录 persist_error 而非 ok。"""
    llm = ScriptedLLM([
        _act("todo", action="add", text="即时落盘已成功"),
        _fin("原本准备正常返回的回复"),
    ])
    mgr, svc, trace = _make_service(tmp_path, llm)

    real_replace = os.replace
    replace_count = 0

    def fail_on_final_save(src, dst):
        """第 1 次替换（写操作即时保存）成功，第 2 次替换（整轮收尾保存）抛 OSError。"""
        nonlocal replace_count
        replace_count += 1
        if replace_count == 2:
            raise OSError("simulated disk error on final turn save")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_on_final_save)
    res = svc.send("u1", "final_save_fail", "记一条待办", create=True)
    assert res.status == PERSIST_FAILED
    assert "保存写操作失败" in res.answer
    assert "即时落盘已成功" in res.answer
    assert "未完成说明" in res.answer

    # 恢复 os.replace 检查磁盘：即时保存的待办完好保留，unfinished_turn 仍为 in_progress
    monkeypatch.setattr(os, "replace", real_replace)
    with mgr.open("u1", "final_save_fail") as s:
        assert s.data.version == 1
        assert [it["text"] for it in s.data.todos["items"]] == ["即时落盘已成功"]
        assert s.data.unfinished_turn is not None
        assert s.data.unfinished_turn["status"] == "in_progress"

    # 检查 Trace：turn_end 必须反映真实的 persist_error 状态，绝不能提前记为 ok
    events = trace.read_events("u1", "final_save_fail")
    turn_ends = [e for e in events if e.get("event") == "turn_end"]
    assert len(turn_ends) == 1
    assert turn_ends[0]["status"] == PERSIST_FAILED
    assert turn_ends[0]["error"]
    assert verify_trace_integrity(events)["valid"] is True


def test_trace_integrity_passes_after_retry_recovery_with_attempt_id(tmp_path: Path):
    """回归 P2-3：正常恢复重试复用原 turn_id 与稳定 call_id，并通过 attempt_id 区分每次尝试，Trace 完整性检查通过。"""
    llm = ScriptedLLM([
        # 第 1 次尝试（attempt_id=1）：step 1 添加待办后模型报错
        _act("todo", action="add", text="买牛奶"),
        LLMError("temporary failure"),
        # 第 2 次尝试（attempt_id=2，复用同一个 turn_id）：step 1 幂等复用待办，step 2 计算器，然后完成
        _act("todo", action="add", text="买牛奶"),
        _act("calculator", expression="10 + 20"),
        _fin("恢复执行完毕"),
    ])
    _, svc, trace = _make_service(tmp_path, llm)

    r1 = svc.send("u1", "retry_trace", "记买牛奶并算 10+20", create=True)
    assert r1.status == LLM_FAILED
    assert r1.attempt_id == 1

    r2 = svc.send("u1", "retry_trace", "继续刚才未完成的请求")
    assert r2.status == OK
    assert r2.turn_id == r1.turn_id
    assert r2.attempt_id == 2
    assert r2.actions[0].call_id == r1.actions[0].call_id
    assert r2.actions[0].reused is True

    events = trace.read_events("u1", "retry_trace")
    report = verify_trace_integrity(events)
    assert report["valid"] is True
    assert report["paired"] is True
    assert report["contiguous_steps"] is True
    assert report["valid_ends"] is True
    assert report["unfinished_calls"] == []


def test_hard_crash_right_after_write_save_recovers_on_restart(tmp_path: Path):
    """重点验收 2：写操作立即保存后进程硬中断（未走收尾保存），重启后仍能恢复待办状态与执行结果。"""
    session_file = tmp_path / "sessions" / "u1" / "crash.json"
    mid_turn_bytes: bytes = b""

    def crash_after_immediate_save(_messages):
        """读取写操作立即落盘后的磁盘字节，模拟此时进程被 SIGKILL/断电终止。"""
        nonlocal mid_turn_bytes
        mid_turn_bytes = session_file.read_bytes()
        raise KeyboardInterrupt("simulated crash after write")

    llm1 = ScriptedLLM([
        _act("todo", action="add", text="交水电费"),
        crash_after_immediate_save,
    ])
    _, svc1, _ = _make_service(tmp_path, llm1)
    with pytest.raises(KeyboardInterrupt):
        svc1.send("u1", "crash", "记一下交水电费", create=True)

    # 将磁盘文件还原为写操作刚落盘、进程突然死亡（连 except BaseException 都没跑完）的瞬间状态
    assert mid_turn_bytes
    session_file.write_bytes(mid_turn_bytes)

    # 模拟进程重启：新建 SessionManager 与 ChatService 恢复该会话
    llm2 = ScriptedLLM([
        _act("todo", action="add", text="交水电费"),
        _fin("已恢复并确认待办：交水电费（#1）"),
    ])
    mgr2, svc2, _ = _make_service(tmp_path, llm2)
    res = svc2.send("u1", "crash", "继续记一下交水电费")
    assert res.status == OK
    assert res.actions[0].reused is True
    assert '"id": 1' in res.actions[0].result

    with mgr2.open("u1", "crash") as s:
        assert s.data.todos["items"] == [{"id": 1, "text": "交水电费", "done": False, "turn": 1}]
        assert s.data.tool_calls[0]["status"] == "ok"
        assert s.data.unfinished_turn is None


def test_uncertain_call_status_reconciles_against_todo_store(tmp_path: Path):
    """执行状态不确定（pending/interrupted）时先核对存储：已生效则直接复用结果，未生效则复用 call_id 重新执行。"""
    llm_init = ScriptedLLM([_fin("初始化会话")])
    mgr, svc, _ = _make_service(tmp_path, llm_init)
    svc.send("u1", "uncertain", "你好", create=True)

    # 情况 A：调用记录处于 interrupted（不确定状态），但 todos 存储中其实已经写入了该待办
    cid_applied = make_call_id("u1", "uncertain", 2, "todo", {"action": "add", "text": "已生效待办"})
    # 情况 B：调用记录处于 pending（不确定状态），且 todos 存储中尚未写入该待办
    cid_not_applied = make_call_id("u1", "uncertain", 2, "todo", {"action": "add", "text": "未生效待办"})

    with mgr.open("u1", "uncertain") as s:
        s.data.next_turn = 3
        s.data.unfinished_turn = {"turn_id": 2, "prompt": "添加两条待办", "status": "interrupted"}
        s.data.todos = {
            "next": 2,
            "items": [{"id": 1, "text": "已生效待办", "done": False, "turn": 2}],
        }
        s.data.tool_calls = [
            {
                "call_id": cid_applied,
                "user_id": "u1",
                "session_id": "uncertain",
                "turn_id": 2,
                "step": 1,
                "tool": "todo",
                "args": {"action": "add", "text": "已生效待办"},
                "status": "interrupted",
                "result": "",
                "error": "interrupted during tool execution",
                "duration_ms": 0.0,
            },
            {
                "call_id": cid_not_applied,
                "user_id": "u1",
                "session_id": "uncertain",
                "turn_id": 2,
                "step": 2,
                "tool": "todo",
                "args": {"action": "add", "text": "未生效待办"},
                "status": "pending",
                "result": "",
                "error": None,
                "duration_ms": 0.0,
            },
        ]
        assert len(s.data.unfinished_calls()) == 2
        s.save()

    llm_recover = ScriptedLLM([
        _act("todo", action="add", text="已生效待办"),
        _act("todo", action="add", text="未生效待办"),
        _fin("两条待办均已妥善处理"),
    ])
    mgr2, svc2, _ = _make_service(tmp_path, llm_recover)
    res = svc2.send("u1", "uncertain", "继续添加这两条待办")
    assert res.status == OK
    # 第 1 条已生效：核对后直接复用，不重复创建
    assert res.actions[0].call_id == cid_applied
    assert res.actions[0].reused is True
    assert '"id": 1' in res.actions[0].result
    # 第 2 条未生效：核对后复用原 call_id 真正执行，创建 #2
    assert res.actions[1].call_id == cid_not_applied
    assert res.actions[1].reused is False
    assert '"id": 2' in res.actions[1].result

    with mgr2.open("u1", "uncertain") as s:
        assert [it["text"] for it in s.data.todos["items"]] == ["已生效待办", "未生效待办"]
        assert s.data.unfinished_calls() == []
        assert all(c["status"] == "ok" for c in s.data.tool_calls)


def test_reconcile_todo_complete_and_delete_states():
    """核对 complete 与 delete 在已生效和未生效两种状态下的返回值。"""
    todos_done = {"next": 3, "items": [{"id": 1, "text": "a", "done": True, "turn": 1}]}
    todos_pending = {"next": 3, "items": [{"id": 1, "text": "a", "done": False, "turn": 1}]}

    rec_comp = {"tool": "todo", "args": {"action": "complete", "id": 1}, "status": "interrupted", "result": ""}
    assert reconcile_todo_call(rec_comp, todos_done, 1) is not None
    assert reconcile_todo_call(rec_comp, todos_pending, 1) is None

    rec_del = {"tool": "todo", "args": {"action": "delete", "id": 2}, "status": "interrupted", "result": ""}
    # id=2 < next=3 且已不在 items 中 -> 说明已删除成功
    assert reconcile_todo_call(rec_del, todos_done, 1) is not None
    # id=1 仍在 items 中 -> 说明删除未生效
    rec_del_1 = {"tool": "todo", "args": {"action": "delete", "id": 1}, "status": "interrupted", "result": ""}
    assert reconcile_todo_call(rec_del_1, todos_done, 1) is None


# ---------------------------------------------------------------------------
# 4. 写入失败回滚、各类异常退出统一说明与 Trace 完整性
# ---------------------------------------------------------------------------


def test_write_persist_failure_rolls_back_uncommitted_change_and_keeps_earlier_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """写操作落盘失败（OSError）时：回滚本次未落盘的内存修改，保留此前已落盘的步骤，并返回 persist_error。"""
    llm = ScriptedLLM([
        _act("todo", action="add", text="第一条成功落盘"),
        _act("todo", action="add", text="第二条落盘失败"),
        _fin("不应该走到这里"),
    ])
    mgr, svc, trace = _make_service(tmp_path, llm)

    real_replace = os.replace
    replace_count = 0

    def flaky_replace(src, dst):
        """第 1 次写操作原子替换放行，第 2 次抛出磁盘写入失败。"""
        nonlocal replace_count
        replace_count += 1
        if replace_count >= 2:
            raise OSError("simulated disk full during atomic replace")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky_replace)
    res = svc.send("u1", "disk_fail", "添加两条待办", create=True)
    assert res.status == PERSIST_FAILED
    assert "保存写操作失败" in res.answer
    assert "第一条成功落盘" in res.answer
    assert "未完成说明" in res.answer
    assert len(res.actions) == 1  # 第 2 条因落盘失败已回滚

    # 恢复 os.replace 后检查磁盘：第一条完好保留，第二条未污染磁盘
    monkeypatch.setattr(os, "replace", real_replace)
    with mgr.open("u1", "disk_fail") as s:
        assert s.data.version == 1
        assert [it["text"] for it in s.data.todos["items"]] == ["第一条成功落盘"]
        assert len(s.data.tool_calls) == 1
        assert s.data.tool_calls[0]["status"] == "ok"

    # Trace 依然完整成对记录了第 2 步的 request 与 error result，以及 turn_end(persist_error)
    events = trace.read_events("u1", "disk_fail")
    report = verify_trace_integrity(events)
    assert report["valid"] is True
    assert events[-1]["event"] == "turn_end"
    assert events[-1]["status"] == PERSIST_FAILED


def test_trace_integrity_across_normal_and_all_failure_paths(tmp_path: Path):
    """验证正常多步、工具异常恢复、模型失败、超步数、格式耗尽、未执行工具、上下文超限与中断的 Trace 完整性。"""
    trace = TraceLogger(tmp_path / "traces")
    mgr = SessionManager(tmp_path / "sessions")

    # 1) 正常多步 + 工具参数错误后纠正
    llm_ok = ScriptedLLM([
        _act("calculator", expression="1 / 0"),  # 工具报错
        _act("calculator", expression="2 + 3"),  # 纠正后成功
        _fin("结果是 5"),
    ])
    svc_ok = ChatService(llm_ok, mgr, lambda s: build_registry(todo_store=s), trace_logger=trace)
    r_ok = svc_ok.send("u", "s_ok", "算一下", create=True)
    assert r_ok.status == OK
    assert verify_trace_integrity(trace.read_events("u", "s_ok"))["valid"] is True

    # 2) max_steps
    llm_max = ScriptedLLM([_act("calculator", expression="1+1"), _act("calculator", expression="2+2")])
    svc_max = ChatService(llm_max, mgr, lambda s: build_registry(todo_store=s), max_steps=1, trace_logger=trace)
    r_max = svc_max.send("u", "s_max", "算两步", create=True)
    assert r_max.status == MAX_STEPS
    assert "未完成说明" in r_max.answer
    assert verify_trace_integrity(trace.read_events("u", "s_max"))["valid"] is True

    # 3) output_retries_exhausted
    llm_parse = ScriptedLLM(["", ""])
    svc_parse = ChatService(
        llm_parse, mgr, lambda s: build_registry(todo_store=s), max_output_retries=1, trace_logger=trace,
    )
    r_parse = svc_parse.send("u", "s_parse", "胡言乱语", create=True)
    assert r_parse.status == OUTPUT_EXHAUSTED
    assert "未完成说明" in r_parse.answer
    ev_parse = trace.read_events("u", "s_parse")
    assert any(e["event"] == "validation_error" for e in ev_parse)
    assert verify_trace_integrity(ev_parse)["valid"] is True

    # 4) tool_not_executed（混入伪造工具输出后直接给 final_answer）
    llm_unexec = ScriptedLLM([
        {"tool_calls": [{"id": "bad", "name": "todo", "arguments": "{broken"}]},
        _fin("我已经帮你添加好了"),
    ])
    svc_unexec = ChatService(llm_unexec, mgr, lambda s: build_registry(todo_store=s), trace_logger=trace, max_output_retries=1)
    r_unexec = svc_unexec.send("u", "s_unexec", "加待办", create=True)
    assert r_unexec.status == TOOL_NOT_EXECUTED
    assert "未完成说明" in r_unexec.answer
    assert verify_trace_integrity(trace.read_events("u", "s_unexec"))["valid"] is True

    # 5) context_limit
    svc_ctx = ChatService(
        ScriptedLLM([_fin("ok")]), mgr, lambda s: build_registry(todo_store=s),
        max_context_tokens=50, trace_logger=trace,
    )
    r_ctx = svc_ctx.send("u", "s_ctx", "超长输入" * 50, create=True)
    assert r_ctx.status == CONTEXT_LIMIT
    assert "未完成说明" in r_ctx.answer
    assert verify_trace_integrity(trace.read_events("u", "s_ctx"))["valid"] is True

    # 6) internal_error（LLM 抛出未预期 RuntimeError）
    llm_internal = ScriptedLLM([RuntimeError("unexpected bug")])
    svc_internal = ChatService(llm_internal, mgr, lambda s: build_registry(todo_store=s), trace_logger=trace)
    r_internal = svc_internal.send("u", "s_internal", "触发内部错误", create=True)
    assert r_internal.status == INTERNAL_ERROR
    assert "未完成说明" in r_internal.answer
    assert verify_trace_integrity(trace.read_events("u", "s_internal"))["valid"] is True


def test_interrupted_tool_call_recorded_and_identifiable_in_trace(tmp_path: Path):
    """重点验收 3：工具执行中途被中断时，Trace 与会话记录均能识别出未完成调用，且请求/结果成对闭合。"""
    trace = TraceLogger(tmp_path / "traces")
    mgr = SessionManager(tmp_path / "sessions")
    store = TodoStore()
    reg = build_registry(todo_store=store)

    @reg.tool("slow_op", "Operation that gets interrupted", {"type": "object", "properties": {}, "required": []})
    def slow_op(_args: dict, _ctx: ToolContext):
        """模拟执行中途收到 KeyboardInterrupt。"""
        raise KeyboardInterrupt("user pressed Ctrl+C during tool execution")

    llm = ScriptedLLM([
        _act("todo", action="add", text="先完成的第一步"),
        _act("slow_op"),
    ])
    svc = ChatService(llm, mgr, lambda _s: reg, trace_logger=trace)
    with pytest.raises(KeyboardInterrupt):
        svc.send("u1", "s_int", "先加待办再执行耗时操作", create=True)

    # 检查 Trace：step 1 (ok) 与 step 2 (interrupted) 请求/结果均成对，且能识别出 step 2 为未完成调用
    events = trace.read_events("u1", "s_int")
    integrity = verify_trace_integrity(events)
    assert integrity["valid"] is True
    unfinished = trace.find_unfinished_calls("u1", "s_int")
    assert len(unfinished) == 1
    assert unfinished[0]["tool"] == "slow_op"
    assert unfinished[0]["status"] == "interrupted"

    # 同时模拟进程在写完 tool_request 后直接硬崩溃（连 tool_result 都没写入）
    trace.log_tool_request("u1", "s_crash_raw", 1, 1, "call-1-todo-abc", "t", "todo", {"action": "add", "text": "x"})
    raw_unfinished = trace.find_unfinished_calls("u1", "s_crash_raw")
    assert len(raw_unfinished) == 1
    assert raw_unfinished[0]["call_id"] == "call-1-todo-abc"
    assert find_unfinished_in_events(trace.read_events("u1", "s_crash_raw"))[0]["tool"] == "todo"


def test_cli_trace_command_displays_session_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    """CLI --trace 可查看指定会话的完整 Trace 日志（含请求、结果、耗时与结束状态）。"""
    data_dir = tmp_path / "data"
    llm = ScriptedLLM([
        _act("todo", action="add", text="CLI Trace 待办"),
        _fin("添加完毕"),
    ])
    mgr = SessionManager(data_dir)
    trace = TraceLogger(data_dir / "traces")
    svc = ChatService(llm, mgr, lambda s: build_registry(todo_store=s), trace_logger=trace)
    svc.send("alice", "s_cli", "测试 CLI Trace", create=True)

    # 缺少 -s 时报错返回 1
    code_no_s = cli.main(["--data-dir", str(data_dir), "-u", "alice", "--trace"])
    assert code_no_s == 1
    assert "--trace requires -s/--session" in capsys.readouterr().err

    # 正常查看 Trace
    code_ok = cli.main(["--data-dir", str(data_dir), "-u", "alice", "-s", "s_cli", "--trace"])
    out = capsys.readouterr().out
    assert code_ok == 0
    assert "[turn 1] START" in out
    assert "REQUEST call_id=call-1-todo-" in out
    assert "RESULT  call_id=call-1-todo-" in out
    assert "[turn 1] END status=ok steps=1" in out


# ---------------------------------------------------------------------------
# 5. 真实 API 核心流程验证（-m live）
# ---------------------------------------------------------------------------


def _live_client_or_skip() -> OpenAIClient:
    """加载真实模型客户端，未配置时跳过。"""
    try:
        s = Settings.from_env()
    except ConfigError as e:
        pytest.skip(f"LLM env not configured: {e}")
    return OpenAIClient(s)


@pytest.mark.live
def test_live_m5_write_persistence_and_trace(tmp_path: Path):
    """真实 API 验收：创建并完成待办，验证 tool_calls 与 todos 落盘且 JSONL Trace 完整有效。"""
    client = _live_client_or_skip()
    mgr, svc, trace = _make_service(tmp_path, client)

    r1 = svc.send("live_u", "live_s", "请帮我添加一条待办：提交M5验收报告", create=True)
    assert r1.status == OK
    assert any(a.tool == "todo" and a.ok for a in r1.actions)

    r2 = svc.send("live_u", "live_s", "把刚才那条待办标记为完成")
    assert r2.status == OK

    with mgr.open("live_u", "live_s") as s:
        assert len(s.data.todos["items"]) == 1
        assert s.data.todos["items"][0]["done"] is True
        assert len(s.data.tool_calls) >= 2
        assert all(c["call_id"] for c in s.data.tool_calls)

    events = trace.read_events("live_u", "live_s")
    integrity = verify_trace_integrity(events)
    assert integrity["valid"] is True
