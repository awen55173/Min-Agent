"""M4 上下文管理与结构化压缩测试：覆盖纯对话追问、带工具追问、轮数/token 压缩、压缩后完成待办及重启恢复。"""
import json

import pytest

import cli
from agent.context import (
    build_context,
    estimate_messages_tokens,
    estimate_tokens,
    parse_summary,
    refresh_summary_todos,
    sanitize_turn,
    split_into_turns,
    summarize_turns,
)
from agent.llm import FakeLLM
from agent.loop import OK
from agent.service import ChatService
from agent.session import SessionCorruptError, SessionManager
from agent.tools import build_registry

WTTR = {
    "current_condition": [
        {
            "temp_C": "20",
            "FeelsLikeC": "19",
            "humidity": "50",
            "weatherDesc": [{"value": "Sunny"}],
        }
    ]
}


def act(tool, thought="secret-thought", **args):
    """构造原生工具调用测试响应。"""
    return {"content": thought, "tool_calls": [{"id": "fixture_call", "type": "function", "function": {"name": tool, "arguments": json.dumps(args, ensure_ascii=False)}}]}


def final(text, thought="secret-thought"):
    """构造直接文本回答。"""
    return text


def factory(store):
    return build_registry(
        "k",
        todo_store=store,
        http_get=lambda u: WTTR,
        http_post=lambda u, b, k: {
            "results": [{"title": "Python 3.11", "url": "https://python.org", "content": "Fast CPython release"}]
        },
    )


def state(tmp_path, user="alice", sid="s1"):
    with SessionManager(tmp_path / "data").open(user, sid) as s:
        return s.data


# ---------------- 基础切分、清理与 token 估算 ----------------
def test_estimate_tokens_and_split_turns():
    assert estimate_tokens("") == 0
    assert estimate_tokens("你好世界") >= 4
    assert estimate_tokens("hello world") >= 3

    history = [
        {"role": "user", "content": "第一问"},
        {"role": "assistant", "content": "bad json"},
        {"role": "user", "content": "Validation error: invalid"},
        {"role": "assistant", **act("weather", thought="思考1", city="Beijing")},
        {'role': 'tool', 'tool_call_id': 'fixture_call', 'name': 'weather', 'content': '{"city":"Beijing","temp_c":"20","description":"Sunny","humidity":"50"}'},
        {"role": "assistant", "content": final("北京20度", thought="思考2")},
        {"role": "user", "content": "第二问"},
        {"role": "assistant", "content": final("回答2", thought="思考3")},
    ]
    turns = split_into_turns(history)
    assert len(turns) == 2
    assert len(turns[0]) == 6 and len(turns[1]) == 2

    cleaned = sanitize_turn(turns[0])
    # 格式错误重试对被丢弃，且历史 thought 被剥离
    assert len(cleaned) == 4
    for m in cleaned:
        assert "思考" not in m["content"]
        assert not m["content"].startswith("Validation error:")
    # 原 history 未被就地修改
    assert "思考1" in history[3]["content"]


# ---------------- 结构化摘要字段提取 ----------------
def test_summarize_turns_extracts_all_five_sections_and_drops_thought():
    turn1 = [
        {"role": "user", "content": "查北京天气、算 6*7、搜 Python 并记待办买牛奶"},
        {"role": "assistant", **act("weather", thought="INTERNAL_THOUGHT_1", city="Beijing")},
        {'role': 'tool', 'tool_call_id': 'fixture_call', 'name': 'weather', 'content': '{"city":"Beijing","temp_c":"20","feels_like_c":"19","humidity":"50","description":"Sunny"}'},
        {"role": "assistant", **act("calculator", thought="INTERNAL_THOUGHT_2", expression="6*7")},
        {'role': 'tool', 'tool_call_id': 'fixture_call', 'name': 'calculator', 'content': '{"expression":"6*7","result":42}'},
        {"role": "assistant", **act("search", thought="INTERNAL_THOUGHT_3", query="python")},
        {'role': 'tool', 'tool_call_id': 'fixture_call', 'name': 'search', 'content': '{"query":"python","results":[{"title":"Py","url":"http://p","snippet":"lang"}]}'},
        {"role": "assistant", **act("todo", thought="INTERNAL_THOUGHT_4", action="add", text="买牛奶")},
        {'role': 'tool', 'tool_call_id': 'fixture_call', 'name': 'todo', 'content': '{"created":true,"item":{"id":1,"text":"买牛奶","done":false}}'},
        {"role": "assistant", "content": final("都办好了", thought="INTERNAL_THOUGHT_5")},
    ]
    turn2 = [
        {"role": "user", "content": "帮我继续做复杂任务"},
        {"role": "assistant", "content": final("已达最大步数（8），任务可能未完成。\n本轮已执行的操作：...", thought="INTERNAL_THOUGHT_6")},
    ]
    summary_obj = summarize_turns([turn1, turn2])
    text = summary_obj.format()

    # 1. 用户目标
    assert "查北京天气、算 6*7、搜 Python 并记待办买牛奶" in text
    # 2. 关键事实
    assert "Beijing" in text and "20°C" in text and "Sunny" in text
    assert "6*7 = 42" in text
    assert "搜索 python" in text
    # 3. 待办 ID 与状态
    assert "#1 [pending/未完成] 买牛奶" in text
    # 4. 已执行操作
    assert "weather(Beijing)" in text and "calculator(6*7)" in text and "todo.add -> #1 买牛奶" in text
    # 5. 未完成事项（含未完成待办 + 超步数未完成请求）
    assert "待办 #1 (pending/未完成): 买牛奶" in text
    assert "未完成请求「帮我继续做复杂任务」" in text
    # 历史 thought 绝不进入摘要
    assert "INTERNAL_THOUGHT" not in text

    # 解析后再格式化保持一致，且同步 TodoStore 状态后更新为已完成
    parsed = parse_summary(text)
    assert parsed.todos[1]["text"] == "买牛奶" and not parsed.todos[1]["done"]
    refreshed = refresh_summary_todos(text, {"next": 2, "items": [{"id": 1, "text": "买牛奶", "done": True}]})
    assert "#1 [done/已完成] 买牛奶" in refreshed
    assert "待办 #1 (pending/未完成)" not in refreshed


# ---------------- 触发压缩：按轮数与按 token 预算 ----------------
def test_build_context_triggers_by_turn_limit_and_token_budget():
    turns = []
    for i in range(1, 5):
        turns.extend([
            {"role": "user", "content": f"问题 {i}：请记住编号 {i * 100}"},
            {"role": "assistant", "content": final(f"已记住编号 {i * 100}", thought=f"thought-{i}")},
        ])

    # 1) 按轮数触发：共 4 轮，max_history_turns=2 -> 压缩前 2 轮，保留后 2 轮
    plan = build_context("SYS", turns, "新问题", max_history_turns=2, max_context_tokens=4000, reserved_tokens=500)
    assert plan.compressed is True
    assert plan.compressed_turn_count == 2 and plan.kept_turn_count == 2
    assert plan.messages[0] == {"role": "system", "content": "SYS"}
    assert plan.messages[1]["role"] == "system" and "[历史对话结构化摘要" in plan.messages[1]["content"]
    assert "100" in plan.summary and "200" in plan.summary
    # 最近 2 轮保留原文（但剥离了 thought）
    recent_contents = [m["content"] for m in plan.recent_messages]
    assert any("问题 3" in c for c in recent_contents) and any("问题 4" in c for c in recent_contents)
    assert not any("问题 1" in c for c in recent_contents)
    assert not any("thought-" in c for c in recent_contents)

    # 2) 按 token 预算触发：虽然 max_history_turns=10，但预留空间后预算仅 80 tokens（小于原 124 tokens）
    tight_plan = build_context("SYS", turns, "新问题", max_history_turns=10, max_context_tokens=200, reserved_tokens=120)
    assert tight_plan.compressed is True
    assert tight_plan.compressed_turn_count >= 1
    assert tight_plan.kept_turn_count < 4
    assert 10 <= estimate_messages_tokens(tight_plan.messages) <= 200


# ---------------- 纯对话追问 & 带工具追问 ----------------
def test_pure_dialogue_and_tool_followups(tmp_path):
    mgr = SessionManager(tmp_path / "data")
    # 第 1 轮：纯对话建立状态
    llm1 = FakeLLM([final("好的，已记住您叫林涛，负责猎户座项目。")])
    ChatService(llm1, mgr, factory).send("alice", "s1", "我叫林涛，负责猎户座项目", create=True)

    # 第 2 轮：纯对话追问
    llm2 = FakeLLM([final("您是林涛，负责猎户座项目。")])
    r2 = ChatService(llm2, mgr, factory).send("alice", "s1", "我是谁？负责什么项目？")
    assert r2.status == OK
    assert any("猎户座项目" in m["content"] for m in llm2.calls[0])

    # 第 3 轮：调用工具记待办
    llm3 = FakeLLM([act("todo", action="add", text="整理猎户座项目周报"), final("已添加待办 #1")])
    ChatService(llm3, mgr, factory).send("alice", "s1", "帮我记个待办：整理猎户座项目周报")

    # 第 4 轮：带工具追问（引用上一轮的待办并完成）
    llm4 = FakeLLM([act("todo", action="complete", id=1), final("已完成待办 #1")])
    r4 = ChatService(llm4, mgr, factory).send("alice", "s1", "把刚才那条待办设为完成")
    assert r4.status == OK
    assert state(tmp_path, "alice", "s1").todos["items"][0]["done"] is True


# ---------------- 核心验收：压缩后追问 & 压缩后仍能完成“刚才那个待办” ----------------
def test_complete_earlier_todo_and_answer_facts_after_compression(tmp_path):
    mgr = SessionManager(tmp_path / "data")

    def run_turn(replies, prompt, create=False):
        llm = FakeLLM(replies)
        svc = ChatService(llm, mgr, factory, max_history_turns=2)
        res = svc.send("alice", "work", prompt, create=create)
        return res, llm

    # Turn 1：查天气 + 记两条待办（这一轮稍后会被挤出最近 2 轮窗口而进入结构化摘要）
    run_turn(
        [
            act("weather", thought="隐藏思考A", city="Beijing"),
            act("todo", thought="隐藏思考B", action="add", text="下午三点提交周报"),
            act("todo", thought="隐藏思考C", action="add", text="买咖啡"),
            final("北京 20°C 晴，已记待办 #1 提交周报、#2 买咖啡", thought="隐藏思考D"),
        ],
        "查北京天气，并记两条待办：下午三点提交周报、买咖啡",
        create=True,
    )
    # Turn 2：先完成 #2 买咖啡
    run_turn(
        [act("todo", action="complete", id=2), final("已完成 #2 买咖啡")],
        "咖啡买好了，把买咖啡那条完成",
    )
    # Turn 3、Turn 4：普通对话轮次，将 Turn 1 和 Turn 2 挤出最近 2 轮窗口
    run_turn([final("计算结果是 100")], "25 乘 4 是多少？")
    run_turn([final("好的，明天见")], "散会后提醒我休息")

    # 此时已有 4 轮历史，max_history_turns=2；发第 5 轮时，Turn 1 & Turn 2 被压缩进摘要
    res5, llm5 = run_turn(
        [
            act("todo", action="complete", id=1),
            final("已为您完成待办 #1（下午三点提交周报），另外之前查的北京天气是 20°C 晴。"),
        ],
        "把刚才那个提交周报的待办完成了，顺便说下之前查的北京几度？",
    )
    assert res5.status == OK

    # 检查第 5 轮发给模型的上下文：
    sent_msgs = llm5.calls[0]
    assert sent_msgs[0]["role"] == "system"
    assert sent_msgs[1]["role"] == "system" and "[历史对话结构化摘要" in sent_msgs[1]["content"]
    summary_sent = sent_msgs[1]["content"]
    # 摘要中保留了待办 #1（未完成）、#2（已完成）、北京天气与初始目标，且不含历史隐藏思考
    assert "#1 [pending/未完成] 下午三点提交周报" in summary_sent
    assert "#2 [done/已完成] 买咖啡" in summary_sent
    assert "Beijing" in summary_sent and "20°C" in summary_sent
    assert "隐藏思考" not in summary_sent
    # Turn 1 的原始 Observation 已不在非摘要消息中
    non_system = [m["content"] for m in sent_msgs[2:]]
    assert not any("Observation (tool=weather" in c for c in non_system)
    assert not any("隐藏思考" in c for c in non_system)

    # 验证待办 #1 确实在存储中被标记为完成，且持久化摘要也同步刷新为 done
    saved = state(tmp_path, "alice", "work")
    items = {i["id"]: i["done"] for i in saved.todos["items"]}
    assert items == {1: True, 2: True}
    assert "#1 [done/已完成] 下午三点提交周报" in saved.summary
    assert "待办 #1 (pending/未完成)" not in saved.summary

    # 验证重启恢复：新建 SessionManager 与 ChatService，即使后续轮次少也继续携带持久化摘要
    fresh_mgr = SessionManager(tmp_path / "data")
    llm6 = FakeLLM([final("您的两条待办（#1 提交周报、#2 买咖啡）都已完成。")])
    res6 = ChatService(llm6, fresh_mgr, factory, max_history_turns=10).send("alice", "work", "我的待办都做完了吗？")
    assert res6.status == OK
    assert "#1 [done/已完成] 下午三点提交周报" in llm6.calls[0][1]["content"]

    # 验证完整历史与受限上下文分离：CLI 交互式恢复会话仍能完整回放第 1 轮到第 6 轮的对话与工具步骤
    assert len(saved.history) >= 12


def test_compression_does_not_affect_cli_history_replay(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "m")

    mgr = SessionManager(tmp_path / "data")
    for idx in range(1, 5):
        llm = FakeLLM([act("calculator", expression=f"{idx}+1"), final(f"第{idx}轮回答")])
        ChatService(llm, mgr, factory, max_history_turns=2).send(
            "alice", "replay", f"第{idx}轮提问", create=(idx == 1)
        )

    # 摘要已生成并落盘
    assert "[历史对话结构化摘要" in state(tmp_path, "alice", "replay").summary

    # CLI 交互式恢复：第 1 轮（已被压缩）至第 4 轮的提问、回答与工具步骤全部可回放
    monkeypatch.setattr(cli, "OpenAIClient", lambda s: FakeLLM([]))
    monkeypatch.setattr("builtins.input", lambda prompt="": "exit")
    assert cli.main(["-u", "alice", "-s", "replay", "-v"]) == 0
    out = capsys.readouterr()
    for idx in range(1, 5):
        assert f"you> 第{idx}轮提问" in out.out
        assert f"第{idx}轮回答" in out.out
        assert f"calculator({{'expression': '{idx}+1'}})" in out.err


def test_corrupt_summary_in_session_file_rejected(tmp_path):
    mgr = SessionManager(tmp_path / "data")
    with mgr.open("alice", "s1", create=True) as s:
        s.save()
    p = mgr._path("alice", "s1")
    d = json.loads(p.read_text(encoding="utf-8"))
    d["summary"] = 12345
    p.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(SessionCorruptError, match="summary"):
        with mgr.open("alice", "s1"):
            pass


# ---------------- live ----------------
@pytest.mark.live
def test_live_complete_todo_after_compression(tmp_path):
    from agent.config import Settings
    from agent.llm import OpenAIClient

    s = Settings.from_env()
    mgr = SessionManager(tmp_path / "data")
    fac = lambda store: build_registry(s.tavily_api_key, todo_store=store)  # noqa: E731

    def ask(prompt, create=False):
        svc = ChatService(OpenAIClient(s), mgr, fac, max_history_turns=2)
        return svc.send("alice", "live_m4", prompt, create=create)

    r1 = ask("帮我添加一条待办：周五下午四点代码评审。完成后告诉我。", create=True)
    assert r1.status == OK
    assert ask("12 加 30 等于多少？直接回答数字。").status == OK
    assert ask("中国的首都是哪里？直接回答城市名。").status == OK

    # 此时第 1 轮已超出 max_history_turns=2 被压缩进摘要；追问完成刚才那个待办
    r4 = ask("把刚才那个代码评审的待办标记为完成，并告诉我结果。")
    assert r4.status == OK
    saved = state(tmp_path, "alice", "live_m4")
    assert "[历史对话结构化摘要" in saved.summary
    assert any("代码评审" in i["text"] and i["done"] for i in saved.todos["items"])
