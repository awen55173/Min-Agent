"""命令行入口：支持用户、会话（新建/按 ID 恢复/列出）与交互模式。"""
from __future__ import annotations

import argparse
import re
import sys

from agent.config import ConfigError, Settings
from agent.llm import LLMClient, OpenAIClient
from agent.loop import OK, Agent, AgentResult
from agent.parser import ParseError, extract_json
from agent.service import ChatService
from agent.session import SessionError, SessionManager
from agent.tools import ToolRegistry, build_registry

_OBS_RE = re.compile(r"^Observation \(tool=([^,]+), status=(ok|error)\): (.*)$", re.S)


def run_once(client: LLMClient, prompt: str, registry: ToolRegistry | None = None) -> AgentResult:
    """用 Agent 循环处理单次输入（无会话持久化）并返回完整结果。"""
    return Agent(client, registry or build_registry()).run(prompt)


def _print_result(result: AgentResult, verbose: bool) -> None:
    """打印回复；verbose 时把工具调用过程输出到 stderr。"""
    if verbose:
        for a in result.actions:
            print(f"[step {a.step}] {a.tool}({a.args}) ok={a.ok} -> {a.result}", file=sys.stderr)
    print(result.answer)


def _print_history(history: list[dict[str, str]], verbose: bool) -> None:
    """回放会话历史中的用户提问与最终回复；verbose 时把历史工具调用输出到 stderr。"""
    step = 0
    pending_args: dict = {}
    for msg in history:
        role, content = msg.get("role"), msg.get("content", "")
        if role == "user":
            m = _OBS_RE.match(content)
            if m:
                if verbose:
                    step += 1
                    tool, status, res = m.group(1), m.group(2), m.group(3)
                    print(f"[step {step}] {tool}({pending_args}) ok={status == 'ok'} -> {res}", file=sys.stderr)
                continue
            if content.startswith("Format error: "):
                continue
            step = 0
            print(f"you> {content}")
        elif role == "assistant":
            try:
                obj = extract_json(content)
            except ParseError:
                continue
            if isinstance(obj.get("action"), str):
                pending_args = obj.get("action_input") if isinstance(obj.get("action_input"), dict) else {}
            elif isinstance(obj.get("final_answer"), str) and obj["final_answer"].strip():
                print(obj["final_answer"].strip())


def _build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    p = argparse.ArgumentParser(description="min_agent")
    p.add_argument("prompt", nargs="?", help="单次输入；省略则进入交互模式")
    p.add_argument("-u", "--user", default="default", help="用户 ID（默认 default）")
    group = p.add_mutually_exclusive_group()
    group.add_argument("-s", "--session", help="按 ID 恢复已有会话")
    group.add_argument("--new", action="store_true", help="新建会话（未指定 --session 时的默认行为）")
    p.add_argument("--list", action="store_true", help="列出该用户的会话 ID 后退出")
    p.add_argument("--data-dir", default="data", help="会话存储目录（默认 data）")
    p.add_argument("-v", "--verbose", action="store_true", help="打印工具调用过程")
    return p


def main(argv: list[str] | None = None) -> int:
    """解析参数、定位或新建会话并运行 Agent；失败时输出错误并返回 1。"""
    args = _build_parser().parse_args(argv)
    manager = SessionManager(args.data_dir)
    try:
        if args.list:
            for sid in manager.list_sessions(args.user):
                print(sid)
            return 0
        settings = Settings.from_env()
        resume = args.session is not None
        session_id = args.session if resume else manager.new_session_id(args.user)
        if resume and not manager.exists(args.user, session_id):
            print(f"error: session '{session_id}' of user '{args.user}' does not exist", file=sys.stderr)
            return 1
        service = ChatService(
            OpenAIClient(settings), manager,
            lambda store: build_registry(settings.tavily_api_key, todo_store=store),
        )
        print(f"[session] user={args.user} id={session_id}", file=sys.stderr)
        create = not resume  # 新会话在首轮保存时落盘
        if args.prompt is not None:
            result = service.send(args.user, session_id, args.prompt, create=create)
            _print_result(result, args.verbose)
            return 0 if result.status == OK else 1
        if resume:
            with manager.open(args.user, session_id) as session:
                _print_history(session.data.history, args.verbose)
        while True:  # 交互模式：空行 / exit / quit / EOF 结束
            try:
                line = input("you> ").strip()
            except EOFError:
                break
            if not line or line.lower() in ("exit", "quit"):
                break
            _print_result(service.send(args.user, session_id, line, create=create), args.verbose)
            create = False
        print(f"[session] resume with: --user {args.user} --session {session_id}", file=sys.stderr)
        return 0
    except (ConfigError, SessionError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
