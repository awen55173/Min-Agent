"""命令行入口：通过 Agent 循环处理一次用户输入。"""
from __future__ import annotations

import argparse
import sys

from agent.config import ConfigError, Settings
from agent.llm import LLMClient, OpenAIClient
from agent.loop import OK, Agent, AgentResult
from agent.tools import ToolRegistry, build_registry


def run_once(client: LLMClient, prompt: str, registry: ToolRegistry | None = None) -> AgentResult:
    """用 Agent 循环处理单次输入并返回完整结果。"""
    return Agent(client, registry or build_registry()).run(prompt)


def main(argv: list[str] | None = None) -> int:
    """解析参数、加载配置并运行 Agent，打印回复；失败时输出错误并返回 1。"""
    parser = argparse.ArgumentParser(description="min_agent")
    parser.add_argument("prompt", nargs="?", default="你好，请介绍一下你自己")
    parser.add_argument("-v", "--verbose", action="store_true", help="打印工具调用过程")
    args = parser.parse_args(argv)
    try:
        settings = Settings.from_env()
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    result = run_once(OpenAIClient(settings), args.prompt, build_registry(settings.tavily_api_key))
    if args.verbose:
        for a in result.actions:
            print(f"[step {a.step}] {a.tool}({a.args}) ok={a.ok} -> {a.result}", file=sys.stderr)
    print(result.answer)
    return 0 if result.status == OK else 1


if __name__ == "__main__":
    sys.exit(main())
