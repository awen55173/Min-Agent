"""提供 M0 命令行入口，验证配置加载到模型回复的完整链路。"""
from __future__ import annotations

import argparse
import sys

from agent.config import ConfigError, Settings
from agent.llm import LLMClient, LLMError, OpenAIClient


def run_once(client: LLMClient, prompt: str) -> str:
    """组合系统提示和用户输入，调用客户端并返回一次回复。"""
    return client.chat(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )


def main(argv: list[str] | None = None) -> int:
    """解析参数并打印模型回复，成功返回 0，配置或调用失败返回 1。"""
    parser = argparse.ArgumentParser(description="min_agent M0 smoke entry")
    parser.add_argument("prompt", nargs="?", default="Reply with exactly: pong")
    args = parser.parse_args(argv)
    try:
        client = OpenAIClient(Settings.from_env())
        print(run_once(client, args.prompt))
    except (ConfigError, LLMError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
