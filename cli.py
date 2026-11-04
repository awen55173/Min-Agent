"""Minimal entry point (M0): verifies the end-to-end call chain env -> client -> LLM."""
from __future__ import annotations

import argparse
import sys

from agent.config import ConfigError, Settings
from agent.llm import LLMClient, LLMError, OpenAIClient


def run_once(client: LLMClient, prompt: str) -> str:
    return client.chat(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )


def main(argv: list[str] | None = None) -> int:
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
