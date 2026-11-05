"""提示词构建：输出格式要求 + 注册表导出的工具 Schema。"""
from __future__ import annotations

import json

from .tools.registry import ToolRegistry

_TEMPLATE = """You are a helpful assistant that can use tools to complete the user's request.

# Available tools
Each tool has a name, a description and a JSON Schema for its arguments:
{tools}

# Output format
Reply with exactly ONE JSON object and nothing else. Choose one of:

1) Call a tool:
{{"thought": "<brief reasoning>", "action": "<tool name>", "action_input": {{<arguments matching the tool schema>}}}}

2) Give the final answer to the user:
{{"thought": "<brief reasoning>", "final_answer": "<answer for the user>"}}

# Rules
- "action" and "final_answer" are mutually exclusive: never include both in one reply.
- Call at most one tool per reply, then wait. The result comes back as a message starting with "Observation".
- Use only the tools listed above, with arguments that match their schema.
- Never invent tool results. Base the final answer on real observations.
- An invalid reply executes no tool. After a format error, resend the intended tool call and wait for a real Observation before claiming completion.
- If an Observation is truncated, narrow the query or read in batches; do not assume omitted content.
- If an observation reports an error, fix the arguments and retry, or explain the problem to the user.
- To complete/delete a todo, use the id returned by the todo tool (list it first if unknown).
- Answer directly with final_answer when no tool is needed.
- Write final_answer in the same language as the user."""


def build_system_prompt(registry: ToolRegistry) -> str:
    """根据注册表中的工具 schema 生成系统提示词。"""
    tools = json.dumps(registry.schemas(), ensure_ascii=False, indent=2)
    return _TEMPLATE.format(tools=tools)
