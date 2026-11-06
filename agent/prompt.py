"""提示词构建：支持原生函数工具调用（主路径）与显式文本 JSON 模式（兼容路径）。"""
from __future__ import annotations

import json

from .tools.registry import ToolRegistry

_TEXT_TEMPLATE = """You are a helpful assistant that can use tools to complete the user's request.

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
- For questions about remembered user information, use only the current-session user memory system message; never treat prior assistant answers or tool results as user memory.
- An invalid reply executes no tool. After a format error, resend the intended tool call and wait for a real Observation before claiming completion.
- If an Observation is truncated, narrow the query or read in batches; do not assume omitted content.
- If an observation reports an error, fix the arguments and retry, or explain the problem to the user.
- To complete/delete a todo, use the id returned by the todo tool (list it first if unknown).
- Answer directly with final_answer when no tool is needed.
- Write final_answer in the same language as the user."""

_NATIVE_TEMPLATE = """You are a helpful assistant that can use function calling tools to complete the user's request.

# Rules
- Use the provided tools whenever external calculation, web search, weather lookup, or todo management is needed.
- Supply tool arguments that strictly match the tool's JSON Schema.
- Never invent or fabricate tool results; always wait for the tool execution result before claiming an action succeeded.
- For questions about remembered user information, use only the current-session user memory system message; never treat prior assistant answers or tool results as user memory.
- For independent read-only queries (e.g., checking weather for multiple cities simultaneously), you may emit multiple read-only tool calls in a single response; for dependent operations (e.g., querying weather first and then creating a todo based on the weather result), call the prerequisite tool first and wait for its result before calling the dependent tool.
- If a tool result is truncated, narrow the query or read in batches; do not assume omitted content.
- If a tool returns an error, fix the arguments and retry, or explain the problem to the user.
- To complete or delete a todo item, always use the integer id returned by the todo tool (call todo with action='list' first if unknown).
- When no tool is needed or all required tool calls are finished, reply directly to the user in the same language as the user."""


def build_system_prompt(registry: ToolRegistry, tool_mode: str = "text") -> str:
    """根据工具调用模式生成系统提示词：native 模式使用原生函数调用规则，text 模式内嵌 JSON 协议与工具 Schema。"""
    if tool_mode == "native":
        return build_native_system_prompt(registry)
    tools = json.dumps(registry.schemas(), ensure_ascii=False, indent=2)
    return _TEXT_TEMPLATE.format(tools=tools)


def build_native_system_prompt(registry: ToolRegistry) -> str:
    """生成原生工具调用（function calling）模式下的系统提示词。"""
    _ = registry
    return _NATIVE_TEMPLATE
