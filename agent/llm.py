"""封装兼容 OpenAI 的模型调用（支持原生 tools 函数调用与纯文本补全），并提供测试用模拟客户端。"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import time
from typing import Any, Protocol

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

from .config import Settings

Message = dict[str, Any]


class LLMError(Exception):
    """表示模型调用失败或响应无有效内容。"""


@dataclass
class ToolCallItem:
    """原生工具调用项：保存模型生成的调用 ID、工具名称与参数 JSON 字符串。"""

    id: str
    name: str
    arguments: str

    def to_openai_dict(self) -> dict[str, Any]:
        """转为 OpenAI assistant 消息中 tool_calls 列表元素的字典格式。"""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            },
        }


@dataclass
class LLMResponse:
    """统一的模型响应结构：包含文本回复 content 与工具调用列表 tool_calls。"""

    content: str = ""
    tool_calls: list[ToolCallItem] = field(default_factory=list)

    def has_tool_calls(self) -> bool:
        """判断当前模型响应是否包含原生工具调用请求。"""
        return bool(self.tool_calls)


def normalize_llm_response(item: Any) -> LLMResponse:
    """将字符串、字典或 LLMResponse 统一归一化为 LLMResponse。"""
    if isinstance(item, LLMResponse):
        return item
    if isinstance(item, str):
        return LLMResponse(content=item, tool_calls=[])
    if isinstance(item, dict):
        content = str(item.get("content", "") or "")
        raw_calls = item.get("tool_calls") or []
        calls: list[ToolCallItem] = []
        for idx, tc in enumerate(raw_calls, start=1):
            if isinstance(tc, ToolCallItem):
                calls.append(tc)
            elif isinstance(tc, dict):
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
                name = str(fn.get("name", ""))
                args_val = fn.get("arguments", tc.get("args", {}))
                if not isinstance(args_val, str):
                    args_str = json.dumps(args_val, ensure_ascii=False)
                else:
                    args_str = args_val
                cid = str(tc.get("id") or f"call_{idx}")
                calls.append(ToolCallItem(id=cid, name=name, arguments=args_str))
        return LLMResponse(content=content, tool_calls=calls)
    raise LLMError(f"unsupported LLM response object: {type(item).__name__}")


class LLMClient(Protocol):
    """约定真实客户端和模拟客户端的统一调用接口。"""

    def chat(self, messages: list[Message]) -> str:
        """接收消息列表并返回模型回复文本。"""
        ...


class OpenAIClient:
    """调用兼容 OpenAI 的接口，支持自定义服务地址与原生函数工具调用。"""

    def __init__(self, settings: Settings, temperature: float = 0.1, backoff: float = 1.0):
        """按配置创建客户端，并保存模型参数和重试策略。"""
        self._model = settings.model
        self._temperature = temperature
        self._max_retries = settings.max_retries
        self._backoff = backoff
        # 关闭 SDK 自动重试，由当前客户端统一控制。
        self._client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.timeout,
            max_retries=0,
        )

    @staticmethod
    def _extract(resp) -> str:
        """提取首个候选纯文本回复，无候选或内容为空时抛出异常。"""
        choices = getattr(resp, "choices", None)
        if not choices:
            raise LLMError("LLM response has no choices")
        content = getattr(choices[0].message, "content", None)
        if content is None or not str(content).strip():
            raise LLMError("LLM returned empty content")
        return content

    @staticmethod
    def _extract_response(resp) -> LLMResponse:
        """从模型响应中提取统一的文本回答与工具调用列表；二者皆空时抛出 LLMError。"""
        choices = getattr(resp, "choices", None)
        if not choices:
            raise LLMError("LLM response has no choices")
        msg = choices[0].message
        raw_content = getattr(msg, "content", None)
        content = str(raw_content) if raw_content is not None else ""
        raw_tool_calls = getattr(msg, "tool_calls", None) or []
        tool_calls: list[ToolCallItem] = []
        for idx, tc in enumerate(raw_tool_calls, start=1):
            fn = getattr(tc, "function", None)
            name = str(getattr(fn, "name", "") or "") if fn is not None else ""
            arguments = str(getattr(fn, "arguments", "") or "{}") if fn is not None else "{}"
            cid = str(getattr(tc, "id", "") or f"call_{idx}")
            tool_calls.append(ToolCallItem(id=cid, name=name, arguments=arguments))
        if not content.strip() and not tool_calls:
            raise LLMError("LLM returned empty content and no tool_calls")
        return LLMResponse(content=content, tool_calls=tool_calls)

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> LLMResponse:
        """发送消息（可选附带原生 tools Schema）并返回统一的 LLMResponse，带指数退避重试。"""
        last_err: Exception | None = None
        use_parallel_param = parallel_tool_calls is not None
        for attempt in range(self._max_retries + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": self._model,
                    "messages": messages,
                    "temperature": self._temperature,
                }
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"
                    if use_parallel_param and parallel_tool_calls is not None:
                        kwargs["parallel_tool_calls"] = parallel_tool_calls
                resp = self._client.chat.completions.create(**kwargs)
                return self._extract_response(resp)
            except (APIConnectionError, APITimeoutError, RateLimitError) as e:
                last_err = e
            except APIStatusError as e:
                if e.status_code == 400 and use_parallel_param and "parallel_tool_calls" in str(e.message):
                    use_parallel_param = False
                    continue
                if e.status_code < 500:
                    raise LLMError(f"LLM request rejected ({e.status_code}): {e.message}") from e
                last_err = e
            if attempt < self._max_retries:
                time.sleep(self._backoff * (2**attempt))
        raise LLMError(f"LLM unavailable after {self._max_retries + 1} attempts: {last_err}") from last_err

    def chat(self, messages: list[Message]) -> str:
        """发送消息并返回文本，对连接、超时、限流和服务端错误退避重试。"""
        last_err: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model, messages=messages, temperature=self._temperature
                )
                return self._extract(resp)
            except (APIConnectionError, APITimeoutError, RateLimitError) as e:
                last_err = e
            except APIStatusError as e:
                if e.status_code < 500:
                    raise LLMError(f"LLM request rejected ({e.status_code}): {e.message}") from e
                last_err = e
            if attempt < self._max_retries:
                time.sleep(self._backoff * (2**attempt))
        raise LLMError(f"LLM unavailable after {self._max_retries + 1} attempts: {last_err}") from last_err


class FakeLLM:
    """按顺序返回预设回复（支持纯文本或带 tool_calls 的 LLMResponse/字典），并记录收到的消息。"""

    def __init__(self, replies: list[ Any]):
        """复制预设回复，并初始化调用记录。"""
        self._replies = list(replies)
        self.calls: list[list[Message]] = []
        self.tools_passed: list[list[dict[str, Any]] | None] = []

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> LLMResponse:
        """记录消息与工具定义，并返回归一化后的下一条预设 LLMResponse。"""
        self.calls.append([dict(m) for m in messages])
        self.tools_passed.append(tools)
        if not self._replies:
            raise LLMError("FakeLLM has no more scripted replies")
        item = self._replies.pop(0)
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            item = item(messages)
        return normalize_llm_response(item)

    def chat(self, messages: list[Message]) -> str:
        """记录消息并返回下一条预设回复文本，回复耗尽时抛出异常。"""
        self.calls.append([dict(m) for m in messages])
        if not self._replies:
            raise LLMError("FakeLLM has no more scripted replies")
        item = self._replies.pop(0)
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            item = item(messages)
        if isinstance(item, LLMResponse):
            return item.content
        return str(item)
