"""封装兼容 OpenAI 的模型调用，并提供测试用模拟客户端。"""
from __future__ import annotations

import time
from typing import Protocol

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError, APIStatusError

from .config import Settings

Message = dict[str, str]


class LLMError(Exception):
    """表示模型调用失败或响应无有效内容。"""


class LLMClient(Protocol):
    """约定真实客户端和模拟客户端的统一调用接口。"""

    def chat(self, messages: list[Message]) -> str:
        """接收消息列表并返回模型回复文本。"""
        ...


class OpenAIClient:
    """调用兼容 OpenAI 的接口，支持自定义服务地址。"""

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
        """提取首个候选回复，无候选或内容为空时抛出异常。"""
        choices = getattr(resp, "choices", None)
        if not choices:
            raise LLMError("LLM response has no choices")
        content = getattr(choices[0].message, "content", None)
        if content is None or not str(content).strip():
            raise LLMError("LLM returned empty content")
        return content

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
    """按顺序返回预设回复，并记录收到的消息。"""

    def __init__(self, replies: list[str]):
        """复制预设回复，并初始化调用记录。"""
        self._replies = list(replies)
        self.calls: list[list[Message]] = []

    def chat(self, messages: list[Message]) -> str:
        """记录消息并返回下一条预设回复，回复耗尽时抛出异常。"""
        self.calls.append([dict(m) for m in messages])
        if not self._replies:
            raise LLMError("FakeLLM has no more scripted replies")
        return self._replies.pop(0)
