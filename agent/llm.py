"""LLM client: OpenAI-compatible chat API wrapper plus a scriptable fake for tests."""
from __future__ import annotations

import time
from typing import Protocol

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError, APIStatusError

from .config import Settings

Message = dict[str, str]


class LLMError(Exception):
    """Raised when the LLM call ultimately fails (after retries)."""


class LLMClient(Protocol):
    def chat(self, messages: list[Message]) -> str: ...


class OpenAIClient:
    """Thin wrapper over any OpenAI-format endpoint (custom base_url supported)."""

    def __init__(self, settings: Settings, temperature: float = 0.0, backoff: float = 1.0):
        self._model = settings.model
        self._temperature = temperature
        self._max_retries = settings.max_retries
        self._backoff = backoff
        # SDK-level retries disabled: we own the retry policy.
        self._client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.timeout,
            max_retries=0,
        )

    def chat(self, messages: list[Message]) -> str:
        last_err: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model, messages=messages, temperature=self._temperature
                )
                content = resp.choices[0].message.content
                if content is None:
                    raise LLMError("LLM returned empty content")
                return content
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
    """Returns scripted replies in order; records the messages it received."""

    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.calls: list[list[Message]] = []

    def chat(self, messages: list[Message]) -> str:
        self.calls.append([dict(m) for m in messages])
        if not self._replies:
            raise LLMError("FakeLLM has no more scripted replies")
        return self._replies.pop(0)
