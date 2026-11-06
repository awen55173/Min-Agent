"""天气工具：支持同步与异步调用 wttr.in (format=j1)，提取精简字段。"""
from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable
from urllib.parse import quote

import httpx2

from .registry import Tool, ToolError

URL = "https://wttr.in/{city}?format=j1"
HttpGet = Callable[[str], Any]  # 返回已解析的 JSON（或协程）
_DEFAULT_TIMEOUT = 15.0


def _default_get(url: str) -> Any:
    """同步请求 wttr.in 并返回 JSON；任何网络/解析失败转为 ToolError。"""
    try:
        r = httpx2.get(url, timeout=_DEFAULT_TIMEOUT, follow_redirects=True)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise ToolError(f"weather service unavailable: {type(e).__name__}") from None


async def _default_aget(url: str) -> Any:
    """异步请求 wttr.in 并返回 JSON；保留取消信号传播，其余失败转为 ToolError。"""
    try:
        async with httpx2.AsyncClient(timeout=_DEFAULT_TIMEOUT, follow_redirects=True) as client:
            r = await client.get(url)
            r.raise_for_status()
            return r.json()
    except asyncio.CancelledError:
        raise
    except Exception as e:
        raise ToolError(f"weather service unavailable: {type(e).__name__}") from None


def parse_weather(city: str, data: Any) -> dict:
    """从 wttr.in j1 响应提取当前天气；结构异常抛 ToolError。"""
    try:
        cur = data["current_condition"][0]
        return {
            "city": city,
            "temp_c": cur["temp_C"],
            "feels_like_c": cur["FeelsLikeC"],
            "humidity": cur["humidity"],
            "description": cur["weatherDesc"][0]["value"],
        }
    except (KeyError, IndexError, TypeError):
        raise ToolError(f"no weather data for '{city}'") from None


def make_weather_tool(http_get: HttpGet = _default_get) -> Tool:
    """创建支持同步与异步执行的天气工具；http_get 可注入以便测试。"""

    def run(args: dict, ctx):
        """同步查询城市当前天气。"""
        _ = ctx
        city = args["city"].strip()
        if not city:
            raise ToolError("city must not be empty")
        data = http_get(URL.format(city=quote(city)))
        if inspect.isawaitable(data):
            data = asyncio.run(data)
        return parse_weather(city, data)

    async def arun(args: dict, ctx):
        """异步查询城市当前天气（兼容异步或同步注入的 http_get）。"""
        _ = ctx
        city = args["city"].strip()
        if not city:
            raise ToolError("city must not be empty")
        target_url = URL.format(city=quote(city))
        if http_get is _default_get:
            data = await _default_aget(target_url)
        elif inspect.iscoroutinefunction(http_get):
            data = await http_get(target_url)
        else:
            res = await asyncio.to_thread(http_get, target_url)
            data = (await res) if inspect.isawaitable(res) else res
        return parse_weather(city, data)

    return Tool(
        name="weather",
        description=(
            "Get current weather (temperature, feels-like, humidity, description) for a city. "
            "Prefer English or Pinyin city names (e.g. 'Beijing' instead of '北京')."
        ),
        parameters={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": (
                        "city name in English or Pinyin "
                        "(prefer English, translate Chinese city names first, e.g. 'Beijing' instead of '北京')"
                    ),
                }
            },
            "required": ["city"],
        },
        func=run,
        afunc=arun,
        read_only=True,
    )
