"""天气工具：调用 wttr.in (format=j1)，提取精简字段。"""
from __future__ import annotations

from typing import Any, Callable
from urllib.parse import quote

import httpx2

from .registry import Tool, ToolError

URL = "https://wttr.in/{city}?format=j1"
HttpGet = Callable[[str], Any]  # 返回已解析的 JSON


def _default_get(url: str) -> Any:
    try:
        r = httpx2.get(url, timeout=15, follow_redirects=True)
        r.raise_for_status()
        return r.json()
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
    def run(args: dict, ctx):
        city = args["city"].strip()
        if not city:
            raise ToolError("city must not be empty")
        return parse_weather(city, http_get(URL.format(city=quote(city))))

    return Tool(
        name="weather",
        description="Get current weather (temperature, feels-like, humidity, description) for a city.",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string", "description": "city name, e.g. Beijing"}},
            "required": ["city"],
        },
        func=run,
    )
