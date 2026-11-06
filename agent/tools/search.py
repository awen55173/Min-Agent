"""搜索工具：支持同步与异步调用 Tavily API，仅保留标题/摘要/URL 并截断。"""
from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable

import httpx2

from .registry import Tool, ToolError

URL = "https://api.tavily.com/search"
HttpPost = Callable[[str, dict, str], Any]  # (url, json_body, api_key) -> 已解析 JSON（或协程）
_SNIPPET_MAX = 300  # 返回摘要内容字数
_DEFAULT_TIMEOUT = 30.0


def _default_post(url: str, body: dict, api_key: str) -> Any:
    """向 Tavily 同步发送 POST 并返回 JSON；失败转为 ToolError。"""
    try:
        r = httpx2.post(url, json=body, headers={"Authorization": f"Bearer {api_key}"}, timeout=_DEFAULT_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise ToolError(f"search service unavailable: {type(e).__name__}") from None


async def _default_apost(url: str, body: dict, api_key: str) -> Any:
    """向 Tavily 异步发送 POST 并返回 JSON；保留取消信号传播，其余失败转为 ToolError。"""
    try:
        async with httpx2.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
            r = await client.post(url, json=body, headers={"Authorization": f"Bearer {api_key}"})
            r.raise_for_status()
            return r.json()
    except asyncio.CancelledError:
        raise
    except Exception as e:
        raise ToolError(f"search service unavailable: {type(e).__name__}") from None


def parse_results(data: Any) -> list[dict]:
    """提取标题/URL/截断摘要；响应结构异常抛 ToolError。"""
    try:
        return [
            {
                "title": str(r.get("title", "")),
                "url": str(r.get("url", "")),
                "snippet": str(r.get("content", ""))[:_SNIPPET_MAX],
            }
            for r in data["results"]
        ]
    except (KeyError, TypeError, AttributeError):
        raise ToolError("unexpected search response") from None


def _validate_search_args(api_key: str | None, args: dict) -> tuple[str, str, int]:
    """校验搜索工具的 API key、查询词与条数上限。"""
    if not api_key:
        raise ToolError("search is not configured: TAVILY_API_KEY is missing")
    query = args["query"].strip()
    if not query:
        raise ToolError("query must not be empty")
    n = args.get("max_results", 3)
    if not 1 <= n <= 10:
        raise ToolError("max_results must be between 1 and 10")
    return api_key, query, n


def make_search_tool(api_key: str | None, http_post: HttpPost = _default_post) -> Tool:
    """创建支持同步与异步执行的搜索工具；api_key 缺失时返回配置错误，http_post 可注入以便测试。"""

    def run(args: dict, ctx):
        """同步执行搜索并返回精简结果。"""
        _ = ctx
        key, query, n = _validate_search_args(api_key, args)
        data = http_post(URL, {"query": query, "max_results": n}, key)
        if inspect.isawaitable(data):
            data = asyncio.run(data)
        return {"query": query, "results": parse_results(data)}

    async def arun(args: dict, ctx):
        """异步执行搜索并返回精简结果（兼容异步或同步注入的 http_post）。"""
        _ = ctx
        key, query, n = _validate_search_args(api_key, args)
        payload = {"query": query, "max_results": n}
        if http_post is _default_post:
            data = await _default_apost(URL, payload, key)
        elif inspect.iscoroutinefunction(http_post):
            data = await http_post(URL, payload, key)
        else:
            res = await asyncio.to_thread(http_post, URL, payload, key)
            data = (await res) if inspect.isawaitable(res) else res
        return {"query": query, "results": parse_results(data)}

    return Tool(
        name="search",
        description="Search the web (Tavily) and return top results with title, url and snippet.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "description": "1-10, default 3"},
            },
            "required": ["query"],
        },
        func=run,
        afunc=arun,
        read_only=True,
    )
