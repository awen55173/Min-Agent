"""搜索工具：调用 Tavily API，仅保留标题/摘要/URL 并截断。"""
from __future__ import annotations

from typing import Any, Callable

import httpx2

from .registry import Tool, ToolError

URL = "https://api.tavily.com/search"
HttpPost = Callable[[str, dict, str], Any]  # (url, json_body, api_key) -> 已解析 JSON
_SNIPPET_MAX = 300 # 返回摘要内容字数


def _default_post(url: str, body: dict, api_key: str) -> Any:
    """向 Tavily 发送 POST 并返回 JSON；失败转为 ToolError。"""
    try:
        r = httpx2.post(url, json=body, headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
        r.raise_for_status()
        return r.json()
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


def make_search_tool(api_key: str | None, http_post: HttpPost = _default_post) -> Tool:
    """创建搜索工具；api_key 缺失时调用返回配置错误，http_post 可注入以便测试。"""

    def run(args: dict, ctx):
        """执行搜索并返回精简结果。"""
        if not api_key:
            raise ToolError("search is not configured: TAVILY_API_KEY is missing")
        query = args["query"].strip()
        if not query:
            raise ToolError("query must not be empty")
        n = args.get("max_results", 3)
        if not 1 <= n <= 10:
            raise ToolError("max_results must be between 1 and 10")
        data = http_post(URL, {"query": query, "max_results": n}, api_key)
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
    )
