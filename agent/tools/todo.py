"""待办工具：增/查/完成/删除，按 (user_id, session_id) 隔离。"""
from __future__ import annotations

import threading
from dataclasses import asdict, dataclass

from .registry import Tool, ToolContext, ToolError


@dataclass
class TodoItem:
    """待办条目；turn 记录创建时所处的对话轮次，仅用于同轮去重，不对外输出。"""

    id: int
    text: str
    done: bool = False
    turn: int | None = None

    def public(self) -> dict:
        """返回对模型可见的字段（不含内部 turn）。"""
        d = asdict(self)
        d.pop("turn")
        return d


class TodoStore:
    """内存待办存储（M3 接入会话持久化）。每个会话独立自增 ID。"""

    def __init__(self) -> None:
        """初始化空存储与锁。"""
        self._data: dict[tuple[str, str], dict] = {}
        self._lock = threading.Lock()

    def _bucket(self, ctx: ToolContext) -> dict:
        """获取（必要时创建）当前用户会话的数据桶。"""
        return self._data.setdefault((ctx.user_id, ctx.session_id), {"next": 1, "items": []})

    def add(self, ctx: ToolContext, text: str) -> tuple[TodoItem, bool]:
        """新增待办；仅当同一轮（turn_id 相同且非 None）内已新增过相同内容的未完成待办时复用，返回 (条目, 是否新建)。"""
        text = text.strip()
        if not text:
            raise ToolError("todo text must not be empty")
        with self._lock:
            b = self._bucket(ctx)
            if ctx.turn_id is not None:
                for it in b["items"]:
                    if not it.done and it.text == text and it.turn == ctx.turn_id:
                        return it, False
            item = TodoItem(b["next"], text, turn=ctx.turn_id)
            b["next"] += 1
            b["items"].append(item)
            return item, True

    def list(self, ctx: ToolContext, status: str = "all") -> list[TodoItem]:
        """按状态（all/pending/done）列出当前会话的待办。"""
        with self._lock:
            items = list(self._bucket(ctx)["items"])
        if status == "pending":
            return [i for i in items if not i.done]
        if status == "done":
            return [i for i in items if i.done]
        return items

    def _find(self, ctx: ToolContext, todo_id: int) -> TodoItem:
        """按 ID 查找当前会话的待办，不存在抛 ToolError。"""
        for it in self._bucket(ctx)["items"]:
            if it.id == todo_id:
                return it
        raise ToolError(f"todo {todo_id} not found")

    def complete(self, ctx: ToolContext, todo_id: int) -> TodoItem:
        """将指定待办标记为完成。"""
        with self._lock:
            it = self._find(ctx, todo_id)
            it.done = True
            return it

    def delete(self, ctx: ToolContext, todo_id: int) -> TodoItem:
        """删除指定待办并返回被删条目。"""
        with self._lock:
            it = self._find(ctx, todo_id)
            self._bucket(ctx)["items"].remove(it)
            return it


def make_todo_tool(store: TodoStore) -> Tool:
    """创建绑定到指定存储的 todo 工具。"""

    def run(args: dict, ctx: ToolContext):
        """按 action 分发待办操作并返回可序列化结果。"""
        action = args["action"]
        if action == "add":
            if "text" not in args:
                raise ToolError("'text' is required for action 'add'")
            item, created = store.add(ctx, args["text"])
            return {"created": created, "item": item.public()}
        if action == "list":
            return {"items": [i.public() for i in store.list(ctx, args.get("status", "all"))]}
        if action in ("complete", "delete"):
            if "id" not in args:
                raise ToolError(f"'id' is required for action '{action}'")
            fn = store.complete if action == "complete" else store.delete
            return {action: fn(ctx, args["id"]).public()}
        raise ToolError(f"unknown action '{action}'")  # 正常情况下已被 enum 校验拦截

    return Tool(
        name="todo",
        description=(
            "Manage the user's todo list. Actions: add (needs text), list (optional status), "
            "complete (needs id), delete (needs id). Always use the todo id returned by list/add."
        ),
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["add", "list", "complete", "delete"]},
                "text": {"type": "string", "description": "todo content, for add"},
                "id": {"type": "integer", "description": "todo id, for complete/delete"},
                "status": {"type": "string", "enum": ["all", "pending", "done"], "description": "filter for list"},
            },
            "required": ["action"],
        },
        func=run,
    )
