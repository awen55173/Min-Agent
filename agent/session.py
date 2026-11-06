"""会话管理：以 (user_id, session_id) 标识会话，JSON 文件原子存储，文件锁 + 进程内锁 + 版本校验。"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_ROLES = {"user", "assistant", "tool"}  # 历史中不保存 system 提示词；支持原生 tool 角色与旧版 user/assistant


class SessionError(Exception):
    """会话相关错误的基类。"""


class InvalidIdError(SessionError):
    """user_id / session_id 不合法（防止路径穿越）。"""


class SessionNotFoundError(SessionError):
    """按 ID 恢复会话时文件不存在。"""


class SessionCorruptError(SessionError):
    """会话文件内容损坏或不符合格式；不会被自动覆盖。"""


class SessionBusyError(SessionError):
    """会话正被其他窗口/进程占用，等待锁超时。"""


class LockLostError(SessionError):
    """持锁期间锁被回收或接管；为避免覆盖他人数据，拒绝保存。"""


class VersionConflictError(SessionError):
    """保存时发现磁盘版本与加载时不一致（兜底的并发保护）。"""


def _now() -> str:
    """返回当前 UTC 时间的 ISO 字符串。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def check_id(kind: str, value: Any) -> str:
    """校验标识仅含字母数字、下划线和短横线（1-64 位），否则抛 InvalidIdError。"""
    if not isinstance(value, str) or not _ID_RE.match(value):
        raise InvalidIdError(f"invalid {kind}: {value!r} (allowed: letters, digits, '_' and '-', 1-64 chars)")
    return value


def _empty_todos() -> dict:
    """返回空的待办状态。"""
    return {"next": 1, "items": []}


@dataclass
class SessionData:
    """会话持久化数据：历史、结构化摘要、待办、工具调用记录、未完成轮次、轮次计数与版本号。"""

    user_id: str
    session_id: str
    version: int = 0
    next_turn: int = 1
    history: list[dict[str, Any]] = field(default_factory=list)
    todos: dict[str, Any] = field(default_factory=_empty_todos)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    summary: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    unfinished_turn: dict[str, Any] | None = None

    def unfinished_calls(self) -> list[dict[str, Any]]:
        """返回会话中状态为 pending 或 interrupted 的未完成工具调用记录。"""
        return [c for c in self.tool_calls if c.get("status") in ("pending", "interrupted")]

    def to_dict(self) -> dict[str, Any]:
        """转为可写入 JSON 的字典。"""
        return {
            "user_id": self.user_id, "session_id": self.session_id, "version": self.version,
            "next_turn": self.next_turn, "history": self.history, "todos": self.todos,
            "summary": self.summary, "tool_calls": self.tool_calls,
            "unfinished_turn": self.unfinished_turn,
            "created_at": self.created_at, "updated_at": self.updated_at,
        }


def _is_int(v: Any) -> bool:
    """判断是否为整数（排除 bool）。"""
    return isinstance(v, int) and not isinstance(v, bool)


def _validate_todos(todos: Any) -> dict:
    """校验待办结构：next 为正整数且大于所有 ID，条目字段类型正确且 ID 不重复。"""
    if not isinstance(todos, dict) or not _is_int(todos.get("next")) or not isinstance(todos.get("items"), list):
        raise SessionCorruptError("invalid 'todos' structure")
    seen: set[int] = set()
    for it in todos["items"]:
        ok = (
            isinstance(it, dict) and _is_int(it.get("id")) and it["id"] >= 1
            and isinstance(it.get("text"), str) and isinstance(it.get("done"), bool)
            and (it.get("turn") is None or isinstance(it.get("turn"), (int, str)))
        )
        if not ok or it["id"] in seen:
            raise SessionCorruptError(f"invalid todo item: {it!r}")
        seen.add(it["id"])
    if todos["next"] < 1 or any(i >= todos["next"] for i in seen):
        raise SessionCorruptError("todos.next must be greater than every todo id")
    return todos


_CALL_STATUSES = {"pending", "ok", "error", "interrupted"}


def _validate_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    """校验持久化的工具调用记录列表。"""
    if not isinstance(tool_calls, list):
        raise SessionCorruptError("invalid 'tool_calls'")
    out: list[dict[str, Any]] = []
    for c in tool_calls:
        ok = (
            isinstance(c, dict)
            and isinstance(c.get("call_id"), str) and bool(c["call_id"])
            and isinstance(c.get("tool"), str) and bool(c["tool"])
            and isinstance(c.get("args"), dict)
            and c.get("status") in _CALL_STATUSES
            and isinstance(c.get("result", ""), str)
        )
        if not ok:
            raise SessionCorruptError(f"invalid tool_call record: {c!r}")
        out.append(dict(c))
    return out


def _validate_unfinished_turn(unfinished: Any) -> dict[str, Any] | None:
    """校验未完成轮次信息（None 或含 turn_id 与 status 的字典）。"""
    if unfinished is None:
        return None
    if not isinstance(unfinished, dict):
        raise SessionCorruptError("invalid 'unfinished_turn'")
    tid = unfinished.get("turn_id")
    if not (_is_int(tid) or (isinstance(tid, str) and bool(tid))):
        raise SessionCorruptError("invalid 'unfinished_turn.turn_id'")
    if not isinstance(unfinished.get("status"), str):
        raise SessionCorruptError("invalid 'unfinished_turn.status'")
    return dict(unfinished)


def _validate_history_message(m: Any) -> dict[str, Any]:
    """校验并规范化单条会话历史消息（兼容旧版文本消息与原生 tool_calls / tool 消息）。"""
    if not isinstance(m, dict) or m.get("role") not in _ROLES or not isinstance(m.get("content"), str):
        raise SessionCorruptError(f"invalid history message: {m!r}")
    role = m["role"]
    out: dict[str, Any] = {"role": role, "content": m["content"]}
    if role == "assistant" and m.get("tool_calls") is not None:
        tcs = m["tool_calls"]
        if not isinstance(tcs, list) or not tcs:
            raise SessionCorruptError(f"invalid assistant tool_calls: {m!r}")
        norm_tcs: list[dict[str, Any]] = []
        for tc in tcs:
            fn = tc.get("function") if isinstance(tc, dict) else None
            ok = (
                isinstance(tc, dict)
                and isinstance(tc.get("id"), str) and bool(tc["id"])
                and tc.get("type", "function") == "function"
                and isinstance(fn, dict)
                and isinstance(fn.get("name"), str) and bool(fn["name"])
                and isinstance(fn.get("arguments"), str)
            )
            if not ok:
                raise SessionCorruptError(f"invalid tool_call entry in history: {tc!r}")
            norm_tcs.append({
                "id": tc["id"],
                "type": "function",
                "function": {"name": fn["name"], "arguments": fn["arguments"]},
            })
        out["tool_calls"] = norm_tcs
    elif role == "tool":
        tc_id = m.get("tool_call_id")
        name = m.get("name", "")
        if not isinstance(tc_id, str) or not tc_id or not isinstance(name, str):
            raise SessionCorruptError(f"invalid tool message in history: {m!r}")
        out["tool_call_id"] = tc_id
        if name:
            out["name"] = name
    return out


def validate_session_dict(raw: Any, user_id: str, session_id: str) -> SessionData:
    """校验文件内容并转为 SessionData；标识须与路径一致，任何不符抛 SessionCorruptError。"""
    if not isinstance(raw, dict):
        raise SessionCorruptError("session file must contain a JSON object")
    if raw.get("user_id") != user_id or raw.get("session_id") != session_id:
        raise SessionCorruptError("session file identity does not match its path")
    if not _is_int(raw.get("version")) or raw["version"] < 0:
        raise SessionCorruptError("invalid 'version'")
    if not _is_int(raw.get("next_turn")) or raw["next_turn"] < 1:
        raise SessionCorruptError("invalid 'next_turn'")
    history = raw.get("history")
    if not isinstance(history, list):
        raise SessionCorruptError("invalid 'history'")
    norm_history = [_validate_history_message(m) for m in history]
    todos = _validate_todos(raw.get("todos"))
    summary = raw.get("summary", "")
    if not isinstance(summary, str):
        raise SessionCorruptError("invalid 'summary'")
    tool_calls = _validate_tool_calls(raw.get("tool_calls", []))
    unfinished_turn = _validate_unfinished_turn(raw.get("unfinished_turn", None))
    for key in ("created_at", "updated_at"):
        if not isinstance(raw.get(key), str):
            raise SessionCorruptError(f"invalid '{key}'")
    return SessionData(
        user_id, session_id, raw["version"], raw["next_turn"],
        norm_history,
        todos, raw["created_at"], raw["updated_at"], summary,
        tool_calls, unfinished_turn,
    )


def _age(path: Path) -> float | None:
    """返回文件距上次修改的秒数，文件不存在返回 None。"""
    try:
        return time.time() - path.stat().st_mtime
    except FileNotFoundError:
        return None


class _FileLock:
    """基于独占创建锁文件的跨进程锁。

    - 持有期间由后台心跳线程定期刷新锁文件 mtime，因此长时间运行的持有者不会被判为过期；
    - 只有 mtime 超过 stale_after（心跳停止，即持有进程崩溃/被杀）才可回收；
    - 回收经由 .reclaim 守卫文件串行化，并在守卫内复核过期，避免多个等待者互相删锁；
    - 释放时校验 token，只删除自己的锁。
    """

    def __init__(self, path: Path, timeout: float, stale_after: float, poll: float = 0.05,
                 heartbeat: float | None = None):
        """记录锁路径、等待超时、过期阈值；心跳间隔默认为 stale_after 的 1/4。"""
        self._path, self._timeout, self._stale_after, self._poll = path, timeout, stale_after, poll
        self._guard = path.with_name(path.name + ".reclaim")
        self._heartbeat = heartbeat if heartbeat is not None else max(stale_after / 4, 0.01)
        self._token: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.lost = False

    def _try_reclaim(self) -> bool:
        """锁已过期时在守卫下复核并删除；成功删除（或锁已消失）返回 True。"""
        age = _age(self._path)
        if age is None:
            return True
        if age <= self._stale_after:
            return False
        try:
            fd = os.open(self._guard, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            guard_age = _age(self._guard)  # 回收者自身崩溃遗留的守卫
            if guard_age is not None and guard_age > self._stale_after:
                try:
                    self._guard.unlink()
                except OSError:
                    pass
            return False
        os.close(fd)
        try:
            age = _age(self._path)  # 守卫内复核：期间可能已被他人回收并重新加锁
            if age is None:
                return True
            if age <= self._stale_after:
                return False
            try:
                self._path.unlink()
            except FileNotFoundError:
                pass
            return True
        finally:
            try:
                self._guard.unlink()
            except OSError:
                pass

    def acquire(self, deadline: float | None = None) -> None:
        """获取锁并启动心跳；超过期限仍被占用则抛 SessionBusyError。"""
        deadline = deadline if deadline is not None else time.monotonic() + self._timeout
        token = uuid.uuid4().hex
        while True:
            try:
                fd = os.open(self._path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if self._try_reclaim():
                    continue
                if time.monotonic() >= deadline:
                    raise SessionBusyError("session is in use by another window/process (lock timeout)") from None
                time.sleep(self._poll)
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"pid": os.getpid(), "token": token, "time": time.time()}, f)
            self._token = token
            self._stop.clear()
            self._thread = threading.Thread(target=self._beat, name=f"lock-heartbeat:{self._path.name}", daemon=True)
            self._thread.start()
            return

    def owned(self) -> bool:
        """锁文件当前是否仍属于自己（未被回收或接管）。"""
        if self._token is None or self.lost:
            return False
        try:
            return json.loads(self._path.read_text(encoding="utf-8")).get("token") == self._token
        except (OSError, ValueError, AttributeError):
            return False

    def _beat(self) -> None:
        """心跳线程：定期刷新锁文件 mtime；发现锁已不属于自己则标记 lost 并退出。"""
        while not self._stop.wait(self._heartbeat):
            if not self.owned():
                self.lost = True
                return
            try:
                os.utime(self._path)
            except OSError:
                self.lost = True
                return

    def release(self) -> None:
        """停止心跳并释放锁：仅当锁文件仍属于自己时才删除，避免误删他人的锁。"""
        if self._token is None:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        try:
            if self.owned():
                self._path.unlink()
        except OSError:
            pass
        finally:
            self._token = None


class Session:
    """已加锁打开的会话句柄：持有 SessionData，保存时做锁归属校验、版本校验与原子写入。"""

    def __init__(self, manager: "SessionManager", data: SessionData, lock: _FileLock | None = None):
        """记录所属管理器、数据、所持文件锁及加载时的版本号。"""
        self._manager = manager
        self.data = data
        self._lock = lock
        self._loaded_version = data.version

    def next_turn_id(self) -> int:
        """分配并占用会话内唯一的轮次 ID（随会话持久化，重启后不会重复）。"""
        turn = self.data.next_turn
        self.data.next_turn += 1
        return turn

    def check_lock(self) -> None:
        """确认仍持有会话锁，否则抛 LockLostError。"""
        if self._lock is not None and not self._lock.owned():
            raise LockLostError("session lock was lost (taken over by another window/process); not saving")

    def save(self) -> None:
        """保存会话：须仍持有锁且磁盘版本等于加载版本，否则抛错；成功后版本号加一。"""
        self._manager._save(self)


class SessionManager:
    """会话管理器：每个会话一个 JSON 文件，位于 root/{user_id}/{session_id}.json。"""

    def __init__(self, root: str | os.PathLike, lock_timeout: float = 10.0, stale_after: float = 60.0):
        """设置存储根目录、加锁等待超时（秒）及锁过期阈值（秒，心跳停止超过该时长才视为持有者崩溃）。"""
        self._root = Path(root)
        self._lock_timeout = lock_timeout
        self._stale_after = stale_after
        self._thread_locks: dict[tuple[str, str], threading.Lock] = {}
        self._guard = threading.Lock()

    def _path(self, user_id: str, session_id: str) -> Path:
        """返回会话文件路径（先校验标识）。"""
        check_id("user_id", user_id)
        check_id("session_id", session_id)
        return self._root / user_id / f"{session_id}.json"

    def _thread_lock(self, key: tuple[str, str]) -> threading.Lock:
        """获取某会话的进程内锁（按需创建）。"""
        with self._guard:
            return self._thread_locks.setdefault(key, threading.Lock())

    def exists(self, user_id: str, session_id: str) -> bool:
        """会话文件是否存在。"""
        return self._path(user_id, session_id).exists()

    def list_sessions(self, user_id: str) -> list[str]:
        """列出某用户已有的会话 ID。"""
        check_id("user_id", user_id)
        d = self._root / user_id
        return sorted(p.stem for p in d.glob("*.json")) if d.exists() else []

    def new_session_id(self, user_id: str) -> str:
        """生成该用户下尚未使用的新会话 ID。"""
        while True:
            sid = uuid.uuid4().hex[:8]
            if not self.exists(user_id, sid):
                return sid

    def _load(self, path: Path, user_id: str, session_id: str) -> SessionData | None:
        """读取并校验会话文件；文件不存在返回 None，损坏抛 SessionCorruptError。"""
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except (OSError, UnicodeDecodeError) as e:
            raise SessionCorruptError(f"cannot read session file: {e}") from None
        try:
            raw = json.loads(text)
        except ValueError as e:
            raise SessionCorruptError(f"session file is not valid JSON: {e}") from None
        return validate_session_dict(raw, user_id, session_id)

    @contextmanager
    def open(self, user_id: str, session_id: str, create: bool = False) -> Iterator[Session]:
        """加锁打开会话：先进程内锁再文件锁，覆盖“加载—执行—保存”；退出时无论成败都释放。

        文件不存在时：create=True 返回全新的内存会话（首次 save 才落盘），否则抛 SessionNotFoundError。
        """
        path = self._path(user_id, session_id)
        deadline = time.monotonic() + self._lock_timeout
        tlock = self._thread_lock((user_id, session_id))
        if not tlock.acquire(timeout=self._lock_timeout):
            raise SessionBusyError("session is in use by another window (lock timeout)")
        flock = _FileLock(path.with_name(path.name + ".lock"), self._lock_timeout, self._stale_after)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            flock.acquire(deadline)
            try:
                data = self._load(path, user_id, session_id)
                if data is None:
                    if not create:
                        raise SessionNotFoundError(f"session '{session_id}' of user '{user_id}' does not exist")
                    data = SessionData(user_id, session_id)
                yield Session(self, data, flock)
            finally:
                flock.release()
        finally:
            tlock.release()

    def _disk_version(self, path: Path, user_id: str, session_id: str) -> int:
        """读取磁盘上当前的版本号，文件不存在为 0。"""
        data = self._load(path, user_id, session_id)
        return 0 if data is None else data.version

    def _save(self, session: Session) -> None:
        """版本校验后原子写入：临时文件 → flush+fsync → os.replace；失败时清理临时文件、原文件不变。"""
        d = session.data
        path = self._path(d.user_id, d.session_id)
        disk = self._disk_version(path, d.user_id, d.session_id)
        session.check_lock()
        if disk != session._loaded_version:
            raise VersionConflictError(f"version conflict: loaded {session._loaded_version}, on disk {disk}")
        new_version = d.version + 1
        payload = {**d.to_dict(), "version": new_version, "updated_at": _now()}
        validate_session_dict(payload, d.user_id, d.session_id)  # 写入前再校验，避免落盘坏数据
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            session.check_lock()  # 替换前再确认一次，缩小窗口
            os.replace(tmp, path)
        except BaseException:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise
        d.version, d.updated_at = new_version, payload["updated_at"]
        session._loaded_version = new_version
