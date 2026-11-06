"""仅从当前用户输入维护会话内可更新、可删除的短事实。"""
from __future__ import annotations

import re


_SEPARATOR = re.compile(r"[，。；;\n]+|(?<!\d)[,.]|[,.](?!\d)")
_KEY_ALIASES = {
    "居住地": "城市", "所在地": "城市", "居住城市": "城市",
    "饮食习惯": "饮食偏好", "饮食": "饮食偏好", "口味": "饮食偏好",
}
_RECALL_RE = re.compile(r"记得|还记得|之前说|上次说|我的信息|我的记忆|保存了什么|remember", re.I)
_QUESTION_RE = re.compile(
    r"我的(?:城市|居住地|所在地|预算|饮食偏好|饮食习惯|偏好|名字)"
    r"(?:是什么|是多少|在哪|哪里|吗)|我(?:是谁|叫什么|住在哪)"
)


def _key(raw: str) -> str:
    """清理用户给出的事实键并合并常见同义词。"""
    key = re.sub(r"\s+", "", raw).removeprefix("我的").removeprefix("我")
    return _KEY_ALIASES.get(key, key)


def _operation(clause: str) -> tuple[str, str, str] | None:
    """只解析用户明确陈述的设置、修改或删除语句。"""
    clause = clause.strip()
    if not clause or "?" in clause or "？" in clause:
        return None
    clause = clause.strip("！! ")
    delete = re.fullmatch(r"(?:请)?(?:忘记|删除|清除)(?:掉)?(?:记忆[:：])?(?:我的)?([\w\u4e00-\u9fff]{1,24})", clause)
    if delete:
        return "delete", _key(delete.group(1)), ""
    update = re.fullmatch(r"(?:请)?(?:把)?(?:我的)?([\w\u4e00-\u9fff]{1,24})(?:改成|改为|更新为|设为)(.{1,200})", clause)
    if update:
        return "set", _key(update.group(1)), update.group(2).strip()
    stated = re.fullmatch(
        r"(?:(?:请)?(?:记住|保存|记录)(?:一下)?[:：]?)?(我的|我)([\w\u4e00-\u9fff]{1,24})(?:是|为|叫|[:：])(.{1,200})",
        clause,
    )
    if stated:
        return "set", _key(stated.group(2)), stated.group(3).strip()
    explicit = re.fullmatch(r"(?:请)?(?:记住|保存|记录)(?:一下)?[:：]?(?:我的)?([\w\u4e00-\u9fff]{1,24})(?:是|为|叫|[:：])(.{1,200})", clause)
    if explicit:
        return "set", _key(explicit.group(1)), explicit.group(2).strip()
    preference = re.fullmatch(r"(?:(?:请)?记住[:：]?)?我(?:喜欢|偏好)(.{1,200})", clause)
    if preference:
        return "set", "偏好", preference.group(1).strip()
    residence = re.fullmatch(r"(?:(?:请)?记住[:：]?)?我住在(.{1,200})", clause)
    if residence:
        return "set", "城市", residence.group(1).strip()
    return None


def apply_user_memory(memories: dict[str, str], user_input: str) -> dict[str, str]:
    """从本轮用户原文提取显式事实变更，并返回更新后的会话记忆。"""
    updated = dict(memories)
    for clause in _SEPARATOR.split(user_input):
        operation = _operation(clause)
        if operation is None:
            continue
        action, key, value = operation
        if action == "delete":
            updated.pop(key, None)
        elif key and value:
            updated[key] = value
    return updated


def strip_memory_declarations(user_input: str) -> str:
    """从历史回放中移除已由会话记忆管理的用户声明。"""
    if not any(_operation(clause) is not None for clause in _SEPARATOR.split(user_input)):
        return user_input
    return "，".join(clause.strip() for clause in _SEPARATOR.split(user_input)
                    if clause.strip() and _operation(clause) is None)


def select_memories(memories: dict[str, str], user_input: str) -> dict[str, str]:
    """仅在当前问题涉及对应键或询问记忆时召回相关短事实。"""
    selected = {key: value for key, value in memories.items()
                if key in user_input or any(alias in user_input for alias, canonical in _KEY_ALIASES.items()
                                           if canonical == key)}
    if selected:
        return selected
    if _RECALL_RE.search(user_input):
        return dict(memories)
    return {}


def wants_memory_recall(user_input: str) -> bool:
    """判断用户是否正在询问此前保存的个人信息。"""
    return bool(_RECALL_RE.search(user_input) or _QUESTION_RE.search(user_input))
