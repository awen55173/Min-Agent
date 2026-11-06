"""仅从明确的用户陈述维护当前会话中的短事实。"""
from __future__ import annotations

import re


_SEPARATOR = re.compile(r"[，。；;\n]+|(?<!\d)[,.]|[,.](?!\d)")
_FIELD = r"饮食偏好|饮食习惯|居住城市|居住地|所在地|姓名|名字|称呼|城市|预算|口味|偏好|饮食"
_ALIASES = {
    "姓名": "名字", "称呼": "名字",
    "居住城市": "城市", "居住地": "城市", "所在地": "城市",
    "饮食习惯": "饮食偏好", "饮食": "饮食偏好", "口味": "饮食偏好",
}
_NO_MEMORY = re.compile(
    r"不要记(?:忆|住|录)|别记(?:忆|住|录)|无需记(?:忆|住|录)|不(?:要|用)保存|请勿保存|"
    r"不(?:要|用)记录|示例|例子|样例|举例|假设|假如|比如|例如|引用|下面这段|以下内容|"
    r"测试文本|演示|范例|模板|请检查|请改写|请翻译|分析以下|"
    r"不是|并非|不在|不喜欢|如果"
)
_RECALL = re.compile(
    r"(?:按|照)(?:刚才|之前|上次)(?:的)?要求|"
    r"(?:之前|上次)说的(?:城市|预算|名字|偏好)|"
    r"(?:还)?记得我(?:的)?(?:名字|城市|预算|偏好|信息)|"
    r"我的信息|我的记忆|保存了什么|remember", re.I,
)
_QUESTION = re.compile(
    r"我的(?:城市|居住地|所在地|预算|饮食偏好|饮食习惯|偏好|名字|姓名|称呼)"
    r"(?:是什么|是多少|在哪|哪里|吗)|我(?:是谁|叫什么|住在哪)"
)
_QUESTION_VALUE = re.compile(r"^(?:什么|多少|哪里|哪儿|谁|几|是否|是不是|吗|怎么|如何|啥|哪个|哪座)")


def _safe_input(user_input: str) -> bool:
    """排除整条输入中的不记忆、示例、假设、引用和否定语境。"""
    return not (_NO_MEMORY.search(user_input) or re.search(r"[“”\"'‘’]|^\s*>", user_input, re.M)
                or chr(96) * 3 in user_input)


def _operation(clause: str) -> tuple[str, str, str] | None:
    """仅识别白名单字段上的明确设置、修改和删除。"""
    clause = clause.strip().strip("！! ")
    if not clause or "?" in clause or "？" in clause:
        return None
    delete = re.fullmatch(rf"(?:请)?(?:忘记|删除|清除)(?:掉)?(?:记忆[:：])?(?:我的)?({_FIELD})", clause)
    if delete:
        field = delete.group(1)
        return "delete", _ALIASES.get(field, field), ""
    update = re.fullmatch(rf"(?:请)?(?:把)?(?:我的)?({_FIELD})(?:改成|改为|更新为|设为)(.{{1,200}})", clause)
    if update:
        field, value = update.groups()
    else:
        stated = re.fullmatch(
            rf"(?:(?:请)?(?:记住|保存|记录)(?:一下)?[:：]?)?我的({_FIELD})(?:是|为|叫|[:：])(.{{1,200}})",
            clause,
        )
        explicit = re.fullmatch(
            rf"(?:请)?(?:记住|保存|记录)(?:一下)?[:：]?(?:我的)?({_FIELD})(?:是|为|叫|[:：])(.{{1,200}})",
            clause,
        )
        match = stated or explicit
        if match:
            field, value = match.groups()
        else:
            name = re.fullmatch(r"(?:(?:请)?记住[:：]?)?我叫(.{1,200})", clause)
            preference = re.fullmatch(r"(?:(?:请)?记住[:：]?)?我(?:喜欢|偏好)(.{1,200})", clause)
            residence = re.fullmatch(r"(?:(?:请)?记住[:：]?)?我住在(.{1,200})", clause)
            if name:
                field, value = "名字", name.group(1)
            elif preference:
                field, value = "偏好", preference.group(1)
            elif residence:
                field, value = "城市", residence.group(1)
            else:
                return None
    value = value.strip()
    if not value or _QUESTION_VALUE.match(value) or value.endswith(("吗", "么", "呢")):
        return None
    return "set", _ALIASES.get(field, field), value


def _operations(user_input: str) -> list[tuple[str, str, str]]:
    """在整条输入通过语境检查后提取少量可信操作。"""
    if not _safe_input(user_input):
        return []
    return [op for clause in _SEPARATOR.split(user_input) if (op := _operation(clause)) is not None]


def apply_user_memory(memories: dict[str, str], user_input: str) -> dict[str, str]:
    """仅根据本轮用户原文更新或删除会话记忆。"""
    updated = dict(memories)
    for action, field, value in _operations(user_input):
        if action == "delete":
            updated.pop(field, None)
        else:
            updated[field] = value
    return updated


def strip_memory_declarations(user_input: str) -> str:
    """从摘要中移除已识别的用户记忆声明，保留其他输入。"""
    if not _operations(user_input):
        return user_input
    return "，".join(clause.strip() for clause in _SEPARATOR.split(user_input)
                    if clause.strip() and _operation(clause) is None)


def filter_historical_user_input(user_input: str, memories: dict[str, str] | None) -> str:
    """回放时只移除已过期或被删除的用户记忆声明。"""
    if memories is None or not _operations(user_input):
        return user_input
    return "，".join(clause.strip() for clause in _SEPARATOR.split(user_input)
                    if clause.strip() and ((op := _operation(clause)) is None
                                           or (op[0] == "set" and memories.get(op[1]) == op[2])))


def select_memories(memories: dict[str, str], user_input: str) -> dict[str, str]:
    """按字段别名、姓名问法或本轮指代召回相关记忆。"""
    if _RECALL.search(user_input):
        return dict(memories)
    selected = {field: value for field, value in memories.items()
                if field in user_input or any(alias in user_input for alias, canonical in _ALIASES.items()
                                              if canonical == field)}
    if "我叫什么" in user_input or "我是谁" in user_input:
        if "名字" in memories:
            selected["名字"] = memories["名字"]
    return selected


def wants_memory_recall(user_input: str) -> bool:
    """判断本轮是否在询问已保存事实或指代先前要求。"""
    return bool(_RECALL.search(user_input) or _QUESTION.search(user_input))
