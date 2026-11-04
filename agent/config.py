"""从环境变量和 .env 文件读取并校验配置。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


class ConfigError(Exception):
    """表示必需配置缺失或配置值不合法。"""


def _num(name: str, default: str, cast, minimum: float) -> float:
    """读取数值配置，空值使用默认值，并校验类型和下限。"""
    raw = os.getenv(name, "").strip() or default
    try:
        value = cast(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from None
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    """保存模型调用和工具所需的配置。"""

    api_key: str
    base_url: str | None
    model: str
    timeout: float = 60.0
    max_retries: int = 2
    tavily_api_key: str | None = None

    @classmethod
    def from_env(cls, dotenv_path: str | os.PathLike | None = None) -> "Settings":
        """加载并校验配置，默认读取当前目录的 .env，保留已有环境变量。"""
        load_dotenv(dotenv_path or Path.cwd() / ".env", override=False)
        api_key = os.getenv("LLM_API_KEY", "").strip()
        model = os.getenv("LLM_MODEL", "").strip()
        missing = [n for n, v in (("LLM_API_KEY", api_key), ("LLM_MODEL", model)) if not v]
        if missing:
            raise ConfigError(f"Missing required env vars: {', '.join(missing)} (see .env.example)")
        return cls(
            api_key=api_key,
            base_url=os.getenv("LLM_BASE_URL", "").strip() or None,
            model=model,
            timeout=_num("LLM_TIMEOUT", "60", float, 0.001),
            max_retries=int(_num("LLM_MAX_RETRIES", "2", int, 0)),
            tavily_api_key=os.getenv("TAVILY_API_KEY", "").strip() or None,
        )
