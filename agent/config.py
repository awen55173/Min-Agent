"""Configuration loaded from environment variables (.env supported)."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


class ConfigError(Exception):
    """Raised when required configuration is missing."""


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str | None
    model: str
    timeout: float = 60.0
    max_retries: int = 2
    tavily_api_key: str | None = None

    @classmethod
    def from_env(cls, dotenv_path: str | None = None) -> "Settings":
        load_dotenv(dotenv_path, override=False)
        api_key = os.getenv("LLM_API_KEY", "").strip()
        model = os.getenv("LLM_MODEL", "").strip()
        missing = [n for n, v in (("LLM_API_KEY", api_key), ("LLM_MODEL", model)) if not v]
        if missing:
            raise ConfigError(f"Missing required env vars: {', '.join(missing)} (see .env.example)")
        return cls(
            api_key=api_key,
            base_url=os.getenv("LLM_BASE_URL", "").strip() or None,
            model=model,
            timeout=float(os.getenv("LLM_TIMEOUT", "60")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
            tavily_api_key=os.getenv("TAVILY_API_KEY", "").strip() or None,
        )
