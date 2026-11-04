import os

import pytest

import cli
from agent.config import ConfigError, Settings
from agent.llm import FakeLLM, LLMError, OpenAIClient


@pytest.fixture
def clean_env(monkeypatch):
    for k in ("LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL", "LLM_TIMEOUT", "LLM_MAX_RETRIES", "TAVILY_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_settings_missing(clean_env, tmp_path):
    with pytest.raises(ConfigError):
        Settings.from_env(str(tmp_path / "none.env"))


def test_settings_from_env(clean_env, tmp_path):
    clean_env.setenv("LLM_API_KEY", "k")
    clean_env.setenv("LLM_MODEL", "m")
    clean_env.setenv("LLM_BASE_URL", "http://x/v1")
    s = Settings.from_env(str(tmp_path / "none.env"))
    assert (s.api_key, s.model, s.base_url) == ("k", "m", "http://x/v1")


def test_fake_llm_records_and_exhausts():
    llm = FakeLLM(["a"])
    assert llm.chat([{"role": "user", "content": "hi"}]) == "a"
    assert llm.calls[0][0]["content"] == "hi"
    with pytest.raises(LLMError):
        llm.chat([])


def test_run_once_uses_client():
    llm = FakeLLM(["pong"])
    assert cli.run_once(llm, "ping") == "pong"
    assert llm.calls[0][-1] == {"role": "user", "content": "ping"}


def test_cli_config_error_returns_1(clean_env, tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no .env here
    assert cli.main(["hi"]) == 1
    assert "error" in capsys.readouterr().err


def test_openai_client_retries_then_fails(clean_env):
    from openai import APIConnectionError

    s = Settings(api_key="k", base_url="http://127.0.0.1:1/v1", model="m", max_retries=1)
    client = OpenAIClient(s, backoff=0)
    calls = {"n": 0}

    def boom(**kw):
        calls["n"] += 1
        raise APIConnectionError(request=object())

    client._client.chat.completions.create = boom
    with pytest.raises(LLMError):
        client.chat([{"role": "user", "content": "x"}])
    assert calls["n"] == 2


@pytest.mark.live
def test_live_ping():
    s = Settings.from_env()
    out = OpenAIClient(s).chat([{"role": "user", "content": "Reply with exactly: pong"}])
    assert "pong" in out.lower()
