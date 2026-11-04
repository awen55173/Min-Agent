import os

import pytest

import cli
from agent.config import ConfigError, Settings
from agent.llm import FakeLLM, LLMError, OpenAIClient


@pytest.fixture
def clean_env(monkeypatch):
    """清除模型和工具配置变量，供测试独立设置环境。"""
    for k in ("LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL", "LLM_TIMEOUT", "LLM_MAX_RETRIES", "TAVILY_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_settings_missing(clean_env, tmp_path):
    """验证缺少必需配置时抛出配置异常。"""
    with pytest.raises(ConfigError):
        Settings.from_env(str(tmp_path / "none.env"))


def test_settings_from_env(clean_env, tmp_path):
    """验证从环境变量读取密钥、模型和服务地址。"""
    clean_env.setenv("LLM_API_KEY", "k")
    clean_env.setenv("LLM_MODEL", "m")
    clean_env.setenv("LLM_BASE_URL", "http://x/v1")
    s = Settings.from_env(str(tmp_path / "none.env"))
    assert (s.api_key, s.model, s.base_url) == ("k", "m", "http://x/v1")


def test_fake_llm_records_and_exhausts():
    """验证模拟客户端记录消息，并在预设回复耗尽时抛出异常。"""
    llm = FakeLLM(["a"])
    assert llm.chat([{"role": "user", "content": "hi"}]) == "a"
    assert llm.calls[0][0]["content"] == "hi"
    with pytest.raises(LLMError):
        llm.chat([])


def test_run_once_uses_client():
    """验证用户输入传入客户端，并返回预设回复。"""
    llm = FakeLLM(["pong"])
    assert cli.run_once(llm, "ping") == "pong"
    assert llm.calls[0][-1] == {"role": "user", "content": "ping"}


def test_cli_config_error_returns_1(clean_env, tmp_path, capsys, monkeypatch):
    """验证缺配置时命令行输出错误并返回失败状态。"""
    monkeypatch.chdir(tmp_path)  # 临时目录中没有 .env 文件。
    assert cli.main(["hi"]) == 1
    assert "error" in capsys.readouterr().err


def test_openai_client_retries_then_fails(clean_env):
    """验证连接失败按配置重试，并在耗尽次数后抛出调用异常。"""
    from openai import APIConnectionError

    s = Settings(api_key="k", base_url="http://127.0.0.1:1/v1", model="m", max_retries=1)
    client = OpenAIClient(s, backoff=0)
    calls = {"n": 0}

    def boom(**kw):
        """记录请求次数并模拟连接失败。"""
        calls["n"] += 1
        raise APIConnectionError(request=object())

    client._client.chat.completions.create = boom
    with pytest.raises(LLMError):
        client.chat([{"role": "user", "content": "x"}])
    assert calls["n"] == 2


@pytest.mark.live
def test_live_ping():
    """调用真实模型，验证回复中包含 pong。"""
    s = Settings.from_env()
    out = OpenAIClient(s).chat([{"role": "user", "content": "Reply with exactly: pong"}])
    assert "pong" in out.lower()


def _settings_env(monkeypatch, tmp_path, **kv):
    """设置测试配置，返回不存在的文件路径以隔离本地 .env。"""
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "m")
    for k, v in kv.items():
        monkeypatch.setenv(k, v)
    return str(tmp_path / "none.env")


@pytest.mark.parametrize("kv", [
    {"LLM_TIMEOUT": "abc"}, {"LLM_TIMEOUT": "-1"}, {"LLM_TIMEOUT": "0"},
    {"LLM_MAX_RETRIES": "-1"}, {"LLM_MAX_RETRIES": "x"},
])
def test_settings_invalid_numbers(clean_env, tmp_path, kv):
    """验证非法超时和重试次数被转换为配置异常。"""
    path = _settings_env(clean_env, tmp_path, **kv)
    with pytest.raises(ConfigError):
        Settings.from_env(path)


def test_settings_does_not_read_project_env_from_other_cwd(clean_env, tmp_path):
    """验证切换目录后不会自动读取项目中的 .env。"""
    clean_env.chdir(tmp_path)
    with pytest.raises(ConfigError):
        Settings.from_env()


class _Msg:
    def __init__(self, content):
        """保存模拟消息的正文。"""
        self.content = content


class _Choice:
    def __init__(self, content):
        """将模拟消息封装为候选回复。"""
        self.message = _Msg(content)


class _Resp:
    def __init__(self, choices):
        """保存模拟响应的候选回复列表。"""
        self.choices = choices


def _client_returning(resp):
    """创建客户端并替换请求方法，使其返回指定的模拟响应。"""
    client = OpenAIClient(Settings(api_key="k", base_url=None, model="m"), backoff=0)
    client._client.chat.completions.create = lambda **kw: resp
    return client


@pytest.mark.parametrize("resp", [
    _Resp([]), _Resp(None), _Resp([_Choice(None)]), _Resp([_Choice("")]), _Resp([_Choice("  \n")]),
])
def test_openai_client_bad_responses(resp):
    """验证缺少候选或回复为空时抛出调用异常。"""
    with pytest.raises(LLMError):
        _client_returning(resp).chat([{"role": "user", "content": "x"}])


def test_openai_client_ok_response():
    """验证从正常响应中提取回复文本。"""
    assert _client_returning(_Resp([_Choice("pong")])).chat([]) == "pong"
