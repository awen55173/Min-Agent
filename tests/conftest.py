import pytest


def pytest_configure(config):
    """注册真实 API 测试标记。"""
    config.addinivalue_line("markers", "live: tests that call the real LLM API (skipped by default)")


def pytest_collection_modifyitems(config, items):
    """默认跳过真实 API 测试，标记表达式含 live 时交由 pytest 筛选。"""
    if "live" in (config.getoption("-m") or ""):
        return
    skip = pytest.mark.skip(reason="live test; run with -m live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
