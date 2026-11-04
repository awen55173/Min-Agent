import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "live: tests that call the real LLM API (skipped by default)")


def pytest_collection_modifyitems(config, items):
    if "live" in (config.getoption("-m") or ""):
        return
    skip = pytest.mark.skip(reason="live test; run with -m live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
