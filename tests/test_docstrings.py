"""检查运行代码中的方法是否包含中文说明。"""

import ast
from pathlib import Path

import pytest


@pytest.mark.parametrize("relative_path", ["agent/config.py", "agent/llm.py", "cli.py"])
def test_runtime_methods_have_chinese_docstrings(relative_path):
    """验证各运行方法都有包含中文的非空文档字符串。"""
    path = Path(__file__).resolve().parents[1] / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            description = ast.get_docstring(node) or ""
            assert any("\u4e00" <= char <= "\u9fff" for char in description), (
                f"{relative_path}:{node.lineno} 的 {node.name} 缺少中文方法说明"
            )
