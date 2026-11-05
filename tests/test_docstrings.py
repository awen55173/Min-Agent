"""检查运行时代码中的方法都带有中文说明。"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted(
    [*(p.relative_to(ROOT).as_posix() for p in (ROOT / "agent").rglob("*.py")), "cli.py"]
)


@pytest.mark.parametrize("relative_path", FILES)
def test_runtime_methods_have_chinese_docstrings(relative_path):
    """验证每个函数/方法（含嵌套函数）都有包含中文的非空说明。"""
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            description = ast.get_docstring(node) or ""
            assert any("\u4e00" <= char <= "\u9fff" for char in description), (
                f"{relative_path}:{node.lineno} 的 {node.name} 缺少中文说明"
            )


def test_docstring_check_covers_tool_modules():
    """确认检查范围包含工具模块。"""
    assert "agent/tools/registry.py" in FILES and "agent/tools/todo.py" in FILES
