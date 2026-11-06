"""检查项目文本文件编码：必须是合法 UTF-8，且不含替换字符（乱码标志）。"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = ["README.md", *sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "docs").glob("*.md"))]


@pytest.mark.parametrize("relative_path", FILES)
def test_docs_are_clean_utf8(relative_path):
    """验证文档能按 UTF-8 严格解码，且不含 U+FFFD。"""
    text = (ROOT / relative_path).read_bytes().decode("utf-8")
    assert "\ufffd" not in text


def test_docs_checked():
    """确认检查范围包含 README 和两份记录文件。"""
    assert "README.md" in FILES
    assert "docs/prompts.md" in FILES and "docs/questions.md" in FILES
