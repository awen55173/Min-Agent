# 问题解决记录

| 问题 | 原因 | 解决方案 |
|---|---|---|
| pytest 报 `No module named 'cli'` | 根目录模块不在 pytest 的 sys.path 中 | pyproject 中配置 `pythonpath = ["."]` |
| 测试里 `import httpx` 失败 | 新版 openai 依赖 httpx2，环境中无 httpx | 测试中不依赖 httpx，用占位对象构造 APIConnectionError |
| PLAN.md 中问题记录文件名与约定不一致 | 早期命名为 troubleshooting.md | 统一改为 docs/questions.md |
| 当前 .env 为空，无法做真实 API 调用 | 尚未填写密钥 | 参考 .env.example 填写后运行 `uv run python cli.py` / `uv run pytest -m live` |
