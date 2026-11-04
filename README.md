# min_agent

从零实现的最小可用 Agent（不依赖 Agent 框架）。详见 [PLAN.md](PLAN.md)。

## 环境（uv + venv）
```powershell
uv venv --python 3.11      # 创建 .venv
uv sync                    # 安装依赖
copy .env.example .env     # 填写 LLM_API_KEY / LLM_MODEL / LLM_BASE_URL
```

## 运行（M0 冒烟）
```powershell
uv run python cli.py "Reply with exactly: pong"
```

## 测试
```powershell
uv run pytest                 # 默认跳过真实 API
uv run pytest -m live         # 真实 API 测试（需 .env）
```
