# min_agent

从零实现的最小可用 Agent（不依赖 Agent 框架）。详见 [PLAN.md](PLAN.md)。

## 环境（uv + venv）
```powershell
uv venv --python 3.11      # 创建 .venv
uv sync                    # 安装依赖
copy .env.example .env     # 填写 LLM_API_KEY / LLM_MODEL / LLM_BASE_URL / TAVILY_API_KEY
```

## 运行
```powershell
# 默认新建独立会话并显示 ID（会话 ID 输出到 stderr）
uv run python cli.py -u alice "查一下北京天气并记一条待办"
# 按 ID 恢复会话继续聊
uv run python cli.py -u alice -s session_id "我有哪些待办？"
# 不带输入进入交互模式（空行 / exit 退出）
uv run python cli.py -u alice
# 列出用户的会话
uv run python cli.py -u alice --list
```
会话数据保存在 `data/{user_id}/{session_id}.json`（可用 `--data-dir` 修改）；`-v` 打印工具调用过程。
请在项目根目录运行（`.env` 从当前目录读取）。

## 会话与存储
- 每个会话一个 JSON，保存完整历史（不含 system 提示词，供交互式回放）、结构化摘要（`summary`）、待办及下一条待办 ID、轮次计数、版本号。
- 写入：临时文件 → flush + fsync → `os.replace` 原子替换；读写前后校验标识与内容，损坏文件不会被覆盖。
- 并发：`ChatService.send` 在“加载—执行—保存”全程持有会话锁（进程内锁 + 锁文件）；持有期间后台心跳定期刷新锁文件 `mtime`，防止长耗时调用被误判过期；
  拿不到锁等待至超时（默认 10s）后报“会话正被占用”；锁文件心跳停止超过 `stale_after`（默认 60s）才视为崩溃遗留并在守卫下回收；保存前再校验锁归属与磁盘版本，不一致则拒绝写入。
- 失败退出（步数耗尽、解析失败、API 失败、内部错误、Ctrl+C）也会保存历史（含已完成的工具调用与 observation）及待办操作。

## 上下文管理与压缩（M4）
- 每次调用 `Agent.run()` 前由 `agent/context.py` 的 `build_context()` 构建发送给模型的有限上下文：`[系统提示词, (可选)结构化摘要, 最近 N 轮对话原文(剥离历史 thought), 本轮输入]`。
- 触发压缩条件：历史轮数超过 `max_history_turns`（默认 6），或估算 token 数超过 `max_context_tokens - reserved_tokens`（默认 4000 - 1200，为本轮工具调用与结果预留空间）。
- 结构化摘要显式保留 5 类信息：用户目标、关键事实、待办 ID 与状态、已执行操作、未完成事项；历史 `thought` 不进入摘要或历史消息；待办状态与 `TodoStore` 实时对齐。
- 完整的持久化历史（`data.history`）与发送给模型的受限上下文分离，压缩不会影响 CLI 交互式会话恢复时的完整历史回放。

## 测试
```powershell
uv run pytest                 # 默认跳过真实 API
uv run pytest -m live         # 真实 API 测试（需 .env）
```
