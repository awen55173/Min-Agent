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
uv run python cli.py -u alice -s <session_id> "我有哪些待办？"
# 不带输入进入交互模式（空行 / exit 退出）
uv run python cli.py -u alice
# 列出用户的会话
uv run python cli.py -u alice --list
```
会话数据保存在 `data/{user_id}/{session_id}.json`（可用 `--data-dir` 修改）；`-v` 打印工具调用过程。
请在项目根目录运行（`.env` 从当前目录读取）。

## 会话与存储
- 每个会话一个 JSON，保存历史（不含 system 提示词）、待办及下一条待办 ID、轮次计数、版本号。
- 写入：临时文件 → flush + fsync → `os.replace` 原子替换；读写前后校验标识与内容，损坏文件不会被覆盖。
- 并发：`ChatService.send` 在“加载—执行—保存”全程持有会话锁（进程内锁 + 锁文件）；拿不到锁等待至超时（默认 10s）后报“会话正被占用”；
  锁文件 mtime 超过 600s 视为崩溃遗留并回收；保存前再校验磁盘版本，不一致则拒绝写入。
- 失败退出（步数耗尽、解析失败、API 失败、内部错误、Ctrl+C）也会保存历史与已完成的待办操作。

## 测试
```powershell
uv run pytest                 # 默认跳过真实 API
uv run pytest -m live         # 真实 API 测试（需 .env）
```
