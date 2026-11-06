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
# 查看指定会话的 JSONL Trace 执行日志
uv run python cli.py -u alice -s session_id --trace
# 根据模型容量调整每次请求的估算上下文上限（默认 16000）
uv run python cli.py --max-context-tokens 24000 -u alice
# 默认使用原生函数工具调用（native），也可显式指定兼容文本模式（text）
uv run python cli.py --tool-mode text -u alice "计算 3 * 9"
```
会话数据保存在 `data/{user_id}/{session_id}.json`（可用 `--data-dir` 修改），Trace 日志保存在 `data/traces/{user_id}/{session_id}.jsonl`（可用 `--trace-dir` 修改）；`-v` 打印工具调用过程。
请在项目根目录运行（`.env` 从当前目录读取）。

## 原生工具调用、异步执行与受控并行调度
- **原生工具调用（主路径）与文本兼容模式**：注册表自动导出 OpenAI `tools` Schema，模型响应统一为 `LLMResponse(content, tool_calls)`，执行后按 `role="tool"` + `tool_call_id` 协议回填；同时保留 `--tool-mode text` 作为显式兼容选项。
- **异步执行链**：`OpenAIClient`（`acomplete`/`achat`）、`search`、`weather`、`Agent.arun` 与 `ChatService.asend` 全链路支持异步执行，统一超时控制、取消信号传播（`CancelledError`）与指数退避重试；同步工具通过线程适配，避免阻塞事件循环。只读同步工具超时后返回“后台执行状态未确定”（线程可能仍在运行）；写工具保持串行，超时后等实际执行结束，再返回真实结果和 `completed_after_timeout` 状态。异步会话锁（`aopen`）与保存（`asave`）避免慢会话阻塞其他并发会话。
- **受控并行调度**：支持单批多个工具调用（`parallel_tool_calls=True`），仅将无依赖只读工具（`weather`/`search`/`calculator`）在并发上限（`max_concurrency`）内并行执行；同一会话的待办写操作（`add`/`complete`/`delete`）及相关读写保持严格串行并立即落盘。单项超时或错误独立记录结果，不影响同批其他调用；并行乱序完成的结果按原调用顺序精准对应各自的 `api_call_id` 与内部幂等 `call_id`，且批量调用逐个计入 `max_steps` 预算。

## 会话与存储
- 每个会话一个 JSON，保存完整历史（不含 system 提示词，供交互式回放）、结构化摘要（`summary`）、用户记忆（`memories`）、待办及下一条待办 ID、工具调用记录（`tool_calls`）、未完成轮次状态（`unfinished_turn`）、轮次计数、版本号。
- 写入：临时文件 → flush + fsync → `os.replace` 原子替换；读写前后校验标识与内容，损坏文件不会被覆盖。
- 写操作立即落盘与幂等恢复：待办写操作（`add`/`complete`/`delete`）成功后立即触发原子保存，将最新待办状态与成功调用记录（含 `call_id`、`user_id`、`session_id`、`turn_id`、参数、状态及结果）在同一事务中落盘，不等整轮结束。若后续模型请求失败或进程中断，恢复重试时自动复用未完成轮次的 `turn_id` 与 `call_id`：已成功的写操作直接返回已有结果，状态不确定（`pending`/`interrupted`）时先与 `TodoStore` 核对，避免重复写入；轮次正常完成后下一轮可正常创建同名待办。
- 并发：`ChatService.send` 在“加载—执行—保存”全程持有会话锁（进程内锁 + 锁文件）；持有期间后台心跳定期刷新锁文件 `mtime`，防止长耗时调用被误判过期；
  拿不到锁等待至超时（默认 10s）后报“会话正被占用”；锁文件心跳停止超过 `stale_after`（默认 60s）才视为崩溃遗留并在守卫下回收；保存前再校验锁归属与磁盘版本，不一致则拒绝写入。
- 失败退出与 Trace：统一处理正常结束与各类异常终止（`max_steps`、`parse_retries_exhausted`、`llm_error`、`internal_error`、`interrupted`、`tool_not_executed`、`context_limit`、`persist_error`），返回已完成操作与未完成说明；`agent/trace.py` 以 JSONL 格式记录每轮开始、工具请求/结果、耗时、错误及终止原因，并可识别中断遗留的未完成调用。

## 上下文管理与压缩
- 每次调用 `Agent.run()` 前由 `agent/context.py` 的 `build_context()` 构建发送给模型的有限上下文：`[系统提示词, (相关时)当前会话用户记忆, (可选)结构化摘要, 最近 N 轮对话原文(剥离历史 thought; 工具调用名、参数,最终答案,工具输出都保留), 本轮输入]`。
- 触发压缩条件：历史轮数超过 `max_history_turns`（默认 6），或估算 token 数超过 `max_context_tokens - reserved_tokens`（默认 16000 - 4000，为本轮工具调用与结果预留空间）。摘要默认限制为 2000 估算 tokens，包含待办的条数和文本长度限制；省略条目时提示通过 `todo.list` 查询完整实时状态。
- 每次模型调用（包括连续工具调用、格式纠正）发送前检查消息与原生 `tools` Schema 的总硬上限；无法容纳时返回 `context_limit` 并保存已执行操作，提示缩短输入或分批处理。系统提示和本轮输入不会被静默截断。此上限使用项目估算器，不等同于模型 tokenizer 的精确计数。
- 非法模型输出不会进入有效历史；纠错明确提示“本次工具未执行”。若模型随后直接声称已完成，返回 `tool_not_executed`，必须重新合法执行工具后才能正常结束。
- 结构化摘要显式保留 5 类信息：用户目标、关键事实、待办 ID 与状态、已执行操作、未完成事项；历史 `thought` 不进入摘要或历史消息；待办状态与 `TodoStore` 实时对齐。
- 完整的持久化历史（`data.history`）与发送给模型的受限上下文分离，压缩不会影响 CLI 交互式会话恢复时的完整历史回放。

## 会话用户记忆
- 只解析当前用户明确写出的姓名、城市、预算和偏好等白名单字段，例如“记住我的城市是苏州”“我的名字是李为民”；“把我的城市改成杭州”更新同一键，“忘记我的城市”删除。疑问、否定、不记忆指令、示例、假设和引用文本不会写入记忆。每个 session 独立保存，重启后仍可用。
- 新问题提及字段、询问姓名或指代“按刚才的要求”时，把相关记忆放在系统提示之后、历史摘要之前。模型回答和工具结果不会写入 `memories`；历史回放保留当前有效的用户声明与示例原文，仅剔除已更新或删除的旧记忆声明。

## 测试
```powershell
uv run pytest                 # 默认跳过真实 API
uv run pytest -m live         # 真实 API 测试（需 .env）
```
