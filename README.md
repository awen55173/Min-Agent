# min_agent

一个用 Python 自行实现 Agent 主循环、工具调度、session 与上下文管理的最小 Agent；不依赖 LangGraph、OpenHands 等 Agent 框架。项目通过 OpenAI-compatible Chat Completions API 调用模型。

## 环境安装

要求 Python 3.11 或更高版本，以及 [uv](https://docs.astral.sh/uv/)。

```powershell
uv venv --python 3.11
uv sync
```

macOS / Linux:

```sh
uv venv --python 3.11
uv sync
```

`uv sync` 会按 `pyproject.toml` 和 `uv.lock` 安装运行及测试依赖。也可不用 uv，使用 Python 3.11 的虚拟环境并安装项目依赖和 pytest。

## 配置

在项目根目录复制 `.env.example` 为 `.env`，填写模型服务配置：

PowerShell:

```powershell
Copy-Item .env.example .env
```

macOS / Linux:

```sh
cp .env.example .env
```

```dotenv
LLM_API_KEY=你的服务密钥
LLM_MODEL=支持工具调用的模型名称
LLM_BASE_URL=https://你的兼容服务地址/v1
LLM_TIMEOUT=60
LLM_MAX_RETRIES=2

# 只有使用 Tavily 搜索时需要
TAVILY_API_KEY=
```

`LLM_API_KEY` 和 `LLM_MODEL` 必填；服务使用默认 OpenAI 地址时可留空 `LLM_BASE_URL`。天气工具访问 wttr.in，不需要额外密钥；搜索工具需要 Tavily key。`.env`、会话数据和日志已加入 `.gitignore`，不要提交真实密钥或个人数据。

## 启动与会话

一次性提问会创建新 session，并在标准错误输出 session ID：

```powershell
uv run python cli.py -u alice "查一下北京天气并记一条待办"
```

用同一用户和 session ID 恢复会话：

```powershell
uv run python cli.py -u alice -s session_id "我有哪些待办？"
uv run python cli.py -u alice -s session_id "继续上次的任务" --retry
```

`--retry` 用于明确恢复上一轮未完成请求；不加时程序也会根据输入判断是否是恢复请求。交互模式会话 ID 会显示在终端，可按提示使用 `-s` 恢复：

```powershell
uv run python cli.py -u alice
uv run python cli.py -u alice --list
```

默认数据路径为 `data/{user_id}/{session_id}.json`，Trace 位于 `data/traces/`。可使用 `--data-dir` 和 `--trace-dir` 指定其他路径。`-v` 在标准错误输出中显示工具调用过程。

## 工具模式与执行

默认 `native` 模式使用原生 function calling；`text` 模式要求模型按 JSON 文本协议输出，便于兼容不支持原生工具的服务：

```powershell
uv run python cli.py --tool-mode native -u alice "计算 3 * 9"
uv run python cli.py --tool-mode text -u alice "计算 3 * 9"
```

注册表提供 calculator、todo、search 和 weather。模型可以在一次响应中请求多个工具。Agent 仅并行执行无依赖的只读调用（weather、search、calculator），并发默认上限为 4；写操作保持串行，涉及同批待办读写时也按序执行。依赖型任务由模型分轮请求，例如先查天气、收到真实结果后再创建待办。并行调度可由 `ChatService(parallel_tool_calls=...)` 控制；CLI 使用默认开启设置。

同步工具通过工作线程运行。只读工具超时后可能仍在后台执行，结果状态为未确定；写工具超时后会等待实际完成再返回，以避免重复副作用。LLM 请求失败会按配置重试；工具调用按 session 和轮次幂等记录，恢复时复用已落盘的写操作。

## Session、上下文与记忆

- Session 按 `(user_id, session_id)` 隔离；JSON 原子保存完整对话、待办、摘要、用户记忆和工具调用状态。写操作完成后立即落盘，进程重启后可以恢复。
- 上下文默认硬上限为 16000 项目估算 tokens，构建历史时预留 4000，摘要最多 2000。每次模型请求都将消息和原生 `tools` Schema 一起检查；无法放入时以 `context_limit` 结束并保存已完成操作。可用 `--max-context-tokens` 调整，例如 `uv run python cli.py --max-context-tokens 24000 -u alice`。项目估算器不等同模型 tokenizer。
- 长历史压缩为结构化摘要，优先保留用户目标、城市/数值/偏好等事实、待办状态、已执行操作和未完成事项；系统规则和本轮输入不会静默截断。
- 记忆仅提取用户当前明确输入中的姓名、城市、预算和偏好等白名单字段。明确的记住/修改/忘记语句可新增、更新或删除当前 session 的记忆；提问、否定、引用、示例或“不记忆”内容不写入。模型回答和工具结果不作为用户记忆。
- 字段相关问题（如“我叫什么？”）以及“按刚才的要求”这类明确指代会按需把记忆放在系统提示后、摘要前。历史中仍有效的用户声明会保留；过期或已删除的声明不会覆盖新记忆。状态型数据（如待办）始终以工具读取结果为准。

## 重试与 Trace

查看指定 session 的事件日志：

```powershell
uv run python cli.py -u alice -s <session_id> --trace
```

Trace 记录轮次、工具请求和结果、耗时、错误及结束状态。恢复未完成轮次时，已成功落盘的写调用会复用；不确定调用会先核对待办状态。非法工具调用及伪造 Observation 不会作为执行结果保存，工具未执行却声称成功时返回 `tool_not_executed`。

## 验收

```powershell
# 常规测试（不访问真实模型，live 测试默认跳过）
uv run pytest -q

# 所有真实 API 测试；需要有效 .env，包含原生调用、并行查询、依赖调用、写操作恢复及记忆 CRUD
uv run pytest -q -m live
```

确定性测试使用 FakeLLM 与注入的工具响应，覆盖工具 CRUD、并行/串行调度、session 隔离与恢复、上下文限制、故障恢复和记忆新增/更新/删除。完整 live 集需要模型服务支持原生工具调用、可访问 wttr.in，并配置 Tavily key 以执行搜索测试；并行天气调度场景使用稳定的注入响应以避免公共天气接口抖动。真实 API 验收结果和限制见 `docs/questions.md`。

## 当前范围与后续方向

已实现单进程 Agent Runtime、同步/异步模型与工具调用、受控并行、session 文件锁和原子保存、上下文估算与结构化压缩、显式会话记忆、失败恢复与 Trace。尚未实现跨 session 或长期语义记忆、LLM 自动提取候选事实、后台任务队列/完成通知、LLM 摘要和多模态输入。
