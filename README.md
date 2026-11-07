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

## 原生工具调用与执行

仅使用原生 function calling：Schema 通过 API `tools` 参数传递，读取响应的 `tool_calls` 后执行，结果以 `role=tool` 和 `tool_call_id` 回填。无工具调用时，`content` 就是直接回答；不要求模型输出 `thought/action/action_input/final_answer` JSON，也不从回答文本解析工具指令。模型服务必须支持原生工具调用。

```powershell
uv run python cli.py -u alice "计算 3 * 9"
```

注册表提供 calculator、todo、search 和 weather。模型可以在一次响应中请求多个工具。Agent 仅并行执行无依赖的只读调用（weather、search、calculator），并发默认上限为 4；写操作保持串行，涉及同批待办读写时也按序执行。依赖型任务由模型分轮请求，例如先查天气、收到真实结果后再创建待办。并行调度可由 `ChatService(parallel_tool_calls=...)` 控制；CLI 使用默认开启设置。

同步工具通过工作线程运行。只读工具超时后可能仍在后台执行，结果状态为未确定；写工具超时后会等待实际完成再返回，以避免重复副作用。LLM 请求失败会按配置重试；工具调用按 session 和轮次幂等记录，恢复时复用已落盘的写操作。

## Agent 循环与预算

主循环在 `agent/loop.py`，每轮使用独立状态，依次调用：

| 方法 | 作用 |
| --- | --- |
| `may_think` | 检查工具步数、累计纠错和模型请求预算 |
| `think` | 检查 messages + tools 总预算，调用模型并区分回答/原生调用 |
| `disputed` | 校验非空文本和非法工具调用后的回答，必要时反馈重试 |
| `perform` | 执行一个只读并行组或单项串行操作 |
| `observe` | 按调用 ID 回填经过序列化与截断的结果，写操作即时保存 |
| `accept` | 接受直接文本回答，保存后记录终止事件 |

`perform` / `observe` 及幂等、中断处理位于 `agent/execution.py`；`agent/tool_call.py` 只解析原生工具的参数 JSON 并按 Schema 校验；`agent/runtime.py` 定义结果和失败类型。`run` 复用异步循环，同步模型接口在工作线程执行；已有事件循环时使用 `await arun(...)`。

默认最多执行 8 次工具调用、允许 3 次纠错重试；纠错次数在本轮累计，不因一次成功工具调用清零。模型最多请求 `max_steps + max_output_retries + 1` 次，最后一次用于生成回答；刚好用满工具步数仍可总结结果，但不能再执行工具。API 网络重试由 `LLM_MAX_RETRIES` 单独控制。超过预算由 `ModelOutputError` 进入统一失败收尾，保存已完成操作；上下文超限返回 `context_limit`。空回答等纠错耗尽状态为 `output_retries_exhausted`，Trace 事件为 `validation_error`。

本次移除了 `--tool-mode`、`tool_mode` 配置和旧文本协议解析；`max_parse_retries` 参数改名为 `max_output_retries`。已有原生会话可继续恢复；早期文本协议的历史正文仅作为普通文本保留和显示，不再解码成工具调用或最终答案，建议这些旧会话新建 session。

## Session、上下文与记忆

- Session 按 `(user_id, session_id)` 隔离；JSON 原子保存完整对话、待办、摘要、用户记忆和工具调用状态。写操作完成后立即落盘，进程重启后可以恢复。
- 上下文默认硬上限为 16000 项目估算 tokens，构建历史时预留 4000，摘要最多 2000。每次模型请求都将消息和原生 `tools` Schema 一起检查；无法放入时以 `context_limit` 结束并保存已完成操作。可用 `--max-context-tokens` 调整，例如 `uv run python cli.py --max-context-tokens 24000 -u alice`。项目估算器不等同模型 tokenizer。
- 长历史压缩为结构化摘要，优先保留用户目标、城市/数值/偏好等事实、待办状态、已执行操作和未完成事项；系统规则和本轮输入不会静默截断。
- 用户记忆独立存储、按需召回，具体流程和边界见下节。状态型数据（如待办）始终以工具读取结果为准。

## Memory：召回时机与放置方式

### 存储内容与来源

| 会话字段 | 作用 |
| --- | --- |
| `history` | 保存完整对话，构建模型上下文时清理、选取近期轮次 |
| `summary` | 压缩历史中的目标、操作和未完成事项等，不代替用户记忆 |
| `memories` | 仅记录当前用户明确输入中的个人事实 |
| `todos` | 保存待办的实际状态，由工具读取和修改 |

`memories` 的规范字段为 `名字`、`城市`、`预算`、`偏好`、`饮食偏好`，支持姓名、称呼、居住地、口味等别名。使用保守的正则规则提取，例如 `我的名字是林涛`、`我住在苏州`、`我的预算是8732元`、`我的饮食偏好是素食`。规则识别到提问、否定、引用、示例、假设或“不记忆”语境时不写入；这不是任意自然语言的语义识别。模型回答、思考和工具结果不写入该字段。

### 每轮处理流程

1. 按 `(user_id, session_id)` 加锁并读取会话，包括已有记忆。
2. 从**本轮用户输入**提取新增、修改或删除操作，先更新内存中的 `memories`。
3. 根据同一条输入选择相关记忆，再构建上下文。因此本轮新增或修改的值可立即参与本轮召回。
4. 将上下文交给 Agent；本轮后续工具循环沿用这些消息并追加工具请求、结果，不从工具结果重新提取用户记忆。
5. 记忆随会话保存到 `data/{user_id}/{session_id}.json` 的 `memories` 字段；正常轮次结束时保存，工具写操作即时保存时也会带上当前记忆。保存成功后，重启并恢复同一用户、同一 session 可继续使用。

召回在**每轮第一次模型请求前**执行。同步 `send` 与异步 `asend` 采用相同的原生调用和记忆规则；不同用户或不同 session 不共享记忆。

### 召回条件

| 本轮输入示例 | 行为 |
| --- | --- |
| `我叫什么？` | 选择已保存的名字 |
| `我的城市是什么？` | 按字段选择城市 |
| `按我的饮食习惯推荐` | 按别名选择饮食偏好 |
| `按刚才的要求安排两天行程`、`我的信息` | 命中明确指代规则，选择当前会话全部记忆，再受注入预算限制 |
| `讲个笑话` | 无相关字段或指代，不添加独立记忆消息 |
| `你记得刚才的待办吗？` | 不因“刚才”自动召回全部个人记忆，待办状态由工具查询 |

若规则判定用户正在询问记忆，但未选中任何已保存字段，会添加提示：`当前会话没有已保存的相关用户记忆；不要从历史回答或工具结果推断个人信息。` 无关输入不添加该提示；仍有效的用户声明可能留在近期历史中。

### 在模型消息中的位置

```text
system：固定系统提示
system：本轮相关用户记忆（或无相关记忆提示，按需出现）
system：结构化历史摘要（有摘要时出现）
user / assistant / tool：清理后的近期历史（保留工具调用配对）
user：本轮用户输入
```

记忆消息以“当前会话中用户明确提供的相关信息（以此为准，勿从历史回答或工具结果补全）”开头，随后逐行列出 `- 城市: 苏州` 等事实。该文本含标题最多占 **800 个项目估算 tokens**，按存储顺序加入完整字段行，下一行超限即停止；未放入的字段仍保存在会话中。记忆、摘要、历史和原生工具 Schema 都计入整体上下文预算。

### 更新、删除与限制

在同一 session 中依次输入 `记住我的城市是苏州` → `我的城市是什么？` → `把我的城市改成杭州` → `我的城市是什么？` → `忘记我的城市` → `我的城市是什么？`，对应记忆应依次为苏州、杭州、无已保存城市。

修改覆盖同一规范字段，删除移除该字段。构建上下文时会过滤规则可识别的过期或已删除用户声明，并在生成摘要时排除可识别的记忆声明；原始历史和 CLI 回放不因此删除。该处理不保证清除历史回答、工具结果或其他表述中的所有旧值，不能等同于彻底擦除对话内容。

当前没有向量检索、模型自动提取或跨 session 长期记忆。实现入口为 `agent/service.py` 的 `send` / `asend`、`agent/memory.py` 的提取与选择函数，以及 `agent/context.py` 的 `build_context`。

## 重试与 Trace

查看指定 session 的事件日志：

```powershell
uv run python cli.py -u alice -s <session_id> --trace
```

Trace 记录轮次、工具请求和结果、耗时、错误及结束状态。恢复未完成轮次时，已成功落盘的写调用会复用；不确定调用会先核对待办状态。非法工具调用不执行、不作为成功事实保存。空回答或非法原生调用回填校验原因并消耗累计纠错预算；非法调用后直接声称完成也会要求重试，耗尽时返回 `tool_not_executed`。工具执行失败仍作为真实 `tool` 结果供模型解释或修正。

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
