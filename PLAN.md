# 最小可用 Agent 实现计划

> 依据 `task_list.md`。原则：先跑通主链路，再逐步完善；不使用 langgraph/openhands 等 Agent 框架，核心 Runtime 自研；使用真实 LLM API（配置见 `.env`）。

## 1. 技术选型
- 语言：Python 3.11+，以 `pyproject.toml` / `uv.lock` 锁定 `openai`（OpenAI 兼容客户端）、`httpx2`、`python-dotenv`、`pytest`。
- LLM 仅提供模型响应，循环、工具调度、会话、上下文均自行实现；仅使用 OpenAI-compatible 原生 function calling，不解析模型正文中的工具指令。
- 持久化：JSON 文件（每个 session 一个文件），见 3.4。

## 2. 目录结构（规划）
```
min_agent/
├─ agent/
│  ├─ llm.py          # LLM 客户端封装（真实 API + 可注入的 Fake 用于测试）
│  ├─ loop.py         # Agent 主循环
│  ├─ execution.py    # 原生工具调度、观察回填、幂等与中断恢复
│  ├─ runtime.py      # 结果、调用记录与预算失败类型
│  ├─ tool_call.py    # 原生工具参数 JSON 解码与 Schema 校验
│  ├─ prompt.py       # 工具使用规则；Schema 通过 API tools 独立传递
│  ├─ tools/
│  │  ├─ registry.py  # 工具注册（name / description / JSON Schema / 参数校验）
│  │  ├─ calculator.py
│  │  ├─ search.py    # Tavily 搜索
│  │  ├─ todo.py      # 必选，按 session 隔离
│  │  └─ weather.py   # wttr.in 天气
│  ├─ session.py      # SessionManager：隔离、原子持久化、并发处理
│  ├─ context.py      # 上下文构建、轮次限制、压缩
│  └─ trace.py        # 执行日志 / trace
├─ cli.py
├─ tests/
├─ README.md
└─ docs/
   ├─ prompts.md          # AI Prompt 记录
   └─ questions.md  # 问题解决记录
```

## 3. 主要功能与设计要点

### 3.1 Agent 循环（`loop.py`）
```python
while agent.may_think():
    reply = await agent.think()
    if reply.is_answer:
        if await agent.disputed(reply):
            continue
        return await agent.accept(reply, actions=agent.actions)
    for action in reply.actions:
        await agent.observe(await agent.perform(action))
raise ModelOutputError("模型调用预算内未生成有效回答")
```

- `may_think`：限制工具步数、累计纠错次数及模型请求次数；`think` 每次请求前检查 messages 与原生 tools Schema 总预算。
- `think` / `prepare_reply`：直接读取模型 `content` / `tool_calls`，校验工具名称、参数与调用 ID，按只读并行组及串行写操作分组。
- `disputed`：空回答、非法调用后直接声称成功时回填错误并消耗纠错预算，不检查回答是否符合自定义 JSON。
- `perform`：异步执行一个调度组；无依赖只读调用受控并行，写操作和同批待办读写串行。
- `observe`：按请求顺序回填序列化、长度受限的 `role=tool` 消息，使用 `tool_call_id` 配对，并在写操作成功后即时保存。
- `accept`：保存直接文本回答并记录结束状态；`observe` 与 `accept` 需要等待保存，因此为异步方法。
- 默认 `max_steps=8`、`max_output_retries=3`，纠错预算不随成功工具调用重置；模型请求上限为两者之和加 1。工具步数刚好用满时仍可请求最终总结。
- 预算耗尽抛出 `ModelOutputError`，运行入口按 3.6 收尾；同步 `run` 复用同一异步循环，每轮状态独立。未实现流式输出。

### 3.2 工具注册与调用
- `@tool(name, description, parameters_schema)` 注册；registry 生成原生 Schema，通过 API `tools` 参数传递，由 LLM 自主决策。
- 调用前按 schema 校验参数；工具异常转为 observation 反馈给模型自我修正。
- 工具：
  - `calculator`：基于 `ast` 的安全求值，禁用 `eval`。
  - `search`：调用 **Tavily** API（key 放 `.env`）；结果只保留标题/摘要/URL 并截断；测试中用 Fake 替代，无 key 时给出明确报错。
  - `todo`：add/list/done/delete，每条待办有稳定 ID，数据挂在 session 下。
  - `weather`：请求 `https://wttr.in/{city}?format=j1`，提取当前温度、天气描述、体感、湿度等精简字段；设置超时，失败返回可读错误。

### 3.3 原生响应与校验边界
- 仅支持原生 `tool_calls` / `role=tool` 协议；没有工具调用时，`content` 即最终回答。
- 删除旧 `parser.py`、文本 JSON 提示模板、`tool_mode` 配置和 CLI `--tool-mode`。
- `tool_call.py` 只解码工具参数，校验名称、参数对象、深度与 Schema；工具调用 ID 必须非空且同批唯一。
- 纯文本、JSON 文本及代码块均可作为回答，不解码 `action` 或 `final_answer` 字段，不把正文当工具执行。
- 纠错参数为 `max_output_retries`，Trace 记录 `validation_error`；空回答也走该预算，网络错误由客户端单独退避重试。
- 工具异常和不可序列化返回值转为真实工具错误；长结果截断后提示缩小范围或分批查询。

### 3.4 Session 管理与存储
- 以 `(user_id, session_id)` 为键隔离：存储路径 `data/{user_id}/{session_id}.json`，因此**不同用户使用相同 session ID 互不影响**。历史、todo 等状态都在 session 内。
- **原子写入**：写临时文件 → `flush + fsync` → `os.replace` 覆盖，避免中途崩溃产生半截文件。
- **同一 session 同时打开**：每个 session 使用文件锁（lock 文件，含超时）；第二个进程/窗口获取不到锁时，明确提示"该会话正被占用"并拒绝写入（只读或稍后重试），不静默覆盖；进程内另用 `threading.Lock`。文件中带 `version` 字段，写前校验，作为兜底。
- 重启后按 id 加载继续对话。
- 已实现显式会话记忆 `memories`：只从本轮用户原文解析姓名、城市、预算、偏好等白名单字段的设置、更新、删除；按字段别名或明确指代按需注入，助手回答与工具结果不作为用户记忆。

### 3.5 Context 管理
- 保留：用户输入、最终答案、工具调用及结果（支持带工具追问）；思考过程只保留当前轮，历史轮丢弃以省 token。
- 限制：单轮最大步数 + 历史最大轮数/估算 token 上限。
- 默认每次模型请求的上下文硬上限为 16000 估算 tokens，历史构建预留 4000，摘要上限 2000；CLI 可用 `--max-context-tokens` 调整。每次请求（含工具循环和回答/调用纠错）发送前统一检查，无法容纳时返回 `context_limit`，不截断系统规则或当前输入、不再请求模型，保存已执行操作。
- **压缩**（超限触发，保留最近 N 轮原文）：较早轮次折叠为结构化摘要，**不能简单截断/拼接**，必须显式保留：
  - 当前任务/用户目标；
  - 关键事实（城市、数值、用户偏好等）；
  - 待办 ID 及状态、已执行的写操作；
  - 未完成事项。
  摘要先用规则按字段抽取（工具调用结果中的 ID/参数直接结构化保存），后续可选 LLM 摘要。
- 状态型数据（todo 列表）以工具实时读取为准，不依赖历史记忆。
- **显式会话记忆**：`SessionData.memories` 只接收当前用户输入中匹配白名单字段（姓名、城市、预算、偏好等）的明确设置、更新或删除命令；忽略疑问、否定、引用、示例和“不记忆”输入。按字段别名/明确指代选取少量事实，在 system prompt 之后、摘要之前注入；历史答案、工具 Observation 不写入用户记忆。当前没有跨 session 长期记忆或 LLM 候选提取。
- 摘要先从完整输入与回答提取金额、地点和偏好等短字段，再收缩文本；摘要中的待办按条数和文本长度受限，省略时明确提示调用 `todo.list` 读取完整 ID 与实时状态，完整存储不会被删减。单条 Observation 限制为 2000 估算 tokens，截断时显式标记并提示分批查询。
- 必须有测试："压缩后仍能完成刚才那个待办"。

### 3.6 异常处理、失败返回与幂等
| 失败场景 | 返回给用户 | 持久化 |
|---|---|---|
| 达到最大步数 | 提示"已达最大步数，任务可能未完成"，并列出已完成的工具操作 | 保存本轮已执行的工具调用及结果，用户可继续追问接着做 |
| 回答/调用纠错耗尽 | 返回 `output_retries_exhausted`，附最近一次原因 | 同上，不保存非法调用为有效动作 |
| 非法工具调用后直接声称完成 | 回填“本次工具未执行”并允许修正，纠错预算耗尽返回 `tool_not_executed` | 不保存被拒绝回答为成功事实；重新成功执行工具后可正常结束 |
| 上下文无法容纳 | 返回 `context_limit`，提示缩短输入或分批处理 | 保存用户输入、已执行工具操作和预算不足原因 |
| LLM API 失败（超时/限流/网络） | 先退避重试；仍失败则提示"服务暂时不可用，请稍后重试" | 保留已完成操作与用户输入，不丢历史 |
| 工具失败 | 作为 observation 回填模型，不直接中断 | 记录到 trace |

- **避免重复添加待办**：
  - 每次工具调用有 `call_id`，执行结果在 session 内记录；写类工具（todo add/done/delete）执行后立即落盘，恢复/重试时同一轮已执行过的调用直接复用结果而不再执行。
  - `todo add` 增加幂等保护：同一轮内相同内容重复调用视为重复，返回已存在的条目。
  - API 失败重试只重试 LLM 调用，**不重放已执行的工具**。

### 3.7 Trace
- 每步记录：session、step、thought、tool、args、result、耗时、错误、call_id，写 JSONL；CLI 可打印。
- 要求完整：每次工具调用必有"请求—结果/错误"成对记录，异常路径同样记录终止原因。

## 4. 逐步实施路线（每步完成后写测试并 git commit）
| 阶段 | 内容 | 验收 |
|---|---|---|
| M0 | 骨架、依赖、LLM 客户端、`.env` | 能调通真实 API |
| M1 | 工具注册 + calculator/todo/search(Tavily)/weather(wttr.in) | 单测通过 |
| M2 | 原生响应处理、工具参数校验 + 分阶段主循环 | 能完成"查天气+记待办" |
| M3 | SessionManager、原子写入、锁、CLI 多窗口 | session 隔离与并发测试通过 |
| M4 | Context：限制、追问、结构化压缩 | 压缩后追问测试通过 |
| M5 | 失败处理、幂等、trace | 异常场景测试通过 |
| M6 | README、docs、本地验收与发布准备 | 文档、常规及 live 核心流程通过；GitHub 发布由仓库维护者决定 |

## 5. 测试计划
默认用 FakeLLM（脚本化输出）+ Fake 搜索/天气，保证确定性。

- 工具：calculator 非法表达式；todo CRUD；weather/search 解析与失败。
- 原生调用：工具参数缺字段、未知工具、非法参数 JSON、深层嵌套、调用 ID 重复；回答正文和代码块原样保留、不执行。
- 循环：多工具连续调用（如 天气 → 记待办 → 列待办）；最大步数；累计纠错耗尽。
- Session：同一用户两个 session 隔离；**不同用户相同 session ID 隔离**；**重启恢复**（重建 manager 后继续追问）；同 session 并发占用；原子写入（模拟写入中断，原文件不损坏）。
- Context：纯对话追问；带工具追问；**压缩后追问**；**压缩后仍能完成刚才那个待办**。
- 评审回归：伪造 Observation 后零执行却声称成功；大量待办或无历史旧摘要的硬预算；连续工具结果和回答/调用纠错请求不超限；关键字段位于长输入/回答末尾；压缩后的关键字段重载；预算不足时完整历史和已执行待办仍保存。
- 失败与幂等：API 失败后重试不重复添加待办；失败时已完成操作被保存。
- Trace：**完整性**检查（调用/结果成对、步数连续、异常终止有记录）。
- 真实 API 测试：标记 `@pytest.mark.live`，默认跳过；**交付前必须至少实际运行一次核心流程**（真实 LLM + 工具 + 追问），并把结果记入 README 或 questions。

## 6. 后续完善方向
- 未实现：LLM 摘要、跨 session 长期/语义记忆、结构化候选事实确认流程、后台任务与完成通知、流式输出。
- 可继续扩充白名单记忆字段和工具，但保留用户原文依据及显式写入边界。

## 7. 当前实现状态
- 已实现并经常规测试验收：M0–M5 核心链路、原生工具调用、异步/并行调度、结构化上下文管理、会话记忆和恢复/Trace。
- 本地文档、配置与真实 API 验收按 `README.md`、`架构设计.md` 和 `docs/questions.md` 维护。
- 尚未执行 GitHub 发布/上传；本仓库保留为本地交付，供维护者自行审阅和推送。

## 8. 交付物清单
- GitHub 代码仓库。
- README：运行方式、系统设计、memory 召回时机与放置方式（历史 → messages，摘要 → system 之后，状态 → 工具实时读取）。
- `docs/prompts.md`（AI Prompt 记录）与 `docs/questions.md`（问题解决记录），两者分开。
