# 最小可用 Agent 实现计划

> 依据 `task_list.md`。原则：先跑通主链路，再逐步完善；不使用 langgraph/openhands 等 Agent 框架，核心 Runtime 自研；使用真实 LLM API（配置见 `.env`）。

## 1. 技术选型
- 语言：Python 3.11+，依赖尽量少：`openai`（OpenAI 兼容客户端）、`httpx`/`requests`、`python-dotenv`、`pytest`。
- LLM 仅作为"文本补全"来源，循环、解析、工具调度、会话、上下文均自行实现。
- 持久化：JSON 文件（每个 session 一个文件），见 3.4。

## 2. 目录结构（规划）
```
min_agent/
├─ agent/
│  ├─ llm.py          # LLM 客户端封装（真实 API + 可注入的 Fake 用于测试）
│  ├─ loop.py         # Agent 主循环
│  ├─ parser.py       # LLM 输出解析与校验
│  ├─ prompt.py       # System prompt 与工具 schema 渲染
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
```
while 预算充足(步数 / token / 重试次数):
    输出 = 模型根据当前 context 回答        # 先同步实现，后续可改 async
    if 输出表示回答结束(final_answer):
        if 输出格式或内容校验失败:
            把校验错误回填给模型; continue   # 消耗重试预算
        return 最终回答
    # 否则为工具调用请求
    结果 = 执行工具(同步，后续可扩展异步/并行)
    校验工具输出(类型、长度截断)，作为 observation 回填 context
预算耗尽 → 走 3.6 的失败处理
```
- 终止条件：得到合法最终答案 / 预算耗尽 / 不可恢复错误。
- 后续补充（不在首版展开）：异步与并行工具调用、流式输出。

### 3.2 工具注册与调用
- `@tool(name, description, parameters_schema)` 注册；registry 生成 schema 注入 prompt，由 LLM 自主决策。
- 调用前按 schema 校验参数；工具异常转为 observation 反馈给模型自我修正。
- 工具：
  - `calculator`：基于 `ast` 的安全求值，禁用 `eval`。
  - `search`：调用 **Tavily** API（key 放 `.env`）；结果只保留标题/摘要/URL 并截断；测试中用 Fake 替代，无 key 时给出明确报错。
  - `todo`：add/list/done/delete，每条待办有稳定 ID，数据挂在 session 下。
  - `weather`：请求 `https://wttr.in/{city}?format=j1`，提取当前温度、天气描述、体感、湿度等精简字段；设置超时，失败返回可读错误。

### 3.3 输出解析与格式规则（`parser.py`）
约定 JSON 输出，字段：`thought`、`action`（工具名）、`action_input`（对象）、`final_answer`；兼容 markdown 代码块包裹。

| 情况 | 处理 |
|---|---|
| 仅有 `action`(+`action_input`) | 视为工具调用 |
| 仅有 `final_answer` | 视为最终答案 |
| `action` 与 `final_answer` 同时出现 | 互斥违规：**不执行工具**，回填"二者只能选其一"让模型重发（计入解析重试） |
| 两者都没有 | 格式错误，回填提示重试 |
| 有 `action` 但缺 `action_input` / 缺必填参数 | 不执行，回填具体缺失字段 |
| `action` 不在注册表 | 回填可用工具列表 |
| 非 JSON / 无法解析 | 回填格式要求重试 |
| 缺 `thought` | 容忍（置空），不视为错误 |
- 解析重试次数有上限（如 3），耗尽走 3.6。

### 3.4 Session 管理与存储
- 以 `(user_id, session_id)` 为键隔离：存储路径 `data/{user_id}/{session_id}.json`，因此**不同用户使用相同 session ID 互不影响**。历史、todo 等状态都在 session 内。
- **原子写入**：写临时文件 → `flush + fsync` → `os.replace` 覆盖，避免中途崩溃产生半截文件。
- **同一 session 同时打开**：每个 session 使用文件锁（lock 文件，含超时）；第二个进程/窗口获取不到锁时，明确提示"该会话正被占用"并拒绝写入（只读或稍后重试），不静默覆盖；进程内另用 `threading.Lock`。文件中带 `version` 字段，写前校验，作为兜底。
- 重启后按 id 加载继续对话。

### 3.5 Context 管理
- 保留：用户输入、最终答案、工具调用及结果（支持带工具追问）；思考过程只保留当前轮，历史轮丢弃以省 token。
- 限制：单轮最大步数 + 历史最大轮数/估算 token 上限。
- 默认每次模型请求的上下文硬上限为 16000 估算 tokens，历史构建预留 4000，摘要上限 2000；CLI 可用 `--max-context-tokens` 调整。每次请求（含工具循环和格式纠正）发送前统一检查，无法容纳时返回 `context_limit`，不截断系统规则或当前输入、不再请求模型，保存已执行操作。
- **压缩**（超限触发，保留最近 N 轮原文）：较早轮次折叠为结构化摘要，**不能简单截断/拼接**，必须显式保留：
  - 当前任务/用户目标；
  - 关键事实（城市、数值、用户偏好等）；
  - 待办 ID 及状态、已执行的写操作；
  - 未完成事项。
  摘要先用规则按字段抽取（工具调用结果中的 ID/参数直接结构化保存），后续可选 LLM 摘要。
- 状态型数据（todo 列表）以工具实时读取为准，不依赖历史记忆。
- 摘要先从完整输入与回答提取金额、地点和偏好等短字段，再收缩文本；摘要中的待办按条数和文本长度受限，省略时明确提示调用 `todo.list` 读取完整 ID 与实时状态，完整存储不会被删减。单条 Observation 限制为 2000 估算 tokens，截断时显式标记并提示分批查询。
- 必须有测试："压缩后仍能完成刚才那个待办"。

### 3.6 异常处理、失败返回与幂等
| 失败场景 | 返回给用户 | 持久化 |
|---|---|---|
| 达到最大步数 | 提示"已达最大步数，任务可能未完成"，并列出已完成的工具操作 | 保存本轮已执行的工具调用及结果，用户可继续追问接着做 |
| 解析重试耗尽 | 提示"模型输出格式多次不合法"，附最近一次原因 | 同上，不保存非法输出为有效动作 |
| 非法工具输出后直接声称完成 | 返回 `tool_not_executed`，明确无法确认完成；纠错反馈必须说明“本次工具未执行” | 隔离非法输出与伪造 Observation，拒绝将后续虚假成功答复保存为事实；重新合法执行工具后才恢复正常结束 |
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
| M2 | parser（含互斥/缺字段规则）+ 主循环 | 能完成"查天气+记待办" |
| M3 | SessionManager、原子写入、锁、CLI 多窗口 | session 隔离与并发测试通过 |
| M4 | Context：限制、追问、结构化压缩 | 压缩后追问测试通过 |
| M5 | 失败处理、幂等、trace | 异常场景测试通过 |
| M6 | README、docs、真实 API 跑通核心流程、上传 GitHub | 文档完整 |

## 5. 测试计划
默认用 FakeLLM（脚本化输出）+ Fake 搜索/天气，保证确定性。

- 工具：calculator 非法表达式；todo CRUD；weather/search 解析与失败。
- Parser：互斥冲突、缺字段、未知工具、非法 JSON、代码块包裹。
- 循环：多工具连续调用（如 天气 → 记待办 → 列待办）；最大步数；解析重试耗尽。
- Session：同一用户两个 session 隔离；**不同用户相同 session ID 隔离**；**重启恢复**（重建 manager 后继续追问）；同 session 并发占用；原子写入（模拟写入中断，原文件不损坏）。
- Context：纯对话追问；带工具追问；**压缩后追问**；**压缩后仍能完成刚才那个待办**。
- 评审回归：伪造 Observation 后零执行却声称成功；大量待办或无历史旧摘要的硬预算；连续工具结果和格式纠正请求不超限；关键字段位于长输入/回答末尾；压缩后的关键字段重载；预算不足时完整历史和已执行待办仍保存。
- 失败与幂等：API 失败后重试不重复添加待办；失败时已完成操作被保存。
- Trace：**完整性**检查（调用/结果成对、步数连续、异常终止有记录）。
- 真实 API 测试：标记 `@pytest.mark.live`，默认跳过；**交付前必须至少实际运行一次核心流程**（真实 LLM + 工具 + 追问），并把结果记入 README 或 questions。

## 6. 后续完善方向
- 原生 function calling 作为备选路径；异步/并行工具调用。
- LLM 摘要式压缩、长期 memory 召回。
- 更多工具与流式输出。

## 7. 交付物清单
- GitHub 代码仓库。
- README：运行方式、系统设计、memory 召回时机与放置方式（历史 → messages，摘要 → system 之后，状态 → 工具实时读取）。
- `docs/prompts.md`（AI Prompt 记录）与 `docs/questions.md`（问题解决记录），两者分开。
