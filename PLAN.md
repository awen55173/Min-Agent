# 最小可用 Agent 实现计划

> 依据 `task_list.md`。原则：先跑通主链路，再逐步完善；不使用 langgraph/openhands 等 Agent 框架，核心 Runtime 自研；使用真实 LLM API（配置见 `.env`）。

## 1. 技术选型
- 语言：Python 3.11+，依赖尽量少：`openai`（兼容各类 OpenAI 风格接口的 HTTP 客户端）、`python-dotenv`、`pytest`。
- LLM 仅作为"文本补全"来源，循环、解析、工具调度、会话、上下文均自行实现。
- 持久化：JSON 文件（每个 session 一个文件），简单可查。

## 2. 目录结构（规划）
```
min_agent/
├─ agent/
│  ├─ llm.py          # LLM 客户端封装（真实 API + 可注入的 Fake 用于测试）
│  ├─ loop.py         # Agent 主循环 (ReAct)
│  ├─ parser.py       # LLM 输出解析：thought / tool_call / final_answer
│  ├─ prompt.py       # System prompt 与工具 schema 渲染
│  ├─ tools/
│  │  ├─ registry.py  # 工具注册机制（name / description / JSON Schema / 参数校验）
│  │  ├─ calculator.py
│  │  ├─ search.py    # mock 搜索
│  │  ├─ todo.py      # 必选，按 session 隔离
│  │  └─ weather.py   # mock 天气
│  ├─ session.py      # SessionManager：user_id + session_id 隔离、持久化
│  ├─ context.py      # 上下文构建、轮次限制、压缩
│  └─ trace.py        # 执行日志 / trace
├─ cli.py             # 命令行入口（多窗口 = 多 session）
├─ tests/
├─ README.md
└─ docs/prompts.md    # AI Prompt 与问题解决记录
```

## 3. 主要功能与设计要点

### 3.1 Agent 循环（`loop.py`）
1. 接收用户输入，追加到 session 历史。
2. 构建 context 调用 LLM。
3. 解析输出：有 `tool_call` → 执行工具；有 `final_answer` → 返回。
4. 工具结果以 observation 写回历史，回到步骤 2。
5. 终止条件：得到最终答案 / 达到 `max_steps`（防死循环）/ 不可恢复错误。

### 3.2 工具注册与调用
- `@tool(name, description, parameters_schema)` 装饰器注册；registry 生成 schema 注入 prompt，LLM 自主决策。
- 调用前做参数校验；工具异常转为 observation 反馈给 LLM，让其自我修正。
- 工具：`calculator`（安全表达式求值，基于 `ast` 而非 `eval`）、`search`（mock 数据）、`todo`（add/list/done/delete，数据挂在 session 下）、`weather`（mock）。

### 3.3 输出解析（`parser.py`）
- 约定结构化输出（JSON，字段 `thought` / `action` / `action_input` / `final_answer`），可兼容 markdown 代码块包裹。
- 解析失败：把格式错误作为反馈让模型重试，限制重试次数。

### 3.4 Session 管理
- 以 `(user_id, session_id)` 为键，历史、todo 等状态完全隔离；可随时按 id 恢复继续对话。
- 落盘 JSON，重启后可恢复。

### 3.5 Context 管理
- 保留：用户输入、assistant 最终答案、工具调用及结果（支持"带工具的追问"）；思考过程只保留当前轮，历史轮丢弃/折叠以节省 token。
- 限制：最大步数（单轮）+ 最大历史轮数/估算 token 上限。
- 基础压缩：超限时，将较早的轮次折叠为摘要（先用规则截断/拼接，后续可选 LLM 摘要），保留最近 N 轮原文。
- 状态型信息（如 todo 列表）通过工具实时读取，不依赖历史记忆，避免过期。

### 3.6 异常处理与 Trace
- 处理：LLM 超时/限流（重试+退避）、解析失败、未知工具、参数错误、工具异常、超出步数。
- Trace：每步记录 step、thought、tool、args、result、耗时、错误，写 JSONL 日志，并可在 CLI 中打印。

## 4. 逐步实施路线（每步完成后写测试并 git commit）
| 阶段 | 内容 | 验收 |
|---|---|---|
| M0 | 项目骨架、依赖、LLM 客户端、`.env` 读取 | 能调通真实 API |
| M1 | 工具注册 + calculator / todo / search / weather | 单测通过 |
| M2 | parser + 主循环（单 session、无压缩） | 能完成"查天气+记待办" |
| M3 | SessionManager + 持久化 + CLI 多窗口 | 两个 session 互不影响 |
| M4 | Context：步数/轮次限制、追问、基础压缩 | 追问与长对话测试通过 |
| M5 | 异常处理、trace 日志 | 异常场景测试通过 |
| M6 | README、AI Prompt 记录、整理上传 GitHub | 文档完整 |

## 5. 测试计划
- 使用 FakeLLM（脚本化输出）做确定性单元/集成测试：工具、parser、循环终止、session 隔离、context 压缩、异常路径。
- 少量真实 API 冒烟测试（标记 `@pytest.mark.live`，默认跳过）。
- 关键用例：计算器非法表达式；todo 跨 session 隔离；纯对话追问；带工具追问（"刚才那个待办完成了"）；超最大步数；LLM 输出非法格式。

## 6. 后续完善方向
- 原生 function calling 支持作为备选解析路径。
- LLM 摘要式压缩、重要信息（用户偏好）长期 memory 召回。
- 工具并行调用、流式输出、更真实的 search/weather 接入。

## 7. 交付物清单
- 代码仓库（GitHub）。
- README：运行方式、系统设计、memory 召回时机与放置方式（历史 → messages，状态 → 工具实时读取，摘要 → system 之后）。
- `docs/prompts.md`：AI Prompt 与问题解决记录（开发中持续追加）。
