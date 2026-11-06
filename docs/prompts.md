# AI Prompt 记录

## 2026-11-04
1. 根据 task_list.md 写实现计划（整体计划、逐步完善方向、主要功能）。
2. 修改 PLAN.md：Tavily/wttr.in、agent loop 伪代码、工具调用与最终答案互斥、结构化压缩、原子写入与并发、失败处理与幂等、补充测试；AGENTS.md 增加 prompts / 问题记录分开维护规则。
3. 按 M0 阶段实施：venv + uv 环境管理，补全缺失文件，配置环境变量，封装 OpenAI 格式模型客户端，提供最小运行入口验证调用链路；统一问题记录文件为 docs/questions.md。

4. 对 M0 阶段内容进行检查、测试和验证。
5. 实施 M1：工具接口与注册、calculator / todo（按会话隔离）/ search(Tavily) / weather(wttr.in)，并测试验收。
6. M1 评审修复：工具结果序列化异常、计算器数值范围、待办去重改为同轮、补全工具模块中文方法说明并扩展检查范围。
7. 实施 M2：prompt.py / parser.py / loop.py，单次请求内的 ReAct 循环，CLI 接入，FakeLLM 与真实 API 验收。

## 2026-11-05
8. 修复 docs 下记录文件的乱码：统一使用 UTF-8 写入，并重写被破坏的条目。
9. 完善 M2：统一轮次 ID 分配规则（全局唯一，避免不同请求误去重）；空白 final_answer 进入格式纠正流程。
10. 实施 M3：SessionManager（JSON 原子存储、文件锁/进程内锁/版本校验）、ChatService 接入 Agent、CLI 支持用户与会话，并补充隔离、重启恢复、并发、版本冲突、写入中断测试。
11. 在天气工具描述中添加英文/拼音城市名限制；完善交互模式恢复会话时在终端回放历史对话（默认展示用户提问与最终回答，传 `-v` 额外回放历史工具调用步骤）。
12. 修复 M3 评审指出的问题：活跃会话锁增加心跳保活、守卫串行回收与保存前归属校验；中断退出时保存本轮已执行的工具调用与 observation 历史；解析器拒绝包含多个顶层 JSON 对象的歧义输出。
13. 完善 parser.py（仅接受完整的顶层 JSON 对象或完整代码块，拦截 JSON 数组、残缺外层 JSON 和深层嵌套输入并转为 ParseError）；实施 M4 上下文管理与结构化压缩（context.py、SessionData.summary 持久化、剥离历史 thought、按轮数与 token 预算压缩、压缩后完成待办测试与完整历史回放隔离）。
14. 修复评审发现的三个问题：隔离非法工具输出和伪造 Observation，防止零执行后声称成功；增加上下文默认预算并对每次模型请求实施硬上限，限制摘要和工具结果，超限时明确结束；在压缩前提取输入末尾的预算、城市和偏好等字段。要求遵循 task_list.md、PLAN.md，补充回归测试、真实 API 验收、问题记录并创建 Git commit。
15. 向用户解释 `agent/context.py` 中摘要预算计算（`summary_cap`）、`agent/service.py` 中 `prefix_len = len(ctx_plan.messages) - 1` 的作用，以及会话摘要的持久化机制。
16. 实施 M5 阶段内容：为工具调用增加 `call_id`（关联用户、会话与 `turn_id`）并持久化工具名、参数、执行状态与结果；待办写操作（`add`/`complete`/`delete`）立即将待办状态与成功调用记录在同一原子保存中落盘；恢复或重试时复用原调用标识实现幂等（不确定状态先核对存储，保留同轮去重并允许跨轮创建同名待办）；统一异常退出说明并新增 `agent/trace.py`（JSONL Trace 日志与 CLI `--trace` 查看）；补充全量与真实 API 测试。
17. 修复 M5 补充验证发现的 3 项缺口：区分未完成轮次的“恢复重试”与“新建请求”（仅重试复用原 `turn_id` 与 `call_id`，要求另外创建同名待办等新请求分配新 `turn_id`）；将整轮收尾保存置于 `turn_end` 日志记录之前，最终保存失败时统一返回 `persist_error` 并记录真实日志状态；新增 `attempt_id` 区分同一 `turn_id` 下的每次尝试，保持稳定 `call_id` 幂等并按 `(turn_id, attempt_id)` 校验 Trace 配对与连续步数。

## 2026-11-06
18. 分阶段实施原生工具调用、异步执行链与受控并行调度：
    - 小阶段 1（接入原生工具调用）：验证当前模型与服务的原生 function calling 支持情况，注册表导出 OpenAI `tools` Schema，将模型响应统一为“文本回答 + 工具调用列表”（`LLMResponse` / `ToolCallItem`），原生模式作为主路径并保留文本模式兼容选项。
    - 小阶段 2（先跑通串行闭环）：限制单次返回一个原生工具调用（`parallel_tool_calls=False`），解析工具名、JSON 参数与模型 `api_call_id`，执行后按 `role="tool"` + `tool_call_id` 协议回填，保留参数校验、步数限制与异常处理，不再依赖自定义 `action` JSON。
