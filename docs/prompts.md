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
