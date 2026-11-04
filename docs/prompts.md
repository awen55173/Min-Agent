# AI Prompt 记录

## 2026-11-04
1. 根据 task_list.md 写实现计划（整体计划、逐步完善方向、主要功能）。
2. 修改 PLAN.md：Tavily/wttr.in、agent loop 伪代码、工具调用与最终答案互斥、结构化压缩、原子写入与并发、失败处理与幂等、补充测试；AGENTS.md 增加 prompts / 问题记录分开维护规则。
3. 按 M0 阶段实施：venv + uv 环境管理，补全缺失文件，配置环境变量，封装 OpenAI 格式模型客户端，提供最小运行入口验证调用链路；统一问题记录文件为 docs/questions.md。
