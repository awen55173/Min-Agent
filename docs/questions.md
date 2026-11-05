# 问题解决记录

| 问题 | 原因 | 解决方案 |
|---|---|---|
| pytest 报 `No module named 'cli'` | 根目录模块不在 pytest 的 sys.path 中 | pyproject 中配置 `pythonpath = ["."]` |
| 测试里 `import httpx` 失败 | 新版 openai 依赖 httpx2，环境中无 httpx | 测试中不依赖 httpx，用占位对象构造 APIConnectionError |
| PLAN.md 中问题记录文件名与约定不一致 | 早期命名为 troubleshooting.md | 统一改为 docs/questions.md |
| 当前 .env 为空，无法做真实 API 调用 | 尚未填写密钥 | 参考 .env.example 填写后运行 `uv run python cli.py` / `uv run pytest -m live` |
| test_cli_config_error 失败：缺配置仍加载到项目 .env | `load_dotenv()` 默认从 config.py 所在目录向上查找 .env，与 cwd 无关 | 默认只读 `cwd/.env`，也可显式传 `dotenv_path`；并加测试 |
| LLM_TIMEOUT=abc 抛 ValueError、负数被接受 | 未校验数值配置 | 统一抛 ConfigError，timeout>0、retries>=0 |
| choices=[] 抛 IndexError、空字符串当成功 | 响应提取无防御 | 新增 `_extract`，异常响应统一抛 LLMError |
| 真实 API 测试因缺 SOCKS 依赖在客户端初始化失败 | 本机设置了 socks 代理，httpx 需要 socksio | 依赖改为 `httpx2[socks]` |
| calculator 用 eval 有注入风险 | eval 可执行任意代码 | 用 ast 白名单节点求值，限制表达式长度与幂指数 |
| 参数 `True` 被当作 integer | bool 是 int 子类 | 校验时对数值类型显式排除 bool |
| 失败重试可能重复添加待办 | add 非幂等 | 相同内容的待办在同一轮内复用已有条目，返回 created=false |
| `(-1)**0.5` / `(10**100)**10` 使 execute 抛 TypeError/ValueError | JSON 序列化在 try 之外；复数、超大整数无法序列化 | 序列化纳入异常处理（`serialize_result`，allow_nan=False），转为 ok=False |
| 嵌套幂产生巨大整数、`1e309` 返回 Infinity | 只限制了单次指数 | 常量与每个中间结果限制为有限实数且 abs<=1e100，整数幂按位数预估拦截，拒绝复数 |
| 待办去重范围大于计划（整个会话） | 按未完成同名条目全局复用 | ToolContext 增加 turn_id，仅同一轮内去重；M2 循环为每个用户轮次设置 turn_id |
| 工具模块缺中文方法说明且检查未覆盖 | 文档测试只列了 M0 文件 | 检查范围改为 agent 下全部 .py + cli.py，补齐说明 |
| 最后一个允许的工具调用后模型无法给出答案 | 预算在工具执行后立即耗尽 | 步数只限制工具执行：第 max_steps+1 次请求工具时才终止，之后仍允许最终答案 |
| 解析失败可能无限循环 | 重试与步数独立 | 解析失败为连续计数，遇到合法输出清零；合法工具调用受 max_steps 限制，因此必然终止 |
| 互斥冲突时可能误执行工具 | 先执行再检查 | 解析阶段拦截，冲突输出不执行任何工具并回填格式错误 |
| PowerShell 中 `2>&1` 使 CLI 的 stderr 显示为 NativeCommandError | PowerShell 将 stderr 包装为错误记录 | 仅为显示问题，程序本身退出码 0 |
| docs 下记录文件出现乱码 | 用 PowerShell 的 `Add-Content` / `Set-Content` 写入中文：Windows PowerShell 5 默认非 UTF-8 编码，且命令行中的中文在传入时已被破坏，事后无法恢复 | 重写受损条目；以后只用 write_file（UTF-8）或 Python 以 `encoding="utf-8"` 写文档，不再用 PowerShell 写中文 |
| 连续两次 run_once 复用注册表时，第二次明确新增的同名待办被合并；显式 turn_id=1 与自动生成冲突 | 轮次号是 Agent 实例内的自增计数，不同实例、重启后都会从 1 开始 | 自动生成 `new_turn_id()`（uuid，全局唯一）；显式传入的 ID 由调用方保证会话内唯一，M3 由会话管理器分配并持久化 |
| 仅含空白的 final_answer 被当作成功，返回空答案 | 解析器只判断了空字符串，strip 在判断之后 | `_present` 对字符串先 strip 再判断，空白视为未提供，走格式纠正与重试流程 |
