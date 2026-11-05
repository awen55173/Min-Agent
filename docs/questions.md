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
| Agent 内部出现未预期异常时，本轮已完成的待办和历史会丢失 | `Agent.run` 未捕获异常，调用方拿不到 messages/actions | `run` 把未预期 `Exception` 转为 `internal_error` 结果；`ChatService` 对 `BaseException`（如 Ctrl+C）也保存待办与用户输入后再抛出 |
| 失败结果（步数耗尽/解析耗尽/API 失败）存入历史后对话轮次不完整 | 失败时最后一条可能是 user 观察或非法 assistant 输出 | 失败提示以 `{"final_answer": ...}` 形式作为 assistant 回复入历史，后续追问可知晓发生了什么 |
| 历史可能重复拼接系统提示词 | Agent 返回的 messages 含 system 与旧历史 | 历史只存 user/assistant；保存时仅取 `messages[1 + len(history):]` 的新增部分，system 每轮重新生成 |
| 跨进程加锁需兼容 Windows | `fcntl` 不可用 | 用 `O_CREAT|O_EXCL` 独占创建锁文件；锁内记录 token，释放时校验持有者；锁文件 mtime 超过 `stale_after`（默认 600s，应大于单轮最长耗时）视为过期并原子改名摘除；该回收为尽力而为，最终由版本校验兜底 |
| 轮次 ID 需要跨重启唯一 | 之前使用随机 uuid，无法与会话恢复衔接 | 会话文件保存 `next_turn`，`Session.next_turn_id()` 在锁内分配并随会话保存 |
| PowerShell 命令行里的中文参数在传给 Python 时可能乱码 | Windows 控制台/参数编码 | 涉及中文的验收放进 UTF-8 测试文件（含 live 测试）执行，不靠命令行传中文 |
| 模型调用 weather 工具时传入中文城市名（如“北京”）导致 wttr.in 连接失败 | 工具描述与参数说明仅写了 `city name, e.g. Beijing`，未明确要求优先使用英文/拼音 | 在 `weather` 工具描述与 `city` 参数说明中明确要求优先使用英文或拼音城市名（先将中文城市名转换为英文如 `Beijing`） |
| 交互模式恢复已有会话（`-s`）时终端只接受新输入，看不到之前的旧对话 | `cli.py` 恢复会话后直接进入 `input("you> ")` 循环，未读取并渲染历史消息 | 新增 `_print_history`，在交互模式恢复会话时回放历史中的用户提问（`you> ...`）与最终回复（`final_answer`），过滤中间 `Observation`/`Format error`，并在开启 `-v` 时将历史工具调用步骤输出到 `stderr` |
| 活跃会话的锁在长耗时调用超过 `stale_after` 时可能被误回收，并留下 `.stale-*` 残留文件 | 仅凭锁文件初始 `mtime` 判断过期，持锁期间未刷新；回收用改名且缺乏互斥 | `_FileLock` 持锁期间启动后台心跳线程定期刷新 `mtime`；只有心跳停止超过 `stale_after`（默认 60s）才在 `.reclaim` 守卫下复核并删除过期锁；`Session.save` 保存前校验锁归属（`check_lock`），失去锁则抛 `LockLostError` 拒绝覆盖 |
| `KeyboardInterrupt` 中断后只保存了用户输入和中断提示，已执行的工具调用与 observation 丢失 | `ChatService.send` 在 `BaseException` 分支未从 `Agent.run` 获取本轮已发生的消息 | `Agent.run` 捕获 `BaseException` 时整理本轮消息（正在执行的工具补写 `status=interrupted` 的 observation，尚未执行的模型输出丢弃）并挂载到 `e.agent_result`；`ChatService.send` 将其连同中断说明完整写入会话历史 |
| 模型一次返回连续两个 JSON 对象时，解析器静默提取第一个并执行工具 | `extract_json` 使用 `raw_decode` 找到第一个 JSON 对象即返回 | 新增 `_scan_objects` 扫描全部顶层 JSON 对象，发现数量大于 1 时抛 `ParseError("ambiguous output...")`，不执行任何工具并回填错误让模型重试 |
| 模型返回 JSON 数组 `[...]`、残缺外层 JSON `{"outer": {...}` 或夹带前后文字时，解析器扫描并执行其中的内部工具调用；深层嵌套输入触发 `RecursionError` 返回 `internal_error` | `extract_json` 在整段解析失败后仍回退到子串扫描 `_scan_objects`；未捕获 `RecursionError` 且未限制 JSON 嵌套深度 | `extract_json` 仅接受完整的顶层 JSON 对象或完整代码块（`_unwrap_fence`），其余一律抛 `ParseError`；捕获 `RecursionError` 并用迭代栈 `_check_depth` 限制最大嵌套深度为 32，使深层嵌套进入格式纠正流程且绝不执行工具 |
| 长对话上下文超限或压缩后丢失早期任务、关键事实与待办 ID，或者压缩破坏了 CLI 会话历史回放 | 未区分持久化全量历史与单次模型调用的受限上下文，也未结构化提取旧轮次信息 | 新增 `agent/context.py`：保留完整 `session.data.history` 供回放，调用 `Agent.run` 前按轮数上限和预留工具空间后的 token 预算构建受限上下文，剥离历史 `thought`，将旧轮次压缩为包含“用户目标、关键事实、待办 ID 与状态、已执行操作、未完成事项”的结构化摘要并随 `session.data.summary` 持久化、与 `TodoStore` 实时对齐 |
| 非法工具调用夹带自行编造的 Observation 被拒绝后，模型直接称待办添加成功，系统返回 ok | 原始非法输出先进入上下文；纠错没有说明零执行；最终答案只校验格式 | 解析通过后才保存 assistant 消息；拒绝模型额外的结果字段；纠错明确“本次工具未执行”；未重新成功执行工具即给最终答案时返回 `tool_not_executed`，拒绝将虚假成功答复持久化为事实。回放、多顶层对象、伪造字段和恢复执行均加入回归 |
| 大量待办使摘要自身超预算；连续工具结果使后续模型请求超限 | 摘要中的待办没有条数/长度限制；最后一次压缩无条件放行；循环仅在首轮构建上下文 | 默认上下文从 4000 提高至 16000、预留从 1200 提高至 4000、摘要从 600 提高至 2000 估算 tokens；CLI 增加 `--max-context-tokens`。摘要限制条数/文本长度并提示读取完整 todo；每条 Observation 限制为 2000 估算 tokens；每次请求发送前统一校验硬上限，放不下时返回 `context_limit` 并保存历史和已执行操作。系统提示、本轮输入不被静默截断 |
| 长输入或长回答末尾的“预算 8732 元、苏州、素食”压缩后丢失 | 普通对话事实只保留用户输入和回答的固定前缀 | 从全文先提取金额、数值、地点和偏好短字段，再对目标和普通文本做长度控制；长文本保留首尾；回归覆盖末尾事实和摘要重载 |

## 2026-11-05 评审修复验收

- 真实 API：`.venv/Scripts/python.exe -m pytest -q -m live tests/test_review_regressions.py`，1 passed；真实 LLM 调用 todo 添加待办，随后在强制压缩历史后追问并实际完成该待办，校验持久化 done 状态。
- 确定性回归覆盖 4000 token 小预算、大量长待办、连续工具结果、格式纠错预算、超长系统/输入拒绝调用、默认 16000 预算、末尾关键字段保留及非法工具输出隔离。
- 最终验证：默认测试集 254 passed、7 skipped（真实 API 测试默认跳过）；单独真实 API 验收 1 passed、20 deselected。记录与文档按 UTF-8 保存。
