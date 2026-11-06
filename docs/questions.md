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
| 添加待办后第二次模型请求失败或进程硬中断，恢复重试时可能重复创建待办或丢失已执行的写操作 | 之前仅在整轮 `ChatService.send` 结束时才调用 `session.save()`，且每次 `send` 都无条件自增 `turn_id`，导致失败轮次重试时 `turn_id` 变化、无法关联同轮已成功的写操作 | 新增确定性 `call_id`（关联 `user_id`、`session_id`、`turn_id`、工具名与规范化参数）与 `SessionData.tool_calls` / `unfinished_turn`；待办 `add`/`complete`/`delete` 成功后通过 `on_step_change` 立即将最新 `todos` 与 `tool_calls` 在同一原子保存中落盘；未完成轮次恢复重试时复用原 `turn_id` 与 `call_id`，已成功调用直接返回结果（`created=false`），`pending`/`interrupted` 状态先通过 `reconcile_todo_call` 与 `TodoStore` 核对，轮次正常完成后清空 `unfinished_turn` 以允许新轮次创建同名待办 |
| 写操作立即落盘时若发生磁盘写入异常（`OSError`），内存中的 `TodoStore` 与磁盘状态可能不一致 | 工具先修改了内存 `TodoStore`，随后 `session.save()` 抛异常 | 在执行待办写操作前保存 `prev_todos` 快照，若立即落盘失败则回滚 `TodoStore` 与会话内存数据、移除未落盘消息，将调用记录标为 `error` 并以 `persist_error` 状态安全退出，保留此前已落盘的步骤 |
| 异常退出或中断时缺乏统一结构化追踪，难以定位未完成调用 | 仅有终端 `-v` 打印，未持久化调用耗时、错误与请求/结果配对关系 | 新增 `agent/trace.py`（`TraceLogger`、`verify_trace_integrity`、`find_unfinished_calls`），按会话写入 JSONL 事件（`turn_start`/`tool_request`/`tool_result`/`parse_error`/`turn_end`），接入 `cli.py --trace` 查看 |
| 失败后的新请求（如用户明确要求“另外再创建一条同名待办”）被误判为旧请求重试，导致被去重 | 只要存在 `unfinished_turn`，下一次输入就无条件复用旧 `turn_id`，未区分恢复重试与新建请求 | 新增 `_is_retry_request`（及可选 `retry` 参数）：仅当输入与原未完成请求相同或包含继续/重试意图且不含“另外/再创建/新增一条”等新建意图时才复用原 `turn_id`，否则作为新请求分配新 `turn_id` |
| 写操作即时保存成功后，若整轮结束时最终保存失败，`OSError` 直接抛出且 Trace 已提前记录 `turn_end=ok` | `Agent.run` 在 `finish()` 中先写 `turn_end` 日志，返回后 `ChatService.send` 才执行最终 `session.save()` 且未捕获 `OSError` | 新增 `on_turn_end` 钩子并在 `finish()` 记录 `turn_end` 前执行最终落盘；若落盘失败则回滚未落盘内存变更、统一转为 `persist_error` 状态并写入 `turn_end(status="persist_error")`，保留此前已即时保存的操作 |
| 正常恢复重试后，Trace 完整性检查返回 `valid=False` | 重试复用了同一个 `turn_id`，但新一次尝试的工具步数重新从 1 开始（如 `[1, 1]`），而原校验按 `turn_id` 汇总所有步数 | 为 Trace 事件增加 `attempt_id` 区分同一 `turn_id` 下的每次尝试，保持 `call_id` 跨尝试稳定用于幂等，并在 `verify_trace_integrity` 中按 `(turn_id, attempt_id)` 校验请求/结果配对、步数从 1 连续递增与终止事件 |
| 原生工具调用响应中 `message.content` 可能为 `None`，原 `_extract` 会判定为空响应并抛 `LLMError` | 模型返回 `tool_calls` 时通常不填充文本 `content` | 新增 `LLMResponse(content, tool_calls)` 与 `_extract_response`，允许仅含 `tool_calls` 或仅含非空 `content` 的响应通过，仅当二者同时为空时抛 `LLMError`；并对不支持 `parallel_tool_calls` 参数的兼容服务端自动降级重试 |
| 切换到原生工具调用主路径后，M0–M5 中大量使用纯文本 `FakeLLM(["not json", ...])` 的旧单元测试如何保持兼容 | 旧测试向 `FakeLLM` 注入自定义 JSON 字符串或故意损坏的非 JSON 文本以测试文本解析器 | `OpenAIClient` 与结构化 `FakeLLM` 标记 `supports_native_tools = True` 走原生主路径；同时保留 `tool_mode="text"` 显式兼容选项，并对仅提供纯字符串回复列表的旧模拟客户端自动兼容文本解析路径 |
| 原生工具调用消息（`assistant.tool_calls` + `role="tool"`）在上下文压缩与历史回放时可能丢失配对或将思考过程带入长期记忆 | 原 `context.py` 和 `session.py` 仅认识 `user`/`assistant` 角色及 `Observation (tool=...)` 文本格式；且 OpenAI 协议要求 `assistant.tool_calls` 必须紧跟配对 `tool_call_id` 的 `role="tool"` 消息 | `session.py` 扩展校验 `assistant.tool_calls` 与 `role="tool"`；`context.py` 在 `sanitize_turn` 中清空 `tool_calls` 消息伴随的思考文本并通过 `_ensure_tool_Pairs` 保证请求与结果成对，在 `summarize_turns` 中通过 `tool_call_id` 关联工具名与参数提取事实和待办状态；`cli.py` 同步支持两种格式的回放 |
| 异步执行链中若直接调用同步文件锁轮询（`time.sleep`）或同步网络 I/O，会阻塞整个事件循环导致其他并发会话卡顿；且 `asyncio.CancelledError` 可能被宽泛捕获吞掉 | 同步 `open()` 使用 `threading.Lock` 与 `time.sleep` 等待锁；工具与模型请求若捕获 `BaseException` 或未区分 `CancelledError` 会破坏取消语义 | `OpenAIClient` 新增 `AsyncOpenAI`（`acomplete`/`achat`），`search`/`weather` 新增 `httpx2.AsyncClient` 异步执行并在 `ToolRegistry.aexecute` 中支持 `asyncio.wait_for` 超时控制与 `CancelledError` 原样透传；`SessionManager.aopen` 与 `Session.asave` 使用 `asyncio.sleep` 非阻塞轮询与 `asyncio.to_thread` 刷盘，确保慢会话不阻塞其他会话 |
| 模型一次返回多个 `tool_calls` 时，若无差别并发执行可能导致同一会话的待办写操作竞争、读写顺序错乱（如 `add` 与 `complete`/`list` 乱序）或单项超时拖垮整批查询 | 缺乏读写属性区分与并发度控制，且若异常直接抛出将取消同批已完成或进行中的独立只读查询 | 新增 `_is_parallel_safe` 与 `_partition_batch_segments`：仅将连续无依赖只读工具合并为 `parallel` 段并在 `asyncio.Semaphore(max_concurrency)` 下并发执行；待办写操作及含写操作批次中的 `todo.list` 均切分为 `serial` 段严格按序串行执行并即时落盘；单项超时或工具错误由 `aexecute` 封装为 `ToolResult(ok=False)`，完整保留同批其他成功结果 |
| 并行批次中各工具完成顺序不确定、中途被取消（`CancelledError`）或写入成功后模型失败重试时，可能导致消息配对残缺、预算漏计或 Trace 校验失败 | 并发任务按完成先后返回，若直接追加会导致顺序漂移；中途取消时未完成项若缺 `role="tool"` 消息或 `tool_result` 事件会破坏协议与 Trace 配对；重试时模型生成的 `api_call_id` 会变化 | 在 `_prepare_batch_items` 中按单次调用逐个扣减 `max_steps` 预算并预分配确定性 `step` 与稳定 `call_id`；并发段内按实际完成时刻记录 `tool_result` Trace，随后按原请求顺序回填 `role="tool"` 消息与 `ActionRecord`；中途取消时在 `_prepare_interrupted_state` 中保留已完成项并把未完成项补齐 `status=interrupted` 的消息与 Trace；恢复重试时通过稳定 `call_id` 复用已落盘写操作并绑定新 `api_call_id` |

| 同步工具在 aexecute 中直接执行，超时也无法打断阻塞调用 | 工具函数占用事件循环，wait_for 无机会触发；超时后线程副作用状态需要区分 | 同步函数改用 asyncio.to_thread。只读调用超时返回 unknown_after_timeout 并说明后台可能仍运行；写调用使用串行锁，超时或取消后先等实际执行结束，再返回真实结果及 completed_after_timeout 状态，避免重复写入与竞态。新增事件循环、超时和串行回归测试 |
| 原生调用的预算仅统计 messages，漏算单独发送的 tools Schema | build_context 和请求前检查未包含原生工具定义 | 统一估算工具 Schema；会话构建、同步及异步原生请求前检查同一份 Schema，超限返回 context_limit 且不调用模型。新增构建阶段、首轮和工具结果后续轮次回归 |
| 会话中缺少可更新、可删除的用户记忆，历史答案或工具结果容易被误当用户事实 | 之前只有混合来源的对话摘要，没有独立的用户事实存储与召回条件 | 新增 session.memories，仅从当前用户明确声明解析设置、修改、删除；相关问题才注入系统消息，历史回放剔除旧记忆声明。模型回答和工具结果不写入该字段；测试覆盖来源隔离、跨 session、更新、删除与重启召回 |

| 无问号提问覆盖城市，姓名“李为民”被拆为错误键值 | 原记忆正则允许任意贪婪字段，并把“是/为”当分隔符回溯；只检查问号，没有识别“什么”等疑问值 | 字段限定为姓名、城市、预算、偏好等白名单，固定字段后再解析分隔符；拒绝疑问词开头及“吗/么/呢”结尾的值，保留姓名内部的“为” |
| “不要记忆”的示例正文仍写入用户记忆 | 原实现先按句拆分，每句独立提取，失去整条输入的否定、示例、引用和假设语境 | 先检查整条用户输入，再从可信语境提取明确声明；示例或引用不写入，也不从历史上下文中错误删去 |
| 姓名问法与“按刚才的要求”未召回，已保存的有效声明又从历史中被移除 | 姓名别名、指代问法和记忆存储键不统一；历史清理无条件剔除所有可解析声明 | 统一姓名别名及指代召回；历史回放仅剔除与当前会话记忆不一致的过期声明，保留有效声明。限制“刚才”召回范围，避免影响待办和天气追问 |

## 2026-11-05 评审修复验收

- 真实 API：`.venv/Scripts/python.exe -m pytest -q -m live tests/test_review_regressions.py`，1 passed；真实 LLM 调用 todo 添加待办，随后在强制压缩历史后追问并实际完成该待办，校验持久化 done 状态。
- 确定性回归覆盖 4000 token 小预算、大量长待办、连续工具结果、格式纠错预算、超长系统/输入拒绝调用、默认 16000 预算、末尾关键字段保留及非法工具输出隔离。
- 最终验证：默认测试集 254 passed、7 skipped（真实 API 测试默认跳过）；单独真实 API 验收 1 passed、20 deselected。记录与文档按 UTF-8 保存。

## 2026-11-06 工具超时、原生预算与会话记忆验收

- 默认测试集：299 passed、11 skipped；新增回归覆盖同步只读超时不阻塞循环、写工具超时与取消后的串行状态、原生 Schema 预算首轮及工具结果后续轮次、记忆来源隔离与更新删除。
- 真实 API 核心流程：tests/test_review_regressions.py 的 live 验收 1 passed、20 deselected，完成待办写入及压缩后追问。

## 2026-11-06 记忆提取与召回修复验收

- 默认测试集：311 passed、11 skipped；覆盖无问号提问、姓名内部“为”、小数及千分位预算、不记忆和示例语境、姓名别名、行程指代、历史有效声明与示例回放。
- 真实 API 核心流程：tests/test_review_regressions.py 的 live 验收 1 passed、20 deselected。

## 2026-11-06 本地交付文档与最终验收

- 文档按代码现状补充安装、配置、CLI 会话恢复、native/text 模式、并发范围、重试、Trace、记忆召回和未实现功能；PLAN 与架构说明区分已交付行为及后续方向。
- 配置核对：`.env.example` 只含占位密钥；`pyproject.toml` 与 `uv.lock` 提供可复现依赖；`.gitignore` 忽略 `.env`、`data/`、`logs/`、虚拟环境和测试缓存。`git ls-files .env data logs '*.jsonl'` 无输出，没有跟踪密钥或会话/Trace 文件。
- 常规测试：`.venv/Scripts/python.exe -m pytest -q`，311 passed、12 skipped（所有真实 API 场景不在默认集执行）。
- 真实 API：`.venv/Scripts/python.exe -m pytest -q -m live`，12 passed、311 deselected。覆盖原生调用闭环、并行天气独立查询、天气后待办依赖操作、待办写入/完成及 Trace、失败写入恢复与真实模型下记忆新增/更新/删除及召回。
- 外部服务限制：并行天气验收使用真实模型决策，但通过注入稳定的异步天气响应隔离 wttr.in 的外部波动；LLM 验收使用本机 `.env` 所指服务与模型。项目估算 token 和记忆白名单规则分别不保证等同模型精确计数及任意自然语言覆盖。
