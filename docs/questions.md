# 问题解决记录

| 问题 | 原因 | 解决方案 |
|---|---|---|
| pytest 报 `No module named 'cli'` | 根目录模块不在 pytest 的 sys.path 中 | pyproject 中配置 `pythonpath = ["."]` |
| 测试里 `import httpx` 失败 | 新版 openai 依赖 httpx2，环境中无 httpx | 测试中不依赖 httpx，用占位对象构造 APIConnectionError |
| PLAN.md 中问题记录文件名与约定不一致 | 早期命名为 troubleshooting.md | 统一改为 docs/questions.md |
| 当前 .env 为空，无法做真实 API 调用 | 尚未填写密钥 | 参考 .env.example 填写后运行 `uv run python cli.py` / `uv run pytest -m live` |
| test_cli_config_error ʧ�ܣ�ȱ�����Լ��ص���Ŀ .env | `load_dotenv()` Ĭ�ϴ� config.py ����Ŀ¼���ϲ��� .env���� cwd �޹� | Ĭ��ֻ�� `cwd/.env`��Ҳ����ʽ�� `dotenv_path`�����Ӳ��� |
| LLM_TIMEOUT=abc �� ValueError������������ | δУ����ֵ���� | ͳһ�� ConfigError��timeout>0��retries>=0 |
| choices=[] �� IndexError�����ַ������ɹ� | ��Ӧ��ȡ�޷��� | ���� `_extract`���쳣��Ӧͳһ�� LLMError |
| ��ʵ API ������ȱ SOCKS �����ڿͻ��˳�ʼ��ʧ�� | ���������� socks �����httpx ��Ҫ socksio | ������Ϊ `httpx2[socks]` |
| calculator �� eval ��ע����� | eval ��ִ��������� | �� ast �������ڵ���ֵ�����Ʊ��ʽ��������ָ�� |
| ���� `True` ������ integer | bool �� int ���� | У��ʱ����ֵ������ʽ�ų� bool |
| ʧ�����Կ����ظ���Ӵ��� | add ���ݵ� | ��ͬ���ݵ�δ��ɴ��츴��������Ŀ������ created=false |
