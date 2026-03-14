# nanobot-opt 全盘分析报告

> 由 Gemini + Codex (gpt-5.4) 于 2026-03-14 对全项目源码进行分析生成。

---

## P1 — 必须修复（真实 bug / 安全风险）

### P1-1 消息静默丢失（loop.py）
`_dispatch` 在处理一条消息期间，新到达的消息被追加到 `_session_queued`，但 `finally` 块直接 `pop` 掉整个队列，这些消息永远不会被处理。
- 位置：`loop.py:377/388/425`
- 影响：正常并发下消息静默丢失，用户无任何提示
- 来源：Codex

### P1-2 SSRF 漏洞（web.py）
`web_fetch` URL 验证只检查 scheme/netloc，未屏蔽 localhost、127.0.0.1、RFC1918 私有 IP、云元数据端点（如 169.254.169.254）。LLM 可被诱导探测内网服务。
- 位置：`web.py:40/227/277`
- 影响：安全边界失效，内网服务可被探测
- 来源：Codex

### P1-3 ExecTool 沙箱可绕过（shell.py）
`working_dir` 由调用方控制，即使设置 `restrict_to_workspace=True`，仍可将 working_dir 设到 workspace 外执行命令。使用 `create_subprocess_shell` + 黑名单正则，可被绕过。
- 位置：`shell.py:78/82/157`
- 影响：LLM 可逃逸 workspace 沙箱
- 来源：Codex

### P1-4 出站队列无界（bus/queue.py）
`MessageBus` 使用无界 `asyncio.Queue`，`ChannelManager._dispatch_outbound` 串行发送所有出站消息。慢 channel 或大量 progress/tool-hint 事件会导致内存无限增长、所有出站流量卡死。
- 位置：`queue.py:16`、`manager.py:107`
- 影响：内存溢出，服务不可用
- 来源：Codex

---

## P2 — 重要改进

### P2-1 Session 持久化 O(n) 全量重写
`SessionManager.save` 每次 turn 重写整个 JSONL 文件。历史增长后 I/O 线性增加，崩溃时可损坏整个 session。
- 位置：`session/manager.py:163/167`
- 建议：改为 append-only + atomic rename 快照
- 来源：Codex

### P2-2 有状态工具并行执行不安全
`_run_agent_loop` 用 `asyncio.gather` 并行执行所有工具调用，包括 `edit_file`+`write_file` 等有状态、相互依赖的工具组合。
- 位置：`loop.py:241`
- 建议：引入工具执行类别（串行/并行），mutating 工具默认串行
- 来源：Codex

### P2-3 System prompt 每次从磁盘重建
`ContextBuilder.build_system_prompt` 每次 turn 重新从磁盘加载 bootstrap 文件、memory、skill 摘要。`estimate_session_prompt_tokens` 也每次重建完整 prompt 做 token 探针。
- 位置：`context.py:27`、`memory.py:342/368`
- 建议：缓存 bootstrap + skill 摘要，memory 变更时才刷新
- 来源：Codex / Gemini

### P2-4 图片 base64 无大小限制
`_build_user_content` 读取所有媒体文件并内联 base64，无大小上限或缩放处理。大图会撑爆 context、RAM 和 provider 请求限制。
- 位置：`context.py:165`
- 建议：限制单图最大尺寸（如 4MB），超出时缩放或拒绝
- 来源：Codex

### P2-5 异步函数中同步文件 I/O
`SessionManager`、`MemoryStore`、`RunLogger` 等在 async 函数中使用同步 `Path.read_text/write_text/open`，高并发下阻塞整个事件循环。
- 位置：多处
- 建议：改用 `asyncio.to_thread` 或 `aiofiles`
- 来源：Gemini / Codex review

### P2-6 pytest 配置损坏
`matrix.py` 硬导入可选依赖导致 collection 失败；`pytest-asyncio` 配置与实际版本不匹配，大量 async 测试无法运行。
- 位置：`matrix.py:11`、`pyproject.toml:63/115`
- 建议：optional 依赖改为懒加载 + skip；安装 pytest-asyncio 并配置 asyncio_mode
- 来源：Codex

---

## P3 — 长期优化

### P3-1 Router 关键词 substring 匹配不精确
`ModelRouter` 使用 `kw in msg_lower` 做关键词匹配，会误判（如包含 `好的` 的长词）。
- 位置：`router.py:183/191`
- 建议：改用词边界正则或 token 级匹配

### P3-2 allow_from=[] 语义不一致
文档说空 allowlist = 公开访问，代码实际 exit 拒绝。
- 位置：`schema.py:199`、`manager.py:54`

### P3-3 缺少流式响应
当前为 turn-based，无端到端 token streaming，影响长回复 UX。

### P3-4 缺少语义记忆
HISTORY.md 只能 grep 关键词搜索，建议引入本地 embedding + RAG。

### P3-5 缺少结构化 Metrics
目前只有 token 统计，缺少延迟、队列深度、错误率、消息丢弃计数等可观测性指标。

---

## 架构评价

**优点：**
- Producer-Consumer 消息总线解耦 channel 与 agent
- 三层记忆（Session/MEMORY.md/HISTORY.md）清晰
- ToolRegistry / ProviderRegistry 扩展性好
- SubagentManager 支持层级 Agent

**主要弱点：**
- `AgentLoop` 承担过多职责（路由/队列/取消/持久化/统计/slash命令/后台整合），正在成为上帝对象
- 建议拆分为：transport/session scheduling、inference orchestration、persistence/memory services
