# nanobot-opt

本仓库是 [nanobot](https://github.com/HKUDS/nanobot) 的个人优化分支，在原版基础上做了稳定性、性能与可观测性方面的改进。所有改动完全向后兼容，不填新配置项则行为与原版一致。

---

## 改动内容

### 1. Per-session 并发锁

原版使用全局锁，多用户场景下 A 用户处理消息时 B 用户必须等待。本分支改为 per-session 字典锁 + 全局信号量（最多 10 个并发 session），各用户会话互不阻塞。新增 drain-merge 队列：同一 session 在处理期间收到的多条消息合并为一次 LLM 调用，避免串行重复请求。

### 2. 工具并行执行

原版对 LLM 一次返回的多个工具调用逐个串行执行。本分支对标记了 `parallel_safe=True` 的工具改用 `asyncio.gather` 并行执行，减少等待时间；有副作用的工具仍按顺序执行。

### 3. Token 用量统计

每轮对话的 token 消耗（输入/输出）累加存入 session 元数据，并持久化到 `workspace/usage_stats.json`。通过 `/usage` 命令查看当前会话的累计用量、路由分布及平均响应延迟。

### 4. 多档模型路由

新增 `nanobot/agent/router.py`，根据消息特征自动选择不同档位的模型，降低 API 费用。

**路由规则（优先级从高到低）：**

| 档位 | 触发条件 | 典型用途 |
|------|---------|----------|
| `heavy` | 消息 ≥300 字，或含编程/分析类关键词，或含代码块/步骤模式 | 代码生成、复杂推理 |
| `normal` | 其他（默认兜底） | 一般任务 |
| `fast` | 消息 ≤60 字 + 历史 ≤2 轮 + 含问候/确认类关键词 | 简单问答、日常闲聊 |

**自动升格**：当模型返回 `finish_reason=length`（回复被截断）时，自动升级到下一档模型重试。

**错误降级**：当路由模型遇到限速/过载错误时，自动回退到配置的默认模型重试。

**启用方式**（`~/.nanobot/config.json`）：

```json
{
  "agents": {
    "defaults": {
      "model": "anthropic/claude-opus-4-5",
      "routingEnabled": true,
      "modelFast": "anthropic/claude-haiku-4-5",
      "modelNormal": "anthropic/claude-sonnet-4-5",
      "modelHeavy": "anthropic/claude-opus-4-5"
    }
  }
}
```

不填 `routingEnabled` 或设为 `false` 则路由不启用，行为与原版完全一致。

### 5. Memory 双轨后端

新增 `memory.backend` 配置项，支持三种模式：

| 模式 | 说明 |
|------|------|
| `file`（默认）| 本地 `MEMORY.md` / `HISTORY.md`，与原版行为一致 |
| `nocturne_mcp` | 仅写入 nocturne_memory MCP 服务，不写本地文件 |
| `hybrid` | 写本地文件（主）+ 异步同步到 MCP（副） |

Memory 文件超过 30 行时自动启用分级加载：仅注入 `# now` 节 + 结构摘要到 system prompt，避免大文件撑满 context window。

### 6. 原生浏览器工具

新增基于 Camoufox 的浏览器工具组，支持打开页面、获取文本/HTML、截图、点击等操作，用于访问 Cloudflare 保护或需要 JS 渲染的页面：

- `browser_open` — 打开 URL
- `page_get_text` — 提取页面纯文本
- `page_get_html` — 获取页面 HTML
- `page_screenshot` — 截图
- `page_click` — 点击元素

### 7. MCP 工具集成增强

- 支持 `stdio` / `sse` / `streamableHttp` 三种传输协议
- 新增 `enabled_tools` 过滤：可按名称白名单控制每个 MCP server 暴露的工具
- `tool_timeout` 可配置，防止单个 MCP 工具调用阻塞整个 agent

### 8. 流式输出（Telegram）

Telegram 频道支持流式输出（edit-message 模式），模型生成过程中实时更新消息，减少等待感。

### 9. 代码质量修复（本次审查）

- `_StreamBuffer`：`asyncio.get_event_loop()` 改为 `asyncio.get_running_loop()`（Python 3.10+ 废弃警告）
- `_dispatch`：`route` 变量初始化为 `None`，消除条件分支下的潜在 NameError
- `MemoryConsolidator`：新增 `configure_nocturne()` 方法，统一 nocturne adapter 赋值路径
- `_update_global_stats`：改为 async + 加 `_stats_lock`，防止多 session 并发写竞争
- `had_error`：改用 `finish_reason == "error"` 替代字符串前缀匹配，提升准确性
- 测试：`test_memory_backend` 工具调用断言对齐 `mcp_nocturne_memory_` 前缀

---

## Agent 自主进化（P0–P3）

本分支在原版基础上新增了一套 Agent 自主进化机制，所有功能默认关闭（kill switch），按需启用。

### P0 — 评估基线

新增 `EvalRunner`，运行固定 Benchmark 任务集并将结果追加到 `memory/BENCHMARK.md`。

**配置（`tools.eval`）：**
```json
{
  "tools": {
    "eval": {
      "enabled": true,
      "run_after_turns": 50,
      "suite": [
        {"id": "e001", "prompt": "ls the workspace", "expect_tool": "exec"}
      ]
    }
  }
}
```

### P1 — 任务反思 + 记忆质量评分

- **反思**：整合完成后触发第二次 LLM 调用，提炼经验写入 `# candidates`，confidence 达标后晋升到 `# lessons`
- **裁剪**：MEMORY.md 超过阈值行数时，三维评分（recency × hits × reliability）低分条目归档到 HISTORY.md

**配置（`tools.memory`）：**
```json
{
  "tools": {
    "memory": {
      "reflection": {"enabled": true, "min_confidence": 0.7},
      "prune": {"enabled": true, "trigger_lines": 100, "min_score": 0.3}
    }
  }
}
```

### P2 — Skill 使用频率追踪 + PITFALLS.md

- **SkillStats**：记录每个 skill 的调用次数和成功率，高频高成功率 skill 在 context 中优先展示
- **PITFALLS.md**：`memory/PITFALLS.md` 存在时自动注入 system prompt，高优先级避坑提示

### P3 — 动态 Skill 生成 + 多 Agent 经验共享

- **动态 Skill 生成**：检测重复命令模式，自动生成草稿 skill（`skills/.drafts/`），达到 `promote_after_uses` 次命中后自动转正
- **多 Agent 经验共享**：子 agent 完成后提取经验，用 `<untrusted-data>` 隔离注入主 agent 记忆候选队列

**配置（`tools.skill_autogen`）：**
```json
{
  "tools": {
    "skill_autogen": {
      "enabled": true,
      "min_pattern_count": 5,
      "min_success_rate": 0.7,
      "promote_after_uses": 3
    }
  }
}
```

---

## 上游项目

本分支基于 [nanobot](https://github.com/HKUDS/nanobot)，遵循 MIT License。原版功能、配置说明、部署文档请参考上游仓库。
