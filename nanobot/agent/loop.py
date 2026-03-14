"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory import MemoryConsolidator, RunLogger
from nanobot.agent.router import FailoverReason, ModelRouter, build_router_from_config, classify_error
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.session.manager import Session, SessionManager

if TYPE_CHECKING:
    from nanobot.config.schema import ChannelsConfig, ExecToolConfig, WebSearchConfig
    from nanobot.cron.service import CronService


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _TOOL_RESULT_MAX_CHARS = 16_000

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 40,
        context_window_tokens: int = 65_536,
        web_search_config: WebSearchConfig | None = None,
        web_proxy: str | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
        router: ModelRouter | None = None,
    ):
        from nanobot.config.schema import ExecToolConfig, WebSearchConfig

        self.bus = bus
        self.channels_config = channels_config
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.context_window_tokens = context_window_tokens
        self.web_search_config = web_search_config or WebSearchConfig()
        self.web_proxy = web_proxy
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace

        self.context = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            web_search_config=self.web_search_config,
            web_proxy=web_proxy,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._session_locks: dict[str, asyncio.Lock] = {}  # per-session locks
        self._session_pending: dict[str, int] = {}  # per-session pending count
        self._session_queued: dict[str, list[InboundMessage]] = {}  # per-session queued msgs for drain merge
        self._max_pending_per_session = 3  # 超过此数量的排队消息直接丢弃
        self._global_semaphore = asyncio.Semaphore(10)  # 全局最大并发 session 数
        self.router = router
        self._run_logger = RunLogger(workspace)
        self.memory_consolidator = MemoryConsolidator(
            workspace=workspace,
            provider=provider,
            model=self.model,
            sessions=self.sessions,
            context_window_tokens=context_window_tokens,
            build_messages=self.context.build_messages,
            get_tool_definitions=self.tools.get_definitions,
        )
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        for cls in (ReadFileTool, WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
            path_append=self.exec_config.path_append,
        ))
        self.tools.register(WebSearchTool(config=self.web_search_config, proxy=self.web_proxy))
        self.tools.register(WebFetchTool(proxy=self.web_proxy))
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from nanobot.agent.tools.mcp import connect_mcp_servers
        try:
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)
            self._mcp_connected = True
        except BaseException as e:
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
        finally:
            self._mcp_connecting = False

    def _set_tool_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Update context for all tools that need routing info."""
        for name in ("message", "spawn", "cron"):
            if tool := self.tools.get(name):
                if hasattr(tool, "set_context"):
                    tool.set_context(channel, chat_id, *([message_id] if name == "message" else []))

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hint, e.g. 'web_search("query")'."""
        def _fmt(tc):
            args = (tc.arguments[0] if isinstance(tc.arguments, list) else tc.arguments) or {}
            val = next(iter(args.values()), None) if isinstance(args, dict) else None
            if not isinstance(val, str):
                return tc.name
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'
        return ", ".join(_fmt(tc) for tc in tool_calls)

    def _get_session_lock(self, key: str) -> asyncio.Lock:
        """Return the per-session lock, creating it if needed."""
        if key not in self._session_locks:
            self._session_locks[key] = asyncio.Lock()
        return self._session_locks[key]

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        model_override: str | None = None,
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> tuple[str | None, list[str], list[dict], dict]:
        """Run the agent iteration loop."""
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []
        total_usage: dict[str, int] = {}
        last_finish_reason: str | None = None
        model = model_override or self.model

        while iteration < self.max_iterations:
            iteration += 1

            tool_defs = self.tools.get_definitions()

            response = await self.provider.chat_with_retry(
                messages=messages,
                tools=tool_defs,
                model=model,
            )

            # Accumulate token usage
            for k, v in (response.usage or {}).items():
                total_usage[k] = total_usage.get(k, 0) + v
            last_finish_reason = response.finish_reason

            if response.has_tool_calls:
                if on_progress:
                    thought = self._strip_think(response.content)
                    if thought:
                        await on_progress(thought)
                    await on_progress(self._tool_hint(response.tool_calls), tool_hint=True)

                tool_call_dicts = [
                    tc.to_openai_tool_call()
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )

                # Execute tool calls in parallel
                async def _exec_one(tc):
                    tools_used.append(tc.name)
                    logger.info("Tool call: {}({})", tc.name, json.dumps(tc.arguments, ensure_ascii=False)[:200])
                    result = await self.tools.execute(tc.name, tc.arguments)
                    return tc, result

                pairs = await asyncio.gather(*[_exec_one(tc) for tc in response.tool_calls])
                for tc, result in pairs:
                    messages = self.context.add_tool_result(
                        messages, tc.id, tc.name, result
                    )
            else:
                clean = self._strip_think(response.content)
                # Don't persist error responses to session history — they can
                # poison the context and cause permanent 400 loops (#1303).
                if response.finish_reason == "error":
                    err_reason = classify_error(clean or "")
                    # Context overflow: don't retry, just report
                    if err_reason == FailoverReason.CONTEXT_OVERFLOW:
                        logger.error("Context overflow, not retrying: {}", (clean or "")[:200])
                        final_content = "对话历史过长，请使用 /new 开启新会话。"
                        break
                    # Auth error: don't retry, credentials won't change
                    if err_reason == FailoverReason.AUTH:
                        logger.error("Auth error, not retrying: {}", (clean or "")[:200])
                        final_content = clean or "认证失败，请检查 API Key 配置。"
                        break
                    # Rate limit / overload: fall back to default model if routed
                    if model_override and model_override != self.model:
                        logger.warning(
                            "Routed model {} failed (reason={}), falling back to default model {}",
                            model_override, err_reason.value, self.model,
                        )
                        model = self.model
                        model_override = None
                        continue
                    logger.error("LLM returned error (reason={}): {}", err_reason.value, (clean or "")[:200])
                    final_content = clean or "Sorry, I encountered an error calling the AI model."
                    break
                messages = self.context.add_assistant_message(
                    messages, clean, reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                final_content = clean
                break

        if final_content is None and iteration >= self.max_iterations:
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            final_content = (
                f"I reached the maximum number of tool call iterations ({self.max_iterations}) "
                "without completing the task. You can try breaking the task into smaller steps."
            )

        return final_content, tools_used, messages, total_usage, last_finish_reason

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            cmd = msg.content.strip().lower()
            if cmd == "/stop":
                await self._handle_stop(msg)
            elif cmd == "/restart":
                await self._handle_restart(msg)
            else:
                task = asyncio.create_task(self._dispatch(msg))
                self._active_tasks.setdefault(msg.session_key, []).append(task)
                task.add_done_callback(lambda t, k=msg.session_key: self._active_tasks.get(k, []) and self._active_tasks[k].remove(t) if t in self._active_tasks.get(k, []) else None)

    async def _handle_stop(self, msg: InboundMessage) -> None:
        """Cancel all active tasks and subagents for the session."""
        tasks = self._active_tasks.pop(msg.session_key, [])
        active_tasks = [t for t in tasks if not t.done()]
        pending_count = self._session_pending.get(msg.session_key, 0)

        cancelled = sum(1 for t in active_tasks if t.cancel())
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        sub_cancelled = await self.subagents.cancel_by_session(msg.session_key)
        self._session_pending.pop(msg.session_key, None)

        total = cancelled + sub_cancelled
        if total:
            parts = [f"已停止 {total} 个任务"]
            if pending_count > 1:
                parts.append(f"（队列中还有 {pending_count - 1} 条消息已清除）")
            content = "。".join(parts) + "。"
        else:
            content = "当前没有正在执行的任务。"
        await self.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=content,
        ))

    async def _handle_restart(self, msg: InboundMessage) -> None:
        """Restart the process in-place via os.execv."""
        await self.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content="Restarting...",
        ))

        async def _do_restart():
            await asyncio.sleep(1)
            # Use -m nanobot instead of sys.argv[0] for Windows compatibility
            # (sys.argv[0] may be just "nanobot" without full path on Windows)
            os.execv(sys.executable, [sys.executable, "-m", "nanobot"] + sys.argv[1:])

        asyncio.create_task(_do_restart())

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message under the per-session lock, with pending queue cap and drain merge."""
        key = msg.session_key
        pending = self._session_pending.get(key, 0)
        if pending >= self._max_pending_per_session:
            logger.warning(
                "Session {} queue full ({} pending), dropping message: {!r}",
                key, pending, msg.content[:60],
            )
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content="消息队列已满，请等待当前任务完成后再发送。",
            ))
            return

        # 如果当前 session 正在处理中（有 pending），把消息加入等待队列
        # 等锁释放后合并处理，避免多次串行调用 LLM
        if pending > 0:
            self._session_queued.setdefault(key, []).append(msg)
            self._session_pending[key] = pending + 1
            logger.debug("Session {} queued message for drain merge: {!r}", key, msg.content[:40])
            return

        self._session_pending[key] = 1
        try:
            async with self._global_semaphore:
                async with self._get_session_lock(key):
                    # 合并等待队列中的消息
                    queued = self._session_queued.pop(key, [])
                    if queued:
                        combined = msg.content + "\n\n" + "\n\n".join(
                            f"[后续消息 {i+1}]: {q.content}" for i, q in enumerate(queued)
                        )
                        logger.info(
                            "Session {} drain-merging {} queued messages into one",
                            key, len(queued),
                        )
                        merged_msg = InboundMessage(
                            channel=msg.channel,
                            sender_id=msg.sender_id,
                            chat_id=msg.chat_id,
                            content=combined,
                            metadata=msg.metadata or {},
                        )
                        process_msg = merged_msg
                    else:
                        process_msg = msg
                    try:
                        response = await self._process_message(process_msg)
                        if response is not None:
                            await self.bus.publish_outbound(response)
                        elif msg.channel == "cli":
                            await self.bus.publish_outbound(OutboundMessage(
                                channel=msg.channel, chat_id=msg.chat_id,
                                content="", metadata=msg.metadata or {},
                            ))
                    except asyncio.CancelledError:
                        logger.info("Task cancelled for session {}", key)
                        raise
                    except Exception:
                        logger.exception("Error processing message for session {}", key)
                        await self.bus.publish_outbound(OutboundMessage(
                            channel=msg.channel, chat_id=msg.chat_id,
                            content="Sorry, I encountered an error.",
                        ))
        finally:
            self._session_pending.pop(key, None)
            self._session_queued.pop(key, None)

    async def close_mcp(self) -> None:
        """Close MCP connections."""
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        # System messages: parse origin from chat_id ("channel:chat_id")
        if msg.channel == "system":
            channel, chat_id = (msg.chat_id.split(":", 1) if ":" in msg.chat_id
                                else ("cli", msg.chat_id))
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            await self.memory_consolidator.maybe_consolidate_by_tokens(session)
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
            history = session.get_history(max_messages=0)
            messages = self.context.build_messages(
                history=history,
                current_message=msg.content, channel=channel, chat_id=chat_id,
            )
            final_content, _, all_msgs, usage, _ = await self._run_agent_loop(messages)
            self._save_turn(session, all_msgs, 1 + len(history), usage)
            self.sessions.save(session)
            await self.memory_consolidator.maybe_consolidate_by_tokens(session)
            return OutboundMessage(channel=channel, chat_id=chat_id,
                                  content=final_content or "Background task completed.")

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)

        # Slash commands
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            try:
                if not await self.memory_consolidator.archive_unconsolidated(session):
                    return OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content="Memory archival failed, session not cleared. Please try again.",
                    )
            except Exception:
                logger.exception("/new archival failed for {}", session.key)
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Memory archival failed, session not cleared. Please try again.",
                )

            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started.")
        if cmd == "/help":
            lines = [
                "🐈 nanobot 命令列表：",
                "/new — 开始新对话",
                "/stop — 停止当前任务",
                "/restart — 重启机器人",
                "/usage — 查看 Token 用量",
                "/help — 显示帮助信息",
            ]
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content="\n".join(lines),
            )
        if cmd == "/usage":
            prompt = session.metadata.get("prompt_tokens", 0)
            completion = session.metadata.get("completion_tokens", 0)
            total = prompt + completion

            def _fmt(n: int) -> str:
                if n >= 1_000_000:
                    return f"{n / 1_000_000:.1f}M"
                if n >= 1_000:
                    return f"{n / 1_000:.1f}k"
                return str(n)

            lines = [
                "📊 本次会话 Token 用量：",
                f"  输入：{_fmt(prompt)}",
                f"  输出：{_fmt(completion)}",
                f"  合计：{_fmt(total)}",
            ]

            # 路由 tier 统计 + 节费估算（仅路由启用时显示）
            if self.router:
                fast = session.metadata.get("route_fast", 0)
                normal = session.metadata.get("route_normal", 0)
                heavy = session.metadata.get("route_heavy", 0)
                total_routed = fast + normal + heavy
                if total_routed > 0:
                    lines.append("")
                    lines.append("🔀 模型路由分布：")
                    lines.append(f"  Fast  ：{fast} 次 ({fast * 100 // total_routed}%)")
                    lines.append(f"  Normal：{normal} 次 ({normal * 100 // total_routed}%)")
                    lines.append(f"  Heavy ：{heavy} 次 ({heavy * 100 // total_routed}%)")
                    # 节费估算：假设无路由时全部走 normal，计算 fast 节省的调用次数
                    # 节省比例 = fast 占总请求的比例（fast 比 normal 便宜）
                    if fast > 0 and total_routed > 0:
                        saved_pct = fast * 100 // total_routed
                        lines.append(f"  💰 路由节省估算：约 {saved_pct}% 的请求用了低价模型")

            # 历史累计统计
            global_stats = self._load_global_stats()
            g_total = global_stats.get("total_tokens", 0)
            if g_total > 0:
                lines.append("")
                lines.append("📈 历史累计 Token 用量：")
                lines.append(f"  合计：{_fmt(g_total)}")
                sessions_count = global_stats.get("sessions", 0)
                if sessions_count:
                    lines.append(f"  会话数：{sessions_count}")

            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content="\n".join(lines),
            )
        await self.memory_consolidator.maybe_consolidate_by_tokens(session)

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        history = session.get_history(max_messages=0)
        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel, chat_id=msg.chat_id,
        )

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
            ))

        # Model routing: select tier based on message complexity
        model_to_use = self.model
        if self.router:
            route = self.router.route(msg.content, len(history))
            model_to_use = route.model
            logger.info(
                "Router: tier={} model={} reason={} msg={!r}",
                route.tier, route.model, route.reason, msg.content[:40],
            )

        current_tier = route.tier if self.router else None
        if self.router and current_tier:
            meta_key = f"route_{current_tier}"
            session.metadata[meta_key] = session.metadata.get(meta_key, 0) + 1
        final_content, _, all_msgs, usage, finish_reason = await self._run_agent_loop(
            initial_messages,
            model_override=model_to_use,
            on_progress=on_progress or _bus_progress,
        )

        # Upgrade to next tier if response was truncated
        if finish_reason == "length" and self.router and current_tier:
            upgraded = self.router.upgrade(current_tier)
            if upgraded:
                logger.warning(
                    "Response truncated (length), upgrading {} -> {} ({})",
                    current_tier, upgraded.tier, upgraded.model,
                )
                final_content, _, all_msgs, usage, _ = await self._run_agent_loop(
                    initial_messages,
                    model_override=upgraded.model,
                    on_progress=on_progress or _bus_progress,
                )

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        self._save_turn(session, all_msgs, 1 + len(history), usage)
        self.sessions.save(session)
        await self.memory_consolidator.maybe_consolidate_by_tokens(session)

        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            return None

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=final_content,
            metadata=msg.metadata or {},
        )

    def _load_global_stats(self) -> dict:
        """Load global cumulative usage stats from workspace."""
        path = self.workspace / "usage_stats.json"
        if not path.exists():
            return {}
        try:
            import json as _json
            return _json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _update_global_stats(self, usage: dict) -> None:
        """Accumulate usage into workspace-level stats file."""
        import json as _json
        path = self.workspace / "usage_stats.json"
        stats = self._load_global_stats()
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            stats[k] = stats.get(k, 0) + usage.get(k, 0)
        # total_tokens may not be provided by all providers
        if "total_tokens" not in usage:
            stats["total_tokens"] = stats.get("total_tokens", 0) + usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        try:
            path.write_text(_json.dumps(stats, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def _save_turn(self, session: Session, messages: list[dict], skip: int, usage: dict | None = None) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        if usage:
            for k, v in usage.items():
                session.metadata[k] = session.metadata.get(k, 0) + v
            self._update_global_stats(usage)
        from datetime import datetime
        session_start = len(session.messages)
        for m in messages[skip:]:
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # skip empty assistant messages — they poison session context
            if role == "tool" and isinstance(content, str) and len(content) > self._TOOL_RESULT_MAX_CHARS:
                entry["content"] = content[:self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
            elif role == "user":
                if isinstance(content, str) and content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                    # Strip the runtime-context prefix, keep only the user text.
                    parts = content.split("\n\n", 1)
                    if len(parts) > 1 and parts[1].strip():
                        entry["content"] = parts[1]
                    else:
                        continue
                if isinstance(content, list):
                    filtered = []
                    for c in content:
                        if c.get("type") == "text" and isinstance(c.get("text"), str) and c["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                            continue  # Strip runtime context from multimodal messages
                        if (c.get("type") == "image_url"
                                and c.get("image_url", {}).get("url", "").startswith("data:image/")):
                            filtered.append({"type": "text", "text": "[image]"})
                        else:
                            filtered.append(c)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()
        # Log the sanitized session entries (not raw messages) to avoid writing
        # unbounded tool outputs or base64 images to the JSONL run log.
        self._run_logger.write_turn(session.key, session.messages[session_start:], usage)

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """Process a message directly (for CLI or cron usage)."""
        await self._connect_mcp()
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        response = await self._process_message(msg, session_key=session_key, on_progress=on_progress)
        return response.content if response else ""
