"""Memory system for persistent agent memory."""

from __future__ import annotations

import asyncio
import json
import weakref
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from nanobot.utils.helpers import ensure_dir, estimate_message_tokens, estimate_prompt_tokens_chain

# ---------------------------------------------------------------------------
# Memory Snapshot (分级加载)
# ---------------------------------------------------------------------------

MEMORY_INLINE_LINES = 30  # 行数阈值：超过此值只注入 # now 节 + 结构摘要


@dataclass
class MemorySnapshot:
    """Snapshot of MEMORY.md with tiered loading based on file size."""
    exists: bool
    full_content: str | None        # ≤30行：完整内容
    first_section: str | None       # >30行：# now 节内容
    outline: list[str] = field(default_factory=list)  # >30行：各节标题
    total_lines: int = 0
    file_path: str = ""


def _extract_section(content: str, heading: str) -> str | None:
    """Extract content of a top-level '# heading' section (until next '# ' heading)."""
    lines = content.splitlines()
    in_section = False
    result = []
    for line in lines:
        if line.strip().lower() == heading.lower():
            in_section = True
            continue
        if in_section and line.startswith("# ") and line.strip().lower() != heading.lower():
            break
        if in_section:
            result.append(line)
    return "\n".join(result).strip() or None


def build_memory_snapshot(memory_file: Path) -> MemorySnapshot:
    """Build a tiered snapshot of MEMORY.md for context injection."""
    if not memory_file.exists():
        return MemorySnapshot(exists=False, full_content=None, first_section=None,
                              file_path=str(memory_file))
    content = memory_file.read_text(encoding="utf-8")
    lines = content.splitlines()
    if len(lines) <= MEMORY_INLINE_LINES:
        return MemorySnapshot(exists=True, full_content=content, first_section=None,
                              total_lines=len(lines), file_path=str(memory_file))
    first_section = _extract_section(content, "# now")
    outline = [l for l in lines if l.startswith("#")][:20]
    return MemorySnapshot(exists=True, full_content=None, first_section=first_section,
                          outline=outline, total_lines=len(lines), file_path=str(memory_file))

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import Session, SessionManager


_SAVE_MEMORY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save the memory consolidation result to persistent storage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "history_entry": {
                        "type": "string",
                        "description": "A paragraph summarizing key events/decisions/topics. "
                        "Start with [YYYY-MM-DD HH:MM]. Include detail useful for grep search.",
                    },
                    "memory_update": {
                        "type": "string",
                        "description": "Full updated long-term memory as markdown. Include all existing "
                        "facts plus new ones. Return unchanged if nothing new.",
                    },
                },
                "required": ["history_entry", "memory_update"],
            },
        },
    }
]


def _ensure_text(value: Any) -> str:
    """Normalize tool-call payload values to text for file storage."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _normalize_save_memory_args(args: Any) -> dict[str, Any] | None:
    """Normalize provider tool-call arguments to the expected dict shape."""
    if isinstance(args, str):
        args = json.loads(args)
    if isinstance(args, list):
        return args[0] if args and isinstance(args[0], dict) else None
    return args if isinstance(args, dict) else None

_TOOL_CHOICE_ERROR_MARKERS = (
    "tool_choice",
    "toolchoice",
    "does not support",
    'should be ["none", "auto"]',
)


def _is_tool_choice_unsupported(content: str | None) -> bool:
    """Detect provider errors caused by forced tool_choice being unsupported."""
    text = (content or "").lower()
    return any(m in text for m in _TOOL_CHOICE_ERROR_MARKERS)


class MemoryStore:
    """Two-layer memory: MEMORY.md (long-term facts) + HISTORY.md (grep-searchable log)."""

    _MAX_FAILURES_BEFORE_RAW_ARCHIVE = 3

    def __init__(self, workspace: Path):
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"
        self._consecutive_failures = 0

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    async def read_long_term_async(self) -> str:
        """Async version of read_long_term — avoids blocking the event loop."""
        return await asyncio.to_thread(self.read_long_term)

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    async def write_long_term_async(self, content: str) -> None:
        """Async version of write_long_term — avoids blocking the event loop."""
        await asyncio.to_thread(self.write_long_term, content)

    def append_history(self, entry: str) -> None:
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

    async def append_history_async(self, entry: str) -> None:
        """Async version of append_history — avoids blocking the event loop."""
        await asyncio.to_thread(self.append_history, entry)

    def get_memory_context(self) -> str:
        snapshot = build_memory_snapshot(self.memory_file)
        if not snapshot.exists:
            return ""
        if snapshot.full_content is not None:
            return f"## Long-term Memory\n{snapshot.full_content}"
        # 大文件：注入 # now 节 + 结构摘要，避免撑满 context
        # 若无 # now 节（legacy 格式），fallback 到全文注入，避免记忆丢失
        if not snapshot.first_section:
            return f"## Long-term Memory\n{snapshot.full_content or self.memory_file.read_text(encoding='utf-8')}"
        parts = []
        parts.append(f"## Memory (# now)\n{snapshot.first_section}")
        if snapshot.outline:
            parts.append(
                f"## Memory Structure ({snapshot.total_lines} lines total — "
                f"read {snapshot.file_path} for full content)\n" +
                "\n".join(snapshot.outline)
            )
        return "\n\n".join(parts)

    @staticmethod
    def _format_messages(messages: list[dict]) -> str:
        lines = []
        for message in messages:
            if not message.get("content"):
                continue
            tools = f" [tools: {', '.join(message['tools_used'])}]" if message.get("tools_used") else ""
            lines.append(
                f"[{message.get('timestamp', '?')[:16]}] {message['role'].upper()}{tools}: {message['content']}"
            )
        return "\n".join(lines)

    async def consolidate(
        self,
        messages: list[dict],
        provider: LLMProvider,
        model: str,
    ) -> bool:
        """Consolidate the provided message chunk into MEMORY.md + HISTORY.md."""
        if not messages:
            return True

        current_memory = await self.read_long_term_async()
        prompt = f"""Process this conversation and call the save_memory tool with your consolidation.

## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation to Process
{self._format_messages(messages)}"""

        chat_messages = [
            {"role": "system", "content": "You are a memory consolidation agent. Call the save_memory tool with your consolidation of the conversation."},
            {"role": "user", "content": prompt},
        ]

        try:
            forced = {"type": "function", "function": {"name": "save_memory"}}
            response = await provider.chat_with_retry(
                messages=chat_messages,
                tools=_SAVE_MEMORY_TOOL,
                model=model,
                tool_choice=forced,
            )

            if response.finish_reason == "error" and _is_tool_choice_unsupported(
                response.content
            ):
                logger.warning("Forced tool_choice unsupported, retrying with auto")
                response = await provider.chat_with_retry(
                    messages=chat_messages,
                    tools=_SAVE_MEMORY_TOOL,
                    model=model,
                    tool_choice="auto",
                )

            if not response.has_tool_calls:
                logger.warning(
                    "Memory consolidation: LLM did not call save_memory "
                    "(finish_reason={}, content_len={}, content_preview={})",
                    response.finish_reason,
                    len(response.content or ""),
                    (response.content or "")[:200],
                )
                return self._fail_or_raw_archive(messages)

            args = _normalize_save_memory_args(response.tool_calls[0].arguments)
            if args is None:
                logger.warning("Memory consolidation: unexpected save_memory arguments")
                return self._fail_or_raw_archive(messages)

            if "history_entry" not in args or "memory_update" not in args:
                logger.warning("Memory consolidation: save_memory payload missing required fields")
                return self._fail_or_raw_archive(messages)

            entry = args["history_entry"]
            update = args["memory_update"]

            if entry is None or update is None:
                logger.warning("Memory consolidation: save_memory payload contains null required fields")
                return self._fail_or_raw_archive(messages)

            entry = _ensure_text(entry).strip()
            if not entry:
                logger.warning("Memory consolidation: history_entry is empty after normalization")
                return self._fail_or_raw_archive(messages)

            await self.append_history_async(entry)
            update = _ensure_text(update)
            if update != current_memory:
                await self.write_long_term_async(update)

            self._consecutive_failures = 0
            logger.info("Memory consolidation done for {} messages", len(messages))
            return True
        except Exception:
            logger.exception("Memory consolidation failed")
            return self._fail_or_raw_archive(messages)

    def _fail_or_raw_archive(self, messages: list[dict]) -> bool:
        """Increment failure count; after threshold, raw-archive messages and return True."""
        self._consecutive_failures += 1
        if self._consecutive_failures < self._MAX_FAILURES_BEFORE_RAW_ARCHIVE:
            return False
        self._raw_archive(messages)
        self._consecutive_failures = 0
        return True

    def _raw_archive(self, messages: list[dict]) -> None:
        """Fallback: dump raw messages to HISTORY.md without LLM summarization."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.append_history(
            f"[{ts}] [RAW] {len(messages)} messages\n"
            f"{self._format_messages(messages)}"
        )
        logger.warning(
            "Memory consolidation degraded: raw-archived {} messages", len(messages)
        )


class MemoryConsolidator:
    """Owns consolidation policy, locking, and session offset updates."""

    _MAX_CONSOLIDATION_ROUNDS = 5

    def __init__(
        self,
        workspace: Path,
        provider: LLMProvider,
        model: str,
        sessions: SessionManager,
        context_window_tokens: int,
        build_messages: Callable[..., list[dict[str, Any]]],
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
    ):
        self.store = MemoryStore(workspace)
        self.provider = provider
        self.model = model
        self.sessions = sessions
        self.context_window_tokens = context_window_tokens
        self._build_messages = build_messages
        self._get_tool_definitions = get_tool_definitions
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()

    def get_lock(self, session_key: str) -> asyncio.Lock:
        """Return the shared consolidation lock for one session."""
        return self._locks.setdefault(session_key, asyncio.Lock())

    async def consolidate_messages(self, messages: list[dict[str, object]]) -> bool:
        """Archive a selected message chunk into persistent memory."""
        return await self.store.consolidate(messages, self.provider, self.model)

    def pick_consolidation_boundary(
        self,
        session: Session,
        tokens_to_remove: int,
    ) -> tuple[int, int] | None:
        """Pick a user-turn boundary that removes enough old prompt tokens."""
        start = session.last_consolidated
        if start >= len(session.messages) or tokens_to_remove <= 0:
            return None

        removed_tokens = 0
        last_boundary: tuple[int, int] | None = None
        for idx in range(start, len(session.messages)):
            message = session.messages[idx]
            if idx > start and message.get("role") == "user":
                last_boundary = (idx, removed_tokens)
                if removed_tokens >= tokens_to_remove:
                    return last_boundary
            removed_tokens += estimate_message_tokens(message)

        return last_boundary

    def estimate_session_prompt_tokens(self, session: Session) -> tuple[int, str]:
        """Estimate current prompt size for the normal session history view."""
        history = session.get_history(max_messages=0)
        channel, chat_id = (session.key.split(":", 1) if ":" in session.key else (None, None))
        probe_messages = self._build_messages(
            history=history,
            current_message="[token-probe]",
            channel=channel,
            chat_id=chat_id,
        )
        return estimate_prompt_tokens_chain(
            self.provider,
            self.model,
            probe_messages,
            self._get_tool_definitions(),
        )

    async def archive_unconsolidated(self, session: Session) -> bool:
        """Archive the full unconsolidated tail for /new-style session rollover."""
        lock = self.get_lock(session.key)
        async with lock:
            snapshot = session.messages[session.last_consolidated:]
            if not snapshot:
                return True
            return await self.consolidate_messages(snapshot)

    async def maybe_consolidate_by_tokens(self, session: Session) -> None:
        """Loop: archive old messages until prompt fits within half the context window."""
        if not session.messages or self.context_window_tokens <= 0:
            return

        lock = self.get_lock(session.key)
        async with lock:
            target = self.context_window_tokens // 2
            estimated, source = self.estimate_session_prompt_tokens(session)
            if estimated <= 0:
                return
            if estimated < self.context_window_tokens:
                logger.debug(
                    "Token consolidation idle {}: {}/{} via {}",
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    source,
                )
                return

            for round_num in range(self._MAX_CONSOLIDATION_ROUNDS):
                if estimated <= target:
                    return

                boundary = self.pick_consolidation_boundary(session, max(1, estimated - target))
                if boundary is None:
                    logger.debug(
                        "Token consolidation: no safe boundary for {} (round {})",
                        session.key,
                        round_num,
                    )
                    return

                end_idx = boundary[0]
                chunk = session.messages[session.last_consolidated:end_idx]
                if not chunk:
                    return

                logger.info(
                    "Token consolidation round {} for {}: {}/{} via {}, chunk={} msgs",
                    round_num,
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    source,
                    len(chunk),
                )
                if not await self.consolidate_messages(chunk):
                    return
                session.last_consolidated = end_idx
                self.sessions.save(session)

                estimated, source = self.estimate_session_prompt_tokens(session)
                if estimated <= 0:
                    return


# ---------------------------------------------------------------------------
# Run Logger (JSONL)
# ---------------------------------------------------------------------------


class RunLogger:
    """Append turn messages to a per-session JSONL file for run replay.

    Files are stored at: {workspace}/runs/{safe_session_key}-{YYYY-MM-DD}.jsonl
    Each line is a JSON object with _ts, _session, messages[], and optional usage{}.
    Write failures are silently ignored to avoid disrupting the main flow.
    """

    def __init__(self, workspace: Path):
        self._runs_dir = ensure_dir(workspace / "runs")

    def get_path(self, session_key: str, date_str: str) -> Path:
        safe_key = session_key.replace(":", "_").replace("/", "_")
        return self._runs_dir / f"{safe_key}-{date_str}.jsonl"

    def write_turn(
        self,
        session_key: str,
        messages: list[dict],
        usage: dict | None = None,
    ) -> None:
        """Append one turn (all new messages) as a single JSONL record.

        The file write is offloaded to a thread-pool executor to avoid
        blocking the asyncio event loop.
        """
        if not messages:
            return
        date_str = datetime.now().strftime("%Y-%m-%d")
        path = self.get_path(session_key, date_str)
        record: dict[str, Any] = {
            "_ts": datetime.now().isoformat(),
            "_session": session_key,
            "messages": messages,
        }
        if usage:
            record["usage"] = usage
        line = json.dumps(record, ensure_ascii=False) + "\n"

        def _write() -> None:
            try:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(line)
            except Exception:
                pass

        try:
            loop = asyncio.get_running_loop()
            loop.run_in_executor(None, _write)
        except RuntimeError:
            # No running loop (e.g. tests) — write synchronously
            _write()
