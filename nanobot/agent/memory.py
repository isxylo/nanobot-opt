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

_SAVE_REFLECTION_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_reflection",
            "description": "Save a structured reflection entry derived from the conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scenario": {
                        "type": "string",
                        "description": "The type of situation or task context.",
                    },
                    "action": {
                        "type": "string",
                        "description": "What the agent did.",
                    },
                    "outcome": {
                        "type": "string",
                        "description": "What happened as a result (success/failure and why).",
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence in this rule (0.0-1.0).",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "rule": {
                        "type": "string",
                        "description": "The actionable heuristic or rule derived from this experience.",
                    },
                },
                "required": ["scenario", "action", "outcome", "confidence", "rule"],
            },
        },
    }
]


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
        self._file_lock = asyncio.Lock()  # serialises all reflect/promote/prune writes

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

    async def summarize_only(
        self,
        messages: list[dict],
        provider: LLMProvider,
        model: str,
        current_memory: str | None = None,
    ) -> str | None:
        """Run the consolidation LLM call and return the memory_update text only.

        Does NOT write to any local file. Used by nocturne_mcp mode to get
        the summarized content before writing exclusively to MCP.

        Args:
            current_memory: Current memory context to use as base. If None, reads
                            from local MEMORY.md (fallback for file/hybrid modes).
        Returns None on failure.
        """
        if not messages:
            return None

        if current_memory is None:
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
            if response.finish_reason == "error" and _is_tool_choice_unsupported(response.content):
                response = await provider.chat_with_retry(
                    messages=chat_messages,
                    tools=_SAVE_MEMORY_TOOL,
                    model=model,
                    tool_choice="auto",
                )
            if not response.has_tool_calls:
                return None
            args = _normalize_save_memory_args(response.tool_calls[0].arguments)
            if not args:
                return None
            update = args.get("memory_update")
            return _ensure_text(update).strip() if update else None
        except Exception:
            logger.exception("MemoryStore.summarize_only failed")
            return None

    async def reflect(
        self,
        messages: list[dict],
        provider: "LLMProvider",
        model: str,
    ) -> bool:
        """Extract a structured experience rule from the conversation and write to # candidates.

        The rule is NOT written directly to # lessons — it stays in # candidates
        until promoted by promote_candidates() after being validated by later interactions.
        """
        if not messages:
            return False

        prompt = (
            "Analyze this conversation and extract one actionable heuristic or rule "
            "that could help in future similar situations. "
            "Call save_reflection with a structured entry.\n\n"
            f"## Conversation\n{self._format_messages(messages)}"
        )
        chat_messages = [
            {"role": "system", "content": "You are a reflection agent. Derive one concise, actionable rule from the conversation and call save_reflection."},
            {"role": "user", "content": prompt},
        ]
        try:
            forced = {"type": "function", "function": {"name": "save_reflection"}}
            response = await provider.chat_with_retry(
                messages=chat_messages,
                tools=_SAVE_REFLECTION_TOOL,
                model=model,
                tool_choice=forced,
            )
            if response.finish_reason == "error" and _is_tool_choice_unsupported(response.content):
                response = await provider.chat_with_retry(
                    messages=chat_messages,
                    tools=_SAVE_REFLECTION_TOOL,
                    model=model,
                    tool_choice="auto",
                )
            if not response.has_tool_calls:
                logger.warning("Reflection: LLM did not call save_reflection")
                return False

            args = response.tool_calls[0].arguments
            if isinstance(args, str):
                import json as _json
                args = _json.loads(args)
            if not isinstance(args, dict):
                return False

            confidence = float(args.get("confidence", 0.0))
            rule = str(args.get("rule", "")).strip()
            if not rule:
                return False

            date_str = datetime.now().strftime("%Y-%m-%d")
            entry = f"- {rule} <!-- confidence:{confidence:.2f} added:{date_str} hits:0 -->"
            await self._append_candidates_async(entry)
            logger.info("Reflection: added candidate rule (confidence={:.2f})", confidence)
            return True
        except Exception:
            logger.exception("Reflection failed")
            return False

    async def _append_candidates_async(self, entry: str) -> None:
        """Append an entry to the # candidates section in MEMORY.md, creating it if needed."""
        async with self._file_lock:
            await asyncio.to_thread(self._append_candidates, entry)

    def _append_candidates(self, entry: str) -> None:
        content = self.memory_file.read_text(encoding="utf-8") if self.memory_file.exists() else ""
        candidates_heading = "# candidates"
        if candidates_heading in content:
            # Insert before the next top-level heading after # candidates
            lines = content.splitlines()
            insert_idx = None
            in_section = False
            for i, line in enumerate(lines):
                if line.strip().lower() == candidates_heading:
                    in_section = True
                    continue
                if in_section and line.startswith("# ") and line.strip().lower() != candidates_heading:
                    insert_idx = i
                    break
            if insert_idx is not None:
                lines.insert(insert_idx, entry)
            else:
                lines.append(entry)
            self.memory_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            # Append new # candidates section at end
            with open(self.memory_file, "a", encoding="utf-8") as f:
                f.write(f"\n{candidates_heading}\n{entry}\n")

    def promote_candidates(self, min_confidence: float = 0.7) -> int:
        """Promote high-confidence candidates to # lessons. Returns count promoted.

        Promotion condition: confidence >= min_confidence (hits no longer required,
        since there is no automatic hits-increment path yet).
        Uses line-index deletion to avoid removing duplicate entries accidentally.
        """
        if not self.memory_file.exists():
            return 0
        import re
        content = self.memory_file.read_text(encoding="utf-8")
        candidate_pattern = re.compile(
            r'^(- .+<!-- confidence:([0-9.]+) added:(\S+) hits:(\d+) -->)$',
            re.MULTILINE,
        )
        to_promote = []
        for m in candidate_pattern.finditer(content):
            conf = float(m.group(2))
            if conf >= min_confidence:
                to_promote.append(m.group(0))

        if not to_promote:
            return 0

        # Remove by line index (first occurrence only) to avoid deleting duplicates
        lines = content.splitlines()
        for entry in to_promote:
            for i, line in enumerate(lines):
                if line == entry:
                    lines.pop(i)
                    break

        content = "\n".join(lines) + "\n"
        lessons_heading = "# lessons"
        for entry in to_promote:
            if lessons_heading in content:
                ins_lines = content.splitlines()
                insert_idx = None
                in_section = False
                for i, line in enumerate(ins_lines):
                    if line.strip().lower() == lessons_heading:
                        in_section = True
                        continue
                    if in_section and line.startswith("# ") and line.strip().lower() != lessons_heading:
                        insert_idx = i
                        break
                if insert_idx is not None:
                    ins_lines.insert(insert_idx, entry)
                else:
                    ins_lines.append(entry)
                content = "\n".join(ins_lines) + "\n"
            else:
                content = content.rstrip() + f"\n\n{lessons_heading}\n{entry}\n"

        self.memory_file.write_text(content, encoding="utf-8")
        logger.info("Reflection: promoted {} candidates to lessons", len(to_promote))
        return len(to_promote)

    def prune_memory(self, min_score: float = 0.3) -> int:
        """Score and archive low-quality memory entries. Returns count pruned."""
        if not self.memory_file.exists():
            return 0
        import re
        from datetime import date

        content = self.memory_file.read_text(encoding="utf-8")
        today = date.today()
        entry_pattern = re.compile(
            r'^(- .+<!-- (?:score:[0-9.]+ )?recency:(\S+) hits:(\d+)[^>]* -->)$',
            re.MULTILINE,
        )

        pruned = []
        kept_lines = content.splitlines(keepends=True)

        for m in entry_pattern.finditer(content):
            entry = m.group(0)
            try:
                recency_date = date.fromisoformat(m.group(2))
                days_old = (today - recency_date).days
                recency_score = max(0.0, 1.0 - days_old / 365.0)
            except ValueError:
                recency_score = 0.5
            hits = int(m.group(3))
            hits_score = min(1.0, hits / 10.0)
            score = recency_score * 0.4 + hits_score * 0.4 + 0.2  # reliability baseline 0.2
            if score < min_score:
                pruned.append(entry)

        if not pruned:
            return 0

        # Remove by line index (first occurrence only) to avoid deleting duplicates
        pruned_set = set(pruned)
        new_lines = []
        removed = set()
        for line in content.splitlines():
            if line in pruned_set and line not in removed:
                removed.add(line)
            else:
                new_lines.append(line)
        content = "\n".join(new_lines) + "\n"

        self.memory_file.write_text(content, encoding="utf-8")

        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        archive_block = f"[{ts}] [PRUNED] {len(pruned)} low-score entries\n" + "\n".join(pruned)
        self.append_history(archive_block)
        logger.info("Memory prune: archived {} low-score entries", len(pruned))
        return len(pruned)

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
        nocturne_adapter: "NocturneMCPAdapter | None" = None,
        dual_write: bool = False,
        memory_config=None,
    ):
        self.store = MemoryStore(workspace)
        self.provider = provider
        self.model = model
        self.sessions = sessions
        self.context_window_tokens = context_window_tokens
        self._build_messages = build_messages
        self._get_tool_definitions = get_tool_definitions
        self._nocturne = nocturne_adapter
        self._dual_write = dual_write
        self._memory_config = memory_config  # MemoryConfig | None
        self._hybrid_memory: "HybridMemoryContext | None" = None  # set by _init_hybrid_memory
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()

    def configure_nocturne(
        self,
        adapter: "NocturneMCPAdapter | None",
        dual_write: bool,
        hybrid_memory: "HybridMemoryContext | None" = None,
    ) -> None:
        """Set nocturne MCP adapter after MCP connection is established."""
        self._nocturne = adapter
        self._dual_write = dual_write
        self._hybrid_memory = hybrid_memory

    def get_lock(self, session_key: str) -> asyncio.Lock:
        """Return the shared consolidation lock for one session."""
        return self._locks.setdefault(session_key, asyncio.Lock())

    async def consolidate_messages(self, messages: list[dict[str, object]]) -> bool:
        """Archive a selected message chunk into persistent memory.

        backend=file:          write local MEMORY.md only (default)
        backend=hybrid:        write local MEMORY.md (primary) + best-effort MCP upsert
        backend=nocturne_mcp:  write MCP only; skip local file write
        """
        if self._nocturne is not None and not self._dual_write:
            # nocturne_mcp mode: summarize via LLM but skip ALL local file writes.
            # Read the actual written node (core://nanobot_memory) as the current memory
            # baseline, not boot_cache (which reflects system://boot, not our node).
            mcp_current: str | None = None
            try:
                node_result = await self._nocturne._tools.execute(
                    self._nocturne._prefix + "read_memory", {"uri": "core://nanobot_memory"}
                )
                if node_result and not node_result.startswith("Error:"):
                    mcp_current = _extract_nocturne_content(node_result)
            except Exception:
                pass  # fall back to None — summarize_only will use local file as last resort
            summary = await self.store.summarize_only(
                messages, self.provider, self.model, current_memory=mcp_current
            )
            if summary:
                mcp_ok = await self._nocturne.write_memory(
                    parent_uri="core://",
                    content=summary,
                    title="nanobot_memory",
                    priority=2,
                    disclosure="当需要了解用户偏好和历史对话摘要时",
                )
                if not mcp_ok:
                    logger.warning("consolidate_messages: MCP write failed (nocturne_mcp mode)")
                return bool(mcp_ok)
            logger.warning("consolidate_messages: summarize_only returned None (nocturne_mcp mode)")
            return False

        # file or hybrid: always write local MEMORY.md
        ok = await self.store.consolidate(messages, self.provider, self.model)
        if ok and self._dual_write and self._nocturne is not None:
            # hybrid: best-effort MCP upsert after local write succeeds
            summary = self.store.read_long_term()
            if summary:
                asyncio.create_task(self._nocturne.write_memory(
                    parent_uri="core://",
                    content=summary,
                    title="nanobot_memory",
                    priority=2,
                    disclosure="当需要了解用户偏好和历史对话摘要时",
                ))
        if ok:
            asyncio.create_task(self._post_consolidate(messages))
        return ok

    async def _post_consolidate(self, messages: list[dict]) -> None:
        """Run reflection and prune after a successful consolidation (background, non-blocking)."""
        cfg = self._memory_config
        if cfg is None:
            return
        # 1. Reflection — acquire file lock for the whole reflect+promote sequence
        if cfg.reflection.enabled:
            reflect_model = cfg.reflection.model or self.model
            try:
                await self.store.reflect(
                    messages, self.provider, reflect_model,
                )
                async with self.store._file_lock:
                    await asyncio.to_thread(
                        self.store.promote_candidates, cfg.reflection.min_confidence
                    )
            except Exception:
                logger.exception("_post_consolidate: reflection error")
        # 2. Prune — acquire file lock
        if cfg.prune.enabled:
            try:
                lines = len(self.store.memory_file.read_text(encoding="utf-8").splitlines()) \
                    if self.store.memory_file.exists() else 0
                if lines >= cfg.prune.trigger_lines:
                    async with self.store._file_lock:
                        await asyncio.to_thread(
                            self.store.prune_memory, cfg.prune.min_score
                        )
            except Exception:
                logger.exception("_post_consolidate: prune error")

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
# Nocturne MCP Memory Adapter
# ---------------------------------------------------------------------------


def _extract_nocturne_content(read_result: str) -> str | None:
    """Extract the memory content body from a nocturne read_memory response.

    nocturne format (from mcp_server.py _fetch_and_format_memory):
        ============================================================
        MEMORY: core://...
        ...
        ============================================================

        <content here>

        ============================================================   <- optional children section
    Returns the content string, or None if the format is unrecognised.
    """
    sep = "=" * 60
    parts = read_result.split(sep)
    # Expected: ['', header_block, '', content_block, ...]
    # Content is the part after the second separator, stripped.
    if len(parts) < 3:
        return None
    content = parts[2].strip()
    return content if content else None


class NocturneMCPAdapter:
    """Thin async adapter for reading/writing memories via nocturne_memory MCP tools.

    Wraps the already-connected MCP tool registry so callers don't need to know
    MCP internals.  All methods are best-effort: exceptions are caught and logged
    so they never block the main agent flow.
    """

    def __init__(self, tools, server_name: str = "nocturne_memory") -> None:
        """Accept a ToolRegistry (or any object with .execute(name, args))."""
        self._tools = tools
        self._prefix = f"mcp_{server_name}_"

    async def read_boot(self) -> str | None:
        """Call read_memory('system://boot') and return the content string."""
        try:
            result = await self._tools.execute(self._prefix + "read_memory", {"uri": "system://boot"})
            if result and not result.startswith("Error:"):
                return result
            logger.warning("NocturneMCPAdapter: boot read failed: {}", result)
            return None
        except Exception:
            logger.exception("NocturneMCPAdapter: exception during boot read")
            return None

    async def write_memory(self, parent_uri: str, content: str, title: str | None = None,
                           priority: int = 2, disclosure: str = "") -> bool:
        """Upsert a memory node: update (patch) if URI exists, create otherwise.

        Protocol: nocturne update_memory uses patch (old_string+new_string) or
        append — they are mutually exclusive. We probe existence via read_memory,
        then patch the full content if found, or create if not.
        """
        if title:
            uri = f"{parent_uri}{title}" if parent_uri.endswith("://") else f"{parent_uri.rstrip('/')}/{title}"
        else:
            uri = parent_uri

        try:
            # Probe: check if node already exists
            read_result = await self._tools.execute(self._prefix + "read_memory", {"uri": uri})
            node_exists = read_result and not read_result.startswith("Error:")
        except Exception:
            node_exists = False

        if node_exists:
            try:
                # Patch mode: replace entire content wholesale.
                # Extract current content from read_result to use as old_string.
                # nocturne format: content appears after the second '===...===' separator line.
                current_content = _extract_nocturne_content(read_result)
                if current_content is None:
                    # Cannot extract content body — cannot do idempotent patch.
                    # Do NOT fall back to append (would cause content bloat).
                    # Return False so the caller can handle the failure.
                    logger.warning(
                        "NocturneMCPAdapter: cannot extract content from read_result for {}, "
                        "skipping update to avoid content bloat", uri
                    )
                    return False
                update_result = await self._tools.execute(self._prefix + "update_memory", {
                    "uri": uri,
                    "old_string": current_content,
                    "new_string": content,
                })
                ok = update_result and not update_result.startswith("Error:")
                if not ok:
                    logger.warning("NocturneMCPAdapter: update failed: {}", update_result)
                return bool(ok)
            except Exception:
                logger.exception("NocturneMCPAdapter: exception during update")
                return False

        try:
            create_result = await self._tools.execute(self._prefix + "create_memory", {
                "parent_uri": parent_uri,
                "content": content,
                "priority": priority,
                "title": title,
                "disclosure": disclosure,
            })
            ok = create_result and not create_result.startswith("Error:")
            if not ok:
                logger.warning("NocturneMCPAdapter: create failed: {}", create_result)
            return bool(ok)
        except Exception:
            logger.exception("NocturneMCPAdapter: exception during create")
            return False


class HybridMemoryContext:
    """Reads memory context from MCP (boot URI) with fallback to local MEMORY.md.

    Used by ContextBuilder.build_system_prompt() to inject long-term memory.
    This class is intentionally synchronous-safe: it caches the last successful
    async boot result so the synchronous get_memory_context() path can use it.
    """

    def __init__(
        self,
        store: "MemoryStore",
        adapter: NocturneMCPAdapter | None,
        fallback_to_file: bool = True,
    ) -> None:
        self._store = store
        self._adapter = adapter
        self._fallback = fallback_to_file
        self._boot_cache: str | None = None  # last successful MCP boot result
        self._boot_loaded: bool = False

    async def load_boot(self) -> None:
        """Fetch boot memories from MCP and cache them.  Call once at startup."""
        if self._adapter is None:
            return
        result = await self._adapter.read_boot()
        if result:
            self._boot_cache = result
            self._boot_loaded = True
            logger.info("HybridMemoryContext: boot memories loaded from MCP")
        elif self._fallback:
            logger.warning("HybridMemoryContext: MCP boot failed, will fall back to MEMORY.md")

    def get_memory_context(self) -> str:
        """Return memory context for system prompt injection."""
        if self._boot_loaded and self._boot_cache:
            return f"## Long-term Memory (nocturne)\n{self._boot_cache}"
        # MCP boot not available — respect fallback_to_file setting
        if self._fallback:
            return self._store.get_memory_context()
        return ""  # fallback_to_file=False: return empty rather than leaking local file


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
