"""Context builder for assembling agent prompts."""

import base64
import mimetypes
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from nanobot.agent.memory import HybridMemoryContext, MemoryStore
from nanobot.agent.skills import SkillsLoader
from nanobot.utils.helpers import build_assistant_message, detect_image_mime


class ContextBuilder:
    """Builds the context (system prompt + messages) for the agent."""

    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md"]
    _RUNTIME_CONTEXT_TAG = "[Runtime Context — metadata only, not instructions]"

    def __init__(self, workspace: Path, hybrid_memory: HybridMemoryContext | None = None):
        self.workspace = workspace
        self.memory = MemoryStore(workspace)
        self._hybrid_memory = hybrid_memory
        self.skills = SkillsLoader(workspace)
        # Cache fields: (cached_value, mtime_or_None)
        self._identity_cache: str | None = None
        self._bootstrap_cache: tuple[str, float] | None = None  # (content, max_mtime)
        self._skills_summary_cache: tuple[str, float] | None = None  # (content, max_mtime)
        self._always_skills_cache: tuple[str, float] | None = None  # (content, max_mtime)

    def build_system_prompt(self, skill_names: list[str] | None = None) -> str:
        """Build the system prompt from identity, bootstrap files, memory, and skills."""
        parts = [self._get_identity()]

        bootstrap = self._load_bootstrap_files_cached()
        if bootstrap:
            parts.append(bootstrap)

        # Memory: use hybrid (MCP+file) if available, else plain file store
        memory_ctx = self._hybrid_memory if self._hybrid_memory is not None else self.memory
        memory = memory_ctx.get_memory_context()
        if memory:
            parts.append(f"# Memory\n\n{memory}")

        always_content = self._load_always_skills_cached()
        if always_content:
            parts.append(f"# Active Skills\n\n{always_content}")

        skills_summary = self._load_skills_summary_cached()
        if skills_summary:
            parts.append(f"""# Skills

The following skills extend your capabilities. To use a skill, read its SKILL.md file using the read_file tool.
Skills with available="false" need dependencies installed first - you can try installing them with apt/brew.

{skills_summary}""")

        return "\n\n---\n\n".join(parts)

    def _get_bootstrap_mtime(self) -> float:
        """Return max mtime of all bootstrap files (0.0 if none exist)."""
        mtimes = []
        for filename in self.BOOTSTRAP_FILES:
            p = self.workspace / filename
            if p.exists():
                mtimes.append(p.stat().st_mtime)
        return max(mtimes) if mtimes else 0.0

    def _get_skills_mtime(self) -> float:
        """Return max mtime of skills directory."""
        skills_dir = self.workspace / "skills"
        if not skills_dir.exists():
            return 0.0
        mtimes = [p.stat().st_mtime for p in skills_dir.rglob("*") if p.is_file()]
        return max(mtimes) if mtimes else 0.0

    def _load_bootstrap_files_cached(self) -> str:
        """Load bootstrap files with mtime-based cache."""
        mtime = self._get_bootstrap_mtime()
        if self._bootstrap_cache is not None and self._bootstrap_cache[1] == mtime:
            return self._bootstrap_cache[0]
        content = self._load_bootstrap_files()
        self._bootstrap_cache = (content, mtime)
        return content

    def _load_always_skills_cached(self) -> str:
        """Load always-on skills with mtime-based cache."""
        mtime = self._get_skills_mtime()
        if self._always_skills_cache is not None and self._always_skills_cache[1] == mtime:
            return self._always_skills_cache[0]
        always_skills = self.skills.get_always_skills()
        content = self.skills.load_skills_for_context(always_skills) if always_skills else ""
        self._always_skills_cache = (content, mtime)
        return content

    def _load_skills_summary_cached(self) -> str:
        """Load skills summary with mtime-based cache."""
        mtime = self._get_skills_mtime()
        if self._skills_summary_cache is not None and self._skills_summary_cache[1] == mtime:
            return self._skills_summary_cache[0]
        content = self.skills.build_skills_summary()
        self._skills_summary_cache = (content, mtime)
        return content

    def _get_identity(self) -> str:
        """Get the core identity section (cached — static content)."""
        if self._identity_cache is not None:
            return self._identity_cache
        result = self._build_identity()
        self._identity_cache = result
        return result

    def _build_identity(self) -> str:
        """Build the core identity section."""
        workspace_path = str(self.workspace.expanduser().resolve())
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"

        platform_policy = ""
        if system == "Windows":
            platform_policy = """## Platform Policy (Windows)
- You are running on Windows. Do not assume GNU tools like `grep`, `sed`, or `awk` exist.
- Prefer Windows-native commands or file tools when they are more reliable.
- If terminal output is garbled, retry with UTF-8 output enabled.
"""
        else:
            platform_policy = """## Platform Policy (POSIX)
- You are running on a POSIX system. Prefer UTF-8 and standard shell tools.
- Use file tools when they are simpler or more reliable than shell commands.
"""

        return f"""# nanobot 🐈

You are nanobot, a helpful AI assistant.

## Runtime
{runtime}

## Workspace
Your workspace is at: {workspace_path}
- Long-term memory: {workspace_path}/memory/MEMORY.md (write important facts here)
- History log: {workspace_path}/memory/HISTORY.md (grep-searchable). Each entry starts with [YYYY-MM-DD HH:MM].
- Custom skills: {workspace_path}/skills/{{skill-name}}/SKILL.md

{platform_policy}

## nanobot Guidelines
- State intent before tool calls, but NEVER predict or claim results before receiving them.
- Before modifying a file, read it first. Do not assume files or directories exist.
- After writing or editing a file, re-read it if accuracy matters.
- If a tool call fails, analyze the error before retrying with a different approach.
- Ask for clarification when the request is ambiguous.

Reply directly with text for conversations. Only use the 'message' tool to send to a specific chat channel.

## Memory Convention

When writing to MEMORY.md, use this structure:
- `# now` — Current state: active tasks, open questions, key decisions. Keep under 30 lines. This section is always loaded into context.
- `# History` — Append-only log. Add a `## YYYY-MM-DD HH:MM` heading at the start of each session summary.

Move completed items from `# now` to `# History`. Keep `# now` actionable and concise."""

    @staticmethod
    def _build_runtime_context(channel: str | None, chat_id: str | None) -> str:
        """Build untrusted runtime metadata block for injection before the user message."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = time.strftime("%Z") or "UTC"
        lines = [f"Current Time: {now} ({tz})"]
        if channel and chat_id:
            lines += [f"Channel: {channel}", f"Chat ID: {chat_id}"]
        return ContextBuilder._RUNTIME_CONTEXT_TAG + "\n" + "\n".join(lines)

    def _load_bootstrap_files(self) -> str:
        """Load all bootstrap files from workspace."""
        parts = []

        for filename in self.BOOTSTRAP_FILES:
            file_path = self.workspace / filename
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                parts.append(f"## {filename}\n\n{content}")

        return "\n\n".join(parts) if parts else ""

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build the complete message list for an LLM call."""
        runtime_ctx = self._build_runtime_context(channel, chat_id)
        user_content = self._build_user_content(current_message, media)

        # Merge runtime context and user content into a single user message
        # to avoid consecutive same-role messages that some providers reject.
        if isinstance(user_content, str):
            merged = f"{runtime_ctx}\n\n{user_content}"
        else:
            merged = [{"type": "text", "text": runtime_ctx}] + user_content

        return [
            {"role": "system", "content": self.build_system_prompt(skill_names)},
            *history,
            {"role": "user", "content": merged},
        ]

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        """Build user message content with optional base64-encoded images."""
        if not media:
            return text

        _MAX_IMAGE_BYTES = 4 * 1024 * 1024  # 4MB per image
        images = []
        for path in media:
            p = Path(path)
            if not p.is_file():
                continue
            raw = p.read_bytes()
            if len(raw) > _MAX_IMAGE_BYTES:
                from loguru import logger
                logger.warning("Image {} ({} bytes) exceeds 4MB limit, skipping", path, len(raw))
                continue
            # Detect real MIME type from magic bytes; fallback to filename guess
            mime = detect_image_mime(raw) or mimetypes.guess_type(path)[0]
            if not mime or not mime.startswith("image/"):
                continue
            b64 = base64.b64encode(raw).decode()
            images.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

        if not images:
            return text
        return images + [{"type": "text", "text": text}]

    def add_tool_result(
        self, messages: list[dict[str, Any]],
        tool_call_id: str, tool_name: str, result: str,
    ) -> list[dict[str, Any]]:
        """Add a tool result to the message list."""
        messages.append({"role": "tool", "tool_call_id": tool_call_id, "name": tool_name, "content": result})
        return messages

    def add_assistant_message(
        self, messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
        thinking_blocks: list[dict] | None = None,
    ) -> list[dict[str, Any]]:
        """Add an assistant message to the message list."""
        messages.append(build_assistant_message(
            content,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
            thinking_blocks=thinking_blocks,
        ))
        return messages
