"""Skills loader for agent capabilities."""

import json
import math
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

# Default builtin skills directory (relative to this file)
BUILTIN_SKILLS_DIR = Path(__file__).parent.parent / "skills"


class SkillStats:
    """Tracks skill usage frequency and success rate in skills/SKILL_STATS.json."""

    def __init__(self, workspace: Path) -> None:
        self._path = workspace / "skills" / "SKILL_STATS.json"
        self._data: dict = self._load()
        self._lock: "asyncio.Lock | None" = None  # lazy-init to avoid event loop issues

    def _get_lock(self) -> "asyncio.Lock":
        import asyncio
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _load(self) -> dict:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_atomic(self) -> None:
        """Write to a temp file then rename for atomic update."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._path)

    async def record_async(self, skill_name: str) -> None:
        """Record one successful invocation of a skill (async, non-blocking)."""
        import asyncio
        async with self._get_lock():
            entry = self._data.setdefault(skill_name, {"calls": 0, "success": 0, "last_used": ""})
            entry["calls"] += 1
            entry["success"] += 1
            entry["last_used"] = datetime.now().strftime("%Y-%m-%d")
            await asyncio.to_thread(self._save_atomic)

    def record(self, skill_name: str, success: bool) -> None:
        """Sync record — for use in non-async contexts only (tests, CLI)."""
        entry = self._data.setdefault(skill_name, {"calls": 0, "success": 0, "last_used": ""})
        entry["calls"] += 1
        if success:
            entry["success"] += 1
        entry["last_used"] = datetime.now().strftime("%Y-%m-%d")
        self._save_atomic()

    def success_rate(self, skill_name: str) -> float:
        entry = self._data.get(skill_name)
        if not entry or entry["calls"] == 0:
            return 0.0
        return entry["success"] / entry["calls"]

    def sorted_by_priority(self, skill_names: list[str]) -> list[str]:
        """Sort skills by composite score: success_rate * log(calls+1), desc."""
        def score(name: str) -> float:
            entry = self._data.get(name)
            if not entry:
                return 0.0
            return self.success_rate(name) * math.log(entry["calls"] + 1)
        return sorted(skill_names, key=score, reverse=True)


class SkillsLoader:
    """
    Loader for agent skills.

    Skills are markdown files (SKILL.md) that teach the agent how to use
    specific tools or perform certain tasks.
    """

    def __init__(self, workspace: Path, builtin_skills_dir: Path | None = None):
        self.workspace = workspace
        self.workspace_skills = workspace / "skills"
        self.builtin_skills = builtin_skills_dir or BUILTIN_SKILLS_DIR

    def list_skills(self, filter_unavailable: bool = True) -> list[dict[str, str]]:
        """
        List all available skills.

        Args:
            filter_unavailable: If True, filter out skills with unmet requirements.

        Returns:
            List of skill info dicts with 'name', 'path', 'source'.
        """
        skills = []

        # Workspace skills (highest priority)
        if self.workspace_skills.exists():
            for skill_dir in self.workspace_skills.iterdir():
                if skill_dir.is_dir():
                    skill_file = skill_dir / "SKILL.md"
                    if skill_file.exists():
                        skills.append({"name": skill_dir.name, "path": str(skill_file), "source": "workspace"})

        # Built-in skills
        if self.builtin_skills and self.builtin_skills.exists():
            for skill_dir in self.builtin_skills.iterdir():
                if skill_dir.is_dir():
                    skill_file = skill_dir / "SKILL.md"
                    if skill_file.exists() and not any(s["name"] == skill_dir.name for s in skills):
                        skills.append({"name": skill_dir.name, "path": str(skill_file), "source": "builtin"})

        # Filter by requirements
        if filter_unavailable:
            return [s for s in skills if self._check_requirements(self._get_skill_meta(s["name"]))]
        return skills

    def load_skill(self, name: str) -> str | None:
        """
        Load a skill by name.

        Args:
            name: Skill name (directory name).

        Returns:
            Skill content or None if not found.
        """
        # Check workspace first
        workspace_skill = self.workspace_skills / name / "SKILL.md"
        if workspace_skill.exists():
            return workspace_skill.read_text(encoding="utf-8")

        # Check built-in
        if self.builtin_skills:
            builtin_skill = self.builtin_skills / name / "SKILL.md"
            if builtin_skill.exists():
                return builtin_skill.read_text(encoding="utf-8")

        return None

    def load_skills_for_context(self, skill_names: list[str]) -> str:
        """
        Load specific skills for inclusion in agent context.

        Args:
            skill_names: List of skill names to load.

        Returns:
            Formatted skills content.
        """
        parts = []
        for name in skill_names:
            content = self.load_skill(name)
            if content:
                content = self._strip_frontmatter(content)
                parts.append(f"### Skill: {name}\n\n{content}")

        return "\n\n---\n\n".join(parts) if parts else ""

    def build_skills_summary(self, skill_stats: "SkillStats | None" = None) -> str:
        """
        Build a summary of all skills (name, description, path, availability).

        This is used for progressive loading - the agent can read the full
        skill content using read_file when needed.

        Args:
            skill_stats: Optional SkillStats to sort high-priority skills first.

        Returns:
            XML-formatted skills summary.
        """
        all_skills = self.list_skills(filter_unavailable=False)
        if not all_skills:
            return ""

        if skill_stats:
            skill_names = [s["name"] for s in all_skills]
            ordered = skill_stats.sorted_by_priority(skill_names)
            skill_map = {s["name"]: s for s in all_skills}
            all_skills = [skill_map[n] for n in ordered]

        def escape_xml(s: str) -> str:
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        lines = ["<skills>"]
        for s in all_skills:
            name = escape_xml(s["name"])
            path = s["path"]
            desc = escape_xml(self._get_skill_description(s["name"]))
            skill_meta = self._get_skill_meta(s["name"])
            available = self._check_requirements(skill_meta)

            lines.append(f"  <skill available=\"{str(available).lower()}\">")
            lines.append(f"    <name>{name}</name>")
            lines.append(f"    <description>{desc}</description>")
            lines.append(f"    <location>{path}</location>")

            # Show missing requirements for unavailable skills
            if not available:
                missing = self._get_missing_requirements(skill_meta)
                if missing:
                    lines.append(f"    <requires>{escape_xml(missing)}</requires>")

            lines.append("  </skill>")
        lines.append("</skills>")

        return "\n".join(lines)

    def _get_missing_requirements(self, skill_meta: dict) -> str:
        """Get a description of missing requirements."""
        missing = []
        requires = skill_meta.get("requires", {})
        for b in requires.get("bins", []):
            if not shutil.which(b):
                missing.append(f"CLI: {b}")
        for env in requires.get("env", []):
            if not os.environ.get(env):
                missing.append(f"ENV: {env}")
        return ", ".join(missing)

    def _get_skill_description(self, name: str) -> str:
        """Get the description of a skill from its frontmatter."""
        meta = self.get_skill_metadata(name)
        if meta and meta.get("description"):
            return meta["description"]
        return name  # Fallback to skill name

    def _strip_frontmatter(self, content: str) -> str:
        """Remove YAML frontmatter from markdown content."""
        if content.startswith("---"):
            match = re.match(r"^---\n.*?\n---\n", content, re.DOTALL)
            if match:
                return content[match.end():].strip()
        return content

    def _parse_nanobot_metadata(self, raw: str) -> dict:
        """Parse skill metadata JSON from frontmatter (supports nanobot and openclaw keys)."""
        try:
            data = json.loads(raw)
            return data.get("nanobot", data.get("openclaw", {})) if isinstance(data, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}

    def _check_requirements(self, skill_meta: dict) -> bool:
        """Check if skill requirements are met (bins, env vars)."""
        requires = skill_meta.get("requires", {})
        for b in requires.get("bins", []):
            if not shutil.which(b):
                return False
        for env in requires.get("env", []):
            if not os.environ.get(env):
                return False
        return True

    def _get_skill_meta(self, name: str) -> dict:
        """Get nanobot metadata for a skill (cached in frontmatter)."""
        meta = self.get_skill_metadata(name) or {}
        return self._parse_nanobot_metadata(meta.get("metadata", ""))

    def get_always_skills(self) -> list[str]:
        """Get skills marked as always=true that meet requirements."""
        result = []
        for s in self.list_skills(filter_unavailable=True):
            meta = self.get_skill_metadata(s["name"]) or {}
            skill_meta = self._parse_nanobot_metadata(meta.get("metadata", ""))
            if skill_meta.get("always") or meta.get("always"):
                result.append(s["name"])
        return result

    def get_skill_metadata(self, name: str) -> dict | None:
        """
        Get metadata from a skill's frontmatter.

        Args:
            name: Skill name.

        Returns:
            Metadata dict or None.
        """
        content = self.load_skill(name)
        if not content:
            return None

        if content.startswith("---"):
            match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
            if match:
                # Simple YAML parsing
                metadata = {}
                for line in match.group(1).split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        metadata[key.strip()] = value.strip().strip('"\'')
                return metadata

        return None


    def list_draft_skills(self) -> list[dict[str, str]]:
        """List skills in the .drafts/ directory (not injected into context)."""
        drafts_dir = self.workspace_skills / ".drafts"
        if not drafts_dir.exists():
            return []
        result = []
        for skill_dir in drafts_dir.iterdir():
            if skill_dir.is_dir():
                skill_file = skill_dir / "SKILL.md"
                if skill_file.exists():
                    result.append({"name": skill_dir.name, "path": str(skill_file), "source": "draft"})
        return result


class SkillWriter:
    """Creates and promotes auto-generated skill drafts."""

    _DRAFT_TEMPLATE = """---
name: {name}
description: {description}
autogenerated: true
version: draft
uses: 0
---
# {name}

{description}

## Example Commands

{examples}

## Notes

This skill was auto-generated from repeated usage patterns.
Review and edit before promoting to production.
"""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self._drafts_dir = workspace / "skills" / ".drafts"

    def write_draft(self, name: str, description: str, example_commands: list[str]) -> Path:
        """Write a new draft skill. Returns the path of the created SKILL.md."""
        draft_dir = self._drafts_dir / name
        draft_dir.mkdir(parents=True, exist_ok=True)
        skill_file = draft_dir / "SKILL.md"
        examples = "\n".join(f"- `{cmd}`" for cmd in example_commands[:5])
        content = self._DRAFT_TEMPLATE.format(
            name=name,
            description=description,
            examples=examples or "(none recorded)",
        )
        skill_file.write_text(content, encoding="utf-8")
        return skill_file

    def increment_uses(self, name: str) -> int:
        """Increment the use counter for a draft skill. Returns new count."""
        import re
        skill_file = self._drafts_dir / name / "SKILL.md"
        if not skill_file.exists():
            return 0
        content = skill_file.read_text(encoding="utf-8")
        match = re.search(r'^uses: (\d+)$', content, re.MULTILINE)
        if match:
            new_count = int(match.group(1)) + 1
            content = content[:match.start()] + f"uses: {new_count}" + content[match.end():]
            skill_file.write_text(content, encoding="utf-8")
            return new_count
        return 0

    def get_uses(self, name: str) -> int:
        """Return the current use count for a draft skill."""
        import re
        skill_file = self._drafts_dir / name / "SKILL.md"
        if not skill_file.exists():
            return 0
        content = skill_file.read_text(encoding="utf-8")
        match = re.search(r'^uses: (\d+)$', content, re.MULTILINE)
        return int(match.group(1)) if match else 0

    def promote(self, name: str) -> Path:
        """Promote a draft skill to the main skills directory."""
        import re
        import shutil
        draft_dir = self._drafts_dir / name
        target_dir = self.workspace / "skills" / name
        if not draft_dir.exists():
            raise FileNotFoundError(f"Draft skill '{name}' not found")
        if target_dir.exists():
            raise FileExistsError(f"Skill '{name}' already exists in skills/")
        shutil.copytree(draft_dir, target_dir)
        skill_file = target_dir / "SKILL.md"
        content = skill_file.read_text(encoding="utf-8")
        content = re.sub(r'^version: draft$', 'version: 1.0', content, flags=re.MULTILINE)
        skill_file.write_text(content, encoding="utf-8")
        shutil.rmtree(draft_dir)
        return skill_file

    def draft_exists(self, name: str) -> bool:
        return (self._drafts_dir / name / "SKILL.md").exists()
