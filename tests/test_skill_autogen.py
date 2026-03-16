"""Tests for P3 SkillWriter and draft skill lifecycle."""

from __future__ import annotations

from pathlib import Path

import pytest

from nanobot.agent.skills import SkillWriter, SkillsLoader


def test_write_draft_creates_file(tmp_path: Path) -> None:
    writer = SkillWriter(tmp_path)
    path = writer.write_draft("git-helper", "Helps with git commands", ["git status", "git log"])
    assert path.exists()
    content = path.read_text()
    assert "git-helper" in content
    assert "git status" in content
    assert "version: draft" in content
    assert "uses: 0" in content


def test_write_draft_stored_in_drafts_dir(tmp_path: Path) -> None:
    writer = SkillWriter(tmp_path)
    writer.write_draft("test-skill", "A test skill", [])
    assert (tmp_path / "skills" / ".drafts" / "test-skill" / "SKILL.md").exists()


def test_draft_exists(tmp_path: Path) -> None:
    writer = SkillWriter(tmp_path)
    assert not writer.draft_exists("missing")
    writer.write_draft("present", "desc", [])
    assert writer.draft_exists("present")


def test_increment_uses(tmp_path: Path) -> None:
    writer = SkillWriter(tmp_path)
    writer.write_draft("skill", "desc", [])
    assert writer.get_uses("skill") == 0
    assert writer.increment_uses("skill") == 1
    assert writer.increment_uses("skill") == 2
    assert writer.get_uses("skill") == 2


def test_increment_uses_missing_skill(tmp_path: Path) -> None:
    writer = SkillWriter(tmp_path)
    assert writer.increment_uses("nonexistent") == 0


def test_promote_moves_to_skills_dir(tmp_path: Path) -> None:
    writer = SkillWriter(tmp_path)
    writer.write_draft("my-skill", "desc", ["cmd"])
    promoted = writer.promote("my-skill")
    assert promoted.exists()
    assert promoted.parent == tmp_path / "skills" / "my-skill"
    assert not writer.draft_exists("my-skill")


def test_promote_updates_version(tmp_path: Path) -> None:
    writer = SkillWriter(tmp_path)
    writer.write_draft("versioned", "desc", [])
    writer.promote("versioned")
    content = (tmp_path / "skills" / "versioned" / "SKILL.md").read_text()
    assert "version: 1.0" in content
    assert "version: draft" not in content


def test_promote_missing_raises(tmp_path: Path) -> None:
    writer = SkillWriter(tmp_path)
    with pytest.raises(FileNotFoundError):
        writer.promote("no-such-skill")


def test_promote_duplicate_raises(tmp_path: Path) -> None:
    writer = SkillWriter(tmp_path)
    writer.write_draft("dup", "desc", [])
    writer.promote("dup")
    writer.write_draft("dup", "desc2", [])
    with pytest.raises(FileExistsError):
        writer.promote("dup")


def test_list_draft_skills(tmp_path: Path) -> None:
    writer = SkillWriter(tmp_path)
    writer.write_draft("alpha", "desc", [])
    writer.write_draft("beta", "desc", [])
    loader = SkillsLoader(tmp_path, builtin_skills_dir=None)
    drafts = loader.list_draft_skills()
    names = {d["name"] for d in drafts}
    assert names == {"alpha", "beta"}


def test_list_draft_skills_empty(tmp_path: Path) -> None:
    loader = SkillsLoader(tmp_path, builtin_skills_dir=None)
    assert loader.list_draft_skills() == []


def test_drafts_not_in_list_skills(tmp_path: Path) -> None:
    writer = SkillWriter(tmp_path)
    writer.write_draft("secret", "draft", [])
    loader = SkillsLoader(tmp_path, builtin_skills_dir=None)
    names = {s["name"] for s in loader.list_skills(filter_unavailable=False)}
    assert "secret" not in names
