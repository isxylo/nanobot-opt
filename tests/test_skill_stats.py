"""Tests for P2 SkillStats and build_skills_summary ordering."""

from __future__ import annotations

from pathlib import Path

import pytest

from nanobot.agent.skills import SkillStats, SkillsLoader


# ---------------------------------------------------------------------------
# SkillStats
# ---------------------------------------------------------------------------

def test_record_creates_entry(tmp_path: Path) -> None:
    stats = SkillStats(tmp_path)
    stats.record("web_search", success=True)
    assert stats._data["web_search"]["calls"] == 1
    assert stats._data["web_search"]["success"] == 1


def test_record_failure(tmp_path: Path) -> None:
    stats = SkillStats(tmp_path)
    stats.record("exec", success=False)
    assert stats._data["exec"]["calls"] == 1
    assert stats._data["exec"]["success"] == 0


def test_record_persists(tmp_path: Path) -> None:
    stats = SkillStats(tmp_path)
    stats.record("web_search", success=True)
    # Reload from disk
    stats2 = SkillStats(tmp_path)
    assert stats2._data["web_search"]["calls"] == 1


def test_success_rate_zero_calls(tmp_path: Path) -> None:
    stats = SkillStats(tmp_path)
    assert stats.success_rate("unknown") == 0.0


def test_success_rate(tmp_path: Path) -> None:
    stats = SkillStats(tmp_path)
    stats.record("exec", success=True)
    stats.record("exec", success=True)
    stats.record("exec", success=False)
    assert abs(stats.success_rate("exec") - 2/3) < 1e-9


def test_sorted_by_priority_orders_correctly(tmp_path: Path) -> None:
    stats = SkillStats(tmp_path)
    # web_search: 10 calls, 100% success — high score
    for _ in range(10):
        stats.record("web_search", success=True)
    # exec: 1 call, 100% success — lower score (fewer calls)
    stats.record("exec", success=True)
    # never_used: no calls — score 0
    ordered = stats.sorted_by_priority(["exec", "never_used", "web_search"])
    assert ordered[0] == "web_search"
    assert ordered[-1] == "never_used"


def test_sorted_by_priority_unknown_skills(tmp_path: Path) -> None:
    stats = SkillStats(tmp_path)
    ordered = stats.sorted_by_priority(["a", "b", "c"])
    # All unknown → all score 0 → order preserved (stable sort)
    assert set(ordered) == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# build_skills_summary ordering
# ---------------------------------------------------------------------------

def _make_skill(skills_dir: Path, name: str) -> None:
    d = skills_dir / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {name} skill\n---\n# {name}\n", encoding="utf-8")


def test_build_skills_summary_with_stats(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    _make_skill(skills_dir, "low_use")
    _make_skill(skills_dir, "high_use")

    stats = SkillStats(tmp_path)
    for _ in range(20):
        stats.record("high_use", success=True)
    stats.record("low_use", success=True)

    loader = SkillsLoader(tmp_path, builtin_skills_dir=None)
    summary = loader.build_skills_summary(skill_stats=stats)

    # high_use should appear before low_use in the summary
    assert summary.index("high_use") < summary.index("low_use")


def test_build_skills_summary_without_stats(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    _make_skill(skills_dir, "alpha")
    _make_skill(skills_dir, "beta")

    loader = SkillsLoader(tmp_path, builtin_skills_dir=None)
    summary = loader.build_skills_summary()
    assert "alpha" in summary
    assert "beta" in summary


# ---------------------------------------------------------------------------
# PITFALLS.md injection (context builder)
# ---------------------------------------------------------------------------

def test_load_pitfalls_exists(tmp_path: Path) -> None:
    from nanobot.agent.context import ContextBuilder
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "PITFALLS.md").write_text("## Avoid\n- Never delete prod DB", encoding="utf-8")
    ctx = ContextBuilder(tmp_path)
    assert "Never delete prod DB" in ctx._load_pitfalls()


def test_load_pitfalls_missing(tmp_path: Path) -> None:
    from nanobot.agent.context import ContextBuilder
    ctx = ContextBuilder(tmp_path)
    assert ctx._load_pitfalls() == ""
