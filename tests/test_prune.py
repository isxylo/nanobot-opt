"""Tests for P1 memory pruning mechanism."""

from __future__ import annotations

import pytest
from pathlib import Path

from nanobot.agent.memory import MemoryStore


def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path)


# ---------------------------------------------------------------------------
# prune_memory
# ---------------------------------------------------------------------------

def test_prune_no_file(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    assert store.prune_memory() == 0


def test_prune_no_scored_entries(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    store.memory_file.parent.mkdir(parents=True, exist_ok=True)
    store.memory_file.write_text("# memory\n- plain entry without metadata\n")
    assert store.prune_memory() == 0


def test_prune_removes_old_zero_hit_entry(tmp_path: Path) -> None:
    """An entry with recency=2020 (very old) and hits=0 should be pruned."""
    store = _make_store(tmp_path)
    store.memory_file.parent.mkdir(parents=True, exist_ok=True)
    store.memory_file.write_text(
        "# memory\n"
        "- stale rule <!-- recency:2020-01-01 hits:0 -->\n"
        "- fresh rule <!-- recency:2026-03-16 hits:10 -->\n"
    )
    pruned = store.prune_memory(min_score=0.3)
    assert pruned == 1
    content = store.memory_file.read_text()
    assert "stale rule" not in content
    assert "fresh rule" in content


def test_prune_archives_to_history(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    store.memory_file.parent.mkdir(parents=True, exist_ok=True)
    store.memory_file.write_text(
        "- stale rule <!-- recency:2020-01-01 hits:0 -->\n"
    )
    store.prune_memory(min_score=0.3)
    assert store.history_file.exists()
    history = store.history_file.read_text()
    assert "[PRUNED]" in history
    assert "stale rule" in history


def test_prune_keeps_high_score_entry(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    store.memory_file.parent.mkdir(parents=True, exist_ok=True)
    store.memory_file.write_text(
        "- good rule <!-- recency:2026-03-16 hits:10 -->\n"
    )
    pruned = store.prune_memory(min_score=0.3)
    assert pruned == 0
    assert "good rule" in store.memory_file.read_text()


def test_prune_multiple_entries(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    store.memory_file.parent.mkdir(parents=True, exist_ok=True)
    store.memory_file.write_text(
        "- stale1 <!-- recency:2019-01-01 hits:0 -->\n"
        "- stale2 <!-- recency:2018-01-01 hits:0 -->\n"
        "- fresh <!-- recency:2026-03-16 hits:5 -->\n"
    )
    pruned = store.prune_memory(min_score=0.3)
    assert pruned == 2
    content = store.memory_file.read_text()
    assert "fresh" in content
    assert "stale1" not in content
    assert "stale2" not in content


# ---------------------------------------------------------------------------
# PruneConfig integration via MemoryConfig
# ---------------------------------------------------------------------------

def test_prune_config_defaults() -> None:
    from nanobot.config.schema import PruneConfig
    cfg = PruneConfig()
    assert cfg.enabled is False
    assert cfg.trigger_lines == 100
    assert cfg.min_score == 0.3


def test_reflection_config_defaults() -> None:
    from nanobot.config.schema import ReflectionConfig
    cfg = ReflectionConfig()
    assert cfg.enabled is False
    assert cfg.min_confidence == 0.7
    assert cfg.model == ""


def test_memory_config_has_reflection_and_prune() -> None:
    from nanobot.config.schema import MemoryConfig
    cfg = MemoryConfig()
    assert hasattr(cfg, "reflection")
    assert hasattr(cfg, "prune")
