"""Tests for P1 reflection mechanism."""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from nanobot.agent.memory import MemoryStore


def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path)


# ---------------------------------------------------------------------------
# _append_candidates / _append_candidates_async
# ---------------------------------------------------------------------------

def test_append_candidates_creates_section(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    store._append_candidates("- rule A <!-- confidence:0.80 added:2026-03-16 hits:0 -->")
    content = store.memory_file.read_text()
    assert "# candidates" in content
    assert "rule A" in content


def test_append_candidates_existing_section(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    store.memory_file.parent.mkdir(parents=True, exist_ok=True)
    store.memory_file.write_text("# candidates\n- rule A <!-- confidence:0.80 added:2026-03-16 hits:0 -->\n")
    store._append_candidates("- rule B <!-- confidence:0.90 added:2026-03-16 hits:0 -->")
    content = store.memory_file.read_text()
    assert "rule A" in content
    assert "rule B" in content


# ---------------------------------------------------------------------------
# promote_candidates
# ---------------------------------------------------------------------------

def test_promote_candidates_moves_high_confidence(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    store.memory_file.parent.mkdir(parents=True, exist_ok=True)
    store.memory_file.write_text(
        "# candidates\n"
        "- rule A <!-- confidence:0.85 added:2026-03-16 hits:2 -->\n"
        "- rule B <!-- confidence:0.60 added:2026-03-16 hits:0 -->\n"
    )
    promoted = store.promote_candidates(min_confidence=0.7)
    assert promoted == 1
    content = store.memory_file.read_text()
    assert "# lessons" in content
    assert "rule A" in content
    # rule B should stay in candidates (hits=0)
    assert "rule B" in content


def test_promote_candidates_no_eligible(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    store.memory_file.parent.mkdir(parents=True, exist_ok=True)
    store.memory_file.write_text(
        "# candidates\n"
        "- rule A <!-- confidence:0.50 added:2026-03-16 hits:0 -->\n"
    )
    promoted = store.promote_candidates(min_confidence=0.7)
    assert promoted == 0


def test_promote_candidates_no_file(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    assert store.promote_candidates() == 0


# ---------------------------------------------------------------------------
# reflect() — mocked LLM
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reflect_writes_candidate(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    store.memory_file.parent.mkdir(parents=True, exist_ok=True)

    mock_tc = MagicMock()
    mock_tc.arguments = {
        "scenario": "user asks for code",
        "action": "used exec tool",
        "outcome": "success",
        "confidence": 0.85,
        "rule": "Always use exec for shell commands",
    }
    mock_response = MagicMock()
    mock_response.has_tool_calls = True
    mock_response.finish_reason = "stop"
    mock_response.tool_calls = [mock_tc]

    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=mock_response)

    messages = [{"role": "user", "content": "run ls", "timestamp": "2026-03-16 10:00"}]
    ok = await store.reflect(messages, provider, "test-model")
    assert ok is True
    content = store.memory_file.read_text()
    assert "Always use exec for shell commands" in content
    assert "# candidates" in content


@pytest.mark.asyncio
async def test_reflect_no_tool_call(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    mock_response = MagicMock()
    mock_response.has_tool_calls = False
    mock_response.finish_reason = "stop"
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=mock_response)
    ok = await store.reflect([], provider, "test-model")
    assert ok is False


@pytest.mark.asyncio
async def test_reflect_empty_messages(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    provider = MagicMock()
    ok = await store.reflect([], provider, "test-model")
    assert ok is False
