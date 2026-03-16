"""Tests for P3 subagent experience sharing."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.subagent import SubagentManager


def _make_manager(tmp_path):
    provider = MagicMock()
    bus = MagicMock()
    bus.publish_inbound = AsyncMock()
    return SubagentManager(
        provider=provider,
        workspace=tmp_path,
        bus=bus,
    )


# ---------------------------------------------------------------------------
# _extract_experience
# ---------------------------------------------------------------------------

def test_extract_experience_last_assistant(tmp_path) -> None:
    mgr = _make_manager(tmp_path)
    messages = [
        {"role": "user", "content": "do something"},
        {"role": "assistant", "content": "I found that using git stash is effective for this pattern."},
    ]
    result = mgr._extract_experience(messages, "git-task")
    assert result is not None
    assert "git-task" in result
    assert "git stash" in result


def test_extract_experience_truncates_long_content(tmp_path) -> None:
    mgr = _make_manager(tmp_path)
    long_content = "x" * 500
    messages = [{"role": "assistant", "content": long_content}]
    result = mgr._extract_experience(messages, "label")
    assert result is not None
    assert len(result) <= 320  # label prefix + 300 chars


def test_extract_experience_no_assistant(tmp_path) -> None:
    mgr = _make_manager(tmp_path)
    messages = [{"role": "user", "content": "hello"}]
    assert mgr._extract_experience(messages, "label") is None


def test_extract_experience_short_content_skipped(tmp_path) -> None:
    mgr = _make_manager(tmp_path)
    messages = [{"role": "assistant", "content": "ok"}]
    assert mgr._extract_experience(messages, "label") is None


# ---------------------------------------------------------------------------
# _announce_result with experience
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_announce_result_includes_experience(tmp_path) -> None:
    mgr = _make_manager(tmp_path)
    await mgr._announce_result(
        task_id="t1",
        label="test-task",
        task="do something",
        result="done",
        origin={"channel": "cli", "chat_id": "direct"},
        status="ok",
        experience="[subagent:test-task] useful heuristic here",
    )
    mgr.bus.publish_inbound.assert_called_once()
    msg = mgr.bus.publish_inbound.call_args[0][0]
    assert "untrusted-data" in msg.content
    assert "useful heuristic here" in msg.content


@pytest.mark.asyncio
async def test_announce_result_no_experience(tmp_path) -> None:
    mgr = _make_manager(tmp_path)
    await mgr._announce_result(
        task_id="t2",
        label="task",
        task="do it",
        result="done",
        origin={"channel": "cli", "chat_id": "direct"},
        status="ok",
        experience=None,
    )
    msg = mgr.bus.publish_inbound.call_args[0][0]
    assert "Experience from subagent" not in msg.content


@pytest.mark.asyncio
async def test_announce_result_error_status(tmp_path) -> None:
    mgr = _make_manager(tmp_path)
    await mgr._announce_result(
        task_id="t3",
        label="fail-task",
        task="do it",
        result="Error: something broke",
        origin={"channel": "cli", "chat_id": "direct"},
        status="error",
        experience=None,
    )
    msg = mgr.bus.publish_inbound.call_args[0][0]
    assert "failed" in msg.content
