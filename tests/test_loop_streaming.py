"""Tests for P3-3 streaming: _StreamBuffer throttling and _run_agent_loop stream callbacks."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.loop import _StreamBuffer


# ---------------------------------------------------------------------------
# _StreamBuffer tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_buffer_accumulates_and_flushes():
    """Buffer accumulates deltas and flushes when FLUSH_INTERVAL elapses."""
    received: list[str] = []

    async def on_delta(content: str) -> None:
        received.append(content)

    buf = _StreamBuffer(on_delta)
    # Manually set last_flush to 0 so first push triggers flush
    buf._last_flush = 0.0

    await buf.push("Hello")
    assert len(received) == 1
    assert received[0] == "Hello"

    # Push within interval — should NOT flush again immediately
    buf._last_flush = asyncio.get_event_loop().time()  # reset to now
    await buf.push(", world")
    # Content accumulated but not flushed yet (within interval)
    assert len(received) == 1
    assert buf._buf == "Hello, world"


@pytest.mark.asyncio
async def test_stream_buffer_flush_final_returns_accumulated():
    """flush_final flushes remaining content and returns the full buffer."""
    received: list[str] = []

    async def on_delta(content: str) -> None:
        received.append(content)

    buf = _StreamBuffer(on_delta)
    buf._last_flush = asyncio.get_event_loop().time()  # prevent auto-flush
    buf._buf = "accumulated text"

    result = await buf.flush_final()
    assert result == "accumulated text"
    assert len(received) == 1
    assert received[0] == "accumulated text"


@pytest.mark.asyncio
async def test_stream_buffer_flush_final_empty():
    """flush_final on empty buffer returns empty string without calling on_delta."""
    received: list[str] = []

    async def on_delta(content: str) -> None:
        received.append(content)

    buf = _StreamBuffer(on_delta)
    result = await buf.flush_final()
    assert result == ""
    assert received == []


@pytest.mark.asyncio
async def test_stream_buffer_throttles_rapid_pushes():
    """Rapid pushes within FLUSH_INTERVAL only flush once."""
    flush_count = 0

    async def on_delta(content: str) -> None:
        nonlocal flush_count
        flush_count += 1

    buf = _StreamBuffer(on_delta)
    # First push flushes (last_flush=0)
    await buf.push("a")
    assert flush_count == 1

    # Subsequent pushes within interval do NOT flush
    for ch in "bcdefgh":
        await buf.push(ch)
    assert flush_count == 1  # still only 1 flush

    # flush_final triggers one more flush
    await buf.flush_final()
    assert flush_count == 2


# ---------------------------------------------------------------------------
# _run_agent_loop streaming integration
# ---------------------------------------------------------------------------

def _make_loop():
    """Create a minimal AgentLoop with mocked dependencies."""
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus
    from pathlib import Path
    import tempfile

    tmpdir = Path(tempfile.mkdtemp())
    bus = MagicMock(spec=MessageBus)
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.chat_with_retry = AsyncMock()
    provider.generation = MagicMock(max_tokens=4096, temperature=0.7, reasoning_effort=None)

    loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=tmpdir,
        model="test-model",
    )
    return loop, provider


@pytest.mark.asyncio
async def test_run_agent_loop_calls_on_stream_for_text_response():
    """on_stream and on_stream_done are called when LLM returns a text response."""
    from nanobot.providers.base import LLMResponse

    loop, provider = _make_loop()
    provider.chat_with_retry.return_value = LLMResponse(
        content="Hello, streaming world!",
        finish_reason="stop",
    )

    stream_deltas: list[str] = []
    stream_done: list[str] = []

    async def on_stream(content: str) -> None:
        stream_deltas.append(content)

    async def on_stream_done(content: str) -> None:
        stream_done.append(content)

    messages = [{"role": "user", "content": "hi"}]
    final, _, _, _, _ = await loop._run_agent_loop(
        messages,
        on_stream=on_stream,
        on_stream_done=on_stream_done,
    )

    assert final == "Hello, streaming world!"
    # on_stream should have been called with progressive chunks
    assert len(stream_deltas) >= 1
    # The last accumulated content should equal full text
    assert stream_deltas[-1] == "Hello, streaming world!"
    # on_stream_done called exactly once with full content
    assert stream_done == ["Hello, streaming world!"]


@pytest.mark.asyncio
async def test_run_agent_loop_no_stream_callbacks_without_on_stream():
    """Without on_stream, no streaming callbacks are triggered."""
    from nanobot.providers.base import LLMResponse

    loop, provider = _make_loop()
    provider.chat_with_retry.return_value = LLMResponse(
        content="Normal response",
        finish_reason="stop",
    )

    # No on_stream passed — should work normally
    messages = [{"role": "user", "content": "hi"}]
    final, _, _, _, _ = await loop._run_agent_loop(messages)

    assert final == "Normal response"


@pytest.mark.asyncio
async def test_run_agent_loop_stream_skipped_for_errors():
    """on_stream_done is NOT called for error responses."""
    from nanobot.providers.base import LLMResponse

    loop, provider = _make_loop()
    provider.chat_with_retry.return_value = LLMResponse(
        content="Error: 401 unauthorized",
        finish_reason="error",
    )

    stream_done: list[str] = []

    async def on_stream_done(content: str) -> None:
        stream_done.append(content)

    messages = [{"role": "user", "content": "hi"}]
    final, _, _, _, _ = await loop._run_agent_loop(
        messages,
        on_stream_done=on_stream_done,
    )

    # Error response — on_stream_done should NOT be called
    assert stream_done == []
