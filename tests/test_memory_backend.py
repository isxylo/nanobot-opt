"""Tests for memory backend (file / nocturne_mcp / hybrid)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.memory import (
    HybridMemoryContext,
    MemoryStore,
    NocturneMCPAdapter,
)
from nanobot.config.schema import MemoryConfig


# ---------------------------------------------------------------------------
# MemoryConfig defaults
# ---------------------------------------------------------------------------


def test_memory_config_defaults():
    cfg = MemoryConfig()
    assert cfg.backend == "file"
    assert cfg.fallback_to_file is True
    assert cfg.mcp_server_name == "nocturne_memory"
    assert cfg.max_recall_items == 5


def test_memory_config_hybrid():
    cfg = MemoryConfig(backend="hybrid")
    assert cfg.backend == "hybrid"


def test_memory_config_nocturne_mcp():
    cfg = MemoryConfig(backend="nocturne_mcp", fallback_to_file=False)
    assert cfg.backend == "nocturne_mcp"
    assert cfg.fallback_to_file is False


# ---------------------------------------------------------------------------
# NocturneMCPAdapter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nocturne_adapter_read_boot_success():
    tools = MagicMock()
    tools.execute = AsyncMock(return_value="# Core Memories\nloaded: 2")
    adapter = NocturneMCPAdapter(tools)
    result = await adapter.read_boot()
    assert result == "# Core Memories\nloaded: 2"
    tools.execute.assert_called_once_with("read_memory", {"uri": "system://boot"})


@pytest.mark.asyncio
async def test_nocturne_adapter_read_boot_error_response():
    tools = MagicMock()
    tools.execute = AsyncMock(return_value="Error: URI not found")
    adapter = NocturneMCPAdapter(tools)
    result = await adapter.read_boot()
    assert result is None


@pytest.mark.asyncio
async def test_nocturne_adapter_read_boot_exception():
    tools = MagicMock()
    tools.execute = AsyncMock(side_effect=RuntimeError("MCP down"))
    adapter = NocturneMCPAdapter(tools)
    result = await adapter.read_boot()
    assert result is None


@pytest.mark.asyncio
async def test_nocturne_adapter_write_memory_success():
    tools = MagicMock()
    tools.execute = AsyncMock(return_value="Success: Memory created at 'core://nanobot_memory'")
    adapter = NocturneMCPAdapter(tools)
    ok = await adapter.write_memory("core://", "some content")
    assert ok is True


@pytest.mark.asyncio
async def test_nocturne_adapter_write_memory_failure():
    tools = MagicMock()
    tools.execute = AsyncMock(return_value="Error: parent not found")
    adapter = NocturneMCPAdapter(tools)
    ok = await adapter.write_memory("core://", "some content")
    assert ok is False


@pytest.mark.asyncio
async def test_nocturne_adapter_write_memory_exception():
    tools = MagicMock()
    tools.execute = AsyncMock(side_effect=Exception("timeout"))
    adapter = NocturneMCPAdapter(tools)
    ok = await adapter.write_memory("core://", "some content")
    assert ok is False


# ---------------------------------------------------------------------------
# HybridMemoryContext
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hybrid_uses_mcp_boot_when_available(tmp_path):
    store = MemoryStore(tmp_path)
    adapter = MagicMock(spec=NocturneMCPAdapter)
    adapter.read_boot = AsyncMock(return_value="MCP memory content")

    ctx = HybridMemoryContext(store=store, adapter=adapter, fallback_to_file=True)
    await ctx.load_boot()

    result = ctx.get_memory_context()
    assert "MCP memory content" in result
    assert "nocturne" in result


@pytest.mark.asyncio
async def test_hybrid_falls_back_to_file_on_mcp_failure(tmp_path):
    store = MemoryStore(tmp_path)
    # Write something to local MEMORY.md
    store.write_long_term("# now\nlocal memory content")

    adapter = MagicMock(spec=NocturneMCPAdapter)
    adapter.read_boot = AsyncMock(return_value=None)  # MCP fails

    ctx = HybridMemoryContext(store=store, adapter=adapter, fallback_to_file=True)
    await ctx.load_boot()

    result = ctx.get_memory_context()
    assert "local memory content" in result


@pytest.mark.asyncio
async def test_hybrid_no_adapter_uses_file(tmp_path):
    store = MemoryStore(tmp_path)
    store.write_long_term("# now\nfile only")

    ctx = HybridMemoryContext(store=store, adapter=None, fallback_to_file=True)
    await ctx.load_boot()  # no-op

    result = ctx.get_memory_context()
    assert "file only" in result


@pytest.mark.asyncio
async def test_hybrid_empty_when_no_file_and_no_mcp(tmp_path):
    store = MemoryStore(tmp_path)
    ctx = HybridMemoryContext(store=store, adapter=None, fallback_to_file=True)
    await ctx.load_boot()
    result = ctx.get_memory_context()
    assert result == ""


# ---------------------------------------------------------------------------
# MemoryConfig in ToolsConfig (schema round-trip)
# ---------------------------------------------------------------------------


def test_tools_config_has_memory():
    from nanobot.config.schema import ToolsConfig
    cfg = ToolsConfig()
    assert hasattr(cfg, "memory")
    assert isinstance(cfg.memory, MemoryConfig)
    assert cfg.memory.backend == "file"


def test_tools_config_memory_from_dict():
    from nanobot.config.schema import ToolsConfig
    cfg = ToolsConfig.model_validate({"memory": {"backend": "hybrid", "maxRecallItems": 10}})
    assert cfg.memory.backend == "hybrid"
    assert cfg.memory.max_recall_items == 10
