"""Tests for memory backend (file / nocturne_mcp / hybrid)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from nanobot.agent.memory import (
    HybridMemoryContext,
    MemoryStore,
    NocturneMCPAdapter,
    _extract_nocturne_content,
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


def test_memory_config_hybrid():
    cfg = MemoryConfig(backend="hybrid")
    assert cfg.backend == "hybrid"


def test_memory_config_nocturne_mcp():
    cfg = MemoryConfig(backend="nocturne_mcp", fallback_to_file=False)
    assert cfg.backend == "nocturne_mcp"
    assert cfg.fallback_to_file is False


# ---------------------------------------------------------------------------
# NocturneMCPAdapter — read_boot
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


# ---------------------------------------------------------------------------
# NocturneMCPAdapter — write_memory (upsert: read→update or create)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _extract_nocturne_content helper
# ---------------------------------------------------------------------------


def test_extract_nocturne_content_normal():
    sep = "=" * 60
    read_result = (
        f"{sep}\n"
        f"MEMORY: core://nanobot_memory\nMemory ID: 42\n"
        f"{sep}\n"
        f"\nThis is the actual content.\n"
        f"\n{sep}\n"
        f"CHILD MEMORIES"
    )
    result = _extract_nocturne_content(read_result)
    assert result == "This is the actual content."


def test_extract_nocturne_content_empty_body():
    sep = "=" * 60
    read_result = f"{sep}\nMEMORY: core://x\n{sep}\n\n"
    result = _extract_nocturne_content(read_result)
    assert result is None  # empty content → None


def test_extract_nocturne_content_unrecognised_format():
    result = _extract_nocturne_content("Error: not found")
    assert result is None


# ---------------------------------------------------------------------------
# NocturneMCPAdapter — write_memory upsert
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_memory_updates_existing_node_with_patch():
    """When node exists, write_memory uses patch (old_string+new_string), not append."""
    sep = "=" * 60
    existing_content = "old memory content"
    read_result = (
        f"{sep}\nMEMORY: core://nanobot_memory\n{sep}\n\n{existing_content}\n\n{sep}\n"
    )
    tools = MagicMock()
    tools.execute = AsyncMock(side_effect=[
        read_result,             # read_memory: node exists with content
        "Success: updated",      # update_memory patch
    ])
    adapter = NocturneMCPAdapter(tools)
    ok = await adapter.write_memory("core://", "new content", title="nanobot_memory")
    assert ok is True
    calls = tools.execute.call_args_list
    assert calls[0][0][0] == "read_memory"
    assert calls[1][0][0] == "update_memory"
    update_args = calls[1][0][1]
    # Must use patch mode (old_string + new_string), NOT append alone
    assert update_args.get("old_string") == existing_content
    assert update_args.get("new_string") == "new content"
    assert "append" not in update_args


@pytest.mark.asyncio
async def test_write_memory_creates_when_node_missing():
    """When read_memory returns error, write_memory calls create_memory."""
    tools = MagicMock()
    tools.execute = AsyncMock(side_effect=[
        "Error: URI not found",                             # read_memory: not found
        "Success: Memory created at 'core://nanobot_memory'",  # create_memory
    ])
    adapter = NocturneMCPAdapter(tools)
    ok = await adapter.write_memory("core://", "new content", title="nanobot_memory")
    assert ok is True
    calls = tools.execute.call_args_list
    assert calls[0][0][0] == "read_memory"
    assert calls[1][0][0] == "create_memory"


@pytest.mark.asyncio
async def test_write_memory_nocturne_rejects_mutual_exclusive_params():
    """Contract test: nocturne returns error when old_string+append are both set.

    This validates our upsert logic never passes both params simultaneously.
    The mock simulates nocturne's actual protocol enforcement.
    """
    tools = MagicMock()

    async def _strict_update(name, args):
        if name == "update_memory":
            if args.get("old_string") is not None and args.get("append") is not None:
                return "Error: Cannot use both old_string/new_string (patch) and append at the same time."
            return "Success: updated"
        if name == "read_memory":
            return "MEMORY: core://x\ncontent"
        return "Success"

    tools.execute = AsyncMock(side_effect=_strict_update)
    adapter = NocturneMCPAdapter(tools)
    ok = await adapter.write_memory("core://", "content", title="x")
    # Our code must not trigger the mutual-exclusion error
    assert ok is True


@pytest.mark.asyncio
async def test_write_memory_both_fail():
    tools = MagicMock()
    tools.execute = AsyncMock(side_effect=[
        "Error: not found",   # read_memory
        "Error: create fail", # create_memory
    ])
    adapter = NocturneMCPAdapter(tools)
    ok = await adapter.write_memory("core://", "content")
    assert ok is False


@pytest.mark.asyncio
async def test_write_memory_exception():
    tools = MagicMock()
    tools.execute = AsyncMock(side_effect=Exception("timeout"))
    adapter = NocturneMCPAdapter(tools)
    ok = await adapter.write_memory("core://", "content")
    assert ok is False


# ---------------------------------------------------------------------------
# HybridMemoryContext — fallback_to_file enforcement
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
async def test_hybrid_falls_back_to_file_when_fallback_true(tmp_path):
    store = MemoryStore(tmp_path)
    store.write_long_term("# now\nlocal memory content")

    adapter = MagicMock(spec=NocturneMCPAdapter)
    adapter.read_boot = AsyncMock(return_value=None)  # MCP fails

    ctx = HybridMemoryContext(store=store, adapter=adapter, fallback_to_file=True)
    await ctx.load_boot()

    result = ctx.get_memory_context()
    assert "local memory content" in result


@pytest.mark.asyncio
async def test_hybrid_returns_empty_when_fallback_false_and_mcp_fails(tmp_path):
    """fallback_to_file=False: must return empty string, not leak local file."""
    store = MemoryStore(tmp_path)
    store.write_long_term("# now\nlocal secret")

    adapter = MagicMock(spec=NocturneMCPAdapter)
    adapter.read_boot = AsyncMock(return_value=None)  # MCP fails

    ctx = HybridMemoryContext(store=store, adapter=adapter, fallback_to_file=False)
    await ctx.load_boot()

    result = ctx.get_memory_context()
    assert result == ""
    assert "local secret" not in result


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
    cfg = ToolsConfig.model_validate({"memory": {"backend": "hybrid"}})
    assert cfg.memory.backend == "hybrid"
