from __future__ import annotations

from types import ModuleType
from typing import Any

import pytest

from nanobot.agent.tools.browser import (
    BrowserOpenTool,
    BrowserSessionManager,
    PageClickTool,
    PageGetHtmlTool,
    PageGetTextTool,
    PageScreenshotTool,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def open(self, **kwargs: Any) -> str:
        self.calls.append(("open", kwargs))
        return "ok-open"

    async def get_text(self, **kwargs: Any) -> str:
        self.calls.append(("get_text", kwargs))
        return "ok-text"

    async def get_html(self, **kwargs: Any) -> str:
        self.calls.append(("get_html", kwargs))
        return "ok-html"

    async def screenshot(self, **kwargs: Any) -> str:
        self.calls.append(("screenshot", kwargs))
        return "ok-shot"

    async def click(self, **kwargs: Any) -> str:
        self.calls.append(("click", kwargs))
        return "ok-click"


@pytest.mark.asyncio
async def test_browser_tool_wrappers_forward_arguments() -> None:
    mgr = _DummyManager()

    assert await BrowserOpenTool(mgr).execute(url="https://example.com", waitUntil="load", timeoutMs=123) == "ok-open"
    assert await PageGetTextTool(mgr).execute(maxChars=321) == "ok-text"
    assert await PageGetHtmlTool(mgr).execute(maxChars=654) == "ok-html"
    assert await PageScreenshotTool(mgr).execute(path="a.png", fullPage=False) == "ok-shot"
    assert await PageClickTool(mgr).execute(selector="#go", timeoutMs=99, waitAfterMs=10) == "ok-click"

    assert mgr.calls == [
        ("open", {"url": "https://example.com", "wait_until": "load", "timeout_ms": 123}),
        ("get_text", {"max_chars": 321}),
        ("get_html", {"max_chars": 654}),
        ("screenshot", {"path": "a.png", "full_page": False}),
        ("click", {"selector": "#go", "timeout_ms": 99, "wait_after_ms": 10}),
    ]


@pytest.mark.asyncio
async def test_browser_open_rejects_non_http_url(tmp_path) -> None:
    manager = BrowserSessionManager(workspace=tmp_path)

    result = await manager.open("file:///etc/passwd")

    assert "Invalid URL" in result


@pytest.mark.asyncio
async def test_page_get_text_requires_open_page(tmp_path) -> None:
    manager = BrowserSessionManager(workspace=tmp_path)

    result = await manager.get_text()

    assert "No page is open" in result


@pytest.mark.asyncio
async def test_browser_open_reports_missing_dependency(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import sys

    # Ensure import camoufox.async_api fails deterministically
    monkeypatch.setitem(sys.modules, "camoufox", ModuleType("camoufox"))
    monkeypatch.delitem(sys.modules, "camoufox.async_api", raising=False)

    manager = BrowserSessionManager(workspace=tmp_path)
    result = await manager.open("https://example.com")

    assert "Camoufox is not installed" in result or "browser_open failed" in result


def test_default_screenshot_path_under_workspace(tmp_path) -> None:
    manager = BrowserSessionManager(workspace=tmp_path)

    path = manager._resolve_screenshot_path(None)

    assert str(path).startswith(str(tmp_path.resolve()))
    assert path.suffix == ".png"


@pytest.mark.asyncio
async def test_manager_close_is_safe_when_not_started(tmp_path) -> None:
    manager = BrowserSessionManager(workspace=tmp_path)
    await manager.close()


def test_agent_loop_registers_native_browser_tools(tmp_path) -> None:
    from unittest.mock import MagicMock

    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"

    loop = AgentLoop(
        bus=MagicMock(spec=MessageBus),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
    )

    for tool_name in (
        "browser_open",
        "page_get_text",
        "page_get_html",
        "page_screenshot",
        "page_click",
    ):
        assert loop.tools.has(tool_name)
