"""Native browser tools backed by Camoufox."""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from nanobot.agent.tools.base import Tool


def _safe_url(url: str) -> bool:
    """Allow only http(s) URLs for browser navigation."""
    p = urlparse(url)
    return p.scheme in {"http", "https"} and bool(p.netloc)


class BrowserSessionManager:
    """Lazy singleton-like browser/page holder for all browser tools."""

    def __init__(
        self,
        workspace: Path,
        *,
        headless: bool = True,
        timeout_ms: int = 30_000,
        max_chars: int = 120_000,
        screenshot_dir: str = "artifacts/browser",
    ) -> None:
        self.workspace = workspace
        self.headless = headless
        self.timeout_ms = timeout_ms
        self.max_chars = max_chars
        self.screenshot_dir = screenshot_dir

        self._lock = asyncio.Lock()
        self._camoufox_cm: Any | None = None
        self._browser: Any | None = None
        self._page: Any | None = None

    async def _ensure_started(self) -> None:
        if self._page is not None:
            return

        try:
            from camoufox.async_api import AsyncCamoufox
        except Exception as exc:  # pragma: no cover - tested through execute path
            raise RuntimeError(
                "Camoufox is not installed. Install with 'pip install camoufox' "
                "and run 'camoufox fetch'."
            ) from exc

        self._camoufox_cm = AsyncCamoufox(headless=self.headless)
        self._browser = await self._camoufox_cm.__aenter__()
        self._page = await self._browser.new_page()
        self._page.set_default_timeout(self.timeout_ms)

    async def open(self, url: str, wait_until: str = "domcontentloaded", timeout_ms: int | None = None) -> str:
        if not _safe_url(url):
            return "Error: Invalid URL. Only http/https URLs are allowed."

        async with self._lock:
            try:
                await self._ensure_started()
                timeout = timeout_ms or self.timeout_ms
                resp = await self._page.goto(url, wait_until=wait_until, timeout=timeout)
                status = resp.status if resp else None
                final_url = self._page.url
                title = await self._page.title()
                return json.dumps(
                    {
                        "ok": True,
                        "url": final_url,
                        "requestedUrl": url,
                        "status": status,
                        "title": title,
                    },
                    ensure_ascii=False,
                )
            except Exception as exc:
                return f"Error: browser_open failed: {type(exc).__name__}: {exc}"

    async def get_text(self, max_chars: int | None = None) -> str:
        async with self._lock:
            if self._page is None:
                return "Error: No page is open. Call browser_open(url) first."
            try:
                text = await self._page.inner_text("body")
                text = re.sub(r"\n{3,}", "\n\n", text).strip()
                text = self._truncate(text, max_chars)
                return text or "(empty text)"
            except Exception as exc:
                return f"Error: page_get_text failed: {type(exc).__name__}: {exc}"

    async def get_html(self, max_chars: int | None = None) -> str:
        async with self._lock:
            if self._page is None:
                return "Error: No page is open. Call browser_open(url) first."
            try:
                html = await self._page.content()
                return self._truncate(html, max_chars)
            except Exception as exc:
                return f"Error: page_get_html failed: {type(exc).__name__}: {exc}"

    async def screenshot(self, path: str | None = None, full_page: bool = True) -> str:
        async with self._lock:
            if self._page is None:
                return "Error: No page is open. Call browser_open(url) first."
            try:
                target = self._resolve_screenshot_path(path)
                target.parent.mkdir(parents=True, exist_ok=True)
                await self._page.screenshot(path=str(target), full_page=full_page)
                return json.dumps({"ok": True, "path": str(target)}, ensure_ascii=False)
            except Exception as exc:
                return f"Error: page_screenshot failed: {type(exc).__name__}: {exc}"

    async def click(
        self,
        selector: str,
        timeout_ms: int | None = None,
        wait_after_ms: int = 800,
    ) -> str:
        async with self._lock:
            if self._page is None:
                return "Error: No page is open. Call browser_open(url) first."
            try:
                timeout = timeout_ms or self.timeout_ms
                await self._page.locator(selector).first.click(timeout=timeout)
                if wait_after_ms > 0:
                    await self._page.wait_for_timeout(wait_after_ms)
                return json.dumps(
                    {"ok": True, "selector": selector, "url": self._page.url},
                    ensure_ascii=False,
                )
            except Exception as exc:
                return f"Error: page_click failed: {type(exc).__name__}: {exc}"

    def _truncate(self, text: str, max_chars: int | None) -> str:
        limit = max_chars or self.max_chars
        if limit <= 0:
            limit = self.max_chars
        if len(text) <= limit:
            return text
        return text[:limit] + f"\n\n...(truncated {len(text) - limit} chars)"

    def _resolve_screenshot_path(self, path: str | None) -> Path:
        if path:
            p = Path(path).expanduser()
            if not p.is_absolute():
                p = self.workspace / p
            return p.resolve()
        ts = time.strftime("%Y%m%d-%H%M%S")
        return (self.workspace / self.screenshot_dir / f"screenshot-{ts}.png").resolve()

    async def close(self) -> None:
        """Close browser process if started."""
        async with self._lock:
            if self._camoufox_cm is None:
                return
            try:
                await self._camoufox_cm.__aexit__(None, None, None)
            finally:
                self._camoufox_cm = None
                self._browser = None
                self._page = None


class _BrowserTool(Tool):
    """Shared base for browser tools."""

    parallel_safe = False

    def __init__(self, manager: BrowserSessionManager):
        self._manager = manager


class BrowserOpenTool(_BrowserTool):
    @property
    def name(self) -> str:
        return "browser_open"

    @property
    def description(self) -> str:
        return "Open URL in a real browser page (Camoufox) and keep the session for follow-up tools."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Target URL (http/https)"},
                "waitUntil": {
                    "type": "string",
                    "enum": ["commit", "domcontentloaded", "load", "networkidle"],
                    "default": "domcontentloaded",
                },
                "timeoutMs": {"type": "integer", "minimum": 1, "maximum": 180000},
            },
            "required": ["url"],
        }

    async def execute(
        self,
        url: str,
        waitUntil: str = "domcontentloaded",
        timeoutMs: int | None = None,
        **kwargs: Any,
    ) -> str:
        return await self._manager.open(url=url, wait_until=waitUntil, timeout_ms=timeoutMs)


class PageGetTextTool(_BrowserTool):
    @property
    def name(self) -> str:
        return "page_get_text"

    @property
    def description(self) -> str:
        return "Get visible text from current browser page."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "maxChars": {"type": "integer", "minimum": 100, "maximum": 500000},
            },
        }

    async def execute(self, maxChars: int | None = None, **kwargs: Any) -> str:
        return await self._manager.get_text(max_chars=maxChars)


class PageGetHtmlTool(_BrowserTool):
    @property
    def name(self) -> str:
        return "page_get_html"

    @property
    def description(self) -> str:
        return "Get HTML source from current browser page."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "maxChars": {"type": "integer", "minimum": 100, "maximum": 1000000},
            },
        }

    async def execute(self, maxChars: int | None = None, **kwargs: Any) -> str:
        return await self._manager.get_html(max_chars=maxChars)


class PageScreenshotTool(_BrowserTool):
    @property
    def name(self) -> str:
        return "page_screenshot"

    @property
    def description(self) -> str:
        return "Take a screenshot of current browser page and return the saved file path."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Optional output path (.png). Relative paths use workspace."},
                "fullPage": {"type": "boolean", "default": True},
            },
        }

    async def execute(self, path: str | None = None, fullPage: bool = True, **kwargs: Any) -> str:
        return await self._manager.screenshot(path=path, full_page=fullPage)


class PageClickTool(_BrowserTool):
    @property
    def name(self) -> str:
        return "page_click"

    @property
    def description(self) -> str:
        return "Click an element on current page using CSS selector."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "selector": {"type": "string", "minLength": 1},
                "timeoutMs": {"type": "integer", "minimum": 1, "maximum": 180000},
                "waitAfterMs": {"type": "integer", "minimum": 0, "maximum": 60000, "default": 800},
            },
            "required": ["selector"],
        }

    async def execute(
        self,
        selector: str,
        timeoutMs: int | None = None,
        waitAfterMs: int = 800,
        **kwargs: Any,
    ) -> str:
        return await self._manager.click(
            selector=selector,
            timeout_ms=timeoutMs,
            wait_after_ms=waitAfterMs,
        )
