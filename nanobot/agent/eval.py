"""P0 评估基线模块：运行固定 Benchmark 任务集并记录结果。"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable, Optional

if TYPE_CHECKING:
    from nanobot.config.schema import EvalConfig


@dataclass
class EvalCase:
    id: str
    prompt: str
    expect_contains: str = ""  # Response must contain this string
    expect_tool: str = ""      # This tool must have been called


@dataclass
class EvalResult:
    case_id: str
    passed: bool
    elapsed_ms: int
    tools_used: list[str] = field(default_factory=list)
    response: str = ""
    error: str = ""


@dataclass
class EvalReport:
    run_at: str
    total: int
    passed: int
    failed: int
    results: list[EvalResult]
    policy_version: str = ""  # Reserved for future version governance

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @staticmethod
    def _md_escape(s: str) -> str:
        return s.replace("|", "\\|").replace("\n", " ").replace("\r", "")

    def to_markdown(self) -> str:
        lines = [
            f"## Eval Run — {self.run_at}",
            f"**Pass rate**: {self.passed}/{self.total} ({self.pass_rate:.0%})",
            "",
            "| ID | Result | Elapsed(ms) | Tools | Note |",
            "|-----|--------|-------------|-------|------|",
        ]
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            tools = self._md_escape(", ".join(r.tools_used) or "-")
            note = self._md_escape(r.error[:60] if r.error else "")
            case_id = self._md_escape(r.case_id)
            lines.append(f"| {case_id} | {status} | {r.elapsed_ms} | {tools} | {note} |")
        lines.append("")
        return "\n".join(lines)


class EvalRunner:
    """Runs the benchmark suite and appends results to BENCHMARK.md."""

    def __init__(self, config: EvalConfig, workspace: Path) -> None:
        self.config = config
        self.workspace = workspace
        self.benchmark_file = self._resolve_benchmark_file(workspace, config.benchmark_file)
        self._turn_counter = 0
        self._running_lock = asyncio.Lock()  # single-flight: at most one eval at a time
        self._write_lock = asyncio.Lock()    # serialize file writes

    @staticmethod
    def _resolve_benchmark_file(workspace: Path, relative: str) -> Path:
        """Resolve benchmark path and ensure it stays inside workspace."""
        resolved = (workspace / relative).resolve()
        try:
            resolved.relative_to(workspace.resolve())
        except ValueError:
            raise ValueError(f"benchmark_file must be inside workspace, got: {relative}")
        return resolved

    def should_run_after_turn(self) -> bool:
        """Return True if auto-trigger threshold has been reached."""
        if not self.config.enabled or self.config.run_after_turns <= 0:
            return False
        self._turn_counter += 1
        return self._turn_counter % self.config.run_after_turns == 0

    async def run(
        self,
        run_agent_fn: Callable[[str], Awaitable[tuple[str, list[str]]]],
    ) -> Optional["EvalReport"]:
        """Execute all benchmark cases and return a report.

        Single-flight: if an eval is already running, skip and return None.

        Args:
            run_agent_fn: async (prompt) -> (response, tools_used)
        """
        if self._running_lock.locked():
            return None  # type: ignore[return-value]  # another eval already running
        async with self._running_lock:
            return await self._run_cases(run_agent_fn)

    async def _run_cases(
        self,
        run_agent_fn: Callable[[str], Awaitable[tuple[str, list[str]]]],
    ) -> EvalReport:
        results: list[EvalResult] = []
        for raw in self.config.suite:
            case = EvalCase(
                id=raw.get("id", "?"),
                prompt=raw.get("prompt", ""),
                expect_contains=raw.get("expect_contains", ""),
                expect_tool=raw.get("expect_tool", ""),
            )
            t0 = time.monotonic()
            try:
                response, tools_used = await run_agent_fn(case.prompt)
                elapsed = int((time.monotonic() - t0) * 1000)
                passed = self._check(case, response, tools_used)
                results.append(EvalResult(
                    case_id=case.id,
                    passed=passed,
                    elapsed_ms=elapsed,
                    tools_used=tools_used,
                    response=response[:200],
                ))
            except Exception as e:
                elapsed = int((time.monotonic() - t0) * 1000)
                results.append(EvalResult(
                    case_id=case.id,
                    passed=False,
                    elapsed_ms=elapsed,
                    error=str(e),
                ))

        report = EvalReport(
            run_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
            total=len(results),
            passed=sum(1 for r in results if r.passed),
            failed=sum(1 for r in results if not r.passed),
            results=results,
        )
        await self._write_report(report)
        return report

    def _check(self, case: EvalCase, response: str, tools_used: list[str]) -> bool:
        if case.expect_contains and case.expect_contains not in response:
            return False
        if case.expect_tool and case.expect_tool not in tools_used:
            return False
        return True

    async def _write_report(self, report: EvalReport) -> None:
        """Append the report to BENCHMARK.md, serialized to prevent concurrent corruption."""
        async with self._write_lock:
            self.benchmark_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.benchmark_file, "a", encoding="utf-8") as f:
                f.write(report.to_markdown())
