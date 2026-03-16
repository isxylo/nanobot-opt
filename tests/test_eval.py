"""Tests for the P0 eval baseline module."""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import AsyncMock

from nanobot.agent.eval import EvalCase, EvalReport, EvalResult, EvalRunner


def _make_config(suite: list[dict], run_after_turns: int = 0, enabled: bool = True):
    from nanobot.config.schema import EvalConfig
    return EvalConfig(
        enabled=enabled,
        benchmark_file="memory/BENCHMARK.md",
        suite=suite,
        run_after_turns=run_after_turns,
    )


# ---------------------------------------------------------------------------
# EvalReport.to_markdown
# ---------------------------------------------------------------------------

def test_report_to_markdown_pass() -> None:
    report = EvalReport(
        run_at="2026-03-16 10:00",
        total=1,
        passed=1,
        failed=0,
        results=[EvalResult(case_id="e001", passed=True, elapsed_ms=42, tools_used=["exec"])],
    )
    md = report.to_markdown()
    assert "PASS" in md
    assert "e001" in md
    assert "1/1" in md


def test_report_to_markdown_fail() -> None:
    report = EvalReport(
        run_at="2026-03-16 10:00",
        total=1,
        passed=0,
        failed=1,
        results=[EvalResult(case_id="e002", passed=False, elapsed_ms=10, error="timeout")],
    )
    md = report.to_markdown()
    assert "FAIL" in md
    assert "timeout" in md


def test_report_pass_rate_zero_total() -> None:
    report = EvalReport(run_at="", total=0, passed=0, failed=0, results=[])
    assert report.pass_rate == 0.0


# ---------------------------------------------------------------------------
# EvalRunner._check
# ---------------------------------------------------------------------------

def test_check_expect_contains_pass(tmp_path: Path) -> None:
    cfg = _make_config([])
    runner = EvalRunner(cfg, tmp_path)
    case = EvalCase(id="c1", prompt="", expect_contains="56088")
    assert runner._check(case, "答案是 56088", []) is True


def test_check_expect_contains_fail(tmp_path: Path) -> None:
    cfg = _make_config([])
    runner = EvalRunner(cfg, tmp_path)
    case = EvalCase(id="c1", prompt="", expect_contains="56088")
    assert runner._check(case, "答案是 999", []) is False


def test_check_expect_tool_pass(tmp_path: Path) -> None:
    cfg = _make_config([])
    runner = EvalRunner(cfg, tmp_path)
    case = EvalCase(id="c2", prompt="", expect_tool="exec")
    assert runner._check(case, "done", ["exec", "web_search"]) is True


def test_check_expect_tool_fail(tmp_path: Path) -> None:
    cfg = _make_config([])
    runner = EvalRunner(cfg, tmp_path)
    case = EvalCase(id="c2", prompt="", expect_tool="exec")
    assert runner._check(case, "done", ["web_search"]) is False


def test_check_no_constraints_always_pass(tmp_path: Path) -> None:
    cfg = _make_config([])
    runner = EvalRunner(cfg, tmp_path)
    case = EvalCase(id="c3", prompt="hello")
    assert runner._check(case, "", []) is True


# ---------------------------------------------------------------------------
# EvalRunner.should_run_after_turn
# ---------------------------------------------------------------------------

def test_should_run_after_turn_disabled(tmp_path: Path) -> None:
    cfg = _make_config([], run_after_turns=1, enabled=False)
    runner = EvalRunner(cfg, tmp_path)
    assert runner.should_run_after_turn() is False


def test_should_run_after_turn_zero(tmp_path: Path) -> None:
    cfg = _make_config([], run_after_turns=0)
    runner = EvalRunner(cfg, tmp_path)
    assert runner.should_run_after_turn() is False


def test_should_run_after_turn_triggers(tmp_path: Path) -> None:
    cfg = _make_config([], run_after_turns=3)
    runner = EvalRunner(cfg, tmp_path)
    assert runner.should_run_after_turn() is False  # turn 1
    assert runner.should_run_after_turn() is False  # turn 2
    assert runner.should_run_after_turn() is True   # turn 3
    assert runner.should_run_after_turn() is False  # turn 4
    assert runner.should_run_after_turn() is False  # turn 5
    assert runner.should_run_after_turn() is True   # turn 6


# ---------------------------------------------------------------------------
# EvalRunner.run
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_all_pass(tmp_path: Path) -> None:
    suite = [
        {"id": "e001", "prompt": "calc", "expect_contains": "42"},
        {"id": "e002", "prompt": "tool", "expect_tool": "exec"},
    ]
    cfg = _make_config(suite)
    runner = EvalRunner(cfg, tmp_path)

    async def mock_agent(prompt: str):
        if prompt == "calc":
            return "answer is 42", []
        return "done", ["exec"]

    report = await runner.run(mock_agent)
    assert report.total == 2
    assert report.passed == 2
    assert report.failed == 0


@pytest.mark.asyncio
async def test_run_partial_fail(tmp_path: Path) -> None:
    suite = [
        {"id": "e001", "prompt": "calc", "expect_contains": "42"},
        {"id": "e002", "prompt": "tool", "expect_tool": "exec"},
    ]
    cfg = _make_config(suite)
    runner = EvalRunner(cfg, tmp_path)

    async def mock_agent(prompt: str):
        return "wrong", []  # both fail

    report = await runner.run(mock_agent)
    assert report.passed == 0
    assert report.failed == 2


@pytest.mark.asyncio
async def test_run_exception_recorded(tmp_path: Path) -> None:
    suite = [{"id": "e001", "prompt": "boom"}]
    cfg = _make_config(suite)
    runner = EvalRunner(cfg, tmp_path)

    async def mock_agent(prompt: str):
        raise RuntimeError("simulated error")

    report = await runner.run(mock_agent)
    assert report.failed == 1
    assert "simulated error" in report.results[0].error


# ---------------------------------------------------------------------------
# EvalRunner._write_report
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_report_creates_file(tmp_path: Path) -> None:
    suite = [{"id": "e001", "prompt": "hi"}]
    cfg = _make_config(suite)
    runner = EvalRunner(cfg, tmp_path)

    async def mock_agent(prompt: str):
        return "hello", []

    await runner.run(mock_agent)
    bench = tmp_path / "memory" / "BENCHMARK.md"
    assert bench.exists()
    content = bench.read_text()
    assert "Eval Run" in content
    assert "e001" in content


@pytest.mark.asyncio
async def test_write_report_appends(tmp_path: Path) -> None:
    """Running twice should append, preserving history."""
    suite = [{"id": "e001", "prompt": "hi"}]
    cfg = _make_config(suite)
    runner = EvalRunner(cfg, tmp_path)

    async def mock_agent(prompt: str):
        return "hello", []

    await runner.run(mock_agent)
    await runner.run(mock_agent)
    bench = tmp_path / "memory" / "BENCHMARK.md"
    content = bench.read_text()
    assert content.count("Eval Run") == 2
