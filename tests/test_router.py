"""Tests for model router: classify_error and routing logic."""
from nanobot.agent.router import (
    ComplexityTier,
    FailoverReason,
    ModelRouter,
    classify_error,
)
import pytest


# ---------------------------------------------------------------------------
# classify_error
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("msg,expected", [
    ("rate limit exceeded 429", FailoverReason.RATE_LIMIT),
    ("429 Too Many Requests", FailoverReason.RATE_LIMIT),
    ("quota exceeded", FailoverReason.RATE_LIMIT),
    ("401 unauthorized", FailoverReason.AUTH),
    ("403 forbidden", FailoverReason.AUTH),
    ("invalid api key", FailoverReason.AUTH),
    ("context length exceeded reduce your input", FailoverReason.CONTEXT_OVERFLOW),
    ("prompt too long", FailoverReason.CONTEXT_OVERFLOW),
    ("maximum context window", FailoverReason.CONTEXT_OVERFLOW),
    ("service overloaded 503", FailoverReason.OVERLOADED),
    ("server busy try again later", FailoverReason.OVERLOADED),
    ("some random unknown error", FailoverReason.UNKNOWN),
])
def test_classify_error_from_string(msg, expected):
    assert classify_error(msg) == expected


def test_classify_error_from_exception():
    err = Exception("429 rate limit")
    assert classify_error(err) == FailoverReason.RATE_LIMIT


def test_classify_error_prefers_context_over_auth():
    # context overflow should take priority over auth-like words
    assert classify_error("context length exceeded") == FailoverReason.CONTEXT_OVERFLOW


def test_classify_error_status_code_attribute():
    class FakeHTTPError(Exception):
        status_code = 429
    assert classify_error(FakeHTTPError("error")) == FailoverReason.RATE_LIMIT


def test_classify_error_status_401():
    class FakeHTTPError(Exception):
        status_code = 401
    assert classify_error(FakeHTTPError("error")) == FailoverReason.AUTH


# ---------------------------------------------------------------------------
# should_fallback
# ---------------------------------------------------------------------------

def test_should_fallback_rate_limit():
    router = ModelRouter({ComplexityTier.FAST: 'f', ComplexityTier.NORMAL: 'n', ComplexityTier.HEAVY: 'h'})
    assert router.should_fallback(FailoverReason.RATE_LIMIT) is True


def test_should_fallback_overloaded():
    router = ModelRouter({ComplexityTier.FAST: 'f', ComplexityTier.NORMAL: 'n', ComplexityTier.HEAVY: 'h'})
    assert router.should_fallback(FailoverReason.OVERLOADED) is True


def test_should_not_fallback_auth():
    router = ModelRouter({ComplexityTier.FAST: 'f', ComplexityTier.NORMAL: 'n', ComplexityTier.HEAVY: 'h'})
    assert router.should_fallback(FailoverReason.AUTH) is False


def test_should_not_fallback_context_overflow():
    router = ModelRouter({ComplexityTier.FAST: 'f', ComplexityTier.NORMAL: 'n', ComplexityTier.HEAVY: 'h'})
    assert router.should_fallback(FailoverReason.CONTEXT_OVERFLOW) is False


# ---------------------------------------------------------------------------
# ModelRouter.route
# ---------------------------------------------------------------------------

@pytest.fixture
def router():
    return ModelRouter({
        ComplexityTier.FAST: 'fast-model',
        ComplexityTier.NORMAL: 'normal-model',
        ComplexityTier.HEAVY: 'heavy-model',
    })


# FAST tier
@pytest.mark.parametrize("msg", ["好的", "谢谢", "嗯嗯", "收到", "是的", "ok", "明白了"])
def test_route_fast_explicit(router, msg):
    assert router.route(msg, 0).tier == ComplexityTier.FAST


# FAST blocked by question words
@pytest.mark.parametrize("msg", [
    "有没有什么可以改进的", "你觉得怎么样", "知道吗", "这样对吗", "中午好", "你是谁",
])
def test_route_not_fast_when_blocked(router, msg):
    assert router.route(msg, 0).tier != ComplexityTier.FAST


# HEAVY tier
@pytest.mark.parametrize("msg", [
    "重构这个系统架构",
    "帮我分析一下这段代码的性能优化方案",
    "a" * 300,  # length trigger
    "implement a full REST API with authentication",
])
def test_route_heavy(router, msg):
    assert router.route(msg, 0).tier == ComplexityTier.HEAVY


# NORMAL default
@pytest.mark.parametrize("msg", ["中午好", "今天天气怎么样", "帮我查一下长沙天气"])
def test_route_normal_default(router, msg):
    assert router.route(msg, 0).tier == ComplexityTier.NORMAL


# History-based upgrade: FAST -> NORMAL when history is long
def test_route_fast_upgrades_to_normal_with_long_history(router):
    # Short message that would normally be FAST
    result_short_history = router.route("好的", 0)
    result_long_history = router.route("好的", 15)
    assert result_short_history.tier == ComplexityTier.FAST
    assert result_long_history.tier == ComplexityTier.NORMAL


# upgrade()
def test_upgrade_fast_to_normal(router):
    result = router.upgrade(ComplexityTier.FAST)
    assert result is not None
    assert result.tier == ComplexityTier.NORMAL
    assert result.model == 'normal-model'


def test_upgrade_normal_to_heavy(router):
    result = router.upgrade(ComplexityTier.NORMAL)
    assert result is not None
    assert result.tier == ComplexityTier.HEAVY


def test_upgrade_heavy_returns_none(router):
    assert router.upgrade(ComplexityTier.HEAVY) is None
