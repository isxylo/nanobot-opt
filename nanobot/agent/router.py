"""Model router: classify message complexity and select model tier."""

from __future__ import annotations

import re
from dataclasses import dataclass

from loguru import logger


class ComplexityTier:
    FAST = "fast"      # 闲聊、简单问答、单步查询
    NORMAL = "normal"  # 一般任务、搜索、文件操作
    HEAVY = "heavy"    # 复杂推理、多步规划、代码生成


@dataclass
class RouteResult:
    tier: str
    model: str
    reason: str  # 调试日志用


class ModelRouter:
    """
    纯规则模型路由器，零延迟，无额外 LLM 调用。

    根据消息特征（长度、关键词、历史轮数）决定使用哪个 tier 的模型。
    保守策略：宁可 normal 不误降级到 fast。
    """

    FAST_MAX_CHARS = 30

    HEAVY_MIN_CHARS = 300
    HEAVY_KEYWORDS = {
        "重构", "架构", "设计方案", "深度分析", "系统设计", "全面",
        "调试", "排查", "性能优化", "安全", "实现一个", "帮我写",
        "refactor", "architecture", "implement", "debug", "analyze",
        "optimize", "design", "comprehensive", "step by step",
    }
    HEAVY_PATTERNS = [
        r"```",
        r"步骤|流程|计划|方案|规划",
        r"第[一二三四五六七八九十]步",
        r"step \d|\d\. ",
        r"\bapi\b|\bsdk\b|\bsql\b",
    ]

    def __init__(self, tier_models: dict[str, str]):
        self.tier_models = tier_models
        self._heavy_re = re.compile("|".join(self.HEAVY_PATTERNS), re.IGNORECASE)

    # Tier upgrade chain
    _UPGRADE = {
        ComplexityTier.FAST: ComplexityTier.NORMAL,
        ComplexityTier.NORMAL: ComplexityTier.HEAVY,
    }

    def upgrade(self, tier: str) -> RouteResult | None:
        """Return next-tier RouteResult, or None if already at HEAVY."""
        next_tier = self._UPGRADE.get(tier)
        if not next_tier:
            return None
        model = self.tier_models.get(next_tier, self.tier_models.get(ComplexityTier.NORMAL, ""))
        return RouteResult(tier=next_tier, model=model, reason=f"upgraded_from={tier}")

    def route(self, message: str, history_len: int) -> RouteResult:
        tier, reason = self._classify(message, history_len)
        model = self.tier_models.get(tier, self.tier_models.get(ComplexityTier.NORMAL, ""))
        logger.debug("ModelRouter: tier={} reason={} model={}", tier, reason, model)
        return RouteResult(tier=tier, model=model, reason=reason)

    def _classify(self, message: str, history_len: int) -> tuple[str, str]:
        msg = message.strip()
        msg_lower = msg.lower()

        # HEAVY 优先检测（避免误判为 FAST）
        if len(msg) >= self.HEAVY_MIN_CHARS:
            return ComplexityTier.HEAVY, f"len={len(msg)}>={self.HEAVY_MIN_CHARS}"

        if any(kw in msg_lower for kw in self.HEAVY_KEYWORDS):
            matched = next(kw for kw in self.HEAVY_KEYWORDS if kw in msg_lower)
            return ComplexityTier.HEAVY, f"keyword={matched!r}"

        if self._heavy_re.search(msg):
            return ComplexityTier.HEAVY, "pattern_match"

        # FAST：短消息且不触发 HEAVY
        if len(msg) <= self.FAST_MAX_CHARS:
            return ComplexityTier.FAST, f"len={len(msg)}<={self.FAST_MAX_CHARS}"

        return ComplexityTier.NORMAL, "default"


def build_router_from_config(defaults: object) -> ModelRouter | None:
    """
    从 AgentDefaults 构建 ModelRouter。
    routing_enabled=False 或配置的 tier 模型少于 2 个时返回 None。
    """
    if not getattr(defaults, "routing_enabled", False):
        return None

    fast = getattr(defaults, "model_fast", None)
    normal = getattr(defaults, "model_normal", None)
    heavy = getattr(defaults, "model_heavy", None)

    configured = {k: v for k, v in {
        ComplexityTier.FAST: fast,
        ComplexityTier.NORMAL: normal,
        ComplexityTier.HEAVY: heavy,
    }.items() if v}

    if len(configured) < 2:
        logger.warning(
            "ModelRouter: routing_enabled=true but fewer than 2 tier models configured, disabling"
        )
        return None

    # 未配置的 tier 继承相邻 tier
    if ComplexityTier.NORMAL not in configured:
        configured[ComplexityTier.NORMAL] = (
            configured.get(ComplexityTier.HEAVY) or configured.get(ComplexityTier.FAST)
        )
    if ComplexityTier.FAST not in configured:
        configured[ComplexityTier.FAST] = configured[ComplexityTier.NORMAL]
    if ComplexityTier.HEAVY not in configured:
        configured[ComplexityTier.HEAVY] = configured[ComplexityTier.NORMAL]

    logger.info(
        "ModelRouter enabled: fast={} normal={} heavy={}",
        configured[ComplexityTier.FAST],
        configured[ComplexityTier.NORMAL],
        configured[ComplexityTier.HEAVY],
    )
    return ModelRouter(configured)
