"""Model router: classify message complexity and select model tier."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from loguru import logger


class FailoverReason(str, Enum):
    """Reason a model call failed, used to decide fallback behavior."""
    RATE_LIMIT = "rate_limit"        # 429 / 速率限制，可降级重试
    OVERLOADED = "overloaded"        # 503 / 服务过载，可降级重试
    AUTH = "auth"                    # 401/403 认证失败，不应重试
    CONTEXT_OVERFLOW = "context_overflow"  # context 超长，不走降级链
    LENGTH = "length"                # finish_reason=length，升级到更大模型
    ERROR = "error"                  # 其他错误，尝试回退默认模型
    UNKNOWN = "unknown"


_RATE_LIMIT_PATTERNS = [
    r"rate.?limit", r"429", r"too many requests", r"request limit",
    r"quota", r"throttl",
]
_OVERLOADED_PATTERNS = [
    r"overload", r"503", r"service unavailable", r"capacity",
    r"server.?busy", r"try again later",
]
_AUTH_PATTERNS = [
    r"401", r"403", r"unauthorized", r"forbidden",
    r"invalid.?api.?key", r"authentication", r"access.?denied",
]
_CONTEXT_PATTERNS = [
    r"context.?(?:length|window|limit|overflow)",
    r"maximum.?(?:context|token)",
    r"prompt.?too.?long",
    r"reduce.?(?:the.?length|your.?input)",
    r"tokens?.+exceed",
]

_RATE_RE = re.compile("|".join(_RATE_LIMIT_PATTERNS), re.IGNORECASE)
_OVERLOAD_RE = re.compile("|".join(_OVERLOADED_PATTERNS), re.IGNORECASE)
_AUTH_RE = re.compile("|".join(_AUTH_PATTERNS), re.IGNORECASE)
_CONTEXT_RE = re.compile("|".join(_CONTEXT_PATTERNS), re.IGNORECASE)


def classify_error(error: Exception | str) -> FailoverReason:
    """
    从异常或错误消息中推断失败原因。

    用于 loop.py 决定是否降级、升级或直接失败。
    """
    msg = str(error).lower()
    # 检查 HTTP 状态码（优先）
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if status:
        s = int(status)
        if s == 429:
            return FailoverReason.RATE_LIMIT
        if s in (401, 403):
            return FailoverReason.AUTH
        if s == 503:
            return FailoverReason.OVERLOADED
    # 文本匹配
    if _CONTEXT_RE.search(msg):
        return FailoverReason.CONTEXT_OVERFLOW
    if _AUTH_RE.search(msg):
        return FailoverReason.AUTH
    if _RATE_RE.search(msg):
        return FailoverReason.RATE_LIMIT
    if _OVERLOAD_RE.search(msg):
        return FailoverReason.OVERLOADED
    return FailoverReason.UNKNOWN


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

    # 明确是简单确认/情感的词，必须整词匹配（避免「好」匹配「中午好」）
    FAST_EXPLICIT = {
        "好的", "嗯", "嗯嗯", "哦", "哦哦", "哈哈", "哈", "呵呵",
        "谢谢", "谢谢你", "感谢", "多谢", "thanks", "thank you", "thx",
        "ok", "okay", "收到", "明白", "明白了", "知道了",
        "再见", "拜拜", "bye", "hi", "hello", "你好", "早安", "晚安",
        "是的", "没错", "exactly", "yes", "yep", "no", "nope",
    }
    # 含这些词说明需要思考，即使消息很短也不走 FAST
    FAST_BLOCKERS = {
        # 疑问词
        "什么", "怎么", "怎样", "如何", "为什么", "为啥",
        "有没有", "能不能", "可不可以", "是不是", "有什么",
        "哪", "哪些", "哪个", "几", "多少", "多久", "多大",
        "吗", "呢", "？", "?",
        "what", "how", "why", "which", "when", "where", "who",
        # 指令动词
        "帮我", "帮你", "给我", "告诉我", "说说", "讲讲", "介绍",
        "分析", "解释", "解释下", "写", "做", "生成", "创建", "实现",
        "查", "搜", "找", "看看", "评价", "比较", "推荐", "建议",
        "explain", "describe", "write", "make", "create", "find",
        "search", "show", "list", "compare", "recommend",
    }

    def __init__(self, tier_models: dict[str, str]):
        self.tier_models = tier_models
        self._heavy_re = re.compile("|".join(self.HEAVY_PATTERNS), re.IGNORECASE)
        # Pre-compile word-boundary regexes for keyword sets to avoid substring false-positives.
        # Chinese words use (?<![\u4e00-\u9fff]) boundaries; English words use \b.
        self._heavy_kw_re = self._build_kw_re(self.HEAVY_KEYWORDS, strict_end=False)
        self._fast_blocker_re = self._build_kw_re(self.FAST_BLOCKERS, strict_end=False)
        self._fast_explicit_re = self._build_kw_re(self.FAST_EXPLICIT, strict_end=True)

    @staticmethod
    def _build_kw_re(keywords: set[str], strict_end: bool = True) -> re.Pattern:
        """Build a regex that matches any keyword with word/CJK boundaries.

        strict_end=True (default): also guard end boundary (for FAST_EXPLICIT/FAST_BLOCKERS).
        strict_end=False: only guard start boundary (for HEAVY_KEYWORDS — a heavy trigger
          like '帮我写' should fire even when followed by more CJK characters).
        """
        parts = []
        for kw in keywords:
            if re.search(r'[\u4e00-\u9fff]', kw):
                if len(kw) == 1:
                    end = r'(?![\u4e00-\u9fff])' if strict_end else ''
                    parts.append(rf'(?<![\u4e00-\u9fff]){re.escape(kw)}{end}')
                else:
                    end = r'(?![\u4e00-\u9fff])' if strict_end else ''
                    parts.append(rf'{re.escape(kw)}{end}')
            else:
                parts.append(rf'\b{re.escape(kw)}\b')
        return re.compile('|'.join(parts), re.IGNORECASE)

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

    def should_fallback(self, reason: FailoverReason) -> bool:
        """
        根据错误原因判断是否应该走降级/重试逻辑。

        - CONTEXT_OVERFLOW：不降级，上下文超长换模型也没用
        - AUTH：不降级，认证失败换模型也无效
        - 其他：允许降级回默认模型
        """
        return reason not in (FailoverReason.CONTEXT_OVERFLOW, FailoverReason.AUTH)

    def route(self, message: str, history_len: int) -> RouteResult:
        tier, reason = self._classify(message, history_len)
        model = self.tier_models.get(tier, self.tier_models.get(ComplexityTier.NORMAL, ""))
        logger.debug("ModelRouter: tier={} reason={} model={}", tier, reason, model)
        return RouteResult(tier=tier, model=model, reason=reason)

    # 历史轮数超过此值时，FAST 自动升为 NORMAL（上下文复杂度上升）
    FAST_MAX_HISTORY = 10

    def _classify(self, message: str, history_len: int) -> tuple[str, str]:
        msg = message.strip()
        msg_lower = msg.lower()

        # HEAVY 优先检测（避免误判为 FAST）
        if len(msg) >= self.HEAVY_MIN_CHARS:
            return ComplexityTier.HEAVY, f"len={len(msg)}>={self.HEAVY_MIN_CHARS}"

        m = self._heavy_kw_re.search(msg_lower)
        if m:
            return ComplexityTier.HEAVY, f"keyword={m.group()!r}"

        if self._heavy_re.search(msg):
            return ComplexityTier.HEAVY, "pattern_match"

        # FAST：必须明确是简单确认/情感词，且不含任何思考性词汇
        if self._fast_blocker_re.search(msg_lower):
            return ComplexityTier.NORMAL, "fast_blocked"

        m = self._fast_explicit_re.search(msg_lower)
        if m:
            matched = m.group()
            # 历史轮数过多时升为 NORMAL（上下文复杂，fast 模型容易丢失上下文）
            if history_len > self.FAST_MAX_HISTORY:
                return ComplexityTier.NORMAL, f"explicit={matched!r},history_upgrade={history_len}"
            return ComplexityTier.FAST, f"explicit={matched!r}"

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
