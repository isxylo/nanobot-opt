#!/usr/bin/env bash
# review.sh — 代码审查脚本（gemini 自动 + codex 手动提示）
# 用法：
#   ./scripts/review.sh              # 审查 uncommitted 变更
#   ./scripts/review.sh HEAD         # 审查最近一次 commit
#   ./scripts/review.sh abc1234      # 审查指定 commit SHA
#
# 作为 pre-push hook：
#   ln -sf ../../scripts/review.sh .git/hooks/pre-push
#
# 手动 codex 交互式审查：
#   codex review --commit HEAD

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"

COMMIT=""
if [[ $# -ge 1 ]]; then
    COMMIT=$(git rev-parse "$1" 2>/dev/null || echo "")
fi

if [[ -n "$COMMIT" ]]; then
    DIFF=$(git diff "$COMMIT"^ "$COMMIT" 2>/dev/null || git show "$COMMIT" --unified=3)
    DIFF_DESC="commit ${COMMIT:0:8}"
else
    DIFF=$(git diff --cached)
    if [[ -z "$DIFF" ]]; then
        DIFF=$(git diff HEAD)
    fi
    DIFF_DESC="uncommitted changes"
fi

if [[ -z "$DIFF" ]]; then
    echo "[review] No changes to review."
    exit 0
fi

DIFF_LINES=$(echo "$DIFF" | wc -l)
echo "[review] Reviewing $DIFF_DESC ($DIFF_LINES lines of diff)"
echo

GEMINI_PROMPT="You are a senior code reviewer. Review the following git diff for:
1. Correctness and logic errors
2. Security vulnerabilities
3. Performance issues
4. Code style and maintainability
5. Missing error handling or edge cases

Be concise. List issues by severity (CRITICAL / WARNING / SUGGESTION).
If no issues found, say 'LGTM' with a brief summary.

--- DIFF ---
$DIFF"

echo "================================================================"
echo "  GEMINI REVIEW"
echo "================================================================"
gemini -p "$GEMINI_PROMPT" 2>&1

echo
echo "================================================================"
echo "[review] Gemini review done."
echo "[review] For interactive codex review: codex review --commit ${COMMIT:-HEAD}"
