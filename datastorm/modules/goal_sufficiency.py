"""Goal 满足度自评模块。

让 agent 在每层探索结束后, 拿研究 goal 回看自己已产出的发现, 判断:
  - 这些发现是否已经足以回答 goal? (sufficient)
  - 如果不够, 还缺哪些角度/维度? (missing_aspects)

判据完全来自 goal 本身 ("我回答了我的问题吗 / 还缺什么"), 不引用任何
ground truth, 因此对任意数据集/任务都成立, 不构成对 benchmark 的过拟合。

用途 (见 ExplorationFramework):
  - sufficient=True  → 提前停止探索, 省 token
  - sufficient=False → 把 missing_aspects 喂给下一层 planner 引导补充探索
max_layers 始终是硬上限。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from datastorm.config import DataSTORMConfig
from datastorm.llm.client import LLMClient
from datastorm.skills import describe_categorical_columns, get_skill_package
from datastorm.types import Insight

logger = logging.getLogger(__name__)

# 本模块消费的 skill（顺序即注入顺序）。
# v8 曾把这两段引导硬编码在 prompt 里，其中：
#   - DIMENSION DECOMPOSITION 写死了 "time, category, agent/owner, priority/group"
#     —— 后三个是 ServiceNow 工单维度名；
#   - ANTI-PREMATURE-STOP 的举例含 "TTR per agent over time"
#     —— TTR (time-to-resolution) 是 ITSM 专有术语。
# 两者都会把自评器往工单场景上带，迁移到其他领域时失效。现改为从 skill 包
# 加载，维度名在运行时由当前数据集的真实分类列注入。
_SUFFICIENCY_SKILL_IDS = [
    "coverage-multidim-selfcheck",
    "anti-premature-stop",
]


@dataclass
class SufficiencyVerdict:
    """自评结果。"""

    sufficient: bool
    missing_aspects: list[str] = field(default_factory=list)
    reasoning: str = ""


class GoalSufficiencyModule:
    """基于 goal 的发现充分性自评。"""

    def __init__(self, llm: LLMClient, config: DataSTORMConfig) -> None:
        self._llm = llm
        self._config = config

    def _render_skill_guidance(self, grouping_columns: list[str] | None) -> str:
        """从 skill 包渲染自评引导，维度名用本数据集的真实分类列填充。"""
        try:
            pkg = get_skill_package()
            return pkg.render_guidance(
                _SUFFICIENCY_SKILL_IDS,
                header="COVERAGE AUDIT RULES",
                categorical_columns=describe_categorical_columns(grouping_columns or []),
            )
        except Exception as e:
            logger.warning("Sufficiency skill render failed (%s) — continuing without", e)
            return ""

    def evaluate(
        self,
        goal: str,
        insights: list[Insight],
        grouping_columns: list[str] | None = None,
    ) -> SufficiencyVerdict:
        """评估当前发现是否足以回答 goal。

        Args:
            goal: 研究目标 (即 topic / 用户查询)
            insights: 截至当前层的全局洞察库
            grouping_columns: 本数据集适合做分组轴的分类列名。用于把
                「按维度分解」的引导具体化到真实列名；为 None 时退回
                数据无关的通用措辞。

        Returns:
            SufficiencyVerdict(sufficient, missing_aspects, reasoning)
            出错时保守返回 sufficient=False (倾向继续探索, 不误停)。
        """
        findings = self._format_findings(insights)
        if not findings.strip() or findings.startswith("No findings"):
            return SufficiencyVerdict(sufficient=False, missing_aspects=[], reasoning="no findings yet")

        prompt = (
            "You are auditing whether a data analysis has gathered enough evidence "
            "to fully answer its research goal.\n\n"
            f"RESEARCH GOAL:\n{goal}\n\n"
            f"FINDINGS SO FAR:\n{findings}\n\n"
            "Judge ONLY against the goal itself — do not assume any external 'correct' "
            "answer. Ask yourself:\n"
            "- Do these findings, taken together, actually answer the goal?\n"
            "- Has every distinct angle the goal asks about been investigated "
            "(e.g. if the goal mentions a trend, was change-over-time examined; if it "
            "mentions a specific subgroup or time window, was that subgroup/window "
            "actually located and analyzed rather than assumed)?\n"
            "- Are there obvious follow-up questions a careful analyst would still ask "
            "before concluding?\n"
            f"{self._render_skill_guidance(grouping_columns)}\n\n"
            "Respond in JSON:\n"
            "{\n"
            '  "sufficient": true | false,\n'
            '  "reasoning": "<one or two sentences>",\n'
            '  "missing_aspects": ["<specific angle still unexplored>", ...]\n'
            "}\n"
            "When sufficient=true, missing_aspects must be empty."
        )

        try:
            result = self._llm.generate_json(prompt, scenario="goal_sufficiency", temperature=0.3)
        except Exception as e:
            logger.warning("Goal-sufficiency eval failed (%s); defaulting to continue", e)
            return SufficiencyVerdict(sufficient=False, missing_aspects=[], reasoning="eval error")

        sufficient = bool(result.get("sufficient", False))
        missing = result.get("missing_aspects", []) or []
        if not isinstance(missing, list):
            missing = [str(missing)]
        missing = [str(m).strip() for m in missing if str(m).strip()]
        reasoning = str(result.get("reasoning", "")).strip()

        logger.info(
            "Goal-sufficiency: sufficient=%s, %d missing aspect(s)%s",
            sufficient, len(missing),
            (": " + "; ".join(missing[:3])) if missing else "",
        )
        return SufficiencyVerdict(sufficient=sufficient, missing_aspects=missing, reasoning=reasoning)

    def _format_findings(self, insights: list[Insight]) -> str:
        if not insights:
            return "No findings available yet."
        lines = []
        for i, insight in enumerate(insights, 1):
            lines.append(
                f"Finding {i}:\n"
                f"  Question: {insight.question}\n"
                f"  Insight: {insight.content}"
            )
        return "\n\n".join(lines)
