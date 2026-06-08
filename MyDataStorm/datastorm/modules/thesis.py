"""论点生成与精炼模块。

论文 Section 3.2.2 (Thesis-Driven Exploration):
- 在第 p 层后, 调用论点生成模块 (Prompt 9)
- 此后每 p 层精炼一次 (Prompt 10)
- 论点可以被 Sharpen / Pivot / Confirm

Dykes (2019): "每个数据故事都需要一个中心洞察"
DataSTORM 将此原则更进一步: 从探索一开始就围绕引导论点组织。
"""

from __future__ import annotations

import json
import logging

from datastorm.config import DataSTORMConfig
from datastorm.llm.client import LLMClient
from datastorm.prompts import renderer, templates
from datastorm.types import Insight, Thesis

logger = logging.getLogger(__name__)


class ThesisModule:
    """论点生成与精炼模块 (论文 Section 3.2.2)。"""

    def __init__(self, llm: LLMClient, config: DataSTORMConfig) -> None:
        self._llm = llm
        self._config = config

    def generate(
        self,
        topic: str,
        insights: list[Insight],
    ) -> Thesis:
        """生成初始论点 (Prompt 9, Table 9)。

        论文 Section 3.2.2:
        在第 p 层后, 从当前洞察库 B_p 生成初始论点 t_p。
        生成至多 3 个候选论点, 选择第一个。

        Args:
            topic: 研究主题
            insights: 当前洞察库

        Returns:
            生成的论点
        """
        context = self._format_findings(insights)

        prompt = renderer.render(
            templates.THESIS_GENERATION,
            db_description=self._config.db_description,
            topic=topic,
            context=context,
        )

        result = self._llm.generate_json(prompt, temperature=0.7)

        # 解析论点候选
        theses = result.get("theses", result.get("thesis_candidates", []))
        logger.debug("Thesis generation: %d candidates returned", len(theses) if theses else 0)
        if not theses:
            # 回退: 尝试其他键
            for key in result:
                if isinstance(result[key], list) and result[key]:
                    theses = result[key]
                    break

        if theses and isinstance(theses[0], dict):
            # 选择第一个论点
            first = theses[0]
            thesis = Thesis(
                title=first.get("thesis", first.get("title", "")),
                research_strategy=first.get("research_strategy", ""),
            )
        else:
            # 回退
            thesis = Thesis(
                title=str(theses[0]) if theses else "Research findings analysis",
                research_strategy="Continue systematic exploration of the database.",
            )

        logger.info("Thesis generated: %s", thesis.title)
        return thesis

    def refine(
        self,
        topic: str,
        current_thesis: Thesis,
        insights: list[Insight],
    ) -> Thesis:
        """精炼现有论点 (Prompt 10, Table 10)。

        论文 Section 3.2.2:
        每 p 层精炼一次, 可以:
        1. Sharpen — 用新的支持证据缩小或深化
        2. Pivot — 转向新发现支持的更好论点
        3. Confirm — 保持不变

        Args:
            topic: 研究主题
            current_thesis: 当前论点
            insights: 当前洞察库

        Returns:
            精炼后的论点
        """
        context = self._format_findings(insights)

        prompt = renderer.render(
            templates.THESIS_REFINEMENT,
            db_description=self._config.db_description,
            topic=topic,
            current_thesis=current_thesis.title,
            current_research_strategy=current_thesis.research_strategy,
            context=context,
        )

        result = self._llm.generate_json(prompt, temperature=0.5)

        refined = Thesis(
            title=result.get("thesis", result.get("title", current_thesis.title)),
            research_strategy=result.get(
                "research_strategy", current_thesis.research_strategy
            ),
        )

        action = result.get("action", "unknown")
        logger.debug("Thesis refinement action: %s", action)
        logger.info("Thesis refined: %s → %s", current_thesis.title, refined.title)
        return refined

    def _format_findings(self, insights: list[Insight]) -> str:
        """格式化洞察列表为发现文本。"""
        if not insights:
            return "No findings available yet."
        lines = []
        for i, insight in enumerate(insights, 1):
            lines.append(
                f"Finding {i} ({insight.source}):\n"
                f"  Question: {insight.question}\n"
                f"  Insight: {insight.content}"
            )
        return "\n\n".join(lines)
