"""Warm-start 互联网预热研究模块。

论文 Section 3.1:
预热阶段提供下游探索的初始支架。目标不是穷尽式深度互联网研究,
而是获得广泛的主题覆盖并浮现有前景的研究方向。

论文使用 Co-STORM (Jiang et al., 2024) 实现。
本实现提供一个简化版本, 通过 Web 搜索 + LLM 合成来生成初始洞察。
"""

from __future__ import annotations

import logging
import uuid

from datastorm.config import DataSTORMConfig
from datastorm.internet.search import WebSearcher
from datastorm.llm.client import LLMClient
from datastorm.types import Insight

logger = logging.getLogger(__name__)

# 用于合成初始报告的 Prompt
_WARMSTART_SYNTHESIS_PROMPT = """\
You are a research assistant preparing an initial briefing on a topic.
Based on the following web search results, synthesize a structured preliminary report
that covers the key aspects of the topic. This report will guide further database exploration.

Topic: {topic}

Web Search Results:
{search_results}

Write a structured report with:
1. Key background context
2. Important actors/entities involved
3. Major trends and patterns identified
4. Open questions worth investigating further
5. Potential analytical angles

Keep the report focused and concise (500-800 words).
"""

# 用于提取初始洞察的 Prompt
_WARMSTART_INSIGHTS_PROMPT = """\
Based on the following preliminary report on the topic "{topic}", extract a list of key insights
that could guide further database exploration. Each insight should be a self-contained finding
or observation.

Report:
{report}

Return a JSON object with an "insights" array, where each item has:
- "content": the insight text
- "source": "internet"
"""


class WarmStartModule:
    """Warm-start 互联网预热研究模块 (论文 Section 3.1)。

    简化版 Co-STORM 实现:
    1. 基于主题执行多轮 Web 搜索
    2. 合成搜索结果为初始报告 r₀
    3. 从报告中提取洞察列表 B₀
    """

    def __init__(
        self,
        llm: LLMClient,
        searcher: WebSearcher,
        config: DataSTORMConfig,
    ) -> None:
        self._llm = llm
        self._searcher = searcher
        self._config = config

    def run(self, topic: str) -> tuple[str, list[Insight]]:
        """执行预热研究。

        论文 Section 3.1:
        Co-STORM 接收 I 作为输入, 产出洞察列表 B₀ = {b₁, ..., b_l}
        以及初步报告 r₀。

        Args:
            topic: 用户查询/主题

        Returns:
            (r₀: 初步报告, B₀: 初始洞察列表)
        """
        logger.info("Starting warm-start internet research for: %s", topic)

        # 1. 执行多角度搜索
        search_queries = self._generate_search_queries(topic)
        logger.debug("Warmstart search queries: %s", search_queries)
        all_results = []
        for query in search_queries:
            results_text = self._searcher.search_and_format(query, num_results=5)
            all_results.append(results_text)

        combined_results = "\n\n---\n\n".join(all_results)

        # 2. 合成初步报告 r₀
        synthesis_prompt = _WARMSTART_SYNTHESIS_PROMPT.format(
            topic=topic,
            search_results=combined_results[:8000],
        )
        report = self._llm.generate(synthesis_prompt, max_completion_tokens=2048)

        # 3. 提取洞察 B₀
        insights_prompt = _WARMSTART_INSIGHTS_PROMPT.format(
            topic=topic,
            report=report,
        )
        insights_data = self._llm.generate_json(insights_prompt)
        insights = []
        for item in insights_data.get("insights", []):
            insights.append(
                Insight(
                    id=f"ws_{uuid.uuid4().hex[:8]}",
                    content=item.get("content", ""),
                    source="internet",
                    layer=0,
                )
            )

        logger.info(
            "Warm-start complete: report (%d chars), %d insights",
            len(report),
            len(insights),
        )
        logger.debug(
            "Warmstart insights: %s",
            [ins.content[:80] for ins in insights],
        )
        return report, insights

    def _generate_search_queries(self, topic: str) -> list[str]:
        """为主题生成多角度搜索查询。"""
        prompt = (
            f"Generate 3-5 diverse web search queries to research the following topic from "
            f"different angles. Output one query per line.\n\nTopic: {topic}"
        )
        response = self._llm.generate(prompt, temperature=0.7)
        queries = [
            line.strip().lstrip("0123456789.-) ")
            for line in response.strip().split("\n")
            if line.strip() and len(line.strip()) > 5
        ]
        return queries[:5] if queries else [topic]
