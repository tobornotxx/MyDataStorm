"""全局洞察库管理模块。

论文 Section 3.2.1:
每一层结束时, DataSTORM 将新证据 a'_{i,j} 与已有洞察 B_{i-1} 合并,
保留预设最大数量的洞察。低质量的已有洞察可能被更强的新发现取代。
使用 Prompt 8 (Table 8) 进行过滤和选择。
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from datastorm.config import DataSTORMConfig
from datastorm.llm.client import LLMClient
from datastorm.prompts import renderer, templates
from datastorm.types import ExecutorResponse, Insight, Thesis

logger = logging.getLogger(__name__)


class InsightBank:
    """全局洞察库 (论文 Section 3.2.1)。"""

    def __init__(self, llm: LLMClient, config: DataSTORMConfig) -> None:
        self._llm = llm
        self._config = config
        self._insights: list[Insight] = []

    @property
    def insights(self) -> list[Insight]:
        """获取当前所有洞察。"""
        return list(self._insights)

    @property
    def size(self) -> int:
        return len(self._insights)

    def initialize(self, insights: list[Insight]) -> None:
        """初始化洞察库 (用于 warm-start 阶段的 B₀)。"""
        self._insights = list(insights)
        logger.info("Insight bank initialized with %d insights", len(self._insights))

    def update(
        self,
        new_responses: list[ExecutorResponse],
        topic: str,
        thesis: Thesis | None = None,
        layer: int = 0,
    ) -> None:
        """合并新证据到洞察库 (Prompt 8, Table 8)。

        论文 Section 3.2.1:
        在每一层结束时, 将新证据 a'_{i,j} 与 B_{i-1} 合并,
        保留最多 max_insights 个洞察。

        Args:
            new_responses: 本层 Executor 的所有响应
            topic: 研究主题
            thesis: 当前论点
            layer: 当前层编号
        """
        max_insights = self._config.exploration.max_insights

        # 构造候选洞察 (包括已有的和新的)
        candidate_input: dict[str, str] = {}
        # 新洞察 ID → 源响应的映射
        new_id_to_response: dict[str, ExecutorResponse] = {}

        # 已有洞察
        for insight in self._insights:
            candidate_input[insight.id] = insight.content

        # 新洞察 (从响应中提取)
        for resp in new_responses:
            if resp.summary_text:
                insight_id = f"new_{uuid.uuid4().hex[:8]}"
                candidate_input[insight_id] = (
                    f"Question: {resp.question}\n"
                    f"Answer: {resp.summary_text}"
                )
                new_id_to_response[insight_id] = resp

        if not candidate_input:
            return

        logger.debug(
            "InsightBank update: %d existing + %d new candidates, max=%d",
            len(self._insights), len(new_id_to_response), max_insights,
        )

        # 调用 LLM 进行过滤 (Prompt 8)
        prompt = renderer.render(
            templates.INSIGHT_BANK_FILTER,
            max_num_insights=max_insights,
            topic=topic,
            db_description=self._config.db_description,
            thesis=thesis.title if thesis else None,
            input=json.dumps(candidate_input, indent=2),
        )

        result = self._llm.generate_json(prompt, temperature=0.3)

        # 更新洞察库
        selected_insights = []
        for node_id, content in result.items():
            # 查找是否是已有洞察
            existing = next((ins for ins in self._insights if ins.id == node_id), None)
            if existing:
                selected_insights.append(existing)
            else:
                # 新洞察: 通过 ID 映射查找源响应
                source_resp = new_id_to_response.get(node_id)

                selected_insights.append(
                    Insight(
                        id=node_id,
                        content=str(content),
                        source="database" if source_resp else "internet",
                        question=source_resp.question if source_resp else "",
                        sql=source_resp.sql if source_resp else "",
                        answer=source_resp.answer if source_resp else "",
                        layer=layer,
                    )
                )

        self._insights = selected_insights[:max_insights]
        logger.info(
            "Insight bank updated: %d insights (layer %d)", len(self._insights), layer
        )
        logger.debug(
            "InsightBank kept IDs: %s",
            [ins.id for ins in self._insights],
        )

    def get_formatted_insights(self) -> str:
        """获取格式化的洞察文本。"""
        if not self._insights:
            return "No insights collected yet."
        lines = []
        for i, insight in enumerate(self._insights, 1):
            lines.append(f"[{insight.id}] ({insight.source}) {insight.content}")
        return "\n\n".join(lines)

    def get_note_digest(self) -> str:
        """获取用于报告生成的证据摘要。"""
        lines = []
        for i, insight in enumerate(self._insights):
            lines.append(
                f"Evidence [{i + 1}]:\n"
                f"Source: {insight.source}\n"
                f"Question: {insight.question}\n"
                f"SQL: {insight.sql}\n"
                f"Finding: {insight.content}\n"
            )
        return "\n".join(lines)

    def get_valid_ids(self) -> list[int]:
        """获取有效的证据 ID 列表。"""
        return list(range(1, len(self._insights) + 1))
