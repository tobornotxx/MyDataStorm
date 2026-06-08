"""DataSTORM 主流水线。

论文 Section 3 (Figure 1):
给定关系数据库 D、互联网语料 I 和用户查询 q,
DataSTORM 生成最终研究报告 r。

三个主要组件:
1. Warm-start (Section 3.1): 生成初步报告 r₀ 和初始洞察库 B₀
2. Multi-agent exploration (Section 3.2): 运行 m 层, 产出 B_m 和 t_m
3. Report generation (Section 3.3): 从 B_m 和 t_m 生成最终报告 r
"""

from __future__ import annotations

import logging
from typing import Any

from datastorm.agents.executor import ExecutorAgent
from datastorm.agents.planner import PlannerAgent
from datastorm.config import DataSTORMConfig
from datastorm.database.connector import DatabaseConnector
from datastorm.internet.search import WebSearcher
from datastorm.llm.client import LLMClient
from datastorm.modules.exploration import ExplorationFramework
from datastorm.modules.insight_bank import InsightBank
from datastorm.modules.report import ReportGenerator
from datastorm.modules.warmstart import WarmStartModule
from datastorm.types import FinalReport, Thesis

logger = logging.getLogger(__name__)


class DataSTORMPipeline:
    """DataSTORM 主流水线 (论文 Figure 1)。

    完整的研究工作流:
    Input: 关系数据库 D, 互联网语料 I, 用户查询 q
    Output: 最终研究报告 r (free-text)
    """

    def __init__(self, config: DataSTORMConfig) -> None:
        self._config = config

        # 初始化组件
        self._llm = LLMClient(config.llm)
        self._db = DatabaseConnector(config.database)
        self._searcher = WebSearcher(config.internet)

        # 初始化 agents
        self._planner = PlannerAgent(self._llm, config)
        self._executor = ExecutorAgent(self._llm, self._db, config)

        # 初始化 modules
        self._insight_bank = InsightBank(self._llm, config)

    def run(self, query: str, output_dir: str | None = None) -> FinalReport:
        """执行完整的 DataSTORM 研究流水线。

        论文 Section 3:
        1. Warm-start → r₀, B₀
        2. Multi-agent exploration → B_m, t_m
        3. Report generation → r

        Args:
            query: 用户研究查询 q
            output_dir: 输出目录 (用于保存问题树日志等中间产物)

        Returns:
            最终研究报告 r
        """
        logger.info("=" * 60)
        logger.info("DataSTORM Pipeline Started")
        logger.info("Query: %s", query)
        logger.info("=" * 60)

        # =========================================================
        # Stage 1: Warm-Start with Internet Research (Section 3.1)
        # =========================================================
        logger.info("--- Stage 1: Warm-Start with Internet Research ---")
        warmstart = WarmStartModule(self._llm, self._searcher, self._config)
        warmstart_report, initial_insights = warmstart.run(query)

        # 初始化全局洞察库 B₀
        self._insight_bank.initialize(initial_insights)
        logger.info(
            "Warm-start complete: report=%d chars, B₀=%d insights",
            len(warmstart_report),
            len(initial_insights),
        )

        # =========================================================
        # Stage 2: Multi-Agent Exploration (Section 3.2)
        # =========================================================
        logger.info("--- Stage 2: Multi-Agent Exploration ---")
        exploration = ExplorationFramework(
            llm=self._llm,
            planner=self._planner,
            executor=self._executor,
            searcher=self._searcher,
            insight_bank=self._insight_bank,
            config=self._config,
            output_dir=output_dir,
        )

        final_insights, final_thesis = exploration.run(
            topic=query,
            warmstart_report=warmstart_report,
        )

        # 如果探索未能生成论点, 创建一个默认的
        if final_thesis is None:
            final_thesis = Thesis(
                title=f"Analysis of {query}",
                research_strategy="Synthesize findings into a coherent narrative.",
            )

        logger.info(
            "Exploration complete: B_m=%d insights, thesis='%s'",
            len(final_insights),
            final_thesis.title,
        )

        # =========================================================
        # Stage 3: Final Report Generation (Section 3.3)
        # =========================================================
        logger.info("--- Stage 3: Final Report Generation ---")
        report_generator = ReportGenerator(
            llm=self._llm,
            searcher=self._searcher,
            insight_bank=self._insight_bank,
            config=self._config,
        )

        report = report_generator.generate(
            topic=query,
            thesis=final_thesis,
            warmstart_report=warmstart_report,
        )

        logger.info("=" * 60)
        logger.info("DataSTORM Pipeline Complete")
        logger.info("Report: %d chars", len(report.markdown))
        logger.info("=" * 60)

        return report

    def cleanup(self) -> None:
        """清理资源。"""
        self._db.close()
