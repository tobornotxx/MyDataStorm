"""多智能体探索框架。

论文 Section 3.2:
DataSTORM 将探索空间组织为层 (layers), 最多 m 层。
每一层 i 接收当前全局洞察库 B_{i-1} 和论点 t_{i-1} 作为输入,
产出更新的洞察库 B_i。

单层流程 (Section 3.2.1):
1. Planner 生成探索问题 q_{i,1}, ..., q_{i,n}
2. Executor 执行每个问题 → SQL 查询 s_{i,j} + 答案 a_{i,j}
3. Query Consistency Module 检测不一致 → 跟进查询 q'_{i,j}
4. Executor 执行跟进查询 → 最终 SQL s'_{i,j} + 答案 a'_{i,j}
5. 汇总统计嵌入到 a'_{i,j}
6. 洞察库更新: 合并 a'_{i,j} 到 B_i
7. 每 p 层: 论点生成/精炼
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy

from datastorm.agents.executor import ExecutorAgent
from datastorm.agents.planner import PlannerAgent
from datastorm.config import DataSTORMConfig
from datastorm.internet.search import WebSearcher
from datastorm.llm.client import LLMClient
from datastorm.modules.consistency import QueryConsistencyModule
from datastorm.modules.insight_bank import InsightBank
from datastorm.modules.statistics import StatisticsModule
from datastorm.modules.thesis import ThesisModule
from datastorm.types import ExecutorResponse, Insight, QuestionDestination, Thesis

logger = logging.getLogger(__name__)


class ExplorationFramework:
    """多智能体探索框架 (论文 Section 3.2)。

    协调 Planner、Executor、一致性检测、统计、洞察库、论点模块,
    执行 m 层迭代探索。
    """

    def __init__(
        self,
        llm: LLMClient,
        planner: PlannerAgent,
        executor: ExecutorAgent,
        searcher: WebSearcher,
        insight_bank: InsightBank,
        config: DataSTORMConfig,
    ) -> None:
        self._llm = llm
        self._planner = planner
        self._executor = executor
        self._searcher = searcher
        self._insight_bank = insight_bank
        self._config = config

        self._consistency = QueryConsistencyModule(llm, config)
        self._statistics = StatisticsModule()
        self._thesis_module = ThesisModule(llm, config)

        self._thesis: Thesis | None = None

    @property
    def thesis(self) -> Thesis | None:
        return self._thesis

    def run(
        self,
        topic: str,
        warmstart_report: str = "",
    ) -> tuple[list[Insight], Thesis | None]:
        """执行完整的多智能体探索。

        论文 Section 3.2:
        运行 m 层探索, 返回最终洞察库 B_m 和论点 t_m。

        Args:
            topic: 用户查询/主题
            warmstart_report: 预热阶段报告 r₀

        Returns:
            (B_m: 最终洞察库, t_m: 最终论点)
        """
        m = self._config.exploration.max_layers
        p = self._config.exploration.thesis_generation_interval

        logger.info("Starting multi-agent exploration: %d layers, thesis every %d layers", m, p)

        for layer in range(1, m + 1):
            logger.info("=== Exploration Layer %d/%d ===", layer, m)

            # Step 1: Planner 生成探索问题
            if layer == 1:
                # 论文: 第一层从预热报告 r₀ 生成 (Prompt 11)
                questions = self._planner.generate_initial_questions(
                    topic=topic,
                    warmstart_report=warmstart_report,
                )
            else:
                # 论文: 后续层从洞察库和论点生成 (Prompt 5)
                questions = self._planner.generate_exploration_questions(
                    topic=topic,
                    insights=self._insight_bank.insights,
                    thesis=self._thesis,
                )

            logger.info("Layer %d: Planner generated %d questions", layer, len(questions))

            # Step 2: Executor 并行执行每个问题
            db_responses: list[ExecutorResponse] = []
            internet_responses: list[ExecutorResponse] = []

            def _run_db_question(question: str) -> ExecutorResponse:
                """每个 executor 独立运行（独立 LLM client, 无共享状态）。"""
                executor = ExecutorAgent(self._llm, self._executor._db, self._config)
                return executor.execute(question)

            db_questions = [q for q in questions if q.destination == QuestionDestination.DATABASE]
            internet_questions = [q for q in questions if q.destination != QuestionDestination.DATABASE]

            # 并行执行数据库查询
            if db_questions:
                with ThreadPoolExecutor(max_workers=len(db_questions)) as pool:
                    futures = {
                        pool.submit(_run_db_question, q.question): i
                        for i, q in enumerate(db_questions)
                    }
                    # 按原始顺序收集结果
                    results_by_idx: dict[int, ExecutorResponse] = {}
                    for future in as_completed(futures):
                        idx = futures[future]
                        try:
                            results_by_idx[idx] = future.result()
                        except Exception as e:
                            logger.error("Executor failed for question %d: %s", idx, e)
                            results_by_idx[idx] = ExecutorResponse(
                                question=db_questions[idx].question,
                                answer=f"Execution failed: {e}",
                                sql="",
                            )
                    db_responses = [results_by_idx[i] for i in range(len(db_questions))]

            # 互联网问题（通常为空，因为 Serper 禁用）
            for q in internet_questions:
                search_result = self._searcher.search_and_format(q.question)
                internet_responses.append(
                    ExecutorResponse(
                        question=q.question,
                        answer=search_result,
                        sql="",
                    )
                )

            # Step 3: Query Consistency Detection (论文 Section 3.2.1)
            if db_responses:
                follow_ups = self._consistency.detect_inconsistencies(
                    responses=db_responses,
                    existing_insights=self._insight_bank.insights,
                )

                # Step 4: 并行执行跟进查询
                follow_up_tasks: list[tuple[int, str, str]] = []
                for i, follow_up in enumerate(follow_ups):
                    if follow_up.follow_up_question:
                        logger.info(
                            "Layer %d: Consistency follow-up for query %d", layer, i
                        )
                        context = (
                            f"Original question: {follow_up.original_question}\n"
                            f"Original SQL: {follow_up.original_sql}\n"
                            f"Follow-up instruction: {follow_up.follow_up_question}"
                        )
                        follow_up_tasks.append((i, follow_up.follow_up_question, context))

                if follow_up_tasks:
                    def _run_follow_up(question: str, context: str) -> ExecutorResponse:
                        executor = ExecutorAgent(self._llm, self._executor._db, self._config)
                        return executor.execute(question, context=context)

                    with ThreadPoolExecutor(max_workers=len(follow_up_tasks)) as pool:
                        fu_futures = {
                            pool.submit(_run_follow_up, q, ctx): idx
                            for idx, q, ctx in follow_up_tasks
                        }
                        for future in as_completed(fu_futures):
                            idx = fu_futures[future]
                            try:
                                db_responses[idx] = future.result()
                            except Exception as e:
                                logger.error("Follow-up failed for query %d: %s", idx, e)

            # Step 5: 嵌入汇总统计 (论文 Section 3.2.1, Bottom-up inductive)
            for i, resp in enumerate(db_responses):
                db_responses[i] = self._statistics.compute_and_embed(resp)

            # Step 6: 更新洞察库 (Prompt 8)
            all_responses = db_responses + internet_responses
            self._insight_bank.update(
                new_responses=all_responses,
                topic=topic,
                thesis=self._thesis,
                layer=layer,
            )

            logger.info(
                "Layer %d: Insight bank size: %d", layer, self._insight_bank.size
            )

            # Step 7: 论点生成/精炼 (论文 Section 3.2.2)
            if layer % p == 0:
                if self._thesis is None:
                    # 首次生成论点 (Prompt 9)
                    self._thesis = self._thesis_module.generate(
                        topic=topic,
                        insights=self._insight_bank.insights,
                    )
                    logger.info("Thesis generated: %s", self._thesis.title)
                else:
                    # 精炼论点 (Prompt 10)
                    self._thesis = self._thesis_module.refine(
                        topic=topic,
                        current_thesis=self._thesis,
                        insights=self._insight_bank.insights,
                    )
                    logger.info("Thesis refined: %s", self._thesis.title)

        return self._insight_bank.insights, self._thesis
