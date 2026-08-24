"""多智能体探索框架 — 基于问题树的改进版。

论文 Section 3.2:
DataSTORM 将探索空间组织为层 (layers), 最多 m 层。

改进 (问题树模式):
- 每一步不再仅基于前一步结果提出新问题,
  而是基于「完整问题树」中所有已提出过的问题,
  同时生成 m 个跟进问题 (follow_up) 和 n 个探索性问题 (exploratory)。
- 问题之间的链和继承关系被记录在 QuestionNode 中,
  并通过 QuestionLogger 持久化到本地文件。
- Stage 3 的报告生成逻辑不变。

单层流程:
1. Layer 1: Planner 从 warmstart report 生成初始问题
2. Layer 2+: Planner 基于完整问题树生成 m 个跟进 + n 个探索问题
3. Executor 执行每个问题 → SQL + 答案
4. 汇总统计嵌入
5. 洞察库更新
6. 每 p 层: 论点生成/精炼
"""

from __future__ import annotations

import logging
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from datastorm.agents.executor import ExecutorAgent
from datastorm.agents.planner import PlannerAgent
from datastorm.config import DataSTORMConfig
from datastorm.internet.search import WebSearcher
from datastorm.llm.client import LLMClient
from datastorm.modules.consistency import QueryConsistencyModule
from datastorm.modules.goal_sufficiency import GoalSufficiencyModule
from datastorm.modules.insight_bank import InsightBank
from datastorm.modules.question_logger import QuestionLogger
from datastorm.modules.statistics import StatisticsModule
from datastorm.modules.thesis import ThesisModule
from datastorm.types import (
    ExecutorResponse,
    ExplorerQuestion,
    Insight,
    QuestionDestination,
    QuestionNode,
    Thesis,
)

logger = logging.getLogger(__name__)


class ExplorationFramework:
    """多智能体探索框架 (论文 Section 3.2) — 问题树改进版。

    协调 Planner、Executor、一致性检测、统计、洞察库、论点模块,
    执行 m 层迭代探索。

    改进: 每一步基于完整问题树而非仅前一步结果提出新问题,
    同时产出跟进问题 (m) 和探索性问题 (n), 并记录完整链。
    """

    def __init__(
        self,
        llm: LLMClient,
        planner: PlannerAgent,
        executor: ExecutorAgent,
        searcher: WebSearcher,
        insight_bank: InsightBank,
        config: DataSTORMConfig,
        output_dir: str | None = None,
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
        self._goal_sufficiency = GoalSufficiencyModule(llm, config)

        self._thesis: Thesis | None = None

        # 问题树: 记录所有已提出的问题及其链和继承关系
        self._question_nodes: list[QuestionNode] = []
        # 问题日志器: 持久化到本地文件
        self._question_logger = QuestionLogger(output_dir=output_dir)

    @property
    def thesis(self) -> Thesis | None:
        return self._thesis

    @property
    def question_nodes(self) -> list[QuestionNode]:
        """获取完整问题树。"""
        return list(self._question_nodes)

    def _grouping_columns(self) -> list[str]:
        """取当前数据源适合做分组轴的分类列（供 skill 引导注入真实列名）。

        由 CsvDatabaseBridge 在加载时按数据结构自动推断；非 bridge 的
        connector 没有这个属性，返回空列表退回通用措辞。
        """
        return list(getattr(self._executor._db, "_grouping_columns", []) or [])

    @staticmethod
    def _classify_outcome(resp: ExecutorResponse) -> str:
        """用规则（非 LLM）判定一次执行的产出质量。

        档 2「run 内轨迹反馈」：把上一层哪些问题有产出、哪些是哑弹反馈给
        下一层 planner。刻意不用 LLM —— 执行失败/空结果/结果过稀疏都能由
        规则准确判定，多一次 LLM 调用没有收益。

        Returns: "productive" | "empty" | "failed" | "sparse"
        """
        answer = (resp.answer or "").strip()
        if not answer or answer.lower().startswith("execution failed"):
            return "failed"
        if "SQL Error" in answer or "Python execution error" in answer:
            return "failed"
        row_count = getattr(resp, "row_count", None)
        if row_count == 0:
            return "empty"
        if isinstance(row_count, int) and 0 < row_count < 2:
            # 单行结果多为退化聚合（例如只取到一个总计），信息量有限
            return "sparse"
        return "productive"

    def _build_outcome_feedback(self, nodes: list[QuestionNode]) -> list[str]:
        """把无产出的问题整理成给下一层 planner 的负面线索。"""
        dead: list[str] = []
        for node in nodes:
            if node.outcome in ("failed", "empty"):
                q = (node.question or "").strip()
                if q:
                    dead.append(f"{q} [{node.outcome}]")
        return dead

    def set_output_dir(self, output_dir: str) -> None:
        """设置输出目录 (用于日志保存)。"""
        self._question_logger.set_output_dir(output_dir)

    def run(
        self,
        topic: str,
        warmstart_report: str = "",
    ) -> tuple[list[Insight], Thesis | None]:
        """执行完整的多智能体探索 (问题树模式)。"""
        m = self._config.exploration.max_layers
        p = self._config.exploration.thesis_generation_interval
        early_stop_patience = self._config.exploration.early_stop_patience

        logger.info(
            "Starting multi-agent exploration (tree mode): %d layers, "
            "follow_up=%d, exploratory=%d, thesis every %d layers, "
            "early_stop_patience=%d",
            m,
            self._config.exploration.follow_up_questions_per_layer,
            self._config.exploration.exploratory_questions_per_layer,
            p,
            early_stop_patience,
        )

        prev_insight_size = 0
        plateau_count = 0
        # 上一层自评得到的"仍缺失的角度", 用于引导本层补充探索
        missing_aspects: list[str] = []
        # 上一层执行后无产出的问题 (规则判定), 用于让本层避开死路（档 2）
        unproductive_questions: list[str] = []

        for layer in range(1, m + 1):
            logger.info("=== Exploration Layer %d/%d ===", layer, m)

            # ── Step 1: Planner 生成探索问题 ──
            if layer == 1:
                questions = self._planner.generate_initial_questions(
                    topic=topic,
                    warmstart_report=warmstart_report,
                )
                question_categories = ["exploratory"] * len(questions)
                question_parents: list[list[str]] = [[] for _ in questions]
            else:
                follow_ups, exploratory = self._planner.generate_tree_based_questions(
                    topic=topic,
                    question_nodes=self._question_nodes,
                    insights=self._insight_bank.insights,
                    thesis=self._thesis,
                    focus_aspects=missing_aspects,
                    unproductive_questions=unproductive_questions,
                )
                questions = follow_ups + exploratory
                question_categories = ["follow_up"] * len(follow_ups) + ["exploratory"] * len(exploratory)
                question_parents = [
                    q.previous_queries if q.previous_queries else []
                    for q in follow_ups
                ] + [[] for _ in exploratory]

            logger.info(
                "Layer %d: Planner generated %d questions (%d follow_up, %d exploratory)",
                layer,
                len(questions),
                sum(1 for c in question_categories if c == "follow_up"),
                sum(1 for c in question_categories if c == "exploratory"),
            )

            # 为每个问题分配唯一 ID 并创建 QuestionNode
            nodes_by_idx: dict[int, QuestionNode] = {}
            for i, q in enumerate(questions):
                node_id = f"q_{layer}_{i:02d}_{uuid.uuid4().hex[:6]}"
                node = QuestionNode(
                    id=node_id,
                    question=q.question,
                    destination=q.destination,
                    layer=layer,
                    parent_ids=question_parents[i],
                    category=question_categories[i],
                )
                nodes_by_idx[i] = node

            # ── Step 2: Executor 并行执行每个问题 ──
            db_questions_with_idx = [
                (idx, q) for idx, q in enumerate(questions)
                if q.destination == QuestionDestination.DATABASE
            ]
            internet_questions_with_idx = [
                (idx, q) for idx, q in enumerate(questions)
                if q.destination != QuestionDestination.DATABASE
            ]

            db_responses: list[ExecutorResponse] = []
            internet_responses: list[ExecutorResponse] = []

            def _run_db_question(question_text: str) -> ExecutorResponse:
                executor = ExecutorAgent(self._llm, self._executor._db, self._config)
                return executor.execute(question_text)

            if db_questions_with_idx:
                with ThreadPoolExecutor(max_workers=len(db_questions_with_idx)) as pool:
                    futures = {
                        pool.submit(_run_db_question, q.question): orig_idx
                        for orig_idx, q in db_questions_with_idx
                    }
                    results_by_orig_idx: dict[int, ExecutorResponse] = {}
                    for future in as_completed(futures):
                        orig_idx = futures[future]
                        try:
                            results_by_orig_idx[orig_idx] = future.result()
                        except Exception as e:
                            logger.error("Executor failed for question idx %d: %s", orig_idx, e)
                            q = questions[orig_idx]
                            results_by_orig_idx[orig_idx] = ExecutorResponse(
                                question=q.question,
                                answer=f"Execution failed: {e}",
                                sql="",
                            )
                    db_responses = [
                        results_by_orig_idx[idx] for idx, _ in db_questions_with_idx
                    ]

            for orig_idx, q in internet_questions_with_idx:
                search_result = self._searcher.search_and_format(q.question)
                resp = ExecutorResponse(
                    question=q.question,
                    answer=search_result,
                    sql="",
                )
                internet_responses.append(resp)
                node = nodes_by_idx[orig_idx]
                node.answer = resp.answer
                node.sql = resp.sql
                self._question_nodes.append(node)

            # ── Step 3-4: 一致性检测 + 跟进执行 ──
            if db_responses:
                consistency_follow_ups = self._consistency.detect_inconsistencies(
                    responses=db_responses,
                    existing_insights=self._insight_bank.insights,
                )

                follow_up_tasks: list[tuple[int, str, str]] = []
                for i, follow_up in enumerate(consistency_follow_ups):
                    if follow_up.follow_up_question:
                        logger.info("Layer %d: Consistency follow-up for query %d", layer, i)
                        context = (
                            f"Original question: {follow_up.original_question}\n"
                            f"Original SQL: {follow_up.original_sql}\n"
                            f"Follow-up instruction: {follow_up.follow_up_question}"
                        )
                        follow_up_tasks.append((i, follow_up.follow_up_question, context))

                if follow_up_tasks:
                    def _run_follow_up(question_text: str, context: str) -> ExecutorResponse:
                        executor = ExecutorAgent(self._llm, self._executor._db, self._config)
                        return executor.execute(question_text, context=context)

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

            # ── Step 5: 嵌入汇总统计 ──
            for i, resp in enumerate(db_responses):
                db_responses[i] = self._statistics.compute_and_embed(resp)

            # ── 更新 QuestionNode (填入执行结果) ──
            for db_i, (orig_idx, _) in enumerate(db_questions_with_idx):
                node = nodes_by_idx[orig_idx]
                resp = db_responses[db_i]
                node.answer = resp.answer
                node.sql = resp.sql
                node.summary_text = resp.summary_text
                node.summary_stats = resp.summary_stats
                node.raw_results = resp.raw_results
                node.row_count = resp.row_count
                # 规则判定产出质量，供下一层 planner 避开已知死路（档 2）
                node.outcome = self._classify_outcome(resp)
                self._question_nodes.append(node)

            # 记录到日志
            all_db_nodes = [nodes_by_idx[idx] for idx, _ in db_questions_with_idx]
            all_internet_nodes = [nodes_by_idx[idx] for idx, _ in internet_questions_with_idx]
            new_nodes = all_db_nodes + all_internet_nodes
            self._question_logger.log_nodes(new_nodes)

            # ── Step 6: 更新洞察库 ──
            all_responses = db_responses + internet_responses
            self._insight_bank.update(
                new_responses=all_responses,
                topic=topic,
                thesis=self._thesis,
                layer=layer,
            )

            logger.info(
                "Layer %d: Insight bank size: %d, Question tree nodes: %d",
                layer, self._insight_bank.size, len(self._question_nodes),
            )

            # ── 轨迹反馈（档 2）：统计本层产出质量，供下一层 planner 参考 ──
            layer_db_nodes = [nodes_by_idx[idx] for idx, _ in db_questions_with_idx]
            unproductive_questions = self._build_outcome_feedback(layer_db_nodes)
            if layer_db_nodes:
                dist = Counter(n.outcome for n in layer_db_nodes)
                logger.info(
                    "Layer %d outcome mix: %s",
                    layer, ", ".join(f"{k}={v}" for k, v in sorted(dist.items())),
                )
                if unproductive_questions:
                    logger.info(
                        "Layer %d: %d unproductive question(s) will be fed back to planner",
                        layer, len(unproductive_questions),
                    )

            # ── 停止决策 ──
            current_size = self._insight_bank.size
            goal_check_on = getattr(self._config.exploration, "goal_sufficiency_check", False)

            if goal_check_on:
                # Goal 满足度自评驱动: agent 自己判断"够不够回答 goal"。
                # 够了 → 提前停 (省 token); 不够 → 记录缺失角度引导下一层补探。
                # max_layers (循环上界) 是唯一硬上限。
                min_layers = getattr(self._config.exploration, "goal_sufficiency_min_layers", 2)
                missing_aspects = []  # 重置; 仅保留本层最新的缺口
                if layer >= min_layers and layer < m:
                    verdict = self._goal_sufficiency.evaluate(
                        goal=topic,
                        insights=self._insight_bank.insights,
                        grouping_columns=self._grouping_columns(),
                    )
                    if verdict.sufficient:
                        logger.info(
                            "Goal-sufficiency: findings answer the goal — stopping early at layer %d",
                            layer,
                        )
                        prev_insight_size = current_size
                        self._question_logger.save()
                        break
                    missing_aspects = verdict.missing_aspects
                    logger.info(
                        "Goal-sufficiency: not yet — continuing with %d focus aspect(s)",
                        len(missing_aspects),
                    )
            elif early_stop_patience > 0 and layer > 1:
                # 回退: 机械 plateau 早停 (洞察库连续 N 层无增长)
                if current_size <= prev_insight_size:
                    plateau_count += 1
                    logger.info(
                        "Insight plateau: %d layers without growth (patience=%d)",
                        plateau_count, early_stop_patience,
                    )
                    if plateau_count >= early_stop_patience:
                        logger.info(
                            "Early stopping at layer %d: no new insights for %d layers",
                            layer, plateau_count,
                        )
                        break
                else:
                    plateau_count = 0
            prev_insight_size = current_size

            # ── 保存问题树日志 ──
            self._question_logger.save()

            # ── Step 7: 论点生成/精炼 ──
            if layer % p == 0:
                if self._thesis is None:
                    self._thesis = self._thesis_module.generate(
                        topic=topic,
                        insights=self._insight_bank.insights,
                    )
                    logger.info("Thesis generated: %s", self._thesis.title)
                else:
                    self._thesis = self._thesis_module.refine(
                        topic=topic,
                        current_thesis=self._thesis,
                        insights=self._insight_bank.insights,
                    )
                    logger.info("Thesis refined: %s", self._thesis.title)

        # 最终保存
        saved_path = self._question_logger.save()
        logger.info("Final question tree saved to %s", saved_path)

        return self._insight_bank.insights, self._thesis
