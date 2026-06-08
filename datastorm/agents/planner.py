"""Planner Agent — 高层探索问题生成。

论文 Section 3.2.1:
Planner 负责生成高层探索问题, 然后交由 Executor 翻译为正式查询。
- 第一层: 从预热报告 r₀ 生成问题 (Prompt 11)
- 后续层: 从全局洞察库 B_{i-1} 和论点 t_{i-1} 生成问题 (Prompt 5)
- 树模式 (改进): 基于完整问题树, 同时生成跟进问题(m)和探索性问题(n)

Planner 只生成高层问题, 不涉及 SQL 翻译。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from datastorm.config import DataSTORMConfig
from datastorm.llm.client import LLMClient
from datastorm.prompts import renderer, templates
from datastorm.types import ExplorerQuestion, Insight, QuestionDestination, QuestionNode, Thesis

logger = logging.getLogger(__name__)


class PlannerAgent:
    """Planner Agent (论文 Section 3.2.1)。

    生成高层探索问题, 路由到 database 或 internet。
    """

    def __init__(self, llm: LLMClient, config: DataSTORMConfig) -> None:
        self._llm = llm
        self._config = config

    def generate_initial_questions(
        self,
        topic: str,
        warmstart_report: str = "",
    ) -> list[ExplorerQuestion]:
        """生成第一层探索问题 (Prompt 11, Table 11)。

        论文 Section 3.2.1: 第一层问题从预热报告 r₀ 生成。

        Args:
            topic: 用户查询/主题
            warmstart_report: 预热阶段生成的报告 r₀

        Returns:
            探索问题列表
        """
        max_q = self._config.exploration.first_layer_max_questions

        prompt = renderer.render(
            templates.INITIAL_QUESTIONS_GENERATION,
            topic=topic,
            db_description=self._config.db_description,
            num_questions=max_q,
            article=warmstart_report if warmstart_report else None,
        )

        # 使用 json_mode 强制结构化输出
        response = self._llm.generate_json(prompt, temperature=0.7)
        questions = self._parse_exploration_questions(response)
        # 安全截断：保证不超过 max_q（即使 LLM 返回更多）
        if len(questions) > max_q:
            logger.warning("Planner returned %d questions, truncating to %d", len(questions), max_q)
            questions = questions[:max_q]
        logger.debug(
            "Initial questions generated: %s",
            [(q.question[:80], q.destination.value) for q in questions],
        )
        return questions

    def generate_exploration_questions(
        self,
        topic: str,
        insights: list[Insight],
        thesis: Thesis | None = None,
        dialogue_turns: str = "",
    ) -> list[ExplorerQuestion]:
        """生成后续层探索问题 (Prompt 5, Table 5)。

        论文 Section 3.2.1: 后续层问题从洞察库 B_{i-1} 和论点 t_{i-1} 生成。

        Args:
            topic: 用户查询/主题
            insights: 当前全局洞察库
            thesis: 当前论点 (可能为 None)
            dialogue_turns: 对话历史

        Returns:
            探索问题列表
        """
        max_q = self._config.exploration.subsequent_layer_max_questions

        # 格式化洞察为文本
        global_insights = self._format_insights(insights)

        prompt = renderer.render(
            templates.EXPLORATION_QUESTIONS_GENERATION,
            max_questions=max_q,
            db_description=self._config.db_description,
            global_insights=global_insights,
            dialogue_turns=dialogue_turns,
            topic=topic,
            thesis=thesis.title if thesis else None,
            research_strategy=thesis.research_strategy if thesis else None,
        )

        response = self._llm.generate_json(prompt, temperature=0.7)
        questions = self._parse_exploration_questions(response)
        # 安全截断：保证不超过 max_q（即使 LLM 返回更多）
        if len(questions) > max_q:
            logger.warning("Planner returned %d questions, truncating to %d", len(questions), max_q)
            questions = questions[:max_q]
        logger.debug(
            "Exploration questions generated: %s",
            [(q.question[:80], q.destination.value) for q in questions],
        )
        return questions

    def generate_tree_based_questions(
        self,
        topic: str,
        question_nodes: list[QuestionNode],
        insights: list[Insight],
        thesis: Thesis | None = None,
    ) -> tuple[list[ExplorerQuestion], list[ExplorerQuestion]]:
        """基于完整问题树生成双模式问题 (树模式改进)。

        每一步同时产出:
        - m 个跟进问题 (follow-up): 基于已有问题深入挖掘
        - n 个探索性问题 (exploratory): 全新方向
        m/n 为下限, 每类上限 5 个。

        Args:
            topic: 用户查询/主题
            question_nodes: 完整的问题树节点列表
            insights: 当前全局洞察库
            thesis: 当前论点 (可能为 None)

        Returns:
            (follow_up_questions, exploratory_questions)
        """
        m = self._config.exploration.follow_up_questions_per_layer
        n = self._config.exploration.exploratory_questions_per_layer

        question_tree_text = self._format_question_tree(question_nodes)
        global_insights = self._format_insights(insights)

        prompt = renderer.render(
            templates.TREE_BASED_QUESTION_GENERATION,
            m=m,
            n=n,
            db_description=self._config.db_description,
            topic=topic,
            question_tree=question_tree_text if question_tree_text else "No previous questions yet.",
            global_insights=global_insights,
            thesis=thesis.title if thesis else None,
            research_strategy=thesis.research_strategy if thesis else None,
        )

        response = self._llm.generate_json(prompt, temperature=0.7)

        follow_ups = self._parse_tree_questions(
            response.get("follow_up_questions", []),
            default_parent_ids=None,
        )
        exploratory = self._parse_tree_questions(
            response.get("exploratory_questions", []),
            default_parent_ids=[],
        )

        # 上限截断: 每类不超过 5 个
        MAX_PER_TYPE = 5
        if len(follow_ups) > MAX_PER_TYPE:
            follow_ups = follow_ups[:MAX_PER_TYPE]
        if len(exploratory) > MAX_PER_TYPE:
            exploratory = exploratory[:MAX_PER_TYPE]

        # 下限检查
        if len(follow_ups) < m:
            logger.warning(
                "Tree-based: only %d follow-up questions (lower bound was %d)",
                len(follow_ups), m,
            )
        if len(exploratory) < n:
            logger.warning(
                "Tree-based: only %d exploratory questions (lower bound was %d)",
                len(exploratory), n,
            )

        logger.info(
            "Tree-based: %d follow-up + %d exploratory questions (bounds: m=%d n=%d)",
            len(follow_ups), len(exploratory), m, n,
        )
        return follow_ups, exploratory

    def _format_question_tree(self, nodes: list[QuestionNode]) -> str:
        """将问题树格式化为文本，展示问题的链和继承关系。"""
        if not nodes:
            return "No previous questions yet."

        lines = []
        by_layer: dict[int, list[QuestionNode]] = {}
        for node in nodes:
            by_layer.setdefault(node.layer, []).append(node)

        for layer in sorted(by_layer.keys()):
            lines.append(f"=== Layer {layer} ===")
            for node in by_layer[layer]:
                parent_info = f" [derived from: {', '.join(node.parent_ids)}]" if node.parent_ids else " [exploratory/root]"
                lines.append(f"  Node [{node.id}] ({node.category}){parent_info}")
                lines.append(f"    Q: {node.question}")
                if node.answer:
                    answer_preview = node.answer[:300] + "..." if len(node.answer) > 300 else node.answer
                    lines.append(f"    A: {answer_preview}")
                else:
                    lines.append(f"    A: [not yet answered]")
                lines.append("")
        return "\n".join(lines)

    def _parse_tree_questions(
        self,
        questions_data: list[dict[str, Any]],
        default_parent_ids: list[str] | None = None,
    ) -> list[ExplorerQuestion]:
        """解析树模式问题列表。"""
        questions = []
        for q in questions_data:
            if not isinstance(q, dict):
                continue
            dest_str = q.get("destination", "database")
            try:
                dest = QuestionDestination(dest_str)
            except ValueError:
                dest = QuestionDestination.DATABASE

            parent_ids = q.get("parent_ids", None)
            if parent_ids is None:
                parent_ids = default_parent_ids or []

            questions.append(
                ExplorerQuestion(
                    question=q.get("question", ""),
                    destination=dest,
                    previous_queries=parent_ids,
                )
            )
        return questions

    def _format_insights(self, insights: list[Insight]) -> str:
        """将洞察列表格式化为文本。"""
        if not insights:
            return "No insights collected yet."
        lines = []
        for i, insight in enumerate(insights, 1):
            lines.append(f"{i}. [{insight.source}] {insight.content}")
        return "\n".join(lines)

    def _parse_initial_questions(self, response: str) -> list[ExplorerQuestion]:
        """解析初始问题生成的响应。"""
        questions = []

        # 尝试 JSON 解析
        try:
            data = json.loads(response)
            if isinstance(data, dict) and "questions" in data:
                for q in data["questions"]:
                    questions.append(
                        ExplorerQuestion(
                            question=q.get("question", q) if isinstance(q, dict) else str(q),
                            destination=QuestionDestination(
                                q.get("destination", "database")
                            )
                            if isinstance(q, dict)
                            else QuestionDestination.DATABASE,
                        )
                    )
                return questions
        except (json.JSONDecodeError, ValueError):
            pass

        # 回退: 按行解析编号列表
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # 移除编号前缀 (1., 2., -, * 等)
            cleaned = re.sub(r"^[\d]+[.)\]]\s*", "", line)
            cleaned = re.sub(r"^[-*•]\s*", "", cleaned)
            if len(cleaned) > 10:  # 至少要有实质内容
                questions.append(
                    ExplorerQuestion(
                        question=cleaned,
                        destination=QuestionDestination.DATABASE,
                    )
                )

        return questions

    def _parse_exploration_questions(self, data: dict[str, Any]) -> list[ExplorerQuestion]:
        """解析探索问题生成的 JSON 响应。"""
        questions = []
        for q in data.get("questions", []):
            if isinstance(q, dict):
                dest_str = q.get("destination", "database")
                try:
                    dest = QuestionDestination(dest_str)
                except ValueError:
                    dest = QuestionDestination.DATABASE

                questions.append(
                    ExplorerQuestion(
                        question=q.get("question", ""),
                        destination=dest,
                    )
                )
        return questions
