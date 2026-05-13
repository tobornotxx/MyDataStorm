"""DataSTORM 评估模块。

对应论文 Section 4.2 和 Appendix C。

实现四类评估:
1. Reference-Induced Criteria (RIC): 参考诱导标准评估 (Prompt 17, 18)
2. Insight Recall: 基于原子洞察的覆盖率评估 (Prompt 19)
3. Insight Attribution: ACLED-derived 洞察归因 (Prompt 20)
4. InsightBench Compatibility: 对 InsightBench 的答案评分与摘要 (Prompt 21, 23)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from datastorm.config import DataSTORMConfig
from datastorm.llm.client import LLMClient
from datastorm.prompts import renderer, templates

logger = logging.getLogger(__name__)


@dataclass
class Criterion:
    """参考诱导评估标准。"""

    name: str
    description: str


@dataclass
class CriterionScore:
    """单个评估标准得分。"""

    name: str
    score: float
    explanation: str = ""


@dataclass
class RICEvaluationResult:
    """Reference-Induced Criteria 评估结果。"""

    criteria: list[Criterion] = field(default_factory=list)
    scores: list[CriterionScore] = field(default_factory=list)

    @property
    def average_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.score for s in self.scores) / len(self.scores)


class DataSTORMEvaluator:
    """DataSTORM 评估器 (论文 Section 4.2)。"""

    def __init__(self, llm: LLMClient, config: DataSTORMConfig) -> None:
        self._llm = llm
        self._config = config

    # -------------------------------------------------------------------------
    # Reference-Induced Criteria (RIC) 评估
    # -------------------------------------------------------------------------

    def generate_reference_criteria(
        self,
        task_prompt: str,
        reference_article: str,
    ) -> list[Criterion]:
        """从参考文章生成评估标准 (Prompt 17, Table 17)。

        论文 Section 4.2:
        通过 LLM 从参考文章中提取评估标准, 然后评分生成文章。
        """
        prompt = renderer.render(
            templates.REFERENCE_CRITERIA_GENERATION,
            task_prompt=task_prompt,
            reference_article=reference_article,
        )

        result = self._llm.generate_json(prompt, temperature=0.0)
        criteria = []
        for c in result.get("criteria", []):
            criteria.append(
                Criterion(
                    name=c.get("name", ""),
                    description=c.get("description", ""),
                )
            )
        return criteria

    def grade_against_criteria(
        self,
        task_prompt: str,
        criteria: list[Criterion],
        generated_article: str,
        score_reminder: str = "",
    ) -> list[CriterionScore]:
        """根据标准评分生成文章 (Prompt 18, Table 18)。"""
        criteria_json = json.dumps(
            [c.__dict__ for c in criteria], indent=2, ensure_ascii=False
        )

        prompt = renderer.render(
            templates.REFERENCE_CRITERIA_GRADING,
            score_reminder=score_reminder,
            task_prompt=task_prompt,
            criteria=criteria_json,
            generated_article=generated_article,
        )

        result = self._llm.generate_json(prompt, temperature=0.0)
        scores = []
        for s in result.get("criterion_scores", []):
            scores.append(
                CriterionScore(
                    name=s.get("name", ""),
                    score=float(s.get("score", 0.0)),
                    explanation=s.get("explanation", ""),
                )
            )
        return scores

    def evaluate_with_reference(
        self,
        task_prompt: str,
        reference_article: str,
        generated_article: str,
    ) -> RICEvaluationResult:
        """完整 RIC 评估流程。"""
        criteria = self.generate_reference_criteria(task_prompt, reference_article)
        scores = self.grade_against_criteria(
            task_prompt, criteria, generated_article
        )
        return RICEvaluationResult(criteria=criteria, scores=scores)

    # -------------------------------------------------------------------------
    # Insight Recall / 原子洞察分解
    # -------------------------------------------------------------------------

    def atomic_breakdown(self, article: str) -> list[str]:
        """将文章分解为原子洞察 (Prompt 19, Table 19)。"""
        prompt = renderer.render(templates.ATOMIC_BREAKDOWN, article=article)
        response = self._llm.generate(prompt, temperature=0.0)

        insights = []
        for line in response.split("\n"):
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^[\d]+[.)\]]\s*", "", line)
            line = re.sub(r"^[-*•]\s*", "", line)
            if len(line) > 20:
                insights.append(line)
        return insights

    def compute_insight_recall(
        self,
        reference_article: str,
        generated_article: str,
    ) -> dict[str, Any]:
        """计算洞察召回率。

        将参考文章分解为原子洞察, 然后判断生成文章覆盖了多少。
        """
        reference_insights = self.atomic_breakdown(reference_article)
        generated_insights = self.atomic_breakdown(generated_article)

        # 使用 LLM 判断覆盖关系
        covered = []
        for ref in reference_insights:
            prompt = (
                f"Determine whether the generated article covers the reference insight.\n"
                f"Reference insight: {ref}\n\n"
                f"Generated article insights:\n{json.dumps(generated_insights, indent=2)}\n\n"
                f"Return JSON: {{\"covered\": true/false, \"matched_insight\": \"...\"}}"
            )
            result = self._llm.generate_json(prompt, temperature=0.0)
            if result.get("covered", False):
                covered.append({"reference": ref, "matched": result.get("matched_insight", "")})

        recall = len(covered) / len(reference_insights) if reference_insights else 0.0
        return {
            "reference_insights": reference_insights,
            "generated_insights": generated_insights,
            "covered": covered,
            "recall": recall,
        }

    # -------------------------------------------------------------------------
    # Insight Attribution
    # -------------------------------------------------------------------------

    def classify_acled_attribution(
        self,
        article_topic: str,
        evidence: str,
    ) -> dict[str, Any]:
        """判断证据是否源自 ACLED 数据 (Prompt 20, Table 20)。"""
        prompt = renderer.render(
            templates.INSIGHT_ATTRIBUTION,
            article_topic=article_topic,
            evidence=evidence,
        )
        prompt += "\n\nReturn JSON: {\"is_acled_derived\": true/false, \"explanation\": \"...\"}"

        return self._llm.generate_json(prompt, temperature=0.0)

    def compute_acled_attribution_rate(
        self,
        article_topic: str,
        article: str,
    ) -> dict[str, Any]:
        """计算文章中 ACLED-derived 洞察比例。"""
        insights = self.atomic_breakdown(article)
        classifications = []
        for insight in insights:
            cls = self.classify_acled_attribution(article_topic, insight)
            classifications.append({"insight": insight, **cls})

        acled_count = sum(1 for c in classifications if c.get("is_acled_derived", False))
        rate = acled_count / len(classifications) if classifications else 0.0
        return {
            "classifications": classifications,
            "acled_count": acled_count,
            "total": len(classifications),
            "rate": rate,
        }

    # -------------------------------------------------------------------------
    # InsightBench 兼容评估
    # -------------------------------------------------------------------------

    def insightbench_score(self, answer: str, gt_answer: str) -> int:
        """InsightBench 1-10 评分 (Prompt 21, Table 21)。"""
        prompt = renderer.render(
            templates.INSIGHTBENCH_EVAL,
            answer=answer,
            gt_answer=gt_answer,
        )
        response = self._llm.generate(prompt, temperature=0.0, max_completion_tokens=64)

        match = re.search(r"<rating>(\d+)</rating>", response)
        if match:
            return int(match.group(1))
        # 回退: 提取第一个数字
        match = re.search(r"\d+", response)
        return int(match.group(0)) if match else 0

    def insightbench_summary(self, insights: list[str]) -> str:
        """InsightBench 洞察摘要 (Prompt 23, Table 23)。"""
        prompt = renderer.render(
            templates.INSIGHTBENCH_SUMMARY,
            insights="\n".join(f"- {i}" for i in insights),
        )
        return self._llm.generate(prompt, temperature=0.3)
