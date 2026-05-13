"""报告生成模块。

论文 Section 3.3 (Final Report Generation):
DataSTORM 使用分阶段编辑流水线, 从最终洞察库 B_m 和论点 t_m 生成报告 r。

五个阶段 (Figure 2):
Stage A: 大纲生成 — 每个章节指定叙事目的、所需证据、关键要点 (Prompt 12)
Stage B: 章节草稿 — 基于策划的证据子集独立起草 (Prompt 13)
Stage C: 引用验证 — 逐句事实核查, 标记未被引用源支持的声明 (Prompt 14)
Stage D: 章节修订 — 根据批评修订章节 (Prompt 15)
Stage E: 最终润色 — 汇总并润色为最终报告 (Prompt 16)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from datastorm.config import DataSTORMConfig
from datastorm.internet.search import WebSearcher
from datastorm.llm.client import LLMClient
from datastorm.modules.insight_bank import InsightBank
from datastorm.prompts import renderer, templates
from datastorm.types import (
    CitationCheck,
    DraftedSection,
    FinalReport,
    Insight,
    ReportOutline,
    SectionSpec,
    Thesis,
)

logger = logging.getLogger(__name__)


class ReportGenerator:
    """报告生成流水线 (论文 Section 3.3, Figure 2)。"""

    def __init__(
        self,
        llm: LLMClient,
        searcher: WebSearcher,
        insight_bank: InsightBank,
        config: DataSTORMConfig,
    ) -> None:
        self._llm = llm
        self._searcher = searcher
        self._insight_bank = insight_bank
        self._config = config

    def generate(
        self,
        topic: str,
        thesis: Thesis,
        warmstart_report: str = "",
    ) -> FinalReport:
        """执行完整的报告生成流水线。

        Args:
            topic: 研究主题
            thesis: 最终论点 t_m
            warmstart_report: 预热报告 (可选补充上下文)

        Returns:
            最终报告
        """
        logger.info("Starting report generation pipeline")

        # 生成标题包
        title_package = self._generate_title_package(topic, thesis)

        # Stage A: 大纲生成 (Prompt 12)
        outline = self._stage_a_outline(topic, thesis, title_package, warmstart_report)
        logger.info("Stage A complete: %d sections planned", len(outline.sections))
        logger.debug(
            "Stage A outline: lede=%r, sections=%s",
            outline.lede_strategy[:100],
            [(s.section_id, s.heading) for s in outline.sections],
        )

        # Stage B: 章节草稿 (Prompt 13)
        drafted_sections = []
        for section_spec in outline.sections:
            web_context = self._fetch_web_context(section_spec.web_queries)
            draft = self._stage_b_draft(topic, thesis, title_package, section_spec, web_context)
            drafted_sections.append(draft)
            logger.debug(
                "Stage B: section %s drafted (%d chars, %d citations)",
                draft.section_id, len(draft.markdown), len(draft.used_citations),
            )
        logger.info("Stage B complete: %d sections drafted", len(drafted_sections))

        # Stage C: 引用验证 (Prompt 14)
        all_criticisms: list[list[CitationCheck]] = []
        for draft in drafted_sections:
            criticisms = self._stage_c_citation_grounding(draft)
            all_criticisms.append(criticisms)
            failed = sum(1 for c in criticisms if not c.is_entailed)
            logger.info(
                "Stage C: Section %s — %d/%d citations failed",
                draft.section_id, failed, len(criticisms),
            )
            if failed:
                logger.debug(
                    "Stage C: Section %s failed sentences: %s",
                    draft.section_id,
                    [(c.sentence[:60], c.issue[:80]) for c in criticisms if not c.is_entailed],
                )

        # Stage D: 章节修订 (Prompt 15)
        revised_sections = []
        for i, (draft, criticisms) in enumerate(zip(drafted_sections, all_criticisms)):
            failed_checks = [c for c in criticisms if not c.is_entailed]
            if failed_checks:
                section_spec = outline.sections[i]
                web_context = self._fetch_web_context(section_spec.web_queries)
                revised = self._stage_d_revision(
                    topic, thesis, title_package, section_spec, draft, failed_checks, web_context
                )
                revised_sections.append(revised)
            else:
                revised_sections.append(draft)
        logger.info("Stage D complete: sections revised")

        # Stage E: 最终润色 (Prompt 16)
        report = self._stage_e_polish(
            topic, thesis, title_package, outline, revised_sections
        )
        logger.info("Stage E complete: final report generated")

        return report

    def _generate_title_package(
        self, topic: str, thesis: Thesis
    ) -> dict[str, str]:
        """生成标题包 (title, subtitle, editorial_angle)。"""
        prompt = (
            f"Generate a publication-ready title package for an analytical report.\n"
            f"Topic: {topic}\n"
            f"Thesis: {thesis.title}\n\n"
            f"Return JSON with: title, subtitle, editorial_angle"
        )
        result = self._llm.generate_json(prompt, temperature=0.7)
        return {
            "title": result.get("title", topic),
            "subtitle": result.get("subtitle", thesis.title),
            "editorial_angle": result.get("editorial_angle", thesis.research_strategy),
        }

    def _stage_a_outline(
        self,
        topic: str,
        thesis: Thesis,
        title_package: dict[str, str],
        warmstart_report: str,
    ) -> ReportOutline:
        """Stage A: 大纲生成 (Prompt 12, Table 12)。"""
        prompt = renderer.render(
            templates.OUTLINE_GENERATION,
            topic=topic,
            thesis=thesis.title,
            title=title_package["title"],
            subtitle=title_package["subtitle"],
            editorial_angle=title_package["editorial_angle"],
            note_digest=self._insight_bank.get_note_digest(),
            warmstart_text=warmstart_report[:3000] if warmstart_report else "",
            valid_ids=self._insight_bank.get_valid_ids(),
        )

        result = self._llm.generate_json(
            prompt, model=self._config.llm.report_model, temperature=0.5
        )

        outline = ReportOutline(
            lede_strategy=result.get("lede_strategy", ""),
            key_findings=result.get("key_findings", []),
            closing_strategy=result.get("closing_strategy", ""),
        )

        for s in result.get("sections", []):
            outline.sections.append(
                SectionSpec(
                    section_id=s.get("section_id", ""),
                    heading=s.get("heading", ""),
                    purpose=s.get("purpose", ""),
                    must_include_evidence_ids=s.get("must_include_evidence_ids", []),
                    key_points=s.get("key_points", []),
                    storytelling_moves=s.get("storytelling_moves", []),
                    web_queries=s.get("web_queries", []),
                )
            )

        return outline

    def _stage_b_draft(
        self,
        topic: str,
        thesis: Thesis,
        title_package: dict[str, str],
        section_spec: SectionSpec,
        web_context: str,
    ) -> DraftedSection:
        """Stage B: 章节草稿 (Prompt 13, Table 13)。"""
        # 构造核心证据包
        core_packet = self._build_evidence_packet(section_spec.must_include_evidence_ids)
        allowed = self._insight_bank.get_valid_ids()

        prompt = renderer.render(
            templates.SECTION_DRAFT,
            topic=topic,
            thesis=thesis.title,
            report_title=title_package["title"],
            section_id=section_spec.section_id,
            heading=section_spec.heading,
            purpose=section_spec.purpose,
            key_points=json.dumps(section_spec.key_points),
            storytelling_moves=json.dumps(section_spec.storytelling_moves),
            allowed=allowed,
            core_packet=core_packet,
            web_packet=web_context,
            target_words=self._config.report.section_target_words,
        )

        result = self._llm.generate_json(
            prompt, model=self._config.llm.report_model, max_completion_tokens=4096
        )

        return DraftedSection(
            section_id=result.get("section_id", section_spec.section_id),
            heading=result.get("heading", section_spec.heading),
            markdown=result.get("section_markdown", ""),
            used_citations=result.get("used_citations", []),
        )

    def _stage_c_citation_grounding(
        self, draft: DraftedSection
    ) -> list[CitationCheck]:
        """Stage C: 引用验证 (Prompt 14, Table 14)。

        论文 Section 3.3:
        章节在引用边界处分块, LLM 检查每个声明是否被引用源支持。
        """
        # 按引用边界分割句子
        sentences = self._split_by_citations(draft.markdown)
        checks = []

        for sentence, citation_ids in sentences:
            if not citation_ids:
                continue  # 无引用的句子跳过

            # 获取引用源
            sources = self._get_citation_sources(citation_ids)
            if not sources:
                continue

            prompt = renderer.render(
                templates.CITATION_GROUNDING,
                sentence=sentence,
                sources=sources,
            )

            result = self._llm.generate_json(
                prompt, model=self._config.llm.report_model, temperature=0.0
            )

            checks.append(
                CitationCheck(
                    sentence=sentence,
                    is_entailed=result.get("is_entailed", True),
                    issue=result.get("issue", ""),
                )
            )

        return checks

    def _stage_d_revision(
        self,
        topic: str,
        thesis: Thesis,
        title_package: dict[str, str],
        section_spec: SectionSpec,
        draft: DraftedSection,
        failed_checks: list[CitationCheck],
        web_context: str,
    ) -> DraftedSection:
        """Stage D: 章节修订 (Prompt 15, Table 15)。"""
        core_packet = self._build_evidence_packet(section_spec.must_include_evidence_ids)
        allowed = self._insight_bank.get_valid_ids()

        criticisms = json.dumps(
            [
                {"original_sentence": c.sentence, "criticism": c.issue}
                for c in failed_checks
            ],
            indent=2,
        )

        prompt = renderer.render(
            templates.SECTION_REVISION,
            topic=topic,
            thesis=thesis.title,
            report_title=title_package["title"],
            section_id=section_spec.section_id,
            heading=section_spec.heading,
            purpose=section_spec.purpose,
            key_points=json.dumps(section_spec.key_points),
            storytelling_moves=json.dumps(section_spec.storytelling_moves),
            allowed=allowed,
            core_packet=core_packet,
            web_packet=web_context,
            previous_draft=draft.markdown,
            criticisms=criticisms,
        )

        result = self._llm.generate_json(
            prompt, model=self._config.llm.report_model, max_completion_tokens=4096
        )

        return DraftedSection(
            section_id=result.get("section_id", draft.section_id),
            heading=result.get("heading", draft.heading),
            markdown=result.get("section_markdown", draft.markdown),
            used_citations=result.get("used_citations", draft.used_citations),
        )

    def _stage_e_polish(
        self,
        topic: str,
        thesis: Thesis,
        title_package: dict[str, str],
        outline: ReportOutline,
        sections: list[DraftedSection],
    ) -> FinalReport:
        """Stage E: 最终润色 (Prompt 16, Table 16)。"""
        # 组装草稿 markdown
        draft_parts = []
        for section in sections:
            draft_parts.append(f"## {section.heading}\n\n{section.markdown}")
        draft_markdown = "\n\n".join(draft_parts)

        # 收集所有使用的引用
        all_citations = set()
        for section in sections:
            all_citations.update(section.used_citations)

        # 构造大纲 JSON
        plan_json = json.dumps(
            {
                "lede_strategy": outline.lede_strategy,
                "closing_strategy": outline.closing_strategy,
                "sections": [
                    {"heading": s.heading, "purpose": s.purpose}
                    for s in outline.sections
                ],
            },
            indent=2,
        )

        prompt = renderer.render(
            templates.FINAL_POLISH,
            topic=topic,
            thesis=thesis.title,
            title=title_package["title"],
            subtitle=title_package["subtitle"],
            plan_json=plan_json,
            allowed_citations=sorted(all_citations),
            draft_markdown=draft_markdown,
            target_total_words=self._config.report.total_target_words,
        )

        result = self._llm.generate_json(
            prompt, model=self._config.llm.report_model, max_completion_tokens=8192
        )

        # 构建参考文献
        references = self._build_references(all_citations)

        return FinalReport(
            title=title_package["title"],
            subtitle=title_package["subtitle"],
            thesis=thesis,
            markdown=result.get("report_markdown", draft_markdown),
            references=references,
        )

    # ---- 辅助方法 ----

    def _build_evidence_packet(self, evidence_ids: list[int]) -> str:
        """构建证据包。"""
        insights = self._insight_bank.insights
        lines = []
        for eid in evidence_ids:
            idx = eid - 1  # 转为 0-based
            if 0 <= idx < len(insights):
                insight = insights[idx]
                lines.append(
                    f"[{eid}] Source: {insight.source}\n"
                    f"Question: {insight.question}\n"
                    f"SQL: {insight.sql}\n"
                    f"Finding: {insight.content}\n"
                )
        return "\n".join(lines) if lines else "No evidence available."

    def _fetch_web_context(self, queries: list[str]) -> str:
        """获取补充 web 上下文。"""
        if not queries:
            return ""
        parts = []
        for query in queries[: self._config.report.max_web_queries_per_section]:
            result = self._searcher.search_and_format(query, num_results=3)
            parts.append(result)
        return "\n\n".join(parts)

    def _split_by_citations(
        self, markdown: str
    ) -> list[tuple[str, list[int]]]:
        """按引用边界分割 markdown 文本。"""
        # 按句子分割, 提取引用号
        sentences: list[tuple[str, list[int]]] = []

        # 按句号/分号/换行分割
        raw_sentences = re.split(r"(?<=[.!?])\s+", markdown)

        for sentence in raw_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            # 提取引用号 [N]
            citation_ids = [int(m) for m in re.findall(r"\[(\d+)\]", sentence)]
            sentences.append((sentence, citation_ids))

        return sentences

    def _get_citation_sources(self, citation_ids: list[int]) -> str:
        """获取引用源文本。"""
        insights = self._insight_bank.insights
        sources = []
        for cid in citation_ids:
            idx = cid - 1
            if 0 <= idx < len(insights):
                insight = insights[idx]
                sources.append(
                    f"[{cid}] {insight.content}\n"
                    f"SQL: {insight.sql}\n"
                    f"Answer: {insight.answer}"
                )
        return "\n\n".join(sources) if sources else ""

    def _build_references(self, citation_ids: set[int]) -> list[dict[str, str]]:
        """构建参考文献列表。"""
        insights = self._insight_bank.insights
        refs = []
        for cid in sorted(citation_ids):
            idx = cid - 1
            if 0 <= idx < len(insights):
                insight = insights[idx]
                refs.append(
                    {
                        "id": str(cid),
                        "source": insight.source,
                        "question": insight.question,
                        "sql": insight.sql,
                    }
                )
        return refs
