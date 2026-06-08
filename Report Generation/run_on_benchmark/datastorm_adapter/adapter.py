"""DataSTORM → InsightBench 适配器。

把 MyDataStorm 的 DataSTORMPipeline 包装成 InsightBench 兼容的 Agent 接口。

InsightBench 期望的接口：
    agent.get_insights(dataset_csv_path, ..., return_summary=True)
    → (pred_insights: list[str], pred_summary: str)

适配策略：
1. 用 CsvDatabaseBridge 替换 PostgreSQL DatabaseConnector
2. 禁用 Serper 网络搜索（InsightBench 是纯 CSV 场景）
3. 手动组装 DataSTORMPipeline 各组件，绕过其 __init__ 中的自动初始化
4. 从 FinalReport 中提取 insights list 和 summary
"""

from __future__ import annotations

import logging
import re
import sys
import os
import json as _json
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 把 MyDataStorm 加入 Python 路径
# 目录结构:
#   D:\DataAgents\
#     MyDataStorm\               ← datastorm 包在这里
#     Report Generation\
#       run_on_benchmark\
#         datastorm_adapter\
#           adapter.py           ← 本文件
_ADAPTER_DIR    = os.path.dirname(os.path.abspath(__file__))   # .../datastorm_adapter
_RUN_ON_BENCH   = os.path.dirname(_ADAPTER_DIR)                # .../run_on_benchmark
_REPORT_GEN_DIR = os.path.dirname(_RUN_ON_BENCH)               # .../Report Generation
_DATAAGENTS_DIR = os.path.dirname(_REPORT_GEN_DIR)             # D:\DataAgents
_MYDATASTORM_DIR = os.path.join(_DATAAGENTS_DIR, "MyDataStorm")

for _path in [_MYDATASTORM_DIR, _RUN_ON_BENCH]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from datastorm.agents.executor import ExecutorAgent
from datastorm.agents.planner import PlannerAgent
from datastorm.config import (
    DataSTORMConfig,
    DatabaseConfig,
    ExplorationConfig,
    InternetConfig,
    LLMConfig,
    ReportConfig,
)
from datastorm.llm.client import LLMClient
from datastorm.internet.search import WebSearcher
from datastorm.modules.exploration import ExplorationFramework
from datastorm.modules.insight_bank import InsightBank
from datastorm.modules.report import ReportGenerator
from datastorm.modules.warmstart import WarmStartModule
from datastorm.pipeline import DataSTORMPipeline
from datastorm.types import FinalReport, Insight, Thesis

from datastorm_adapter.csv_db_bridge import CsvDatabaseBridge


class DataStormAdapter:
    """将 MyDataStorm 包装为 InsightBench 兼容的 Agent。

    参数：
        model_name:      LLM 模型名称（None=使用 llm_config.json 中的配置）
        max_layers:      探索层数（默认 3，比论文的 5 少以节省 token）
        openai_api_key:  OpenAI API key（None=使用 llm_config.json 中的配置）
        savedir:         结果保存目录（可选）
        verbose:         是否输出详细日志
    """

    def __init__(
        self,
        model_name: str | None = None,
        max_layers: int = 3,
        questions_per_layer: int = 2,
        follow_up_per_layer: int | None = None,
        exploratory_per_layer: int | None = None,
        openai_api_key: str | None = None,
        api_base: str | None = None,
        savedir: str | None = None,
        verbose: bool = False,
    ) -> None:
        self.model_name = model_name
        self.max_layers = max_layers
        self.questions_per_layer = questions_per_layer
        self.follow_up_per_layer = follow_up_per_layer if follow_up_per_layer is not None else questions_per_layer
        self.exploratory_per_layer = exploratory_per_layer if exploratory_per_layer is not None else questions_per_layer
        self.savedir = savedir

        if verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

        # 构造 LLMConfig：只覆盖显式传入的参数，其余由 llm_config.json + 环境变量提供
        llm_kwargs: dict = {}
        if model_name:
            llm_kwargs["exploration_model"] = model_name
            llm_kwargs["report_model"] = model_name
        if openai_api_key:
            llm_kwargs["api_key"] = openai_api_key
        if api_base:
            llm_kwargs["api_base"] = api_base

        # 构造配置（不含 DB URL，DB 由 CsvDatabaseBridge 提供）
        self._base_config = DataSTORMConfig(
            llm=LLMConfig(**llm_kwargs),
            database=DatabaseConfig(url="sqlite:///:memory:", database_type="SQLite"),
            internet=InternetConfig(serper_api_key=""),  # 禁用网络搜索
            exploration=ExplorationConfig(
                max_layers=max_layers,
                first_layer_max_questions=questions_per_layer,
                subsequent_layer_max_questions=questions_per_layer,
                follow_up_questions_per_layer=self.follow_up_per_layer,
                exploratory_questions_per_layer=self.exploratory_per_layer,
                executor_max_turns=5,
            ),
            report=ReportConfig(
                section_target_words=400,
                total_target_words=2000,
                max_web_queries_per_section=0,  # 禁用报告阶段的 web 查询
                skip_citation_check=True,  # 跳过引用验证，减少 API 请求
            ),
        )
        self._llm = LLMClient(self._base_config.llm)

    # ------------------------------------------------------------------
    # 主接口：与 InsightBench Agent.get_insights() 签名一致
    # ------------------------------------------------------------------

    def get_insights(
        self,
        dataset_csv_path: str,
        user_dataset_csv_path: str | None = None,
        goal: str = "Find interesting trends and patterns in this dataset",
        dataset_description: str = "",
        return_summary: bool = True,
    ) -> tuple[list[str], str] | list[str]:
        """在 CSV 数据集上运行 DataSTORM，返回 InsightBench 格式的结果。

        Args:
            dataset_csv_path:      主 CSV 文件路径
            user_dataset_csv_path: 可选的第二张表 CSV 路径
            goal:                  分析目标（来自 InsightBench metadata.goal）
            dataset_description:   数据集描述（来自 InsightBench metadata.dataset_description）
            return_summary:        是否同时返回 summary 字符串

        Returns:
            return_summary=True:  (pred_insights, pred_summary)
            return_summary=False: pred_insights
        """
        # 1. 构建查询字符串
        query = self._build_query(goal, dataset_description)
        logger.info("DataStormAdapter: query=%r", query[:200])

        # 2. 建立 CSV → SQLite 桥接
        bridge = CsvDatabaseBridge(
            csv_path=dataset_csv_path,
            table_name="main_table",
            user_csv_path=user_dataset_csv_path,
            user_table_name="user_table",
        )

        # 3. 手动组装 pipeline（注入 bridge 替代 PostgreSQL connector）
        config = self._base_config
        # 把完整的 schema 信息注入 db_description，这样：
        # - Executor 不需要浪费轮次调 get_tables() / retrieve_tables_details()
        # - Planner 能看到表结构来生成更精准的问题
        schema_context = bridge.get_schema_context()
        config.db_description = (
            f"{dataset_description}\n\n"
            f"DATABASE SCHEMA (SQLite):\n{schema_context}\n\n"
            f"AVAILABLE PYTHON PACKAGES for execute_python_from_sql:\n"
            f"  numpy (np), pandas (pd), scipy, scipy.stats (stats), scipy.optimize,\n"
            f"  sklearn (scikit-learn), statsmodels, sympy, networkx, xgboost, lightgbm,\n"
            f"  polars, duckdb, lifelines, pingouin, ruptures, category_encoders, imblearn\n"
            f"  The sql_results variable is a pandas DataFrame.\n"
            f"NOTE: You already have the full schema above. "
            f"Skip get_tables() and retrieve_tables_details() — go directly to execute_sql()."
        )

        pipeline = self._build_pipeline(config, bridge)

        try:
            report: FinalReport = pipeline.run(query, output_dir=self.savedir)
        finally:
            bridge.close()

        # 保存完整报告到 savedir（中间产物）
        if self.savedir:
            self._save_report(report, query)

        # 4. 从 FinalReport 提取 insights 和 summary
        pred_insights = self._extract_insights(report)
        if return_summary:
            pred_summary = self._extract_summary(report)
            return pred_insights, pred_summary

        return pred_insights

    def _save_report(self, report: FinalReport, query: str) -> None:
        """保存 DataSTORM 完整报告到 savedir。"""
        import json as _json_local
        savedir_path = Path(self.savedir) if not isinstance(self.savedir, Path) else self.savedir
        savedir_path.mkdir(parents=True, exist_ok=True)

        # 完整报告 markdown
        md_path = savedir_path / "datastorm_report.md"
        parts = [
            f"# {report.title}",
            f"*{report.subtitle}*",
            "",
            report.markdown,
            "",
            "## Sources",
            "",
        ]
        for ref in report.references:
            ref_id = ref.get("id", "")
            source = ref.get("source", "")
            question = ref.get("question", "")
            sql = ref.get("sql", "")
            parts.append(f"[{ref_id}] {source}: {question}")
            if sql:
                parts.append(f"```sql\n{sql}\n```")
            parts.append("")
        md_path.write_text("\n".join(parts), encoding="utf-8")
        logger.info("Report markdown saved to %s", md_path)

        # 报告元数据
        meta_path = savedir_path / "datastorm_report.json"
        meta = {
            "query": query,
            "title": report.title,
            "subtitle": report.subtitle,
            "thesis": {
                "title": report.thesis.title,
                "research_strategy": report.thesis.research_strategy,
            },
            "n_references": len(report.references),
        }
        meta_path.write_text(_json_local.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Report metadata saved to %s", meta_path)

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _build_query(self, goal: str, description: str) -> str:
        """把 InsightBench 的 goal + description 拼成 DataSTORM 的 query。"""
        parts = [goal]
        if description:
            parts.append(f"\nDataset context: {description[:500]}")
        return "\n".join(parts)

    def _build_pipeline(
        self, config: DataSTORMConfig, bridge: CsvDatabaseBridge
    ) -> DataSTORMPipeline:
        """手动组装 DataSTORMPipeline，用 bridge 替换 DatabaseConnector。

        绕过 DataSTORMPipeline.__init__ 中的 psycopg2 初始化，
        直接注入 CsvDatabaseBridge。
        """
        llm = self._llm
        searcher = WebSearcher(config.internet)  # serper_api_key="" → 自动返回空结果

        planner = PlannerAgent(llm, config)
        executor = ExecutorAgent(llm, bridge, config)  # 注入 bridge

        insight_bank = InsightBank(llm, config)

        # 用 object.__new__ 跳过 __init__，再手动赋值各属性
        pipeline = object.__new__(DataSTORMPipeline)
        pipeline._config = config
        pipeline._llm = llm
        pipeline._db = bridge
        pipeline._searcher = searcher
        pipeline._planner = planner
        pipeline._executor = executor
        pipeline._insight_bank = insight_bank

        return pipeline

    def _extract_insights(self, report: FinalReport) -> list[str]:
        """从 FinalReport 中提取 insights list。

        把 report.markdown + references（SQL 查询结果）拼接后由 LLM 浓缩为简洁的 insight statements。
        """
        # 拼接所有可用内容
        parts = []
        if report.markdown:
            parts.append(report.markdown)
        if report.references:
            for ref in report.references:
                answer = ref.get("answer", "")
                sql = ref.get("sql", "")
                if answer:
                    parts.append(f"SQL result: {answer}")
                if sql:
                    parts.append(f"(SQL: {sql})")

        combined = "\n\n".join(parts)
        if not combined.strip():
            return [report.thesis.title]

        try:
            insights = self._condense_insights(combined)
            if insights:
                return insights
        except Exception:
            logger.warning("LLM condensation failed, falling back to sentence extraction")

        return self._extract_sentences_from_markdown(combined) if combined else [report.thesis.title]

    # ── 用 LLM 将叙事报告浓缩为 insight statements ─────────────────────

    _CONDENSE_PROMPT = (
        "You are extracting key analytical findings from a data analysis report. "
        "Your task is to produce a list of insight statements that describe PATTERNS, "
        "TRENDS, CORRELATIONS, and ANOMALIES found in the data.\n\n"
        "Rules:\n"
        "- Each insight should describe a qualitative observation about the data "
        "(e.g., trends over time, comparisons between categories, correlations or lack thereof).\n"
        "- Focus on WHAT the data reveals at a high level, not raw numbers.\n"
        "- Good style: \"The volume of X incidents is increasing over time\", "
        "\"There is no correlation between A and B\", "
        "\"Category X has significantly higher resolution time than others\", "
        "\"Performance is uniform across all agents despite volume changes\".\n"
        "- BAD style: \"The mean is 38.5\" or \"In April 2023 there were 46 incidents\" "
        "(too specific/quantitative).\n"
        "- Include actionable observations where the data suggests them "
        "(e.g., \"Specific hardware issues are predominantly mentioned in incident descriptions\").\n"
        "- Extract 4-8 insights total. Cover different aspects of the analysis.\n"
        "- Each insight should be 1-2 sentences.\n\n"
        "Return a JSON object with an \"insights\" array of strings."
    )

    def _condense_insights(self, markdown: str) -> list[str]:
        """调用 LLM 将报告 markdown 浓缩为 terse insight statements。"""
        prompt = self._CONDENSE_PROMPT + "\n\nReport:\n" + markdown[:6000]
        result = self._llm.generate_json(
            prompt, temperature=0.3, max_completion_tokens=1024
        )
        raw = result.get("insights", [])
        if isinstance(raw, list):
            return [s.strip() for s in raw if isinstance(s, str) and len(s.strip()) > 15]
        return []

    def _extract_sentences_from_markdown(self, markdown: str) -> list[str]:
        """从 markdown 正文中提取有意义的句子作为 insights。"""
        lines = []
        for line in markdown.split("\n"):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith(">"):
                continue
            line = re.sub(r"\[\^?\d+\]", "", line).strip()
            if len(line) > 40:
                lines.append(line)

        sentences: list[str] = []
        for line in lines:
            for sent in re.split(r"(?<=[.!?])\s+", line):
                sent = sent.strip()
                if len(sent) > 30:
                    sentences.append(sent)

        return sentences[:20]

    def _extract_summary(self, report: FinalReport) -> str:
        """从 FinalReport 提取 summary 字符串。

        使用 LLM 将完整报告浓缩为一段简短的总结段落，
        风格与 InsightBench 的 GT summary 对齐。
        """
        # 尝试用 LLM 生成高质量 summary
        if report.markdown and len(report.markdown) > 100:
            try:
                prompt = (
                    "Summarize the following analytical report into a single concise paragraph "
                    "(100-200 words). Focus on:\n"
                    "- The main finding or thesis\n"
                    "- Key patterns and trends discovered\n"
                    "- Actionable implications or recommendations\n\n"
                    "Report:\n" + report.markdown[:4000]
                )
                summary = self._llm.generate(prompt, temperature=0.3, max_completion_tokens=512)
                if summary and len(summary) > 50:
                    return summary
            except Exception:
                pass

        # 回退：拼接 thesis 信息
        summary_parts = [report.thesis.title]
        if report.thesis.research_strategy:
            summary_parts.append(report.thesis.research_strategy)
        if report.subtitle:
            summary_parts.append(report.subtitle)
        return " ".join(summary_parts)
