"""DataSTORM 核心数据类型定义。

定义系统中流转的所有数据结构, 对应论文中的各类实体:
- Insight (洞察 b): 存储在全局洞察库 B 中
- ExplorerQuestion (探索问题 q): Planner 生成的问题
- ExecutorResponse (执行器响应 a): Executor 返回的答案
- Thesis (论点 t): 论点生成模块产出
- Section / Report: 报告生成流水线产出
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QuestionDestination(str, Enum):
    """问题路由目标 (论文 Section 3.2.1)。"""

    DATABASE = "database"
    INTERNET = "internet"


@dataclass
class ExplorerQuestion:
    """Planner 生成的探索问题 (论文 Section 3.2.1, q_{i,j})。

    Attributes:
        question: 自然语言问题文本
        destination: 路由目标 (database / internet)
        previous_queries: 上下文中的前序查询
    """

    question: str
    destination: QuestionDestination = QuestionDestination.DATABASE
    previous_queries: list[str] = field(default_factory=list)


@dataclass
class QuestionNode:
    """问题树节点，记录问题的完整链和继承关系。

    每个问题在提出时即分配唯一 ID，执行后记录答案。
    parent_ids 记录该问题由哪些已有问题派生而来，
    category 区分"跟进问题 (follow_up)"和"探索性问题 (exploratory)"。

    Attributes:
        id: 唯一标识 (UUID)
        question: 问题文本
        destination: 路由目标
        layer: 所属探索层编号
        parent_ids: 派生来源的问题 ID 列表
        category: "follow_up" (基于已有问题的深入) 或 "exploratory" (新探索方向)
        answer: 自然语言答案 (执行后填充)
        sql: 执行的 SQL (执行后填充)
        summary_text: 包含统计的摘要文本 (执行后填充)
        summary_stats: 列级统计 (执行后填充)
        raw_results: 原始查询结果
        row_count: 结果行数
    """

    id: str
    question: str
    destination: QuestionDestination = QuestionDestination.DATABASE
    layer: int = 0
    parent_ids: list[str] = field(default_factory=list)
    category: str = "exploratory"
    answer: str = ""
    sql: str = ""
    summary_text: str = ""
    summary_stats: list["SummaryStatistics"] = field(default_factory=list)
    raw_results: Any = None
    row_count: int = 0


@dataclass
class SummaryStatistics:
    """列级别汇总统计 (论文 Section 3.2.1, 自底向上归纳)。

    包含: distinct_percentage, min, max, median, mean, top_values
    """

    column_name: str
    distinct_percentage: float | None = None
    min_val: Any = None
    max_val: Any = None
    median_val: Any = None
    mean_val: Any = None
    top_values: list[dict[str, Any]] = field(default_factory=list)

    def to_text(self) -> str:
        """转为文本嵌入到 Executor 响应中。"""
        parts = [f"{self.column_name}"]
        if self.distinct_percentage is not None:
            parts.append(f"distinct_percentage: {self.distinct_percentage:.2f}%")
        if self.min_val is not None:
            parts.append(f"min: {self.min_val}")
        if self.max_val is not None:
            parts.append(f"max: {self.max_val}")
        if self.median_val is not None:
            parts.append(f"median: {self.median_val}")
        if self.mean_val is not None:
            parts.append(f"mean: {self.mean_val}")
        if self.top_values:
            parts.append(f"top_values: {self.top_values}")
        return " ".join(parts)


@dataclass
class ExecutorResponse:
    """Executor Agent 返回的响应 (论文 Section 3.2.1, a_{i,j} / a'_{i,j})。

    Attributes:
        question: 原始问题
        answer: 自然语言答案
        sql: 执行的 SQL 查询 (s_{i,j})
        summary_text: 包含汇总统计的摘要文本
        summary_stats: 各列汇总统计
        raw_results: 原始查询结果
        row_count: 结果行数
    """

    question: str
    answer: str
    sql: str = ""
    summary_text: str = ""
    summary_stats: list[SummaryStatistics] = field(default_factory=list)
    raw_results: Any = None
    row_count: int = 0


@dataclass
class Insight:
    """全局洞察库中的洞察条目 (论文 Section 3.2.1, b ∈ B)。

    Attributes:
        id: 唯一标识
        content: 洞察内容文本
        source: 来源 ("database" / "internet")
        question: 产生该洞察的问题
        sql: 关联的 SQL 查询
        answer: 原始答案
        layer: 产生该洞察的层编号
    """

    id: str
    content: str
    source: str = "database"
    question: str = ""
    sql: str = ""
    answer: str = ""
    layer: int = 0


@dataclass
class Thesis:
    """论点 (论文 Section 3.2.2, t)。

    Attributes:
        title: 论点标题 (最多10个词)
        research_strategy: 研究策略
    """

    title: str
    research_strategy: str


@dataclass
class ConsistencyFollowUp:
    """查询一致性模块的跟进查询 (论文 Section 3.2.1)。

    Attributes:
        original_question: 原始问题
        original_sql: 原始 SQL
        follow_up_question: 跟进问题 (None 表示无需跟进)
    """

    original_question: str
    original_sql: str
    follow_up_question: str | None = None


@dataclass
class SectionSpec:
    """报告章节规格 (论文 Section 3.3, Stage A)。

    Attributes:
        section_id: 章节标识
        heading: 标题
        purpose: 叙事目的
        must_include_evidence_ids: 必须引用的证据 ID
        key_points: 关键要点
        storytelling_moves: 叙事手法
        web_queries: 补充 web 查询
    """

    section_id: str
    heading: str
    purpose: str
    must_include_evidence_ids: list[int] = field(default_factory=list)
    key_points: list[str] = field(default_factory=list)
    storytelling_moves: list[str] = field(default_factory=list)
    web_queries: list[str] = field(default_factory=list)


@dataclass
class ReportOutline:
    """报告大纲 (论文 Section 3.3, Stage A)。"""

    lede_strategy: str = ""
    key_findings: list[str] = field(default_factory=list)
    sections: list[SectionSpec] = field(default_factory=list)
    closing_strategy: str = ""


@dataclass
class DraftedSection:
    """已起草的章节 (论文 Section 3.3, Stage B-D)。

    Attributes:
        section_id: 章节标识
        heading: 标题
        markdown: Markdown 内容
        used_citations: 使用的引用编号
    """

    section_id: str
    heading: str
    markdown: str
    used_citations: list[int] = field(default_factory=list)


@dataclass
class CitationCheck:
    """引用验证结果 (论文 Section 3.3, Stage C)。"""

    sentence: str
    is_entailed: bool
    issue: str = ""


@dataclass
class FinalReport:
    """最终报告 (论文 Section 3.3, Stage E)。

    Attributes:
        title: 报告标题
        subtitle: 副标题
        thesis: 中心论点
        markdown: 最终 Markdown 正文
        references: 参考文献列表
    """

    title: str
    subtitle: str
    thesis: Thesis
    markdown: str
    references: list[dict[str, str]] = field(default_factory=list)
