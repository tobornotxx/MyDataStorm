"""DataSTORM 配置管理。

所有超参数均严格遵照论文 Section 4.1 设定:
- m = 5 (最大探索层数)
- 第一层 n = 2 个查询, 后续层 n = 5 个查询
- Executor 最多 15 轮 ReAct 循环
- 论点每 p 层生成/精炼一次
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    """LLM 调用配置。"""

    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    # 论文: gpt-5-2025-08-07 for exploration, gpt-5.1-2025-11-13 for report
    exploration_model: str = "gpt-4o"
    report_model: str = "gpt-4o"
    temperature: float = 0.7
    max_completion_tokens: int = 4096


@dataclass
class DatabaseConfig:
    """数据库连接配置。"""

    url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", ""))
    database_type: str = "PostgreSQL"


@dataclass
class InternetConfig:
    """互联网搜索配置。"""

    serper_api_key: str = field(default_factory=lambda: os.getenv("SERPER_API_KEY", ""))
    max_results_per_query: int = 10
    blocked_domains: list[str] = field(default_factory=list)
    date_restrict: str | None = None


@dataclass
class ExplorationConfig:
    """多智能体探索框架配置 (论文 Section 3.2, 4.1)。"""

    # 论文 Section 4.1: m = 5
    max_layers: int = 5
    # 论文 Section 4.1: 第一层 2 个查询, 后续层 5 个查询
    first_layer_max_questions: int = 2
    subsequent_layer_max_questions: int = 5
    # Executor ReAct 最大轮数 (论文 Appendix B)
    executor_max_turns: int = 15
    # 论点生成频率: 每 p 层生成/精炼一次 (论文 Section 3.2.2)
    thesis_generation_interval: int = 1
    # 全局洞察库最大容量
    max_insights: int = 50
    # 结果表截断行数
    max_table_rows: int = 50


@dataclass
class ReportConfig:
    """报告生成配置 (论文 Section 3.3)。"""

    # 每个章节目标字数
    section_target_words: int = 600
    # 最终报告总字数上限
    total_target_words: int = 3000
    # 每个章节最多 web 查询数
    max_web_queries_per_section: int = 3


@dataclass
class DataSTORMConfig:
    """DataSTORM 系统总配置。"""

    llm: LLMConfig = field(default_factory=LLMConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    internet: InternetConfig = field(default_factory=InternetConfig)
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    # 数据库内容描述 (提供给 Planner 作为上下文)
    db_description: str = ""
