"""DataSTORM 配置管理。

所有超参数均严格遵照论文 Section 4.1 设定:
- m = 5 (最大探索层数)
- 第一层 n = 2 个查询, 后续层 n = 5 个查询
- Executor 最多 15 轮 ReAct 循环
- 论点每 p 层生成/精炼一次

LLM 配置优先级（由低到高）:
    1. datastorm/llm_config.json   ← 全局默认
    2. 环境变量 (OPENAI_API_KEY, OPENAI_API_BASE)
    3. 代码中显式传入的参数         ← 最终覆盖
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── 加载全局 LLM 配置文件 ─────────────────────────────────────────────
_CFG_PATH = Path(__file__).parent / "llm_config.json"

def _load_json_config() -> dict:
    """加载 llm_config.json，文件不存在或解析失败返回空 dict。"""
    try:
        if _CFG_PATH.is_file():
            return json.loads(_CFG_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

_JSON_CFG: dict = _load_json_config()


def _json_or_env(json_key: str, env_key: str, fallback: str = "") -> str:
    """按「JSON 文件 → 环境变量」优先级读取字符串配置。"""
    return os.getenv(env_key) or _JSON_CFG.get(json_key, "") or fallback


def _json_or_fallback(json_key: str, fallback: str = "") -> str:
    """从 JSON 文件读取，缺失时回退到 fallback。"""
    return _JSON_CFG.get(json_key, "") or fallback


@dataclass
class LLMConfig:
    """LLM 调用配置。

    api_base / api_key / model_name 的默认值由 llm_config.json 统一管理，
    环境变量 (OPENAI_API_KEY / OPENAI_API_BASE) 可覆盖 JSON 中的值。
    """

    api_key: str = field(
        default_factory=lambda: _json_or_env("api_key", "OPENAI_API_KEY")
    )
    api_base: str = field(
        default_factory=lambda: _json_or_env("api_base", "OPENAI_API_BASE")
    )
    exploration_model: str = field(
        default_factory=lambda: _json_or_fallback("model_name", "gpt-5.4-mini")
    )
    report_model: str = field(
        default_factory=lambda: _json_or_fallback("model_name", "gpt-5.4-mini")
    )
    temperature: float = field(
        default_factory=lambda: _JSON_CFG.get("temperature", 0.7)
    )
    max_completion_tokens: int = field(
        default_factory=lambda: int(_JSON_CFG.get("max_completion_tokens", 4096))
    )


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
