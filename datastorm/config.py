"""DataSTORM 配置管理。

所有超参数均严格遵照论文 Section 4.1 设定:
- m = 5 (最大探索层数)
- 第一层 n = 2 个查询, 后续层 n = 5 个查询
- Executor 最多 15 轮 ReAct 循环
- 论点每 p 层生成/精炼一次

LLM 配置优先级（由低到高）:
    1. datastorm/llm_config.json 的 default 块   ← 全局默认
    2. 环境变量 (OPENAI_API_KEY, OPENAI_API_BASE) ← 覆盖 default 的 api_key/api_base
    3. datastorm/llm_config.json 的 scenarios[name] ← 按场景覆盖任意字段
    4. 代码中显式传入的参数                       ← 最终覆盖

每个场景 (warmstart/planner/executor/thesis/insight_bank/consistency/
goal_sufficiency/report/evaluation) 可在 scenarios 中单独配置
model_name / api_base / api_key / temperature / max_completion_tokens，
未填写的字段自动继承 default。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

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

# default 块: 顶层默认配置 (受环境变量覆盖); scenarios 块: 各场景覆盖项
_DEFAULT_CFG: dict = _JSON_CFG.get("default") or {}


def _json_or_env(json_key: str, env_key: str, fallback: str = "") -> str:
    """按「JSON default 块 → 环境变量」优先级读取字符串配置。"""
    return os.getenv(env_key) or _DEFAULT_CFG.get(json_key, "") or fallback


def _json_or_fallback(json_key: str, fallback: str = "") -> str:
    """从 JSON default 块读取，缺失时回退到 fallback。"""
    return _DEFAULT_CFG.get(json_key, "") or fallback


@dataclass
class ScenarioConfig:
    """单个场景解析后的有效 LLM 配置。

    由 ``LLMConfig.scenario(name)`` 合并 ``default`` 与该场景的覆盖项后产生。
    每个场景可独立指定 model_name / api_base / api_key / temperature /
    max_completion_tokens，从而让不同环节调用不同的模型或服务端点。
    """

    model_name: str = ""
    api_base: str = ""
    api_key: str = ""
    temperature: float = 0.7
    max_completion_tokens: int = 4096


# 所有受支持的场景名 (供参考/校验; 未列出的场景名仍可使用, 会回退到 default)。
SUPPORTED_SCENARIOS: tuple[str, ...] = (
    "default",
    "warmstart",
    "planner",
    "executor",
    "thesis",
    "insight_bank",
    "consistency",
    "goal_sufficiency",
    "report",
    "evaluation",
)


@dataclass
class LLMConfig:
    """LLM 调用配置。

    顶层字段 (api_key / api_base / model_name / temperature /
    max_completion_tokens) 即 ``default`` 场景的取值，环境变量
    (OPENAI_API_KEY / OPENAI_API_BASE) 可覆盖 JSON 中的 api_key / api_base。

    各场景可在 ``llm_config.json`` 的 ``scenarios`` 字段里单独覆盖任意字段，
    未覆盖的字段自动继承 ``default``。通过 :meth:`scenario` 取得合并后的配置。
    """

    api_key: str = field(
        default_factory=lambda: _json_or_env("api_key", "OPENAI_API_KEY")
    )
    api_base: str = field(
        default_factory=lambda: _json_or_env("api_base", "OPENAI_API_BASE")
    )
    model_name: str = field(
        default_factory=lambda: _json_or_fallback("model_name", "deepseek-v4-pro")
    )
    temperature: float = field(
        default_factory=lambda: _DEFAULT_CFG.get("temperature", 0.7)
    )
    max_completion_tokens: int = field(
        default_factory=lambda: int(_DEFAULT_CFG.get("max_completion_tokens", 4096))
    )
    # 场景覆盖表: {scenario_name: {field: value, ...}}
    scenarios: dict[str, dict[str, Any]] = field(
        default_factory=lambda: dict(_JSON_CFG.get("scenarios") or {})
    )

    _SCENARIO_FIELDS: ClassVar[tuple[str, ...]] = (
        "model_name", "api_base", "api_key", "temperature", "max_completion_tokens",
    )

    def scenario(self, name: str | None = None) -> ScenarioConfig:
        """返回某场景合并后的有效配置。

        合并优先级 (由低到高):
            1. default 场景顶层字段 (受环境变量覆盖)
            2. JSON 中 ``scenarios.default`` 的覆盖项
            3. JSON 中 ``scenarios[name]`` 的覆盖项 (本场景)
        空字符串 / None 的覆盖项会被忽略, 以便只覆盖部分字段。
        """
        merged: dict[str, Any] = {
            "model_name": self.model_name,
            "api_base": self.api_base,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_completion_tokens,
        }
        # 先应用 default 场景覆盖, 再应用目标场景覆盖
        for src in ("default", name or "default"):
            override = self.scenarios.get(src) or {}
            for key in self._SCENARIO_FIELDS:
                val = override.get(key)
                if val not in (None, ""):
                    merged[key] = val
        return ScenarioConfig(**merged)

    # ── 向后兼容: 旧的 exploration_model / report_model 字段 ──────────────
    # 代码中可能仍有引用, 这里委托给对应场景, 保证旧行为不变。
    @property
    def exploration_model(self) -> str:
        return self.scenario("executor").model_name

    @property
    def report_model(self) -> str:
        return self.scenario("report").model_name


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
    # 每层基于已有问题提出的跟进问题数 (m, 下限)
    follow_up_questions_per_layer: int = 3
    # 每层提出的全新探索性问题数 (n, 下限)
    exploratory_questions_per_layer: int = 2
    # Executor ReAct 最大轮数 (论文 Appendix B)
    executor_max_turns: int = 15
    # 论点生成频率: 每 p 层生成/精炼一次 (论文 Section 3.2.2)
    thesis_generation_interval: int = 1
    # 全局洞察库最大容量
    max_insights: int = 50
    # 结果表截断行数
    max_table_rows: int = 50
    # 早期停止: 连续 N 层洞察库未增长则停止 (0 = 禁用)
    early_stop_patience: int = 2
    # Goal 满足度自评: 每层结束 agent 自评"发现是否已回答 goal"。
    #   sufficient=True  → 提前停止 (省 token)
    #   sufficient=False → 用反馈的缺失角度引导下一层补充探索
    # max_layers 始终是硬上限。False = 禁用, 回退到机械 plateau 早停。
    goal_sufficiency_check: bool = True
    # 自评前的最小层数: 在此之前不评 (避免第 1 层信息太少就误判"够了")
    goal_sufficiency_min_layers: int = 2


@dataclass
class ReportConfig:
    """报告生成配置 (论文 Section 3.3)。"""

    # 每个章节目标字数
    section_target_words: int = 600
    # 最终报告总字数上限
    total_target_words: int = 3000
    # 每个章节最多 web 查询数
    max_web_queries_per_section: int = 3
    # 是否跳过引用验证 (Stage C) 和修订 (Stage D)，可显著减少 API 请求
    skip_citation_check: bool = False


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
