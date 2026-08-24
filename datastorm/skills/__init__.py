"""Skill 包加载与渲染。

设计意图（见 run_on_benchmark/SKILL_PACKAGE_DESIGN.md）：
把原先硬编码在 csv_db_bridge.py / goal_sufficiency.py 里的分析引导，
重构成声明式、数据无关、可冻结、可审计的 skill 包。

为什么必须数据无关：
    v8 的硬编码引导里举例写了 `category, assigned_to, priority, assignment_group`
    和 `TTR per agent over time` —— 这些是 ServiceNow / ITSM 专有字段和术语。
    LLM 见到这些具体词会被诱导往工单场景靠，迁移到政府开放数据（氡浓度、
    人口加权）时会去找根本不存在的列。

解决方式不是简单删掉举例（那会让引导变空泛、削弱效果），而是把具体列名
变成 {categorical_columns} 占位符，运行时从**当前数据集的真实 schema** 注入。
这样既消除了 benchmark 特化，又保留了「指名道姓」的引导强度——
对任意数据集都给出该数据集自己的列名。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_PACKAGE = Path(__file__).parent / "skill_package_v9.json"

# 占位符找不到真实列名时的兜底措辞（仍然数据无关）
_FALLBACK_CATEGORICAL = "the primary categorical columns shown in the schema above"


@dataclass
class Skill:
    """单条声明式 skill。"""

    id: str
    trigger: str
    action: str
    rationale: str = ""
    provenance: str = ""
    purity: str = ""
    validated: dict[str, Any] = field(default_factory=dict)

    def render(self, **ctx: str) -> str:
        """用运行时上下文填充 action 中的占位符。

        未提供的占位符保持原样而不抛错——缺一个上下文不应让整条 skill 失效。
        """
        action = self.action
        for key, val in ctx.items():
            action = action.replace("{" + key + "}", val)
        return action


@dataclass
class SkillPackage:
    """声明式技能包。"""

    meta: dict[str, Any]
    skills: list[Skill]
    episodes: list[dict[str, Any]] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path | None = None) -> "SkillPackage":
        """从 JSON 加载 skill 包。加载失败返回空包（降级为无引导，不崩主流程）。"""
        p = Path(path) if path else _DEFAULT_PACKAGE
        try:
            raw = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Skill package load failed (%s): %s — running without skills", p, e)
            return cls(meta={}, skills=[])

        skills = [
            Skill(
                id=s.get("id", ""),
                trigger=s.get("trigger", ""),
                action=s.get("action", ""),
                rationale=s.get("rationale", ""),
                provenance=s.get("provenance", ""),
                purity=s.get("purity", ""),
                validated=s.get("validated", {}) or {},
            )
            for s in raw.get("skills", [])
        ]
        pkg = cls(
            meta=raw.get("meta", {}) or {},
            skills=skills,
            episodes=raw.get("episodes", []) or [],
            config=raw.get("config", {}) or {},
        )
        logger.info(
            "Skill package loaded: version=%s, %d skill(s), %d episode(s)",
            pkg.meta.get("version", "?"), len(pkg.skills), len(pkg.episodes),
        )
        return pkg

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def get(self, skill_id: str) -> Skill | None:
        for s in self.skills:
            if s.id == skill_id:
                return s
        return None

    def subset(self, skill_ids: list[str]) -> list[Skill]:
        """按 id 取子集，保持给定顺序，跳过不存在的（便于 ablation 关掉某条）。"""
        out = []
        for sid in skill_ids:
            s = self.get(sid)
            if s is not None:
                out.append(s)
            else:
                logger.debug("Skill '%s' not in package — skipped", sid)
        return out

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------

    def render_guidance(
        self,
        skill_ids: list[str],
        header: str = "ANALYTICAL GUIDANCE (generic — applies to any question on this data)",
        **ctx: str,
    ) -> str:
        """把若干 skill 渲染成可注入 prompt 的引导文本块。

        Args:
            skill_ids: 要包含的 skill id（顺序即输出顺序）
            header: 文本块标题
            **ctx: 占位符上下文，如 categorical_columns="a, b, c"

        Returns:
            形如 "\\n\\nHEADER:\\n- <action>\\n- <action>" 的文本；无 skill 时返回 ""。
        """
        selected = self.subset(skill_ids)
        if not selected:
            return ""
        ctx.setdefault("categorical_columns", _FALLBACK_CATEGORICAL)
        lines = [f"\n\n{header}:"]
        for s in selected:
            lines.append(f"- {s.render(**ctx)}")
        return "\n".join(lines)


# ----------------------------------------------------------------------
# 模块级单例（避免每次建 bridge 都重读 JSON）
# ----------------------------------------------------------------------

_cached: SkillPackage | None = None


def get_skill_package(path: str | Path | None = None) -> SkillPackage:
    """取全局 skill 包（首次调用时加载并缓存）。"""
    global _cached
    if _cached is None or path is not None:
        _cached = SkillPackage.load(path)
    return _cached


def describe_categorical_columns(names: list[str], max_n: int = 4) -> str:
    """把列名列表格式化成可嵌入 prompt 的字符串。

    空列表时返回数据无关的兜底措辞，保证 prompt 永不出现空括号。
    """
    picked = [n for n in names if n][:max_n]
    if not picked:
        return _FALLBACK_CATEGORICAL
    return ", ".join(picked)
