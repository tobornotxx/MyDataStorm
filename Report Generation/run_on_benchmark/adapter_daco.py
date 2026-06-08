"""
adapter_daco.py — DACO 适配器

将我们的 Agent（CodeAgent + LLM）接入 DACO 的数据格式。
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd

import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from llm import OpenAILikeLLM, LLMConfig
from code_agent import CodeAgent
from utils.data_inspector import describe_dataframes_schema
from utils import logger


def run_agent_on_instance(
    db_path: str,
    query: str,
    max_queries: int = 5,
    code_agent_model: Optional[str] = None,
    code_agent_max_steps: int = 3,
) -> Dict[str, Any]:
    """
    在一条 DACO 样本上运行 Agent。

    Args:
        db_path: 数据库目录路径，内含一个或多个 CSV 文件
        query:   分析查询，格式 "As a [role], I want to [intention]"
        max_queries: LLM 规划的分析步骤数
        code_agent_model: CodeAgent 使用的模型
        code_agent_max_steps: 每步最大代码执行次数

    Returns:
        {
            "findings": [str, ...],
            "suggestions": [str, ...],
            "code_trajectory": [{"step": str, "result": str}, ...],
        }
    """
    db_path = Path(db_path)

    # ---- 1. 读入所有表 ----
    tables = _load_database(db_path)
    if not tables:
        raise FileNotFoundError(f"No CSV/data files found in {db_path}")

    logger.info(f"[DACO] 加载了 {len(tables)} 个表: {list(tables.keys())}")

    # ---- 2. 构造 Schema ----
    schema = describe_dataframes_schema(tables, max_sample_rows=3, max_unique_values=10)

    # ---- 3. 规划分析步骤 ----
    llm = OpenAILikeLLM(config=LLMConfig())
    steps = _plan_analysis(llm, schema, query, max_queries)
    logger.info(f"[DACO] 规划了 {len(steps)} 步分析")

    # ---- 4. 逐步执行代码分析 ----
    agent_kwargs = {}
    if code_agent_model:
        agent_kwargs["model"] = code_agent_model

    agent = CodeAgent(**agent_kwargs)
    code_trajectory: List[Dict[str, str]] = []
    analysis_results: List[str] = []

    for i, step in enumerate(steps, 1):
        step_desc = step.get("purpose", step.get("query", f"Step {i}"))
        logger.info(f"[DACO] 执行步骤 {i}/{len(steps)}: {step_desc[:80]}")

        # 筛选相关表
        requested_tables = step.get("tables", step.get("sheets", []))
        if requested_tables:
            relevant = {}
            for t in requested_tables:
                if t in tables:
                    relevant[t] = tables[t]
                else:
                    # 模糊匹配
                    for k in tables:
                        if t.lower() in k.lower() or k.lower() in t.lower():
                            relevant[k] = tables[k]
            if not relevant:
                relevant = tables
        else:
            relevant = tables

        instruction = (
            f"你有以下 pandas DataFrame 变量（通过 dfs 字典访问）:\n"
            f"{describe_dataframes_schema(relevant, max_sample_rows=0)}\n\n"
            f"用户查询: {query}\n"
            f"当前分析步骤: {step_desc}\n\n"
            f"请编写 Python 代码执行分析，用 print 输出关键结果。\n"
            f"数据通过 dfs 字典访问，例如 dfs['表名']。"
        )

        result = agent.run(
            input=instruction,
            max_steps=code_agent_max_steps,
            additional_args={"dfs": relevant},
        )

        code_trajectory.append({
            "step": step_desc,
            "result": result or "执行失败",
        })
        if result:
            analysis_results.append(f"[{step_desc}]\n{result}")

    # ---- 5. 汇总生成 Findings + Suggestions ----
    output = _synthesize_report(llm, query, analysis_results)
    output["code_trajectory"] = code_trajectory

    return output


# ============================================================
# 内部辅助函数
# ============================================================

def _load_database(db_path: Path) -> Dict[str, pd.DataFrame]:
    """加载数据库目录下的所有数据文件。"""
    tables = {}

    # CSV 文件
    for f in sorted(db_path.glob("*.csv")):
        try:
            tables[f.stem] = pd.read_csv(f)
        except Exception as e:
            logger.warning(f"[DACO] 读取 {f.name} 失败: {e}")

    # 也支持 Excel
    for f in sorted(db_path.glob("*.xlsx")):
        try:
            xls = pd.ExcelFile(f)
            for sheet in xls.sheet_names:
                key = f"{f.stem}__{sheet}" if len(xls.sheet_names) > 1 else f.stem
                tables[key] = pd.read_excel(f, sheet_name=sheet)
        except Exception as e:
            logger.warning(f"[DACO] 读取 {f.name} 失败: {e}")

    # 支持 SQLite（DACO 有些数据库是 .sqlite）
    for f in sorted(db_path.glob("*.sqlite")):
        try:
            import sqlite3
            conn = sqlite3.connect(str(f))
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            for (table_name,) in cursor.fetchall():
                tables[table_name] = pd.read_sql_query(f"SELECT * FROM `{table_name}`", conn)
            conn.close()
        except Exception as e:
            logger.warning(f"[DACO] 读取 {f.name} 失败: {e}")

    return tables


def _plan_analysis(
    llm: OpenAILikeLLM,
    schema: str,
    query: str,
    max_steps: int,
) -> List[Dict[str, Any]]:
    """让 LLM 根据 Schema + Query 规划多步分析。"""

    prompt = f"""You are a data analysis expert. Plan a multi-step analysis for the following query.

User Query: {query}

Available Data:
{schema}

Generate exactly {max_steps} analysis steps. Each step should specify:
1. "purpose": what to analyze in this step
2. "tables": list of table names needed (use exact names from the schema)

Return as JSON array:
[
  {{"purpose": "Examine the overall distribution of ...", "tables": ["table1"]}},
  {{"purpose": "Compare ... across ...", "tables": ["table1", "table2"]}},
  ...
]

Return ONLY the JSON array, no other text."""

    response = llm.chat(prompt)
    return _parse_json_list(response.content, max_steps)


def _synthesize_report(
    llm: OpenAILikeLLM,
    query: str,
    analysis_results: List[str],
) -> Dict[str, List[str]]:
    """汇总分析结果，生成 Findings + Suggestions。"""

    if not analysis_results:
        return {"findings": ["No analysis results available."], "suggestions": []}

    all_results = "\n\n---\n\n".join(analysis_results)

    prompt = f"""You are writing a data analysis report.

User Query: {query}

Analysis Results:
{all_results[:8000]}

Based on the analysis above, produce a structured report with:
1. "findings": 3-8 data-driven findings, each backed by specific numbers from the analysis
2. "suggestions": 3-8 actionable recommendations based on the findings

Return as JSON:
{{"findings": ["Finding 1...", "Finding 2..."], "suggestions": ["Suggestion 1...", "Suggestion 2..."]}}

Return ONLY the JSON, no other text."""

    response = llm.chat(prompt)

    try:
        result = json.loads(_extract_json_obj(response.content))
        if "findings" in result and "suggestions" in result:
            return result
    except (json.JSONDecodeError, TypeError):
        pass

    # 回退：把整个回复当作 findings
    return {
        "findings": [response.content.strip()],
        "suggestions": [],
    }


def _extract_json_obj(text: str) -> str:
    """从文本中提取 JSON 对象。"""
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text.strip()


def _parse_json_list(text: str, max_items: int) -> List[Dict[str, Any]]:
    """从 LLM 输出中解析 JSON 列表。"""
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    try:
        items = json.loads(text.strip())
        if isinstance(items, list):
            return items[:max_items]
    except json.JSONDecodeError:
        pass

    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            items = json.loads(text[start : end + 1])
            if isinstance(items, list):
                return items[:max_items]
        except json.JSONDecodeError:
            pass

    logger.warning("[DACO] 无法解析分析步骤，使用默认步骤")
    return [
        {"purpose": "Explore the basic statistics and distributions of key columns", "tables": []},
        {"purpose": "Identify trends, correlations, or patterns relevant to the query", "tables": []},
        {"purpose": "Perform deeper analysis and draw conclusions", "tables": []},
    ]
