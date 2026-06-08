# 在 InsightBench / DACO 上运行自有 Agent 的实操指南

## 概述

本文档描述：如果要将我们的 Agent 接入 InsightBench 和 DACO benchmark 进行评测，Agent 接口需要怎么设计、读入什么、输出什么、评估器怎么搭建。

我们的 Agent 当前架构：`data_analysis.py` 中的 `analyze_region()` → 生成查询指令 → `CodeAgent` 多轮执行代码 → 拼接结果 → `DocWriter` 生成报告。核心能力是"读表 + 生成代码分析 + 输出文本报告"，与两个 benchmark 的需求高度吻合。

---

## 一、InsightBench

### 1. 数据获取

```bash
git clone https://github.com/ServiceNow/insight-bench
cd insight-bench
```

目录结构：
```
insight-bench/
├── data/
│   ├── dataset_1/
│   │   ├── data.csv              # 500 行 CSV 数据
│   │   ├── goal.txt              # SMART 目标
│   │   ├── notebook.ipynb        # Ground-Truth 分析 Notebook
│   │   └── metadata.json         # 难度、主题等元信息
│   ├── dataset_2/
│   │   └── ...
│   └── ... (共 100 个)
├── evaluation/
│   └── evaluate.py               # 评估脚本
└── agents/
    └── agent_poirot.py           # 基线 Agent 实现
```

### 2. Agent 接口规范

#### 输入

Agent 在每个 dataset 上被调用时，收到以下输入：

| 输入项 | 类型 | 内容 | 来源 |
|--------|------|------|------|
| `dataset_csv_path` | `str` | CSV 文件路径 | `data/dataset_X/data.csv` |
| `goal` | `str` | 分析目标文本 | `data/dataset_X/goal.txt` |
| `schema` | `str`（可选） | 数据表的结构描述（列名、类型、统计值） | 自行从 CSV 提取 |

对应到我们的代码：`dataset_csv_path` 对应 `assessment_df`，`goal` 对应我们目前的 `region_name` + 隐式目标。

#### 输出

Agent 需要输出一个结构化的 JSON/dict，包含：

```python
{
    "insights": [
        {
            "question": "What is the distribution of incident categories?",
            "code": "import pandas as pd\ndf = pd.read_csv('data.csv')\n...",
            "insight": "Hardware incidents account for 62.8% of all incidents (314 out of 500).",
            "type": "descriptive"  # descriptive / diagnostic / predictive / prescriptive
        },
        {
            "question": "What is the trend in TTR over time?",
            "code": "...",
            "insight": "TTR increases linearly from 113h in Jan 2023 to 3150h in Jun 2024.",
            "type": "diagnostic"
        },
        # ... 通常 10-15 条
    ],
    "summary": "Overall, incident resolution times show a clear linear increase, especially in the Hardware category. Root cause analysis points to printer and server malfunctions. Recommended actions: (1) increase Hardware team staffing, (2) establish preventive maintenance, (3) open a root cause investigation ticket."
}
```

#### 我们需要做的适配

```python
# benchmark_adapter_insightbench.py

import pandas as pd
import json
from pathlib import Path

def run_agent_on_insightbench_dataset(dataset_dir: str) -> dict:
    """
    在一个 InsightBench dataset 上运行我们的 Agent。
    
    Args:
        dataset_dir: 如 "insight-bench/data/dataset_1"
    
    Returns:
        dict: {"insights": [...], "summary": "..."}
    """
    dataset_dir = Path(dataset_dir)
    
    # ---- 1. 读入 ----
    csv_path = dataset_dir / "data.csv"
    goal_path = dataset_dir / "goal.txt"
    
    df = pd.read_csv(csv_path)
    goal = goal_path.read_text(encoding="utf-8").strip()
    
    # ---- 2. 构造 Schema（复用现有的 describe_dataframes_schema）----
    from utils.data_inspector import describe_dataframes_schema
    schema = describe_dataframes_schema({"data": df}, max_sample_rows=3, max_unique_values=10)
    
    # ---- 3. 调用 LLM 生成分析问题（类似 _generate_query_instructions）----
    # 需要把 goal 作为分析目标传入 prompt
    # prompt 中应包含：schema + goal + 要求生成多角度问题（descriptive/diagnostic/predictive/prescriptive）
    
    from llm import OpenAILikeLLM, LLMConfig
    llm = OpenAILikeLLM(config=LLMConfig())
    
    questions = generate_analysis_questions(llm, schema, goal)  # 需要实现
    
    # ---- 4. 逐个问题执行代码分析（复用 CodeAgent）----
    from code_agent import CodeAgent
    agent = CodeAgent()
    
    insights = []
    for q in questions:
        result = agent.run(
            input=f"数据文件路径: {csv_path}\n分析目标: {goal}\n当前问题: {q['question']}\n请编写Python代码分析并给出insight。",
            additional_args={"df": df},
        )
        insights.append({
            "question": q["question"],
            "code": "",  # 可从 CodeAgent 日志中提取
            "insight": result or "分析失败",
            "type": q.get("type", "descriptive"),
        })
    
    # ---- 5. 生成总结 ----
    all_insights_text = "\n".join([f"- {ins['insight']}" for ins in insights])
    summary = llm.chat(
        f"Goal: {goal}\n\nInsights found:\n{all_insights_text}\n\n"
        f"Please summarize all insights and provide actionable recommendations."
    ).content
    
    return {
        "insights": insights,
        "summary": summary,
    }
```

### 3. 评估器搭建

#### 方法 A：使用官方评估脚本

```bash
cd insight-bench
python evaluation/evaluate.py \
    --predictions_dir ./my_agent_outputs/ \
    --ground_truth_dir ./data/ \
    --evaluator llama3  # 或 gpt4
```

预测文件格式：每个 dataset 一个 JSON 文件，放在 `my_agent_outputs/dataset_X/predictions.json`。

#### 方法 B：自己实现评估器

评估器的核心逻辑：

```python
# evaluator_insightbench.py

import json
from typing import List, Dict

def evaluate_dataset(
    predicted: Dict,        # Agent 输出的 {"insights": [...], "summary": "..."}
    ground_truth: Dict,     # 从 notebook.ipynb 解析出的 GT
    evaluator_llm,          # LLaMA-3-70b 或 GPT-4
) -> Dict[str, float]:
    """
    对一个 dataset 计算 Insight 级别分 + Summary 级别分。
    """
    gt_insights: List[str] = ground_truth["insights"]  # GT 的每条 insight 文本
    pred_insights: List[str] = [ins["insight"] for ins in predicted["insights"]]
    gt_summary: str = ground_truth["summary"]
    pred_summary: str = predicted["summary"]
    
    # ---- Insight 级别：One-to-Many Matching ----
    insight_scores = []
    for gt_ins in gt_insights:
        # 对每条 GT insight，找 pred 中最高匹配分
        best_score = 0.0
        for pred_ins in pred_insights:
            score = llm_eval_score(evaluator_llm, gt_ins, pred_ins)
            best_score = max(best_score, score)
        insight_scores.append(best_score)
    
    avg_insight_score = sum(insight_scores) / len(insight_scores) if insight_scores else 0.0
    
    # ---- Summary 级别 ----
    summary_score = llm_eval_score(evaluator_llm, gt_summary, pred_summary)
    
    return {
        "insight_score": avg_insight_score,
        "summary_score": summary_score,
        "overall": (avg_insight_score + summary_score) / 2,
    }


def llm_eval_score(evaluator_llm, ground_truth_text: str, predicted_text: str) -> float:
    """
    用 LLM（LLaMA-3-70b）评估两段文本的语义 + 事实相似度。
    返回 0.0 ~ 1.0 的分数。
    
    Prompt 模板（参考 G-Eval）:
    """
    prompt = f"""You are evaluating the quality of a data analysis insight.

Ground Truth Insight:
{ground_truth_text}

Predicted Insight:
{predicted_text}

Rate how well the predicted insight captures the key information in the ground truth insight.
Consider:
1. Factual accuracy - does it state the same facts?
2. Completeness - does it cover the main finding?
3. Specificity - does it include relevant details (numbers, trends, etc.)?

Score (0.0 to 1.0, where 1.0 means the predicted insight perfectly captures the ground truth):"""
    
    response = evaluator_llm.chat(prompt)
    # 解析分数（从 response 中提取 float）
    score = parse_score(response.content)
    return score
```

#### 评估指标汇总

| 指标 | 计算方式 | 分数范围 | 说明 |
|------|---------|---------|------|
| **Insight Score** | 每条 GT insight 与所有 pred insights 中最高匹配分的平均 | 0.0 - 1.0 | 主指标之一 |
| **Summary Score** | GT summary 与 pred summary 的 LLM-Eval 分数 | 0.0 - 1.0 | 主指标之一 |
| **Overall Score** | (Insight Score + Summary Score) / 2 | 0.0 - 1.0 | 最终报告的分数 |
| **ROUGE-1** | n-gram 重合度 | 0.0 - 1.0 | 辅助指标 |

#### 评估器选择

| 评估器 | 部署方式 | 成本 | 稳定性 |
|--------|---------|------|--------|
| **LLaMA-3-70b（推荐）** | vLLM 部署在 4×A100 上，温度 0 | 免费（自有 GPU） | 高（权重固定） |
| GPT-4o | API 调用 | ~$0.01/题 | 中（模型可能更新） |

---

## 二、DACO

### 1. 数据获取

```bash
git clone https://github.com/shirley-wu/daco
cd daco
```

目录结构：
```
daco/
├── data/
│   ├── databases/            # 440 个数据库目录
│   │   ├── db_001/
│   │   │   ├── table1.csv
│   │   │   ├── table2.csv
│   │   │   └── schema.json   # 表结构信息
│   │   └── ...
│   ├── train.jsonl           # 1,558 条训练数据
│   ├── dev.jsonl             # 100 条验证数据
│   ├── test_a.jsonl          # 284 条自动标注测试数据
│   └── test_h.jsonl          # 100 条人工精标测试数据（主要评测集）
├── evaluation/
│   └── evaluate.py
└── src/
    └── pipeline.py           # GPT-4 标注管线参考实现
```

每条 JSONL 数据的格式：
```json
{
    "db_id": "db_001",
    "query": "As an advertising executive, I want to select the channels for targeted ad placements.",
    "answer": {
        "findings": [
            "Social media channels reach 68% of the 18-34 age demographic...",
            "Email campaigns have the highest ROI at $42 per $1 spent..."
        ],
        "suggestions": [
            "Prioritize social media for brand awareness campaigns targeting younger demographics...",
            "Allocate 30% of budget to email campaigns for retention..."
        ]
    },
    "code_trajectory": [
        {"code": "import pandas as pd\n...", "output": "..."},
        {"code": "...", "output": "..."}
    ]
}
```

### 2. Agent 接口规范

#### 输入

| 输入项 | 类型 | 内容 |
|--------|------|------|
| `db_path` | `str` | 数据库目录路径，内含一个或多个 CSV/数据表 |
| `query` | `str` | 应用驱动的分析查询，格式 "As a [角色], I want to [意图]" |
| `tables` | `Dict[str, pd.DataFrame]` | 从数据库目录读取的所有表 |

#### 输出

Agent 必须输出：

```python
{
    "findings": [
        "Finding 1: ...",
        "Finding 2: ...",
        # 通常 3-8 条
    ],
    "suggestions": [
        "Suggestion 1: ...",
        "Suggestion 2: ...",
        # 通常 3-8 条
    ],
    # 以下为可选（用于分析 Agent 行为，不参与评分）
    "code_trajectory": [
        {"code": "...", "output": "..."},
        # 每轮代码和执行结果
    ]
}
```

#### 我们需要做的适配

```python
# benchmark_adapter_daco.py

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List

def run_agent_on_daco_instance(db_path: str, query: str) -> dict:
    """
    在一条 DACO 数据上运行我们的 Agent。
    
    Args:
        db_path: 数据库目录路径，如 "daco/data/databases/db_001"
        query:   分析查询，如 "As an advertising executive, I want to..."
    
    Returns:
        dict: {"findings": [...], "suggestions": [...], "code_trajectory": [...]}
    """
    db_path = Path(db_path)
    
    # ---- 1. 读入所有表 ----
    tables: Dict[str, pd.DataFrame] = {}
    for csv_file in db_path.glob("*.csv"):
        tables[csv_file.stem] = pd.read_csv(csv_file)
    
    # ---- 2. 构造 Schema ----
    from utils.data_inspector import describe_dataframes_schema
    schema = describe_dataframes_schema(tables, max_sample_rows=3, max_unique_values=10)
    
    # ---- 3. 调用 LLM 规划分析步骤 ----
    # 关键差异：DACO 的 query 带角色设定，需要传入 prompt
    from llm import OpenAILikeLLM, LLMConfig
    llm = OpenAILikeLLM(config=LLMConfig())
    
    planning_prompt = f"""你是一位数据分析专家。

用户查询: {query}

可用数据表:
{schema}

请规划 3-5 步分析过程，每步指定：
1. 分析目的
2. 需要用到的表
3. 具体的分析操作（如统计、对比、相关性分析等）

以 JSON 列表输出。"""
    
    plan = llm.chat(planning_prompt).content
    steps = json.loads(extract_json(plan))  # 需要解析
    
    # ---- 4. 逐步执行代码分析 ----
    from code_agent import CodeAgent
    agent = CodeAgent()
    
    code_trajectory = []
    analysis_results = []
    
    for step in steps:
        # 筛选相关表
        relevant_tables = {k: v for k, v in tables.items() if k in step.get("tables", tables.keys())}
        
        result = agent.run(
            input=f"用户需求: {query}\n当前分析步骤: {step['purpose']}\n请编写Python代码执行分析。",
            additional_args={"dfs": relevant_tables},  # 传入 DataFrame
        )
        
        code_trajectory.append({
            "step": step["purpose"],
            "result": result or "执行失败",
        })
        if result:
            analysis_results.append(result)
    
    # ---- 5. 汇总生成 Findings + Suggestions ----
    all_results = "\n\n".join(analysis_results)
    
    synthesis_prompt = f"""用户查询: {query}

以下是多轮数据分析的结果:
{all_results}

请基于以上分析，输出结构化的分析报告：
1. Findings（数据发现）: 3-8 条，每条基于具体数据证据
2. Suggestions（建议）: 3-8 条，每条具有可操作性

以 JSON 格式输出: {{"findings": [...], "suggestions": [...]}}"""
    
    final_response = llm.chat(synthesis_prompt).content
    output = json.loads(extract_json(final_response))
    output["code_trajectory"] = code_trajectory
    
    return output


def run_all_daco_test(test_file: str, db_base_dir: str, output_file: str):
    """
    批量运行整个 DACO 测试集。
    """
    results = []
    with open(test_file, "r") as f:
        for line in f:
            item = json.loads(line)
            db_path = Path(db_base_dir) / item["db_id"]
            query = item["query"]
            
            pred = run_agent_on_daco_instance(str(db_path), query)
            results.append({
                "db_id": item["db_id"],
                "query": query,
                "prediction": pred,
            })
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
```

### 3. 评估器搭建

#### 方法 A：使用官方评估脚本

```bash
cd daco
python evaluation/evaluate.py \
    --predictions ./my_predictions.json \
    --ground_truth ./data/test_h.jsonl \
    --evaluators gpt-4o-mini,claude-3.5-sonnet,llama-3-8b
```

#### 方法 B：自己实现评估器

DACO 的评估核心是 **Pairwise Comparison**——把 Agent 输出和 Ground-Truth 并列，让评判 LLM 选哪个更好。

```python
# evaluator_daco.py

import json
from typing import Dict, List

def evaluate_helpfulness(
    prediction: Dict,      # Agent 输出 {"findings": [...], "suggestions": [...]}
    ground_truth: Dict,    # GT 的 {"findings": [...], "suggestions": [...]}
    query: str,            # 原始查询
    db_title: str,         # 数据库标题
    evaluator_llm,         # 评判 LLM
) -> str:
    """
    Pairwise comparison: Agent 输出 vs Ground-Truth，返回 "Report-1" 或 "Report-2"。
    为避免位置偏差，随机交换 Report-1/Report-2 的顺序。
    """
    import random
    
    # 格式化两份报告
    report_pred = format_report(prediction)
    report_gt = format_report(ground_truth)
    
    # 随机交换顺序
    if random.random() < 0.5:
        report_1, report_2 = report_pred, report_gt
        pred_is_1 = True
    else:
        report_1, report_2 = report_gt, report_pred
        pred_is_1 = False
    
    prompt = f"""I have a database of {db_title}. {query}

I have hired two data analysts to perform the analysis, and they gave me two different reports (listed below). Which one is more helpful to my analysis?

When evaluating helpfulness, you should consider the following three rubrics in decreasing priority:
(1) relevance to my analysis goal
(2) insightfulness
(3) diversity of perspectives, especially for suggestions

Your response should be in the following format. Note: <answer> should be either Report-1 or Report-2
* Answer: <answer>
* Reasoning: <explain your reasoning here>

The reports are as follows:

# Report-1
{report_1}

# Report-2
{report_2}"""
    
    response = evaluator_llm.chat(prompt).content
    winner = parse_winner(response)  # 提取 "Report-1" 或 "Report-2"
    
    # 还原：prediction 赢了还是 GT 赢了
    if pred_is_1:
        pred_wins = (winner == "Report-1")
    else:
        pred_wins = (winner == "Report-2")
    
    return pred_wins


def format_report(report: Dict) -> str:
    """将 findings + suggestions 格式化为文本报告。"""
    lines = ["## Findings"]
    for i, f in enumerate(report.get("findings", []), 1):
        lines.append(f"{i}. {f}")
    lines.append("\n## Suggestions")
    for i, s in enumerate(report.get("suggestions", []), 1):
        lines.append(f"{i}. {s}")
    return "\n".join(lines)


def evaluate_all(predictions_file: str, ground_truth_file: str, evaluator_llms: list) -> dict:
    """
    在整个测试集上计算 Helpfulness（胜率）。
    使用多个 evaluator LLM 取平均。
    """
    with open(predictions_file) as f:
        predictions = json.load(f)
    
    gt_data = {}
    with open(ground_truth_file) as f:
        for line in f:
            item = json.loads(line)
            gt_data[item["db_id"] + "|" + item["query"]] = item
    
    results_per_evaluator = {str(llm): [] for llm in evaluator_llms}
    
    for pred_item in predictions:
        key = pred_item["db_id"] + "|" + pred_item["query"]
        gt_item = gt_data[key]
        
        for eval_llm in evaluator_llms:
            pred_wins = evaluate_helpfulness(
                prediction=pred_item["prediction"],
                ground_truth=gt_item["answer"],
                query=gt_item["query"],
                db_title=gt_item["db_id"],
                evaluator_llm=eval_llm,
            )
            results_per_evaluator[str(eval_llm)].append(1 if pred_wins else 0)
    
    # 计算每个 evaluator 的胜率
    scores = {}
    for eval_name, wins in results_per_evaluator.items():
        scores[eval_name] = sum(wins) / len(wins) * 100  # 胜率百分比
    
    # 最终分数 = 多个 evaluator 的平均
    avg_score = sum(scores.values()) / len(scores)
    
    return {
        "per_evaluator": scores,
        "average_helpfulness": avg_score,  # 50 = 与人类标注持平
    }
```

#### 评估指标汇总

| 指标 | 计算方式 | 分数范围 | 说明 |
|------|---------|---------|------|
| **Helpfulness（主指标）** | Pairwise: Agent 输出 vs GT，多 LLM 评判胜率取平均 | 0 - 100 | 50 = 与人类标注持平。GPT-4 约 42 |
| **BLEU** | `nltk.translate.bleu_score` | 0 - 100 | 辅助，字面相似度 |
| **Entailment** | NLI 模型（如 `roberta-large-mnli`）计算 P(pred entailed by GT) | 0 - 100 | 辅助，事实正确性 |
| **Point-wise Helpfulness** | 人类逐条打分（仅人类评估时用） | 0 / 1 / 2 | 0=没用, 1=边界, 2=有用 |

#### 推荐的评判 LLM 组合

| 评判器 | 说明 | 注意 |
|--------|------|------|
| **GPT-4o-mini** | OpenAI API | 便宜，效果好 |
| **Claude 3.5 Sonnet** | Anthropic API | 与 GPT-4o-mini 互补 |
| **Llama-3-8B-Instruct** | 本地部署 | 开源可复现 |

三个评判器取平均（论文中 Spearman 相关 0.90，排名高度一致）。

---

## 三、两个 Benchmark 的关键差异与适配要点

| 维度 | InsightBench | DACO |
|------|-------------|------|
| **数据格式** | 单个 CSV 文件 (500 行) | 多表关系型数据库 (平均 2.3 表) |
| **任务指令** | SMART Goal（分析目标） | 角色驱动查询 "As a [role], I want to..." |
| **Agent 需要做什么** | 自主提问 + 写代码 + 生成 Insight + 汇总 | 多轮写代码分析 + 输出 Findings + Suggestions |
| **输出格式** | insights 列表（含 question/code/insight/type）+ summary | findings 列表 + suggestions 列表 |
| **评估方式** | LLM-as-Judge（每条 GT 找最佳匹配，0-1 打分） | Pairwise Comparison（二选一，计算胜率） |
| **评估指标含义** | 0.60 = 与 GT 的平均匹配度 60% | 42 = 42% 的情况下比人类标注更好 |
| **主要适配工作** | 需要新增"自主生成问题"和"分类 Insight 类型"的能力 | 需要适配多表输入和角色驱动的 prompt |

### 我们 Agent 的改动清单

#### InsightBench 适配

1. **新增 Schema 提取逻辑** — 复用 `describe_dataframes_schema`，增加统计信息（min/max/mean/std/唯一值）
2. **新增问题生成模块** — 给 LLM 一个 prompt，让它根据 Schema + Goal 生成 3 个高层问题 + 每个问题 4 个追问（覆盖 descriptive/diagnostic/predictive/prescriptive 四类）
3. **复用 CodeAgent** — 每个问题调用 `CodeAgent.run()` 执行代码
4. **新增 Insight 提取模块** — 代码执行完后，让 LLM 解读输出，生成一句话 Insight
5. **新增 Summary 生成模块** — 汇总所有 Insight，复用 `DocWriter` 或单独 prompt

#### DACO 适配

1. **多表输入支持** — 我们现有的 `read_all_excel` + `describe_dataframes_schema` 已支持多表，基本可复用
2. **角色驱动的 prompt** — 把 DACO 的 "As a [role], I want to..." 格式直接传入我们的分析 prompt
3. **输出格式转换** — 将 `DocWriter` 的输出解析为 `{"findings": [...], "suggestions": [...]}`
4. **多轮代码交互** — 我们的 `CodeAgent` 已支持多步执行+错误重试，可直接复用
5. **去掉"地区"相关硬编码** — 当前 `analyze_region` 按地区名查找文件，需要抽象为通用的"按数据库路径读取"

---

## 四、快速开始脚本

```python
# run_benchmarks.py

"""
一键运行两个 benchmark 的评测。

Usage:
    python run_benchmarks.py --benchmark insightbench --data_dir ./insight-bench/data
    python run_benchmarks.py --benchmark daco --data_dir ./daco/data --test_file test_h.jsonl
"""

import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["insightbench", "daco"], required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--test_file", default="test_h.jsonl", help="DACO 测试文件名")
    parser.add_argument("--output_dir", default="./benchmark_results")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.benchmark == "insightbench":
        from benchmark_adapter_insightbench import run_agent_on_insightbench_dataset
        
        data_dir = Path(args.data_dir)
        all_results = {}
        
        for dataset_dir in sorted(data_dir.glob("dataset_*")):
            print(f"Running on {dataset_dir.name}...")
            result = run_agent_on_insightbench_dataset(str(dataset_dir))
            all_results[dataset_dir.name] = result
        
        with open(output_dir / "insightbench_predictions.json", "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"Done! {len(all_results)} datasets processed.")
        print(f"Next: run evaluator on {output_dir / 'insightbench_predictions.json'}")
    
    elif args.benchmark == "daco":
        from benchmark_adapter_daco import run_all_daco_test
        
        test_file = Path(args.data_dir) / args.test_file
        db_base_dir = Path(args.data_dir) / "databases"
        output_file = output_dir / "daco_predictions.json"
        
        run_all_daco_test(str(test_file), str(db_base_dir), str(output_file))
        
        print(f"Done! Predictions saved to {output_file}")
        print(f"Next: run evaluator on {output_file}")

if __name__ == "__main__":
    main()
```
