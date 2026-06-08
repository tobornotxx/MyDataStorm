"""
evaluator.py — InsightBench / DACO 评估器

使用 G-Eval (LLM-as-Judge) 对预测结果与 Ground Truth 进行语义比对打分。
底层调用统一打分器 unified_scorer.py，配置来自 MyDataStorm/datastorm/llm_config.json。
"""

import json
import re
from pathlib import Path
from typing import Dict, Any

import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils import logger


def evaluate_insightbench(
    predictions_file: str,
    data_dir: str,
) -> Dict[str, Any]:
    """
    评估 InsightBench 预测结果。

    Args:
        predictions_file: Agent 预测结果 JSON 文件路径
        data_dir: InsightBench 数据集根目录

    Returns:
        dict: 包含 avg_insight_score, avg_summary_score, overall, per_dataset 等
    """
    from run_on_benchmark.unified_scorer import (
        score_insights as _score_insights,
        score_summary as _score_summary,
        get_scorer_config,
    )
    from run_on_benchmark.adapter_insightbench import load_ground_truth

    with open(predictions_file, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    data_path = Path(data_dir)
    scorer_cfg = get_scorer_config()
    logger.info(f"[Eval] Scorer: G-Eval | Judge: {scorer_cfg['model']} | logprobs: {scorer_cfg['logprobs']}")

    per_dataset = {}
    total_insight_score = 0.0
    total_summary_score = 0.0
    n_evaluated = 0

    for ds_name, pred_data in predictions.items():
        gt_path = data_path / "data" / "notebooks" / f"{ds_name}.json"
        if not gt_path.exists():
            gt_path = data_path / f"{ds_name}.json"
        if not gt_path.exists():
            gt_path = data_path / "data" / ds_name
        if not gt_path.exists():
            logger.warning(f"[Eval] GT not found for {ds_name}, skipping")
            continue

        try:
            gt = load_ground_truth(str(gt_path))
        except Exception as e:
            logger.warning(f"[Eval] Failed to load GT for {ds_name}: {e}")
            continue

        gt_insights = gt.get("insights", [])
        gt_summary = gt.get("summary", "")
        pred_insights_raw = pred_data.get("insights", [])
        pred_summary = pred_data.get("summary", "")

        # 统一转成纯字符串列表
        pred_insights: list[str] = [
            p.get("insight", "") if isinstance(p, dict) else str(p)
            for p in pred_insights_raw
        ]
        gt_insight_strs: list[str] = [
            g if isinstance(g, str) else str(g) for g in gt_insights
        ]

        avg_insight = _score_insights(pred_insights, gt_insight_strs)
        summary_score = _score_summary(pred_summary, gt_summary) if (gt_summary and pred_summary) else 0.0

        per_dataset[ds_name] = {
            "insight_score": round(avg_insight, 4),
            "summary_score": round(summary_score, 4),
            "n_gt_insights": len(gt_insights),
            "n_pred_insights": len(pred_insights),
        }

        total_insight_score += avg_insight
        total_summary_score += summary_score
        n_evaluated += 1

    avg_insight_score = total_insight_score / n_evaluated if n_evaluated > 0 else 0.0
    avg_summary_score = total_summary_score / n_evaluated if n_evaluated > 0 else 0.0

    return {
        "avg_insight_score": round(avg_insight_score, 4),
        "avg_summary_score": round(avg_summary_score, 4),
        "overall": round((avg_insight_score + avg_summary_score) / 2, 4),
        "n_datasets_evaluated": n_evaluated,
        "per_dataset": per_dataset,
    }


# ============================================================
# DACO 评估
# ============================================================

def evaluate_daco(
    predictions_file: str,
    ground_truth_file: str,
) -> Dict[str, Any]:
    """
    评估 DACO 预测结果。
    """
    from run_on_benchmark.unified_scorer import score_summary as _score_summary

    with open(predictions_file, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    gt_items = []
    with open(ground_truth_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                gt_items.append(json.loads(line))

    scores = []
    for pred_item in predictions:
        db_id = pred_item.get("db_id", "")
        query = pred_item.get("query", "")
        prediction = pred_item.get("prediction", {})

        gt_match = None
        for gt in gt_items:
            if gt.get("db_id") == db_id and gt.get("query") == query:
                gt_match = gt
                break

        if gt_match is None:
            continue

        pred_text = json.dumps(prediction, ensure_ascii=False)
        gt_text = json.dumps(gt_match, ensure_ascii=False)

        score = _score_summary(pred_text, gt_text)
        scores.append(score)

    avg_helpfulness = sum(scores) / len(scores) * 100 if scores else 0.0

    return {
        "average_helpfulness": round(avg_helpfulness, 1),
        "n_evaluated": len(scores),
    }