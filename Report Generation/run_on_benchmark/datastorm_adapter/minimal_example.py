"""单条数据集快速验证脚本。

在跑完整 benchmark 之前，先用这个脚本验证适配层是否正常工作。

用法（任意目录均可）：
    python "D:/DataAgents/Report Generation/run_on_benchmark/datastorm_adapter/minimal_example.py"

API Key 来源（优先级从高到低）：
    1. 环境变量 OPENAI_API_KEY
    2. MyDataStorm/datastorm/llm_config.json 中的 api_key
"""

from __future__ import annotations

import os
import sys

# 路径计算：所有路径都基于脚本自身位置，与运行目录无关
_script_dir   = os.path.dirname(os.path.abspath(__file__))   # .../datastorm_adapter
_run_on_bench = os.path.dirname(_script_dir)                  # .../run_on_benchmark
_insight_bench = os.path.join(_run_on_bench, "insight-bench") # .../insight-bench

for _p in [_run_on_bench, _insight_bench]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from insightbench import benchmarks
from datastorm_adapter.adapter import DataStormAdapter
from unified_scorer import score_insights as g_eval_insights, score_summary as g_eval_summary, get_scorer_config

# ── 配置 ──────────────────────────────────────────────────────────────
_DATA_DIR    = os.path.join(_insight_bench, "data", "notebooks")
DATASET_JSON = os.path.join(_DATA_DIR, "flag-1.json")
MODEL_NAME   = None   # None = 使用 llm_config.json 中的配置
MAX_LAYERS   = 2   # 快速验证用 2 层，正式跑用 3-5
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    if not os.path.exists(DATASET_JSON):
        print(f"ERROR: 找不到数据文件 {DATASET_JSON}")
        print(f"       请确认 insight-bench 目录存在于 {_insight_bench}")
        sys.exit(1)

    # API Key: 环境变量 > llm_config.json
    if not os.environ.get("OPENAI_API_KEY"):
        import json as _json
        from pathlib import Path as _Path
        _json_path = _Path(__file__).resolve().parents[3] / "MyDataStorm" / "datastorm" / "llm_config.json"
        try:
            if _json_path.is_file():
                _cfg = _json.loads(_json_path.read_text(encoding="utf-8"))
                if _cfg.get("api_key"):
                    os.environ["OPENAI_API_KEY"] = _cfg["api_key"]
        except Exception:
            pass

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: 未找到 API Key。")
        print("       请在 MyDataStorm/datastorm/llm_config.json 中设置 api_key")
        print("       或设置环境变量 OPENAI_API_KEY")
        sys.exit(1)

    print(f"Loading dataset: {DATASET_JSON}")
    dataset_dict = benchmarks.load_dataset_dict(DATASET_JSON)
    metadata = dataset_dict.get("metadata", {})

    # JSON 里的 csv 路径是相对于 insight-bench 目录的，转成绝对路径
    csv_path = os.path.join(_insight_bench, dataset_dict["dataset_csv_path"])
    user_csv = dataset_dict.get("user_dataset_csv_path")
    if user_csv:
        user_csv = os.path.join(_insight_bench, user_csv)

    print(f"Goal: {metadata.get('goal', 'N/A')}")
    print(f"CSV:  {csv_path}")
    print(f"GT insights: {len(dataset_dict['insights'])}")
    print()

    adapter = DataStormAdapter(
        model_name=MODEL_NAME,
        max_layers=MAX_LAYERS,
        verbose=True,
    )

    print("Running DataSTORM adapter...")
    pred_insights, pred_summary = adapter.get_insights(
        dataset_csv_path=csv_path,
        user_dataset_csv_path=user_csv,
        goal=metadata.get("goal", "Find interesting trends in this dataset"),
        dataset_description=metadata.get("dataset_description", ""),
        return_summary=True,
    )

    print("\n── Predicted Insights ──────────────────────────────────────")
    for i, ins in enumerate(pred_insights, 1):
        print(f"[{i}] {ins[:200]}")

    print("\n── Predicted Summary ───────────────────────────────────────")
    print(pred_summary[:500])

    print("\n── Evaluation ──────────────────────────────────────────────")
    scorer_cfg = get_scorer_config()
    print(f"Scorer: G-Eval | Judge: {scorer_cfg['model']} | logprobs: {scorer_cfg['logprobs']}")
    score_insights = g_eval_insights(
        pred_insights=pred_insights,
        gt_insights=dataset_dict["insights"],
    )
    score_summary = g_eval_summary(
        pred_summary=pred_summary,
        gt_summary=dataset_dict.get("summary", ""),
    )
    print(f"score_insights: {score_insights:.4f}")
    print(f"score_summary:  {score_summary:.4f}")


if __name__ == "__main__":
    main()
