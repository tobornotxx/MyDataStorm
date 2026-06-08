"""在 InsightBench 上运行 MyDataStorm 的入口脚本。

用法：
    # 从 insight-bench 目录运行
    cd "Report Generation/run_on_benchmark/insight-bench"

    python ../datastorm_adapter/run_benchmark.py \
        --openai_api_key sk-... \
        --savedir results/datastorm \
        --benchmark_type toy

    # 或者在 MyDataStorm/datastorm/llm_config.json 中配好 api_key 后直接运行
    python ../datastorm_adapter/run_benchmark.py --benchmark_type standard

参数说明：
    --benchmark_type  toy(5条) / standard(30条) / full(100条)，默认 toy
    --n_datasets      只跑前 N 条数据集（覆盖 benchmark_type 的上限），默认跑全部
    --datadir         InsightBench 数据目录，默认 data/notebooks
    --savedir_base    结果保存根目录，默认 results/datastorm
    --openai_api_key  OpenAI API key（也可用环境变量）
    --model_name      LLM 模型（不指定则使用 llm_config.json 中的配置）
    --max_layers      DataSTORM 探索层数，默认 3
    --score_name      评估指标 rouge1 / g_eval，默认 rouge1
    --start_from      从第几个 flag 开始（断点续跑），默认 1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path

# 路径计算：所有路径都基于脚本自身位置，与运行目录无关
_script_dir    = os.path.dirname(os.path.abspath(__file__))   # .../datastorm_adapter
_run_on_bench  = os.path.dirname(_script_dir)                  # .../run_on_benchmark
_insight_bench = os.path.join(_run_on_bench, "insight-bench")  # .../insight-bench

for _p in [_run_on_bench, _insight_bench]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MyDataStorm on InsightBench"
    )
    parser.add_argument("--openai_api_key", default=None)
    parser.add_argument("--api_base", default=None,
                        help="OpenAI API base URL（覆盖 llm_config.json 中的值）")
    parser.add_argument("--benchmark_type", default="toy",
                        choices=["toy", "standard", "full"])
    parser.add_argument("--n_datasets", type=int, default=None,
                        help="只跑前 N 条数据集，不指定则跑 benchmark_type 对应的全部")
    parser.add_argument("--only", type=int, default=None,
                        help="只运行指定的 flag 编号（如 --only 4 只跑 flag-4）")
    parser.add_argument("--datadir", default=None,
                        help="InsightBench 数据目录，默认自动定位到 insight-bench/data/notebooks")
    parser.add_argument("--savedir_base", default=None,
                        help="结果保存根目录，默认 Report Generation/results/datastorm")
    parser.add_argument("--model_name", default=None,
                        help="LLM 模型名（不指定则使用 llm_config.json 中的配置）")
    parser.add_argument("--max_layers", type=int, default=3)
    parser.add_argument("--questions_per_layer", type=int, default=2,
                        help="每层 Planner 生成的问题数（默认 2）")
    parser.add_argument("--follow_up", type=int, default=None,
                        help="每层跟进问题数 m（默认等于 questions_per_layer）")
    parser.add_argument("--exploratory", type=int, default=None,
                        help="每层探索问题数 n（默认等于 questions_per_layer）")
    parser.add_argument("--start_from", type=int, default=1,
                        help="从第几个 flag 开始（断点续跑）")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 路径解析（不依赖运行目录）
    _report_gen_dir = os.path.dirname(_run_on_bench)  # .../Report Generation
    datadir     = args.datadir     or os.path.join(_insight_bench, "data", "notebooks")
    savedir_base_str = args.savedir_base or os.path.join(_report_gen_dir, "results", "datastorm")

    # 创建保存目录（提前创建，用于放日志文件）
    savedir_base = Path(savedir_base_str)
    savedir_base.mkdir(parents=True, exist_ok=True)

    # 配置 logging：同时输出到终端和文件
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    # 确定日志文件名
    if args.only:
        log_filename = f"run_flag-{args.only}.log"
    else:
        log_filename = "run.log"
    log_path = savedir_base / log_filename

    # 设置 root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # 捕获所有级别

    # 终端 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(console_handler)

    # 文件 handler (始终 DEBUG 级别，记录所有细节)
    file_handler = logging.FileHandler(str(log_path), mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(file_handler)

    logger.info("Log file: %s", log_path)

    if not os.path.exists(datadir):
        print(f"ERROR: 数据目录不存在: {datadir}")
        sys.exit(1)

    from insightbench import benchmarks
    from datastorm_adapter.adapter import DataStormAdapter
    from unified_scorer import score_insights as g_eval_insights, score_summary as g_eval_summary, get_scorer_config

    # 设置 API key
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key

    # 检查 api_key 来源：CLI > 环境变量 > llm_config.json
    if not os.environ.get("OPENAI_API_KEY"):
        # 也尝试从 llm_config.json 读取
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
        print("ERROR: 需要提供 OpenAI API key")
        print("       方式1: 设置环境变量 OPENAI_API_KEY")
        print("       方式2: --openai_api_key 参数")
        print("       方式3: MyDataStorm/datastorm/llm_config.json 中设置 api_key")
        sys.exit(1)

    # 加载 benchmark 数据集列表
    dataset_paths = benchmarks.get_benchmark(args.benchmark_type, datadir)
    if args.n_datasets is not None:
        dataset_paths = dataset_paths[: args.n_datasets]
    logger.info(
        "Benchmark: %s (%d datasets)", args.benchmark_type, len(dataset_paths)
    )

    # 初始化适配器（复用同一个实例，避免重复初始化）
    adapter = DataStormAdapter(
        model_name=args.model_name,
        max_layers=args.max_layers,
        questions_per_layer=args.questions_per_layer,
        follow_up_per_layer=args.follow_up,
        exploratory_per_layer=args.exploratory,
        openai_api_key=args.openai_api_key,
        api_base=args.api_base,
        verbose=args.verbose,
    )

    all_scores: list[dict] = []
    summary_path = savedir_base / "summary.json"

    # 加载已有结果（断点续跑）
    if summary_path.exists():
        with open(summary_path) as f:
            all_scores = json.load(f)
        logger.info("Loaded %d existing results from %s", len(all_scores), summary_path)

    completed_flags = {r["flag"] for r in all_scores if r.get("status") == "ok"}

    for dataset_json_path in dataset_paths:
        # 解析 flag id
        flag_id = Path(dataset_json_path).stem  # e.g. "flag-1"
        flag_num = int(flag_id.split("-")[1])

        # --only: 只运行指定的 flag
        if args.only is not None and flag_num != args.only:
            continue
        if flag_num < args.start_from:
            continue
        if flag_id in completed_flags and args.only is None:
            logger.info("Skipping %s (already completed)", flag_id)
            continue

        logger.info("=" * 60)
        logger.info("Processing %s (%s)", flag_id, dataset_json_path)

        dataset_dict = benchmarks.load_dataset_dict(dataset_json_path)
        metadata = dataset_dict.get("metadata", {})

        savedir = savedir_base / flag_id
        savedir.mkdir(parents=True, exist_ok=True)
        adapter.savedir = str(savedir)

        # JSON 里的 csv 路径是相对于 insight-bench 目录的，转成绝对路径
        csv_path = os.path.join(_insight_bench, dataset_dict["dataset_csv_path"])
        user_csv = dataset_dict.get("user_dataset_csv_path")
        if user_csv:
            user_csv = os.path.join(_insight_bench, user_csv)

        try:
            pred_insights, pred_summary = adapter.get_insights(
                dataset_csv_path=csv_path,
                user_dataset_csv_path=user_csv,
                goal=metadata.get("goal", "Find interesting trends in this dataset"),
                dataset_description=metadata.get("dataset_description", ""),
                return_summary=True,
            )

            score_insights = g_eval_insights(
                pred_insights=pred_insights,
                gt_insights=dataset_dict["insights"],
            )
            score_summary = g_eval_summary(
                pred_summary=pred_summary,
                gt_summary=dataset_dict.get("summary", ""),
            )

            result = {
                "flag": flag_id,
                "score_insights": float(score_insights),
                "score_summary": float(score_summary),
                "n_pred_insights": len(pred_insights),
                "n_gt_insights": len(dataset_dict["insights"]),
                "pred_summary": pred_summary[:300],
                "scorer": get_scorer_config(),
                "status": "ok",
            }

            # 保存单条结果
            with open(savedir / "result.json", "w") as f:
                json.dump({
                    **result,
                    "pred_insights": pred_insights,
                    "gt_insights": dataset_dict["insights"],
                }, f, indent=2, ensure_ascii=False)

            logger.info(
                "%s → score_insights=%.4f, score_summary=%.4f",
                flag_id, score_insights, score_summary,
            )

        except Exception as e:
            logger.error("Failed on %s: %s", flag_id, e)
            traceback.print_exc()
            result = {
                "flag": flag_id,
                "score_insights": 0.0,
                "score_summary": 0.0,
                "status": f"error: {e}",
            }

        # 替换同一 flag 的旧记录（如果有），避免重复
        all_scores = [r for r in all_scores if r.get("flag") != flag_id]
        all_scores.append(result)

        # 每条结果后立即保存 summary（防止中途崩溃丢失进度）
        with open(summary_path, "w") as f:
            json.dump(all_scores, f, indent=2, ensure_ascii=False)

    # 打印汇总统计
    ok_results = [r for r in all_scores if r.get("status") == "ok"]
    scorer_cfg = get_scorer_config()
    run_info = {
        "benchmark_type": args.benchmark_type,
        "model_name": args.model_name,
        "max_layers": args.max_layers,
        "scorer": scorer_cfg,
        "n_datasets": len(dataset_paths),
        "n_completed": len(ok_results),
        "n_errors": len(all_scores) - len(ok_results),
    }

    # 保存 run_info.json
    with open(savedir_base / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)

    if ok_results:
        avg_insights = sum(r["score_insights"] for r in ok_results) / len(ok_results)
        avg_summary = sum(r["score_summary"] for r in ok_results) / len(ok_results)
        print("\n" + "=" * 60)
        print(f"Benchmark: {args.benchmark_type} | Model: {args.model_name}")
        print(f"Scorer: G-Eval | Judge: {scorer_cfg['model']} | logprobs: {scorer_cfg['logprobs']}")
        print(f"Completed: {len(ok_results)}/{len(dataset_paths)}")
        print(f"Avg score_insights: {avg_insights:.4f}")
        print(f"Avg score_summary:  {avg_summary:.4f}")
        print(f"Results saved to: {savedir_base.resolve()}")
        print("=" * 60)
    else:
        print("No successful results.")


if __name__ == "__main__":
    main()
