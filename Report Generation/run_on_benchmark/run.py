"""
run.py — Benchmark 评测统一入口

Usage:
    # 跑 InsightBench（全部 100 个数据集）
    python -m run_on_benchmark.run --benchmark insightbench --data_dir ./run_on_benchmark/insight-bench

    # 跑 InsightBench（只跑前 5 个，用于调试）
    python -m run_on_benchmark.run --benchmark insightbench --data_dir ./run_on_benchmark/insight-bench --limit 5

    # 跑 DACO Test-H（100 条人工精标）
    python -m run_on_benchmark.run --benchmark daco --data_dir ./run_on_benchmark/daco

    # 跑 DACO（指定测试文件和输出目录）
    python -m run_on_benchmark.run --benchmark daco --data_dir ./run_on_benchmark/daco --test_file test_h.jsonl --output_dir ./my_results

    # 只跑评估（已经有了预测结果）
    python -m run_on_benchmark.run --benchmark insightbench --eval_only --predictions ./benchmark_results/insightbench_predictions.json --data_dir ./run_on_benchmark/insight-bench
"""

import argparse
import json
import sys
import time
from pathlib import Path

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="在 InsightBench / DACO 上运行 Agent 并评估",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--benchmark",
        choices=["insightbench", "daco"],
        required=True,
        help="要运行的 benchmark 名称",
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="benchmark 仓库的根目录（clone 后的路径）",
    )
    parser.add_argument(
        "--output_dir",
        default="./benchmark_results",
        help="输出目录（默认 ./benchmark_results）",
    )
    parser.add_argument(
        "--test_file",
        default="test_h.jsonl",
        help="[DACO] 测试文件名（默认 test_h.jsonl）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="限制运行的数据集/样本数量（0=不限制，调试时设为 3-5）",
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=5,
        help="每个数据集最多生成的分析查询数（默认 5）",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="仅运行评估，不运行 Agent（需要 --predictions）",
    )
    parser.add_argument(
        "--predictions",
        default="",
        help="[eval_only] 已有的预测结果文件路径",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.benchmark == "insightbench":
        _run_insightbench(args, output_dir)
    elif args.benchmark == "daco":
        _run_daco(args, output_dir)


def _run_insightbench(args, output_dir: Path):
    from run_on_benchmark.adapter_insightbench import (
        run_agent_on_dataset,
        load_ground_truth,
    )
    from run_on_benchmark.evaluator import evaluate_insightbench

    data_dir = Path(args.data_dir)
    predictions_file = output_dir / "insightbench_predictions.json"

    if not args.eval_only:
        # ---- 运行 Agent ----
        dataset_items = sorted(data_dir.glob("data/dataset_*"))
        if not dataset_items:
            # 兼容直接指向 data/ 目录的情况
            dataset_items = sorted(data_dir.glob("dataset_*"))
        if not dataset_items:
            # 兼容新版 InsightBench: data/notebooks/flag-*.json
            dataset_items = sorted(data_dir.glob("data/notebooks/flag-*.json"))
        if not dataset_items:
            print(
                f"[ERROR] 在 {data_dir} 下未找到可用数据集。"
                f"支持目录: data/dataset_* 或 dataset_*；"
                f"支持文件: data/notebooks/flag-*.json。"
            )
            return

        if args.limit > 0:
            dataset_items = dataset_items[: args.limit]

        print(f"[InsightBench] 共 {len(dataset_items)} 个数据集")
        all_results = {}
        start_time = time.time()

        for i, ds_item in enumerate(dataset_items, 1):
            ds_name = ds_item.stem if ds_item.is_file() else ds_item.name
            print(f"  [{i}/{len(dataset_items)}] {ds_name} ...", end=" ", flush=True)
            t0 = time.time()
            try:
                result = run_agent_on_dataset(
                    dataset_dir=str(ds_item),
                    max_queries=args.max_queries,
                )
                all_results[ds_name] = result
                print(f"OK ({time.time() - t0:.1f}s, {len(result.get('insights', []))} insights)")
            except Exception as e:
                print(f"FAILED: {e}")
                all_results[ds_name] = {"insights": [], "summary": "", "error": str(e)}

        with open(predictions_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        elapsed = time.time() - start_time
        print(f"\n[InsightBench] Agent 运行完成: {len(all_results)} 个数据集, 耗时 {elapsed:.0f}s")
        print(f"  预测结果保存到: {predictions_file}")
    else:
        predictions_file = Path(args.predictions) if args.predictions else predictions_file

    # ---- 运行评估 ----
    if predictions_file.exists():
        print(f"\n[InsightBench] 开始评估 ...")
        scores = evaluate_insightbench(
            predictions_file=str(predictions_file),
            data_dir=str(data_dir),
        )
        scores_file = output_dir / "insightbench_scores.json"
        with open(scores_file, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2, ensure_ascii=False)
        print(f"  Insight Score:  {scores.get('avg_insight_score', 0):.3f}")
        print(f"  Summary Score:  {scores.get('avg_summary_score', 0):.3f}")
        print(f"  Overall Score:  {scores.get('overall', 0):.3f}")
        print(f"  详细分数保存到: {scores_file}")


def _run_daco(args, output_dir: Path):
    from run_on_benchmark.adapter_daco import run_agent_on_instance
    from run_on_benchmark.evaluator import evaluate_daco

    data_dir = Path(args.data_dir)
    predictions_file = output_dir / "daco_predictions.json"

    if not args.eval_only:
        # ---- 加载测试数据 ----
        test_file = data_dir / "data" / args.test_file
        if not test_file.exists():
            test_file = data_dir / args.test_file
        if not test_file.exists():
            print(f"[ERROR] 测试文件 {test_file} 不存在")
            return

        db_base_dir = data_dir / "data" / "databases"
        if not db_base_dir.exists():
            db_base_dir = data_dir / "databases"

        test_items = []
        with open(test_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    test_items.append(json.loads(line))

        if args.limit > 0:
            test_items = test_items[: args.limit]

        print(f"[DACO] 共 {len(test_items)} 条测试样本")
        all_results = []
        start_time = time.time()

        for i, item in enumerate(test_items, 1):
            db_id = item["db_id"]
            query = item["query"]
            db_path = db_base_dir / db_id
            print(f"  [{i}/{len(test_items)}] {db_id}: {query[:60]}...", end=" ", flush=True)
            t0 = time.time()
            try:
                pred = run_agent_on_instance(
                    db_path=str(db_path),
                    query=query,
                    max_queries=args.max_queries,
                )
                all_results.append({
                    "db_id": db_id,
                    "query": query,
                    "prediction": pred,
                })
                n_findings = len(pred.get("findings", []))
                n_suggestions = len(pred.get("suggestions", []))
                print(f"OK ({time.time() - t0:.1f}s, {n_findings}F+{n_suggestions}S)")
            except Exception as e:
                print(f"FAILED: {e}")
                all_results.append({
                    "db_id": db_id,
                    "query": query,
                    "prediction": {"findings": [], "suggestions": [], "error": str(e)},
                })

        with open(predictions_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        elapsed = time.time() - start_time
        print(f"\n[DACO] Agent 运行完成: {len(all_results)} 条, 耗时 {elapsed:.0f}s")
        print(f"  预测结果保存到: {predictions_file}")
    else:
        predictions_file = Path(args.predictions) if args.predictions else predictions_file

    # ---- 运行评估 ----
    if predictions_file.exists():
        test_file = data_dir / "data" / args.test_file
        if not test_file.exists():
            test_file = data_dir / args.test_file
        print(f"\n[DACO] 开始评估 ...")
        scores = evaluate_daco(
            predictions_file=str(predictions_file),
            ground_truth_file=str(test_file),
        )
        scores_file = output_dir / "daco_scores.json"
        with open(scores_file, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2, ensure_ascii=False)
        print(f"  Helpfulness: {scores.get('average_helpfulness', 0):.1f}%")
        print(f"  详细分数保存到: {scores_file}")


if __name__ == "__main__":
    main()
