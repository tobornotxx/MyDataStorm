# run_on_benchmark

在 InsightBench / DACO benchmark 上运行和评估我们的 Agent。

## 目录结构

```
run_on_benchmark/
├── README.md               # 本文档
├── __init__.py
├── run.py                  # 统一入口（命令行）
├── adapter_insightbench.py # InsightBench 适配器
├── adapter_daco.py         # DACO 适配器
├── evaluator.py            # 统一评估器
│
├── insight-bench/          # ← git clone 到这里
│   └── ...
└── daco/                   # ← git clone 到这里
    └── ...
```

## 快速开始

### 0. 准备

```bash
# 克隆 benchmark 仓库到本目录下
cd run_on_benchmark
git clone https://github.com/ServiceNow/insight-bench
git clone https://github.com/shirley-wu/daco

# 确保项目的 .env 文件已配置好 LLM API
# 需要的环境变量：MODEL_DEFAULT, API_BASE_DEFAULT, API_KEY_DEFAULT
```

### 1. 运行 InsightBench

```bash
# 从项目根目录执行

# 完整运行（100 个数据集）
python -m run_on_benchmark.run --benchmark insightbench --data_dir ./run_on_benchmark/insight-bench

# 调试模式（只跑 3 个）
python -m run_on_benchmark.run --benchmark insightbench --data_dir ./run_on_benchmark/insight-bench --limit 3

# 只跑评估（已有预测结果）
python -m run_on_benchmark.run --benchmark insightbench --eval_only --predictions ./benchmark_results/insightbench_predictions.json --data_dir ./run_on_benchmark/insight-bench
```

### 2. 运行 DACO

```bash
# 完整运行 Test-H（100 条人工精标）
python -m run_on_benchmark.run --benchmark daco --data_dir ./run_on_benchmark/daco

# 调试模式
python -m run_on_benchmark.run --benchmark daco --data_dir ./run_on_benchmark/daco --limit 5

# 跑 Test-A（284 条自动标注）
python -m run_on_benchmark.run --benchmark daco --data_dir ./run_on_benchmark/daco --test_file test_a.jsonl
```

### 3. 输出

运行后会在 `benchmark_results/` 目录生成：

```
benchmark_results/
├── insightbench_predictions.json   # Agent 的预测结果
├── insightbench_scores.json        # 评估分数
├── daco_predictions.json
└── daco_scores.json
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--benchmark` | 必填 | `insightbench` 或 `daco` |
| `--data_dir` | 必填 | benchmark 仓库根目录路径 |
| `--output_dir` | `./benchmark_results` | 输出目录 |
| `--test_file` | `test_h.jsonl` | [DACO] 测试文件名 |
| `--limit` | `0` | 限制样本数（0=不限制，调试时设 3-5） |
| `--max_queries` | `5` | 每个数据集的分析问题数 |
| `--eval_only` | `false` | 仅评估，不运行 Agent |
| `--predictions` | `""` | [eval_only] 预测文件路径 |
| `--evaluator` | `gpt-4o-mini` | 评估器模型名称 |

## 单个数据集的运行流程

### InsightBench（每个 dataset）

```
输入:
  ├── data.csv (500 行)
  └── goal.txt ("Analyze...")

Agent 流程:
  1. describe_dataframes_schema(df) → schema 文本
  2. LLM(schema + goal) → 5 个分析问题 (descriptive/diagnostic/predictive/prescriptive)
  3. 对每个问题:
     CodeAgent.run(问题 + df) → 代码执行结果
     LLM(问题 + 执行结果) → 一句话 insight
  4. LLM(所有 insights + goal) → summary

输出:
  {
    "insights": [{"question", "insight", "type"}, ...],  # 5 条
    "summary": "..."
  }
```

### DACO（每条样本）

```
输入:
  ├── db_path/ (多个 CSV / SQLite)
  └── query: "As a [role], I want to [intention]"

Agent 流程:
  1. 加载所有表 → describe_dataframes_schema → schema 文本
  2. LLM(schema + query) → 5 步分析计划 [{"purpose", "tables"}, ...]
  3. 对每步:
     CodeAgent.run(步骤描述 + 相关表) → 代码执行结果
  4. LLM(所有结果 + query) → {"findings": [...], "suggestions": [...]}

输出:
  {
    "findings": ["...", ...],      # 3-8 条
    "suggestions": ["...", ...],   # 3-8 条
    "code_trajectory": [...]       # 可选
  }
```

## 评估逻辑

### InsightBench 评估

**方法**: LLM-as-Judge, One-to-Many Matching

```
对每条 GT insight:
  score_i = max over all pred insights of LLM_score(gt_i, pred_j)

Insight Score = mean(score_i for i in GT)
Summary Score = LLM_score(gt_summary, pred_summary)
Overall       = (Insight Score + Summary Score) / 2
```

**分数范围**: 0.0 - 1.0，参考基线 AgentPoirot (GPT-4o) = 0.52

### DACO 评估

**方法**: Pairwise Comparison

```
对每条样本:
  将 Agent 报告和 GT 报告随机排序为 Report-1 / Report-2
  LLM 评判哪个更好
  记录 Agent 是否胜出

Helpfulness = wins / total × 100%
```

**分数范围**: 0 - 100，50 = 与人类标注持平，参考基线 GPT-4 = 42

## Token 消耗估算

### InsightBench（100 个数据集）

| 阶段 | 每 dataset 调用次数 | 每次 tokens（估算） | 100 dataset 总 tokens |
|------|-------------------|-------------------|----------------------|
| Schema 生成 | 1 | ~500 (输出) | — （本地计算，不消耗 API） |
| **问题生成** | 1 | ~800 in + ~500 out = **1,300** | 130K |
| **CodeAgent 执行** | 5 问题 × ~2 轮 = 10 | ~1,500 in + ~800 out = **2,300** | 2,300K |
| **Insight 提取** | 5 | ~1,000 in + ~200 out = **1,200** | 600K |
| **Summary 生成** | 1 | ~1,500 in + ~500 out = **2,000** | 200K |
| **Agent 小计** | | | **~3,230K tokens** |
| **评估 (LLM-Eval)** | ~5 GT × 5 pred = 25 匹配 + 1 summary | ~600 each = **15,600** | 1,560K |
| **评估小计** | | | **~1,560K tokens** |
| **总计** | | | **~4,800K tokens** |

### DACO Test-H（100 条样本）

| 阶段 | 每条调用次数 | 每次 tokens（估算） | 100 条总 tokens |
|------|------------|-------------------|----------------|
| Schema 生成 | 1 | — （本地） | — |
| **分析规划** | 1 | ~1,200 in + ~500 out = **1,700** | 170K |
| **CodeAgent 执行** | 5 步 × ~2 轮 = 10 | ~2,000 in + ~800 out = **2,800** | 2,800K |
| **报告汇总** | 1 | ~3,000 in + ~800 out = **3,800** | 380K |
| **Agent 小计** | | | **~3,350K tokens** |
| **评估 (Pairwise)** | 1 | ~2,000 in + ~100 out = **2,100** | 210K |
| **评估小计** | | | **~210K tokens** |
| **总计** | | | **~3,560K tokens** |

### 费用估算（按 GPT-4o-mini 价格，$0.15/1M input + $0.60/1M output）

| Benchmark | 总 tokens | 预估费用 |
|-----------|----------|---------|
| InsightBench 全量 (100 datasets) | ~4.8M | **~$1.5 - $3** |
| DACO Test-H 全量 (100 samples) | ~3.6M | **~$1 - $2.5** |
| 两个都跑完 | ~8.4M | **~$2.5 - $5.5** |

> 以上为 GPT-4o-mini 价格估算。如果用 GPT-4o 则约贵 20 倍（~$50-100）。
> 如果用本地部署的开源模型（如 Qwen、DeepSeek），API 费用为 0，仅消耗 GPU 算力。
> 调试建议：先用 `--limit 3` 跑 3 个样本验证流程，消耗约 150K tokens（< $0.05）。

### 时间估算

| 场景 | 预估耗时 |
|------|---------|
| InsightBench 全量 (100 datasets, GPT-4o-mini) | 30-60 分钟 |
| DACO Test-H 全量 (100 samples, GPT-4o-mini) | 30-60 分钟 |
| 调试 3 个样本 | 2-5 分钟 |
| 评估 (InsightBench 全量) | 15-30 分钟 |
| 评估 (DACO 全量) | 5-10 分钟 |

## 注意事项

1. **InsightBench 仓库目录结构可能有变** — 代码中做了多路径兼容（`data/dataset_*` 和 `dataset_*`），如果还不对需手动检查
2. **DACO 的数据库格式多样** — 支持 CSV / Excel / SQLite，但有些数据库可能有特殊编码问题
3. **评估器的稳定性** — 建议评估时用 `temperature=0`（代码已默认设置），同一模型多次评估的分数差异 < 0.02
4. **CodeAgent 的超时** — 默认 60 秒，对大表可能不够，可在 `.env` 或代码中调整
5. **.gitignore** — 建议把 `insight-bench/` 和 `daco/` 加入 `.gitignore`，避免把大数据集提交到仓库
