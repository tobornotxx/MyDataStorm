"""
交互式数据查询脚本
在终端中输入自然语言查询，由 Code Agent 执行并返回结果。
用法: python human_validation/get_agent_result.py
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import dotenv

# 确保项目根目录在 sys.path 中，以便导入项目模块
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

dotenv.load_dotenv(_PROJECT_ROOT / ".env")

from utils import logger
from utils.data_inspector import (
    describe_dataframes_schema,
    DataInspectorMCPTool,
)
from utils.file_io import read_all_excel

# ============================================================
# 变量设置区 — 按需修改
# ============================================================

# 地区名称
region_name = "江北区"

# 考核评估总表
assessment_file = _PROJECT_ROOT / "data" / "overview_data" / "考核评估总表.xlsx"
assessment_header = [0, 1, 2]  # 表头行配置（MultiIndex 三层）
assessment_ignore_columns: list = [0, 1]  # 不参与排名的列索引

# 补充材料目录（自动查找文件名包含 region_name 的 Excel 文件）
detailed_data_dir = _PROJECT_ROOT / "data" / "detailed_data"
# 补充材料表头行配置（List[List[int]]，按 sheet 顺序对应）
supplementary_header = [
    [2, 3, 4], [3, 4], [0, 1], [0, 1], [0, 1, 2],
    [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
]

# Code Agent 配置
code_agent_model = os.getenv("CODE_AGENT_MODEL_NAME")  # 从环境变量读取，或直接填写模型名
code_agent_kwargs: Dict[str, Any] = {}  # 额外参数，如 temperature, top_p 等
max_steps = 3  # Agent 最大执行步数


# ============================================================
# 数据加载
# ============================================================

def _find_supplementary_files(region: str, data_dir: Path) -> List[Path]:
    """在目录中查找文件名包含地区名称的 Excel 文件（与 data_analysis.py 逻辑一致）。"""
    if not data_dir.exists():
        logger.warning(f"补充材料目录不存在: {data_dir}")
        return []
    matched = [
        f for f in data_dir.iterdir()
        if f.is_file()
        and f.suffix.lower() in (".xlsx", ".xls")
        and region in f.stem
    ]
    return sorted(matched, key=lambda p: p.name)


def _add_ranking_columns(df: pd.DataFrame, ignore_columns: list) -> pd.DataFrame:
    """为数值列添加排名列（与 main.py 逻辑一致）。"""
    ignore_set = set(ignore_columns) if isinstance(ignore_columns, list) else {ignore_columns}
    new_data = {}
    for col_idx, col in enumerate(df.columns):
        series = df.iloc[:, col_idx]
        new_data[col] = series
        if col_idx in ignore_set:
            continue
        if pd.api.types.is_numeric_dtype(series):
            rank_series = series.rank(ascending=False, method="min").astype("Int64")
            if isinstance(df.columns, pd.MultiIndex):
                new_name = (*col[:-1], str(col[-1]) + "排名")
            else:
                new_name = str(col) + "排名"
            new_data[new_name] = rank_series
    return pd.DataFrame(new_data)


def load_data() -> Dict[str, pd.DataFrame]:
    """加载考核评估总表 + 自动查找补充材料，返回 {sheet_name: DataFrame} 字典。"""
    all_dfs: Dict[str, pd.DataFrame] = {}

    # ---- 读取考核评估总表 ----
    if assessment_file.exists():
        try:
            assessment_dfs = read_all_excel(assessment_file, header=assessment_header)
            # 取第一个 sheet，并添加排名列（与 main.py 一致）
            assessment_df = list(assessment_dfs.values())[0]
            assessment_df = _add_ranking_columns(assessment_df, assessment_ignore_columns)
            all_dfs["考核评估数据"] = assessment_df
            logger.info(f"已加载考核评估总表: {assessment_file.name}, shape={assessment_df.shape}")
        except Exception as e:
            logger.warning(f"读取考核评估总表失败: {e}")
    else:
        logger.warning(f"考核评估总表不存在: {assessment_file}")

    # ---- 自动查找并读取补充材料（按地区名匹配）----
    supp_files = _find_supplementary_files(region_name, detailed_data_dir)
    logger.info(f"[{region_name}] 找到 {len(supp_files)} 个补充材料文件: {[f.name for f in supp_files]}")

    for file_path in supp_files:
        try:
            file_dfs = read_all_excel(file_path, header=supplementary_header)
            file_name = file_path.stem
            for sheet_name, df in file_dfs.items():
                all_dfs[f"{file_name}__{sheet_name}"] = df
            logger.info(f"已加载补充材料: {file_path.name}, {len(file_dfs)} 个 Sheet")
        except Exception as e:
            logger.warning(f"读取补充材料失败 {file_path.name}: {e}")

    return all_dfs


def run_query(mcp_tool: DataInspectorMCPTool, dfs: Dict[str, pd.DataFrame], query_text: str) -> str:
    """执行单条自然语言查询，返回结果字符串。"""
    effective_max_steps = code_agent_kwargs.get("max_steps", max_steps)
    agent_kwargs = {k: v for k, v in code_agent_kwargs.items() if k != "max_steps"}

    result = mcp_tool.run({
        "action": "query",
        "dfs": dfs,
        "instruction": query_text,
        "model": code_agent_model,
        "max_steps": effective_max_steps,
        "agent_kwargs": agent_kwargs,
    })

    if "result" in result:
        return result["result"]
    else:
        return f"[查询失败] {result.get('error', '未知错误')}"


# ============================================================
# 主程序：交互式查询循环
# ============================================================

if __name__ == "__main__":
    print(f"=== 交互式数据查询 ({region_name}) ===")
    print("加载数据中...")

    all_dfs = load_data()

    if not all_dfs:
        print("错误: 没有加载到任何数据，请检查变量设置区的文件路径配置。")
        sys.exit(1)

    # 打印数据概览
    schema = describe_dataframes_schema(all_dfs)
    print(f"\n已加载 {len(all_dfs)} 个 Sheet:")
    for name, df in all_dfs.items():
        print(f"  - {name}: {df.shape}")
    print()

    # 建立索引到 sheet 名的映射，方便用户按编号选择
    sheet_index = list(all_dfs.keys())

    mcp_tool = DataInspectorMCPTool()
    results: List[str] = []
    query_count = 0

    print('输入自然语言查询，按回车执行。输入 "quit" 或 "exit" 退出。')
    print("-" * 60)

    while True:
        try:
            query_text = input("\n查询> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if not query_text:
            continue
        if query_text.lower() in ("quit", "exit", "q"):
            break

        # 让用户选择本次查询相关的 sheet（与主逻辑 data_analysis.py 的 sheet 筛选一致）
        print("\n可用 Sheet 列表:")
        for idx, name in enumerate(sheet_index):
            print(f"  [{idx}] {name}: {all_dfs[name].shape}")
        try:
            sheet_input = input("选择 Sheet 编号 (逗号分隔, 如 0,3,5; 直接回车=全部): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if sheet_input:
            try:
                selected_indices = [int(x.strip()) for x in sheet_input.split(",")]
                filtered_dfs = {}
                for i in selected_indices:
                    if 0 <= i < len(sheet_index):
                        name = sheet_index[i]
                        filtered_dfs[name] = all_dfs[name]
                    else:
                        print(f"  警告: 索引 {i} 超出范围，已跳过")
                if not filtered_dfs:
                    print("  未选中任何有效 Sheet，回退使用全部数据")
                    filtered_dfs = all_dfs
            except ValueError:
                print("  输入格式错误，使用全部 Sheet")
                filtered_dfs = all_dfs
        else:
            filtered_dfs = all_dfs

        print(f"本次查询使用 {len(filtered_dfs)} 个 Sheet: {list(filtered_dfs.keys())}")

        query_count += 1
        logger.info(f"[{region_name}] 执行查询 {query_count}: {query_text[:80]}...")

        result_text = run_query(mcp_tool, filtered_dfs, query_text)

        print(f"\n--- 查询 {query_count} 结果 ---")
        print(result_text)
        print("-" * 60)

        results.append(f"### 查询 {query_count}: {query_text}\n\n{result_text}")
        logger.info(f"[{region_name}] 查询 {query_count} 完成")

    # 汇总所有结果
    if results:
        final_report = f"# {region_name} 数据查询结果\n\n" + "\n\n---\n\n".join(results)
        print(f"\n=== 共完成 {len(results)} 条查询 ===")