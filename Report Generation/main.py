"""
主入口
从终端接收地区名，依次执行 数据分析 → 报告撰写 → 文本改写，
最终将报告保存到 output/ 目录。
"""

import sys
import os
from pathlib import Path
from typing import Union, List

import pandas as pd

from llm import OpenAILikeLLM, LLMConfig
from data_analysis import analyze_region
from doc_writing import DocWriter
from rewriting import Rewriter
from utils import logger
from utils.file_io import read_all_excel, data_save
import dotenv

dotenv.load_dotenv()

# ============================================================
# 配置
# ============================================================

# 考核评估总表路径（按实际情况修改）
ASSESSMENT_FILE = Path("data/overview_data/考核评估总表.xlsx")
ASSESSMENT_HEADER = [0, 1, 2]  # 表头配置，按实际情况修改
# 读取时忽略的列索引（int 或 List[int]），这些列不参与排名
ASSESSMENT_IGNORE_COLUMNS: Union[int, List[int]] = [0, 1]

# 输出目录
OUTPUT_DIR = Path("output")


def _add_ranking_columns(
    df: pd.DataFrame,
    ignore_columns: Union[int, List[int]],
) -> pd.DataFrame:
    """
    为数值列添加排名列（从高到低，1 为最高），插入在原列右侧。
    支持 MultiIndex 列名：上层保持不变，最后一级列名后加「排名」。

    Args:
        df: 原始 DataFrame
        ignore_columns: 要忽略的列索引（int 或 List[int]），这些列不参与排名

    Returns:
        添加了排名列的新 DataFrame
    """
    if isinstance(ignore_columns, int):
        ignore_set = {ignore_columns}
    else:
        ignore_set = set(ignore_columns)

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


def _create_planning_llm() -> OpenAILikeLLM:
    """
    创建用于"数据分析规划"阶段的 LLM 客户端。
    此阶段使用默认模型（环境变量）即可。
    """
    return OpenAILikeLLM(config=LLMConfig())


def _create_writing_llm() -> OpenAILikeLLM:
    """
    创建用于"报告撰写"阶段的高级闭源 LLM 客户端。

    """
    model_name = os.getenv("ADVANCED_MODEL_NAME")
    api_base = os.getenv("API_BASE_ADVANCED")
    api_key = os.getenv("API_KEY_ADVANCED")
    return OpenAILikeLLM(config=LLMConfig(
        model=model_name,
        api_base=api_base,
        api_key=api_key,
        temperature=0.7,
    ))


def _create_rewriting_llm() -> OpenAILikeLLM:
    """
    创建用于"文本改写/润色"阶段的高级闭源 LLM 客户端。
    """
    model_name = os.getenv("ADVANCED_MODEL_NAME")
    api_base = os.getenv("API_BASE_ADVANCED")
    api_key = os.getenv("API_KEY_ADVANCED")
    return OpenAILikeLLM(config=LLMConfig(
        model=model_name,
        api_base=api_base,
        api_key=api_key,
        temperature=0.7,
    ))


# ============================================================
# 主流程
# ============================================================

def run(region_name: str) -> Path:
    """
    对指定地区执行完整的报告生成流程。

    Args:
        region_name: 地区名称（如 "渝北区"）

    Returns:
        Path: 最终报告保存路径
    """
    logger.info(f"===== 开始处理: {region_name} =====")

    # ---- 1. 读取考核评估总表 ----
    logger.info(f"[1/4] 读取考核评估数据: {ASSESSMENT_FILE}")
    dfs = read_all_excel(ASSESSMENT_FILE, header=ASSESSMENT_HEADER)
    # 取第一个 sheet（或按需调整）
    assessment_df = list(dfs.values())[0]
    # 为数值列添加排名（从高到低），忽略 ignore_columns 指定的列
    assessment_df = _add_ranking_columns(
        assessment_df,
        ignore_columns=ASSESSMENT_IGNORE_COLUMNS,
    )
    logger.info(f"考核数据 shape: {assessment_df.shape}")
    logger.info(f"考核数据 columns: {assessment_df.head(3)}")

    # ---- 2. 数据分析 ----
    logger.info(f"[2/4] 数据分析: {region_name}")
    planning_llm = _create_planning_llm()
    analysis_result = analyze_region(
        assessment_df=assessment_df,
        region_name=region_name,
        supplementary_header=[[2,3,4],[3,4],[0,1],[0,1],[0,1,2],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]],
        llm=planning_llm,
        code_agent_kwargs={"max_steps": 3},
    )
    logger.info(f"分析结果长度: {len(analysis_result)} 字符")
    logger.info(f"分析结果：{analysis_result}")

    # ---- 3. 报告撰写 ----
    logger.info(f"[3/4] 生成报告初稿: {region_name}")
    writing_llm = _create_writing_llm()
    writer = DocWriter(llm=writing_llm)
    draft = writer.write(
        analysis_result=analysis_result,
        assessment_df=assessment_df,
        region_name=region_name,
    )
    logger.info(f"初稿长度: {len(draft)} 字符")
    logger.info(f"初稿: {draft}")

    # ---- 4. 文本改写/润色 ----
    logger.info(f"[4/4] 改写润色: {region_name}")
    rewriting_llm = _create_rewriting_llm()
    rewriter = Rewriter(llm=rewriting_llm)
    final_report = rewriter.rewrite(draft)
    logger.info(f"最终报告长度: {len(final_report)} 字符")

    # ---- 5. 保存 ----
    output_path = data_save(
        data=final_report,
        file_path=OUTPUT_DIR / f"{region_name}_报告",
        file_type="md",
    )
    logger.info(f"报告已保存至: {output_path}")
    logger.info(f"===== 完成: {region_name} =====\n")

    return output_path


def main():
    if len(sys.argv) > 1:
        region_name = sys.argv[1]
    else:
        region_name = input("请输入地区名称: ").strip()

    if not region_name:
        print("错误: 地区名称不能为空")
        sys.exit(1)

    output_path = run(region_name)
    print(f"\n报告已生成: {output_path}")


if __name__ == "__main__":
    main()
