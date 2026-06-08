"""
Doc Writing 模块
根据数据分析结果和原始考核 DataFrame，使用 LLM 生成报告初稿。

用法:
    from llm import OpenAILikeLLM, LLMConfig
    from doc_writing import DocWriter

    llm = OpenAILikeLLM(config=LLMConfig(
        model="your-model-name",
        api_base="https://your-api-endpoint",
        api_key="your-api-key",
    ))

    writer = DocWriter(llm=llm)
    draft = writer.write(
        analysis_result="上一步的分析结果字符串...",
        assessment_df=assessment_df,
        region_name="渝北区",
    )
    print(draft)
"""

from typing import Optional

import pandas as pd

from llm import BaseLLM
from utils import logger
from utils.prompt_renderer import render_prompt


class DocWriter:
    """
    报告初稿生成器。

    将数据分析结果 + 原始考核 DataFrame 发送给 LLM，生成结构化的报告初稿。

    Args:
        llm: BaseLLM 实例（必须传入，不使用默认模型）
        system_prompt: 系统提示词字符串。如果不传，则从 j2 模板渲染。
        system_template: 系统提示词模板文件名，默认为 doc_writing_system.j2
    """

    def __init__(
        self,
        llm: BaseLLM,
        system_prompt: Optional[str] = None,
        system_template: str = "doc_writing_system.j2",
    ):
        self.llm = llm

        # 加载 system prompt
        if system_prompt is not None:
            self._system_prompt = system_prompt
        else:
            self._system_prompt = render_prompt(system_template)

        logger.info(
            f"DocWriter initialized: model={self.llm.config.model}, "
            f"prompt_len={len(self._system_prompt)}"
        )

    def write(
        self,
        analysis_result: str,
        assessment_df: pd.DataFrame,
        region_name: str = "",
        **kwargs,
    ) -> str:
        """
        根据分析结果和原始考核数据生成报告初稿。

        Args:
            analysis_result: 上一步 data_analysis.analyze_region 返回的分析结果字符串
            assessment_df: 原始多维考核指标 DataFrame
            region_name: 地区名称（可选，用于报告标题）
            **kwargs: 覆盖 LLM 生成参数（temperature, max_tokens 等）

        Returns:
            str: 报告初稿文本
        """
        # 将 DataFrame 转为可读文本
        df_text = _dataframe_to_text(assessment_df)

        user_prompt = render_prompt(
            "doc_writing_user.j2",
            analysis_result=analysis_result,
            df_text=df_text,
            region_name=region_name,
        )

        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        response = self.llm.generate(messages, **kwargs)
        logger.info(
            f"DocWriter.write done: region={region_name}, "
            f"input_len={len(user_prompt)}, output_len={len(response.content)}"
        )
        return response.content


# ============================================================
# 辅助函数
# ============================================================

def _dataframe_to_text(
    df: pd.DataFrame,
    max_rows: int = 200,
) -> str:
    """
    将 DataFrame 转换为适合放入 prompt 的文本格式。

    对于行数超过 max_rows 的大表，截取前后各 max_rows//2 行并标注省略。

    Args:
        df: 待转换的 DataFrame
        max_rows: 最大展示行数

    Returns:
        str: 表格的文本表示
    """
    if len(df) <= max_rows:
        return df.to_string(index=True)

    half = max_rows // 2
    head_str = df.head(half).to_string(index=True)
    tail_str = df.tail(half).to_string(index=True)
    omitted = len(df) - max_rows
    return (
        f"{head_str}\n"
        f"\n... 省略中间 {omitted} 行 ...\n\n"
        f"{tail_str}"
    )
