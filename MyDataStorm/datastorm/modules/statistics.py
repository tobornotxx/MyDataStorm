"""自底向上归纳统计模块。

论文 Section 3.2.1 (Bottom-up inductive insight surfacing):
DataSTORM 自动为返回表的每一列计算汇总统计,
直接嵌入到每个答案 a'_{i,j} 中。

统计包含:
- distinct_percentage: 唯一值占总行数的比例
- top_5 values: 最频繁的 5 个值
- 对数值列: min, max, median, mean
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from datastorm.types import ExecutorResponse, SummaryStatistics

logger = logging.getLogger(__name__)


class StatisticsModule:
    """自底向上归纳统计计算模块。"""

    def compute_and_embed(self, response: ExecutorResponse) -> ExecutorResponse:
        """为 Executor 响应计算汇总统计并嵌入。

        论文 Section 3.2.1:
        对返回表的每一列计算 distinct_percentage, top-5 values,
        以及数值列的 min, max, median, mean。

        Args:
            response: 原始 Executor 响应

        Returns:
            带有嵌入统计的更新响应
        """
        df = response.raw_results
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            logger.debug("Statistics: skipped (no DataFrame or empty)")
            return response

        stats_list = []
        for col in df.columns:
            stats = self._compute_column_stats(df, col)
            stats_list.append(stats)

        logger.debug(
            "Statistics: computed for %d columns (%d rows), columns=%s",
            len(df.columns), len(df), list(df.columns),
        )

        # 生成统计文本
        stats_text_parts = ["Summary Statistics:"]
        for s in stats_list:
            stats_text_parts.append(s.to_text())

        stats_text = "\n".join(stats_text_parts)

        # 嵌入到响应中
        response.summary_stats = stats_list
        response.summary_text = f"{response.answer}\n\n{stats_text}"

        return response

    def _compute_column_stats(self, df: pd.DataFrame, col: str) -> SummaryStatistics:
        """计算单列的汇总统计。"""
        series = df[col].dropna()
        total = len(df)

        stats = SummaryStatistics(column_name=col)

        if total == 0:
            return stats

        # distinct_percentage
        n_unique = series.nunique()
        stats.distinct_percentage = (n_unique / total) * 100

        # top_values (top 5)
        value_counts = series.value_counts().head(5)
        stats.top_values = [
            {"value": str(val), "count": int(count)}
            for val, count in value_counts.items()
        ]

        # 数值列统计
        if pd.api.types.is_numeric_dtype(series):
            try:
                stats.min_val = self._safe_convert(series.min())
                stats.max_val = self._safe_convert(series.max())
                stats.median_val = self._safe_convert(series.median())
                stats.mean_val = round(float(series.mean()), 2)
            except (TypeError, ValueError):
                pass

        return stats

    @staticmethod
    def _safe_convert(val: Any) -> Any:
        """安全转换 numpy 类型为 Python 原生类型。"""
        if isinstance(val, (np.integer,)):
            return int(val)
        if isinstance(val, (np.floating,)):
            return round(float(val), 2)
        return val
