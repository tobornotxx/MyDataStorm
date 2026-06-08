"""CSV → SQLite 数据库桥接器。

实现与 datastorm.database.connector.DatabaseConnector 完全相同的接口，
但底层使用 SQLite + pandas，不依赖 PostgreSQL。

InsightBench 的数据是 CSV 文件，DataSTORM 的 ExecutorAgent 期望一个
DatabaseConnector 对象。这个类作为 drop-in 替换，让 ExecutorAgent
可以直接在 CSV 数据上运行 SQL 查询。
"""

from __future__ import annotations

import io
import logging
import sqlite3
import tempfile
import os
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class CsvDatabaseBridge:
    """将 CSV 文件暴露为可查询的 SQLite 数据库。

    接口与 datastorm.database.connector.DatabaseConnector 完全一致，
    可直接替换传入 ExecutorAgent。

    用法：
        bridge = CsvDatabaseBridge(
            csv_path="data.csv",
            table_name="incidents",
            user_csv_path="sysuser.csv",   # 可选第二张表
        )
        # 之后当作普通 DatabaseConnector 使用
        bridge.get_tables()
        bridge.execute_sql("SELECT * FROM incidents LIMIT 5")
    """

    def __init__(
        self,
        csv_path: str,
        table_name: str = "main_table",
        user_csv_path: str | None = None,
        user_table_name: str = "user_table",
    ) -> None:
        self._csv_path = csv_path
        self._table_name = table_name
        self._user_csv_path = user_csv_path
        self._user_table_name = user_table_name

        # 使用临时文件存储 SQLite DB（避免多进程冲突）
        self._db_fd, self._db_path = tempfile.mkstemp(suffix=".db")
        os.close(self._db_fd)

        self._conn: sqlite3.Connection | None = None
        self._table_descriptions: dict[str, str] = {}
        self._load_csvs()

    # ------------------------------------------------------------------
    # 内部：加载 CSV 到 SQLite
    # ------------------------------------------------------------------

    def _load_csvs(self) -> None:
        """把 CSV 文件加载进 SQLite 数据库。"""
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)

        df = pd.read_csv(self._csv_path)
        df.to_sql(self._table_name, self._conn, if_exists="replace", index=False)
        self._table_descriptions[self._table_name] = (
            f"Main dataset loaded from {os.path.basename(self._csv_path)} "
            f"({len(df)} rows, {len(df.columns)} columns)"
        )
        self._schema_cache = self._build_schema_text(self._table_name, df)
        logger.info(
            "Loaded CSV '%s' → table '%s' (%d rows)",
            self._csv_path, self._table_name, len(df),
        )

        if self._user_csv_path and os.path.exists(self._user_csv_path):
            df_user = pd.read_csv(self._user_csv_path)
            df_user.to_sql(self._user_table_name, self._conn, if_exists="replace", index=False)
            self._table_descriptions[self._user_table_name] = (
                f"User dataset loaded from {os.path.basename(self._user_csv_path)} "
                f"({len(df_user)} rows, {len(df_user.columns)} columns)"
            )
            self._schema_cache += "\n\n" + self._build_schema_text(self._user_table_name, df_user)
            logger.info(
                "Loaded user CSV '%s' → table '%s' (%d rows)",
                self._user_csv_path, self._user_table_name, len(df_user),
            )

        self._conn.commit()

    def _build_schema_text(self, table_name: str, df: pd.DataFrame) -> str:
        """生成可直接注入 prompt 的表结构描述。"""
        lines = [f"Table: {table_name} ({len(df)} rows)"]
        lines.append("Columns:")
        for col in df.columns:
            dtype = str(df[col].dtype)
            n_unique = df[col].nunique()
            n_null = df[col].isnull().sum()
            sample_vals = df[col].dropna().unique()[:5].tolist()
            sample_str = ", ".join(repr(v) for v in sample_vals)
            lines.append(f"  - {col} ({dtype}, {n_unique} unique, {n_null} nulls) — samples: [{sample_str}]")
        return "\n".join(lines)

    def get_schema_context(self) -> str:
        """返回完整的数据库 schema 上下文（用于注入 executor prompt）。"""
        return self._schema_cache

    # ------------------------------------------------------------------
    # DatabaseConnector 兼容接口
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """兼容接口：SQLite 连接在 __init__ 中已建立。"""
        pass

    def close(self) -> None:
        """关闭 SQLite 连接并清理临时文件。"""
        if self._conn:
            self._conn.close()
            self._conn = None
        try:
            os.unlink(self._db_path)
        except OSError:
            pass

    def get_tables(self) -> str:
        """获取所有表及描述（对应 Executor action: get_tables）。"""
        lines = []
        for name, desc in self._table_descriptions.items():
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)

    def retrieve_tables_details(self, table_names: list[str]) -> str:
        """获取表的详细列信息（对应 Executor action: retrieve_tables_details）。"""
        assert self._conn is not None
        results = []
        for table_name in table_names:
            cursor = self._conn.execute(
                f"PRAGMA table_info('{table_name}')"
            )
            columns = cursor.fetchall()
            if not columns:
                results.append(f"Table '{table_name}': not found or no columns.")
                continue

            lines = [f"Table '{table_name}':"]
            for col in columns:
                # PRAGMA table_info: (cid, name, type, notnull, dflt_value, pk)
                nullable = "NOT NULL" if col[3] else "NULL"
                default = f" DEFAULT {col[4]}" if col[4] is not None else ""
                lines.append(f"  - {col[1]}: {col[2] or 'TEXT'} {nullable}{default}")

            count_row = self._conn.execute(
                f"SELECT COUNT(*) FROM \"{table_name}\""
            ).fetchone()
            row_count = count_row[0] if count_row else 0
            lines.append(f"  Row count: {row_count}")
            results.append("\n".join(lines))

        return "\n\n".join(results)

    def execute_sql(self, sql: str, max_rows: int = 50) -> tuple[pd.DataFrame, str]:
        """执行 SQL 查询（对应 Executor action: execute_sql）。"""
        assert self._conn is not None
        try:
            df = pd.read_sql_query(sql, self._conn)
        except Exception as e:
            empty = pd.DataFrame()
            return empty, f"SQL Error: {e}\n\nSQL: {sql}"

        total_rows = len(df)
        display_df = df.head(max_rows)
        omitted = total_rows - len(display_df)

        summary_parts = [f"Observed {total_rows} rows"]
        if omitted > 0:
            summary_parts.append(f"({omitted} omitted)")

        text = display_df.to_string(index=False) if not display_df.empty else "(empty result)"
        summary = ". ".join(summary_parts) + f".\n\nSQL: {sql}\nResult:\n{text}"
        return df, summary

    def execute_python_from_sql(self, sql: str, python_code: str) -> str:
        """基于 SQL 结果执行 Python 代码（对应 Executor action: execute_python_from_sql）。"""
        df, _ = self.execute_sql(sql, max_rows=10000)

        output_buffer = io.StringIO()

        def captured_print(*args: Any, **kwargs: Any) -> None:
            kwargs["file"] = output_buffer
            print(*args, **kwargs)

        # 注入常用数据科学包
        import numpy as np
        try:
            import scipy
            import scipy.stats
            import scipy.optimize
        except ImportError:
            scipy = None
        try:
            import sklearn
        except ImportError:
            sklearn = None
        try:
            import statsmodels
            import statsmodels.api
        except ImportError:
            statsmodels = None

        local_vars: dict[str, Any] = {
            "sql_results": df,
            "pd": pd,
            "np": np,
            "numpy": np,
            "print": captured_print,
        }
        if scipy is not None:
            local_vars["scipy"] = scipy
            local_vars["stats"] = scipy.stats
        if sklearn is not None:
            local_vars["sklearn"] = sklearn
        if statsmodels is not None:
            local_vars["statsmodels"] = statsmodels
            local_vars["sm"] = statsmodels.api

        safe_builtins = {
            "print": captured_print,
            "len": len, "range": range, "str": str, "int": int,
            "float": float, "list": list, "dict": dict, "sorted": sorted,
            "min": min, "max": max, "sum": sum, "round": round,
            "abs": abs, "enumerate": enumerate, "zip": zip,
            "isinstance": isinstance, "type": type, "tuple": tuple,
            "set": set, "bool": bool, "map": map, "filter": filter,
            "any": any, "all": all, "reversed": reversed,
        }
        try:
            exec(python_code, {"__builtins__": safe_builtins}, local_vars)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            return (
                f"Python execution error: {type(e).__name__}: {e}\n\n"
                f"Traceback:\n{tb}\n\n"
                f"Available variables: sql_results (DataFrame with {len(df)} rows, columns: {list(df.columns)})\n"
                f"Available packages: numpy (np), pandas (pd), scipy, scipy.stats (stats), "
                f"sklearn, statsmodels (sm)\n"
                f"Tip: Fix the error and retry. Check column names match the DataFrame."
            )

        return output_buffer.getvalue()
