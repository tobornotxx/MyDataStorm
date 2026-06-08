"""PostgreSQL 数据库连接器。

提供论文 Appendix B 中 Executor Agent 所需的所有数据库操作:
1. get_tables()              → 获取所有表及简要描述
2. retrieve_tables_details() → 获取表的详细信息 (列, 类型, 约束等)
3. execute_sql()             → 执行 SQL 查询并返回结果
4. execute_python_from_sql() → 基于 SQL 结果执行 Python 代码
"""

from __future__ import annotations

import io
import logging
from contextlib import contextmanager
from typing import Any, Generator

import pandas as pd
import psycopg2
import psycopg2.extras

from datastorm.config import DatabaseConfig

logger = logging.getLogger(__name__)


class DatabaseConnector:
    """PostgreSQL 数据库连接器。"""

    def __init__(self, config: DatabaseConfig) -> None:
        self._config = config
        self._conn: psycopg2.extensions.connection | None = None

    def connect(self) -> None:
        """建立数据库连接。"""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self._config.url)
            self._conn.set_session(readonly=True, autocommit=True)
            logger.info("Database connection established")

    def close(self) -> None:
        """关闭数据库连接。"""
        if self._conn and not self._conn.closed:
            self._conn.close()
            logger.info("Database connection closed")

    @contextmanager
    def _cursor(self) -> Generator[psycopg2.extras.RealDictCursor, None, None]:
        """获取数据库游标的上下文管理器。"""
        self.connect()
        assert self._conn is not None
        cur = self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            yield cur
        finally:
            cur.close()

    def get_tables(self) -> str:
        """获取所有表及描述 (对应 Executor action: get_tables)。"""
        query = """
        SELECT table_name,
               obj_description((quote_ident(table_schema) || '.' || quote_ident(table_name))::regclass)
               AS description
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name;
        """
        with self._cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()

        lines = []
        for row in rows:
            name = row["table_name"]
            desc = row.get("description") or "No description"
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)

    def retrieve_tables_details(self, table_names: list[str]) -> str:
        """获取表的详细列信息 (对应 Executor action: retrieve_tables_details)。"""
        results = []
        for table_name in table_names:
            query = """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s
            ORDER BY ordinal_position;
            """
            with self._cursor() as cur:
                cur.execute(query, (table_name,))
                columns = cur.fetchall()

            if not columns:
                results.append(f"Table '{table_name}': not found or no columns.")
                continue

            lines = [f"Table '{table_name}':"]
            for col in columns:
                nullable = "NULL" if col["is_nullable"] == "YES" else "NOT NULL"
                default = f" DEFAULT {col['column_default']}" if col["column_default"] else ""
                lines.append(f"  - {col['column_name']}: {col['data_type']} {nullable}{default}")

            # 获取行数
            count_query = f'SELECT COUNT(*) as cnt FROM "{table_name}"'
            with self._cursor() as cur:
                cur.execute(count_query)
                count_row = cur.fetchone()
                row_count = count_row["cnt"] if count_row else 0
            lines.append(f"  Row count: {row_count}")

            results.append("\n".join(lines))

        return "\n\n".join(results)

    def execute_sql(self, sql: str, max_rows: int = 50) -> tuple[pd.DataFrame, str]:
        """执行 SQL 查询 (对应 Executor action: execute_sql)。

        Args:
            sql: SQL 查询字符串
            max_rows: 最大返回行数

        Returns:
            (DataFrame 结果, 文本摘要)
        """
        with self._cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()

        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        total_rows = len(df)

        # 截断并生成摘要
        display_df = df.head(max_rows)
        omitted = total_rows - len(display_df)

        summary_parts = [f"Observed {total_rows} rows"]
        if omitted > 0:
            summary_parts.append(f"({omitted} omitted)")

        text = display_df.to_string(index=False) if not display_df.empty else "(empty result)"
        summary = ". ".join(summary_parts) + f".\n\nSQL: {sql}\nResult:\n{text}"

        return df, summary

    def execute_python_from_sql(self, sql: str, python_code: str) -> str:
        """基于 SQL 结果执行 Python 代码 (对应 Executor action: execute_python_from_sql)。

        Args:
            sql: SQL 查询
            python_code: 要执行的 Python 代码, 可引用 sql_results (DataFrame)

        Returns:
            Python 代码输出
        """
        df, _ = self.execute_sql(sql, max_rows=10000)

        # 在受限环境中执行 Python
        output_buffer = io.StringIO()
        local_vars: dict[str, Any] = {
            "sql_results": df,
            "pd": pd,
        }

        # 重定向 print 到 buffer
        import builtins
        original_print = builtins.print

        def captured_print(*args: Any, **kwargs: Any) -> None:
            kwargs["file"] = output_buffer
            original_print(*args, **kwargs)

        local_vars["print"] = captured_print

        try:
            exec(python_code, {"__builtins__": {"print": captured_print, "len": len, "range": range, "str": str, "int": int, "float": float, "list": list, "dict": dict, "sorted": sorted, "min": min, "max": max, "sum": sum, "round": round, "abs": abs, "enumerate": enumerate, "zip": zip}}, local_vars)
        except Exception as e:
            return f"Error executing Python code: {e}"

        return output_buffer.getvalue()
