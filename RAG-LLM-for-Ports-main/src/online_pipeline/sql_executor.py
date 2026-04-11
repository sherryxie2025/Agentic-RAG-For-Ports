# src/online_pipeline/sql_executor.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
import pandas as pd


@dataclass
class SQLExecutionOutput:
    sql: str
    execution_ok: bool
    rows: List[Dict[str, Any]]
    row_count: int
    columns: List[str]
    error: Optional[str] = None


class DuckDBExecutor:
    """
    Execute SQL safely against local DuckDB.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)

    def execute(
        self,
        sql: str,
        limit_rows: int = 20,
    ) -> SQLExecutionOutput:
        sql_clean = sql.strip().rstrip(";")
        if not self._is_safe_select(sql_clean):
            return SQLExecutionOutput(
                sql=sql_clean,
                execution_ok=False,
                rows=[],
                row_count=0,
                columns=[],
                error="Only read-only SELECT/WITH queries are allowed.",
            )

        try:
            conn = duckdb.connect(str(self.db_path), read_only=True)
            try:
                df = conn.execute(sql_clean).fetchdf()
            finally:
                conn.close()

            if df is None:
                return SQLExecutionOutput(
                    sql=sql_clean,
                    execution_ok=True,
                    rows=[],
                    row_count=0,
                    columns=[],
                    error=None,
                )

            preview_df = df.head(limit_rows)
            rows = preview_df.to_dict(orient="records")

            return SQLExecutionOutput(
                sql=sql_clean,
                execution_ok=True,
                rows=rows,
                row_count=len(df),
                columns=list(df.columns),
                error=None,
            )
        except Exception as e:
            return SQLExecutionOutput(
                sql=sql_clean,
                execution_ok=False,
                rows=[],
                row_count=0,
                columns=[],
                error=str(e),
            )

    def explain(self, sql: str) -> tuple[bool, Optional[str]]:
        """
        Run DuckDB EXPLAIN on a SQL query without executing it.
        Returns (ok, error). Used for pre-check before expensive execution
        so that LLM-generated SQL with GROUP BY / type errors can fall
        back to rule-based before paying the execution cost.
        """
        sql_clean = sql.strip().rstrip(";")
        if not self._is_safe_select(sql_clean):
            return False, "Only read-only SELECT/WITH queries are allowed."
        try:
            conn = duckdb.connect(str(self.db_path), read_only=True)
            try:
                conn.execute(f"EXPLAIN {sql_clean}").fetchall()
            finally:
                conn.close()
            return True, None
        except Exception as e:
            return False, str(e)

    @staticmethod
    def _is_safe_select(sql: str) -> bool:
        lowered = sql.lower().strip()

        forbidden = [
            "insert ",
            "update ",
            "delete ",
            "drop ",
            "alter ",
            "create ",
            "replace ",
            "truncate ",
            "attach ",
            "copy ",
        ]
        if any(tok in lowered for tok in forbidden):
            return False

        return lowered.startswith("select") or lowered.startswith("with")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    db_path = project_root / "storage" / "sql" / "port_ops.duckdb"
    executor = DuckDBExecutor(db_path)

    demo_sql = """
    SELECT AVG(crane_productivity_mph) AS avg_crane_productivity_mph
    FROM crane_operations
    """
    result = executor.execute(demo_sql)
    print(result)