# src/online_pipeline/build_sql_db.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import duckdb

from .source_registry import SourceRegistry


class DuckDBBuilder:
    """
    Build a local analytical SQL database from CSV files.

    Robust version:
    - imports CSVs into DuckDB
    - inspects actual columns after import
    - builds views using real existing columns
    """

    def __init__(self, project_root: str | Path) -> None:
        self.registry = SourceRegistry.from_project_root(project_root)
        self.sql_storage_dir = self.registry.project_root / "storage" / "sql"
        self.sql_storage_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.sql_storage_dir / "port_ops.duckdb"

        self.table_to_file: Dict[str, str] = {
            "vessel_calls": "POLB_vessel_calls_2015.csv",
            "berth_operations": "POLB_berth_operations_2015.csv",
            "crane_operations": "POLB_crane_operations_2015.csv",
            "yard_operations": "POLB_yard_operations_2015.csv",
            "gate_operations": "POLB_gate_operations_2015.csv",
            "environment": "environment_timeline_2015_2024.csv",
        }

    def build(self, overwrite: bool = True) -> Path:
        if overwrite and self.db_path.exists():
            self.db_path.unlink()

        conn = duckdb.connect(str(self.db_path))

        try:
            for table_name, file_name in self.table_to_file.items():
                csv_path = self.registry.sql_data_dir / file_name
                if not csv_path.exists():
                    raise FileNotFoundError(f"Missing CSV for table '{table_name}': {csv_path}")

                conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                conn.execute(
                    f"""
                    CREATE TABLE {table_name} AS
                    SELECT *
                    FROM read_csv_auto('{csv_path.as_posix()}', HEADER=TRUE)
                    """
                )

            self._create_views(conn)
            self._basic_sanity_checks(conn)

        finally:
            conn.close()

        return self.db_path

    def _create_views(self, conn: duckdb.DuckDBPyConnection) -> None:
        berth_cols = self._get_columns(conn, "berth_operations")
        crane_cols = self._get_columns(conn, "crane_operations")
        yard_cols = self._get_columns(conn, "yard_operations")

        berth_time_col = self._pick_first_existing(
            berth_cols, ["atb", "ata", "etb", "eta", "operation_date"]
        )
        berth_terminal_col = self._pick_first_existing(
            berth_cols, ["terminal_code", "terminal_name"]
        )
        yard_block_col = self._pick_first_existing(
            yard_cols, ["yard_block", "yard_block_id"]
        )
        crane_terminal_col = self._pick_first_existing(
            crane_cols, ["terminal_code"]
        )
        crane_date_col = self._pick_first_existing(
            crane_cols, ["operation_date", "ata", "atb"]
        )
        yard_date_col = self._pick_first_existing(
            yard_cols, ["operation_date", "ata", "atb"]
        )

        conn.execute("DROP VIEW IF EXISTS v_port_ops_joined")

        select_parts: List[str] = [
            "b.call_id AS call_id",
        ]

        if berth_time_col:
            select_parts.append(f"b.{berth_time_col} AS berth_time")
        if berth_terminal_col:
            select_parts.append(f"b.{berth_terminal_col} AS berth_terminal")
        if "berth_id" in berth_cols:
            select_parts.append("b.berth_id AS berth_id")
        if "berth_productivity_mph" in berth_cols:
            select_parts.append("b.berth_productivity_mph AS berth_productivity_mph")
        if "containers_actual" in berth_cols:
            select_parts.append("b.containers_actual AS containers_actual")
        if "arrival_delay_hours" in berth_cols:
            select_parts.append("b.arrival_delay_hours AS arrival_delay_hours")
        if "berth_delay_hours" in berth_cols:
            select_parts.append("b.berth_delay_hours AS berth_delay_hours")

        if "crane_id" in crane_cols:
            select_parts.append("c.crane_id AS crane_id")
        if crane_terminal_col:
            select_parts.append(f"c.{crane_terminal_col} AS crane_terminal")
        if crane_date_col:
            select_parts.append(f"c.{crane_date_col} AS crane_time")
        if "crane_productivity_mph" in crane_cols:
            select_parts.append("c.crane_productivity_mph AS crane_productivity_mph")
        if "total_moves" in crane_cols:
            select_parts.append("c.total_moves AS total_moves")
        if "breakdown_minutes" in crane_cols:
            select_parts.append("c.breakdown_minutes AS breakdown_minutes")

        if yard_block_col:
            select_parts.append(f"y.{yard_block_col} AS yard_block")
        if yard_date_col:
            select_parts.append(f"y.{yard_date_col} AS yard_time")
        if "teu_received" in yard_cols:
            select_parts.append("y.teu_received AS teu_received")
        if "average_dwell_days" in yard_cols:
            select_parts.append("y.average_dwell_days AS average_dwell_days")

        select_sql = ",\n                ".join(select_parts)

        conn.execute(
            f"""
            CREATE VIEW v_port_ops_joined AS
            SELECT
                {select_sql}
            FROM berth_operations b
            LEFT JOIN crane_operations c
                ON b.call_id = c.call_id
            LEFT JOIN yard_operations y
                ON b.call_id = y.call_id
            """
        )

        conn.execute("DROP VIEW IF EXISTS v_crane_berth_summary")
        conn.execute(
            """
            CREATE VIEW v_crane_berth_summary AS
            SELECT
                c.call_id,
                AVG(c.crane_productivity_mph) AS avg_crane_productivity_mph,
                SUM(c.total_moves) AS total_crane_moves,
                SUM(c.breakdown_minutes) AS total_breakdown_minutes,
                AVG(b.berth_productivity_mph) AS avg_berth_productivity_mph,
                AVG(b.arrival_delay_hours) AS avg_arrival_delay_hours
            FROM crane_operations c
            LEFT JOIN berth_operations b
                ON c.call_id = b.call_id
            GROUP BY c.call_id
            """
        )

    @staticmethod
    def _get_columns(conn: duckdb.DuckDBPyConnection, table_name: str) -> List[str]:
        rows = conn.execute(f"DESCRIBE {table_name}").fetchall()
        return [r[0] for r in rows]

    @staticmethod
    def _pick_first_existing(columns: List[str], candidates: List[str]) -> str | None:
        for c in candidates:
            if c in columns:
                return c
        return None

    def _basic_sanity_checks(self, conn: duckdb.DuckDBPyConnection) -> None:
        required_tables = [
            "vessel_calls",
            "berth_operations",
            "crane_operations",
            "yard_operations",
            "gate_operations",
            "environment",
        ]
        for table_name in required_tables:
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            if count <= 0:
                raise ValueError(f"Table {table_name} is empty after build.")

    def inspect(self) -> None:
        conn = duckdb.connect(str(self.db_path))
        try:
            print(f"DB path: {self.db_path}")
            tables = conn.execute("SHOW TABLES").fetchall()
            print("Tables / views:")
            for t in tables:
                print("-", t[0])

            for table_name in [
                "vessel_calls",
                "berth_operations",
                "crane_operations",
                "yard_operations",
                "gate_operations",
                "environment",
                "v_port_ops_joined",
                "v_crane_berth_summary",
            ]:
                print("=" * 100)
                print(table_name)
                print(conn.execute(f"DESCRIBE {table_name}").fetchdf())
                print(conn.execute(f"SELECT COUNT(*) AS row_count FROM {table_name}").fetchdf())
        finally:
            conn.close()


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    builder = DuckDBBuilder(project_root)
    db_path = builder.build(overwrite=True)
    print(f"Built DuckDB database at: {db_path}")
    builder.inspect()