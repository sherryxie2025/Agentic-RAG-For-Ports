# src/online_pipeline/sql_loader.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .source_registry import SourceRegistry


@dataclass
class TableSchema:
    table_name: str
    file_name: str
    primary_time_column: Optional[str]
    join_key: Optional[str]
    columns: List[str]
    description: str


class CSVSQLLoader:
    """
    Loads the port operational CSV files into pandas DataFrames and exposes
    a lightweight schema registry for the online pipeline.
    """

    def __init__(self, registry: SourceRegistry) -> None:
        self.registry = registry

        self.table_file_map: Dict[str, str] = {
            "vessel_calls": "POLB_vessel_calls_2015.csv",
            "berth_operations": "POLB_berth_operations_2015.csv",
            "crane_operations": "POLB_crane_operations_2015.csv",
            "yard_operations": "POLB_yard_operations_2015.csv",
            "gate_operations": "POLB_gate_operations_2015.csv",
            "environment": "environment_timeline_2015_2024.csv",
        }

        self.schemas: Dict[str, TableSchema] = self._build_schema_registry()
        self._tables: Dict[str, pd.DataFrame] = {}

    def _build_schema_registry(self) -> Dict[str, TableSchema]:
        return {
            "vessel_calls": TableSchema(
                table_name="vessel_calls",
                file_name="POLB_vessel_calls_2015.csv",
                primary_time_column="ata",
                join_key="call_id",
                columns=[],
                description="Vessel call level operational records.",
            ),
            "berth_operations": TableSchema(
                table_name="berth_operations",
                file_name="POLB_berth_operations_2015.csv",
                primary_time_column="operation_date",
                join_key="call_id",
                columns=[],
                description="Berth-level vessel operation outcomes including productivity and delays.",
            ),
            "crane_operations": TableSchema(
                table_name="crane_operations",
                file_name="POLB_crane_operations_2015.csv",
                primary_time_column="operation_date",
                join_key="call_id",
                columns=[],
                description="Crane assignment and crane productivity records.",
            ),
            "yard_operations": TableSchema(
                table_name="yard_operations",
                file_name="POLB_yard_operations_2015.csv",
                primary_time_column="operation_date",
                join_key="call_id",
                columns=[],
                description="Yard block utilization and dwell-time records.",
            ),
            "gate_operations": TableSchema(
                table_name="gate_operations",
                file_name="POLB_gate_operations_2015.csv",
                primary_time_column="gate_date",
                join_key=None,
                columns=[],
                description="Daily terminal gate transaction records.",
            ),
            "environment": TableSchema(
                table_name="environment",
                file_name="environment_timeline_2015_2024.csv",
                primary_time_column="timestamp",
                join_key=None,
                columns=[],
                description="Environmental timeline including weather, wave, tide, and event flags.",
            ),
        }

    def load_table(self, table_name: str) -> pd.DataFrame:
        if table_name in self._tables:
            return self._tables[table_name]

        if table_name not in self.table_file_map:
            raise ValueError(f"Unknown table name: {table_name}")

        path = self.registry.sql_data_dir / self.table_file_map[table_name]
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        df = pd.read_csv(path)

        # best-effort datetime parsing for common time columns
        for candidate in ["timestamp", "operation_date", "gate_date", "ata", "atd", "eta", "etd"]:
            if candidate in df.columns:
                try:
                    df[candidate] = pd.to_datetime(df[candidate], errors="coerce")
                except Exception:
                    pass

        self._tables[table_name] = df

        # cache actual columns into schema
        schema = self.schemas[table_name]
        schema.columns = list(df.columns)

        return df

    def load_all(self) -> Dict[str, pd.DataFrame]:
        for table_name in self.table_file_map:
            self.load_table(table_name)
        return self._tables

    def get_schema_summary(self) -> Dict[str, Dict[str, object]]:
        summary: Dict[str, Dict[str, object]] = {}
        for table_name in self.table_file_map:
            df = self.load_table(table_name)
            schema = self.schemas[table_name]
            summary[table_name] = {
                "file_name": schema.file_name,
                "row_count": len(df),
                "primary_time_column": schema.primary_time_column,
                "join_key": schema.join_key,
                "columns": list(df.columns),
                "description": schema.description,
            }
        return summary


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    registry = SourceRegistry.from_project_root(PROJECT_ROOT)
    loader = CSVSQLLoader(registry)

    summary = loader.get_schema_summary()
    for table_name, info in summary.items():
        print("=" * 100)
        print(table_name)
        print(info["description"])
        print("rows:", info["row_count"])
        print("time column:", info["primary_time_column"])
        print("join key:", info["join_key"])
        print("columns:", info["columns"][:20])