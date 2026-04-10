import json
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd


class SQLSchemaParser:
    """
    Parse CSV-based port operation datasets into a structured schema registry
    for downstream rule extraction and query planning.
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

        # 你后面可以继续扩展
        self.known_relationships = [
            {
                "left_table": "berth_operations",
                "left_key": "call_id",
                "right_table": "crane_operations",
                "right_key": "call_id",
                "relationship": "one_to_many",
            },
            {
                "left_table": "berth_operations",
                "left_key": "call_id",
                "right_table": "yard_operations",
                "right_key": "call_id",
                "relationship": "one_to_many",
            },
        ]

        self.filename_table_map = {
            "vessel_calls": "vessel_calls",
            "berth_operations": "berth_operations",
            "crane_operations": "crane_operations",
            "yard_operations": "yard_operations",
            "gate_operations": "gate_operations",
            "environment": "environment",
        }

    def discover_files(self) -> List[Path]:
        return sorted(self.data_dir.glob("*.csv"))

    def infer_table_name(self, file_path: Path) -> str:
        name = file_path.stem.lower()
        for key, table_name in self.filename_table_map.items():
            if key in name:
                return table_name
        return name

    def infer_semantic_type(self, col: str, dtype: str) -> str:
        c = col.lower()

        if c.endswith("_time") or c in {"eta", "ata", "etb", "atb", "etd", "atd", "gate_date", "operation_date"}:
            return "timestamp"

        if "date" in c:
            return "date"

        if c.endswith("_id") or c in {"mmsi", "call_id", "berth_id", "terminal_code", "vessel_imo", "crane_id"}:
            return "identifier"

        if any(x in c for x in ["speed", "gust", "height", "temp", "pressure", "duration", "delay", "moves", "containers", "teu", "hours", "minutes", "pct", "capacity", "loa", "tide"]):
            return "measure"

        if any(x in c for x in ["status", "type", "category", "operator", "line", "name", "pier", "source"]):
            return "category"

        if dtype.startswith("int") or dtype.startswith("float"):
            return "measure"

        return "text"

    def infer_business_role(self, col: str) -> str:
        c = col.lower()

        if c in {"call_id"}:
            return "join_key"
        if c in {"mmsi", "vessel_imo", "berth_id", "crane_id", "yard_block", "terminal_code"}:
            return "entity_key"
        if c in {"eta", "ata", "etb", "atb", "etd", "atd", "gate_date", "operation_date"}:
            return "time_key"
        if c.startswith("event_"):
            return "event_flag"
        if c.endswith("_source"):
            return "data_provenance"
        return "attribute"

    def is_rule_candidate(self, col: str, semantic_type: str) -> bool:
        c = col.lower()

        if col.endswith("_source"):
            return False

        if semantic_type not in {"measure", "category", "event_flag"}:
            if not c.startswith("event_"):
                return False

        blocked = {
            "call_id", "yard_operation_id", "crane_operation_id",
            "vessel_name", "terminal_name", "operator", "source"
        }
        if c in blocked:
            return False

        return True

    def build_synonyms(self, col: str) -> List[str]:
        c = col.lower()
        synonyms = {
            "wind_speed_ms": ["wind speed", "wind", "average wind speed"],
            "wind_gust_ms": ["wind gust", "gust speed"],
            "wave_height_m": ["wave height", "sea state"],
            "tide_ft": ["tide", "tidal height"],
            "event_storm": ["storm", "storm condition"],
            "vessel_loa_meters": ["vessel length", "loa", "ship length"],
            "containers_actual": ["actual container moves", "actual teu", "container throughput"],
            "peak_occupancy_pct": ["yard occupancy", "yard utilization", "peak occupancy"],
            "average_turn_time_minutes": ["truck turn time", "gate turn time"],
            "berth_productivity_mph": ["berth productivity", "moves per hour"],
            "crane_productivity_mph": ["crane productivity", "quay crane productivity"],
        }
        return synonyms.get(c, [])

    def inspect_csv(self, file_path: Path) -> Dict[str, Any]:
        table_name = self.infer_table_name(file_path)
        df = pd.read_csv(file_path, nrows=500)

        columns = []
        primary_key_candidates = []
        time_columns = []
        join_keys = []

        for col in df.columns:
            dtype = str(df[col].dtype)
            semantic_type = self.infer_semantic_type(col, dtype)
            business_role = self.infer_business_role(col)

            if business_role == "time_key":
                time_columns.append(col)
            if business_role == "join_key":
                join_keys.append(col)

            nunique = df[col].nunique(dropna=True)
            notna = df[col].notna().sum()

            if nunique == len(df) and notna == len(df):
                if col.endswith("_id") or col in {"call_id", "mmsi"}:
                    primary_key_candidates.append(col)

            columns.append(
                {
                    "name": col,
                    "dtype_inferred": dtype,
                    "semantic_type": semantic_type,
                    "business_role": business_role,
                    "nullable": bool(df[col].isna().any()),
                    "unique_ratio_sample": round(nunique / max(len(df), 1), 4),
                    "sample_values": [x for x in df[col].dropna().astype(str).head(3).tolist()],
                    "rule_candidate": self.is_rule_candidate(col, semantic_type),
                    "synonyms": self.build_synonyms(col),
                }
            )

        return {
            "table_name": table_name,
            "file_name": file_path.name,
            "row_count_sampled": len(df),
            "primary_key_candidates": primary_key_candidates,
            "join_keys": join_keys,
            "time_columns": time_columns,
            "columns": columns,
        }

    def parse_all(self) -> Dict[str, Any]:
        files = self.discover_files()
        tables = []

        for f in files:
            tables.append(self.inspect_csv(f))

        registry = {
            "source_type": "csv_port_operations",
            "data_dir": str(self.data_dir),
            "tables": tables,
            "relationships": self.known_relationships,
        }

        return registry

    def build_rule_variable_catalog(self, registry: Dict[str, Any]) -> Dict[str, Any]:
        variables = []

        for table in registry["tables"]:
            for col in table["columns"]:
                if col["rule_candidate"]:
                    variables.append(
                        {
                            "table": table["table_name"],
                            "column": col["name"],
                            "semantic_type": col["semantic_type"],
                            "business_role": col["business_role"],
                            "synonyms": col["synonyms"],
                        }
                    )

        return {
            "variables": variables
        }

    @staticmethod
    def save_json(obj: Dict[str, Any], output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)