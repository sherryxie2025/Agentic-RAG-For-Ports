# src/online_pipeline/sql_agent_v2.py

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import logging

import duckdb

from .llm_client import get_openai_client, get_model_name

logger = logging.getLogger("online_pipeline.sql_agent")
from .sql_executor import DuckDBExecutor
from .state_schema import PortQAState, SQLExecutionResult


class SQLAgentV2:
    """
    Phase 2 SQL agent:
    NL query -> SQL generation -> DuckDB execution

    Improvements in this version:
    - schema-aware via DuckDB introspection
    - dynamic time-column selection
    - returns used_tables directly
    - LLM SQL generation with rule-based fallback
    """

    def __init__(
        self,
        db_path: str | Path,
        use_llm_sql: bool = False,
        model_name: str | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.executor = DuckDBExecutor(self.db_path)

        self.use_llm_sql = use_llm_sql
        self.model_name = model_name or get_model_name()
        self.client = get_openai_client() if self.use_llm_sql else None

        self.schema_info = self._inspect_schema()
        self.schema_text = self._build_schema_text()

    # =========================================================
    # Public API
    # =========================================================

    def run(self, query: str) -> SQLExecutionResult:
        sql: Optional[str] = None
        used_tables: List[str] = []
        generation_mode = "rule_based"

        if self.use_llm_sql and self.client is not None:
            llm_result = self._generate_sql_llm(query)
            if llm_result is not None:
                sql = llm_result["sql"]
                used_tables = llm_result["used_tables"]
                generation_mode = "llm"

        if not sql:
            rb_result = self._generate_sql_rule_based(query)
            sql = rb_result["sql"]
            used_tables = rb_result["used_tables"]
            generation_mode = "rule_based"

        logger.info("SQL_GEN: mode=%s tables=%s", generation_mode, used_tables)
        logger.debug("SQL: %s", sql.strip()[:200])

        # Pre-check: use DuckDB EXPLAIN to catch syntax/GROUP BY/type errors
        # BEFORE spending execution cost. If EXPLAIN fails and we're in LLM
        # mode, skip straight to rule-based fallback.
        if generation_mode == "llm":
            explain_ok, explain_err = self.executor.explain(sql)
            if not explain_ok:
                logger.warning(
                    "SQL_EXPLAIN: LLM SQL failed pre-check (%s), pre-empting with rule-based",
                    (explain_err or "")[:80],
                )
                rb_result = self._generate_sql_rule_based(query)
                sql = rb_result["sql"]
                used_tables = rb_result["used_tables"]
                generation_mode = "rule_based_preempt"

        execution = self.executor.execute(sql)

        # Auto-fallback: if SQL still failed execution (rule-based SQL can
        # also fail on edge cases), try rule-based as a second rescue.
        if not execution.execution_ok and generation_mode == "llm":
            logger.warning(
                "SQL_EXEC: LLM SQL failed (%s), falling back to rule-based",
                (execution.error or "")[:80],
            )
            rb_result = self._generate_sql_rule_based(query)
            fallback_sql = rb_result["sql"]
            fallback_tables = rb_result["used_tables"]
            fallback_exec = self.executor.execute(fallback_sql)
            if fallback_exec.execution_ok:
                logger.info("SQL_FALLBACK: rule-based rescue succeeded")
                sql = fallback_sql
                used_tables = fallback_tables
                generation_mode = "rule_based_fallback"
                execution = fallback_exec

        if execution.execution_ok:
            logger.info("SQL_EXEC: ok rows=%d", execution.row_count)
        else:
            logger.error("SQL_EXEC: FAILED error=%s", execution.error)
        if execution.rows:
            logger.debug("SQL first row: %s", execution.rows[0] if execution.rows else "empty")

        return {
            "plan": {
                "nl_query": query,
                "target_tables": used_tables,
                "target_columns": [],
                "filters": {},
                "aggregation": self._infer_aggregation_from_sql(sql),
                "generated_sql": sql,
                "generation_mode": generation_mode,
            },
            "rows": [{"data": row} for row in execution.rows],
            "row_count": execution.row_count,
            "execution_ok": execution.execution_ok,
            "error": execution.error,
        }

    def update_state(self, state: PortQAState) -> PortQAState:
        query = state.get("user_query", "")
        result = self.run(query)

        existing = state.get("sql_results", [])
        existing.append(result)
        state["sql_results"] = existing

        trace = state.get("reasoning_trace", [])
        plan = result.get("plan", {})
        trace.append(
            f"SQLAgentV2 => mode={plan.get('generation_mode')} "
            f"tables={plan.get('target_tables')} rows={result.get('row_count', 0)} "
            f"ok={result.get('execution_ok')}"
        )
        state["reasoning_trace"] = trace
        return state

    # =========================================================
    # Schema inspection
    # =========================================================

    def _inspect_schema(self) -> Dict[str, Dict[str, Any]]:
        info: Dict[str, Dict[str, Any]] = {}

        conn = duckdb.connect(str(self.db_path), read_only=True)
        try:
            tables = conn.execute("SHOW TABLES").fetchall()
            table_names = [t[0] for t in tables]

            for table_name in table_names:
                rows = conn.execute(f"DESCRIBE {table_name}").fetchall()
                columns = [r[0] for r in rows]
                info[table_name] = {
                    "columns": columns,
                    "time_column": self._pick_time_column(columns),
                }
        finally:
            conn.close()

        return info

    @staticmethod
    def _pick_time_column(columns: List[str]) -> Optional[str]:
        candidates = [
            "timestamp",
            "operation_date",
            "gate_date",
            "ata",
            "atd",
            "atb",
            "etb",
            "eta",
            "etd",
        ]
        for c in candidates:
            if c in columns:
                return c
        return None

    def _build_schema_text(self) -> str:
        lines: List[str] = []
        for table_name, meta in self.schema_info.items():
            cols = ", ".join(meta["columns"])
            time_col = meta.get("time_column")
            if time_col:
                lines.append(f"- {table_name} (time_column={time_col}): {cols}")
            else:
                lines.append(f"- {table_name}: {cols}")
        return "\n".join(lines)

    # =========================================================
    # LLM generation
    # =========================================================

    def _generate_sql_llm(self, query: str) -> Optional[Dict[str, Any]]:
        try:
            system_prompt = f"""
You are a SQL generation assistant for a port operations analytics database in DuckDB.

Available tables and fields:
{self.schema_text}

Requirements:
1. Output JSON only.
2. Use this exact schema:
{{
  "sql": "<SQL query>",
  "used_tables": ["table1", "table2"]
}}
3. Only generate a read-only SELECT or WITH query.
4. Use valid table names and valid column names only.
5. Prefer simple, correct SQL.
6. For year filters, use the actual table time column shown in schema_text.
7. Limit non-aggregated preview queries to 20 rows.
8. If the question is partly policy-oriented, generate the best supporting analytical SQL query instead of inventing policy logic.
"""
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt.strip()},
                    {"role": "user", "content": query},
                ],
                temperature=0.1,
                timeout=45,
                max_tokens=600,
            )

            text = response.choices[0].message.content.strip()
            text = self._strip_code_fence(text)

            import json
            parsed = json.loads(text)

            sql = parsed.get("sql")
            used_tables = parsed.get("used_tables", [])

            if not sql or not self._looks_like_sql(sql):
                return None

            used_tables = [t for t in used_tables if t in self.schema_info]
            if not used_tables:
                used_tables = self._extract_table_names_from_sql(sql)

            return {
                "sql": sql,
                "used_tables": used_tables,
            }
        except Exception:
            return None

    # =========================================================
    # Rule-based fallback
    # =========================================================

    def _generate_sql_rule_based(self, query: str) -> Dict[str, Any]:
        q = query.lower().strip()

        crane_time = self._get_time_col("crane_operations")
        env_time = self._get_time_col("environment")
        berth_time = self._get_time_col("berth_operations")
        gate_time = self._get_time_col("gate_operations")

        # 1. average crane productivity in 2015
        if "average crane productivity" in q:
            if crane_time:
                return {
                    "sql": f"""
                    SELECT AVG(crane_productivity_mph) AS avg_crane_productivity_mph
                    FROM crane_operations
                    WHERE EXTRACT(YEAR FROM {crane_time}) = 2015
                    """,
                    "used_tables": ["crane_operations"],
                }
            return {
                "sql": """
                SELECT AVG(crane_productivity_mph) AS avg_crane_productivity_mph
                FROM crane_operations
                """,
                "used_tables": ["crane_operations"],
            }

        # 2. average wave height in 2015
        if "average wave height" in q:
            if env_time:
                return {
                    "sql": f"""
                    SELECT AVG(wave_height_m) AS avg_wave_height_m
                    FROM environment
                    WHERE EXTRACT(YEAR FROM {env_time}) = 2015
                    """,
                    "used_tables": ["environment"],
                }
            return {
                "sql": """
                SELECT AVG(wave_height_m) AS avg_wave_height_m
                FROM environment
                """,
                "used_tables": ["environment"],
            }

        # 3. average turn time in 2015
        if "average turn time" in q:
            if gate_time:
                return {
                    "sql": f"""
                    SELECT AVG(average_turn_time_minutes) AS avg_turn_time_minutes
                    FROM gate_operations
                    WHERE EXTRACT(YEAR FROM {gate_time}) = 2015
                    """,
                    "used_tables": ["gate_operations"],
                }
            return {
                "sql": """
                SELECT AVG(average_turn_time_minutes) AS avg_turn_time_minutes
                FROM gate_operations
                """,
                "used_tables": ["gate_operations"],
            }

        # 4. top highest arrival delay
        if ("top" in q or "highest" in q or "largest" in q) and "arrival delay" in q:
            limit_n = self._extract_top_k(q, default=5)
            order = "DESC"
            return {
                "sql": f"""
                SELECT
                    call_id,
                    berth_id,
                    terminal_code,
                    arrival_delay_hours,
                    berth_productivity_mph
                FROM berth_operations
                ORDER BY arrival_delay_hours {order}
                LIMIT {limit_n}
                """,
                "used_tables": ["berth_operations"],
            }

        # 5. berth delay + crane slowdown / breakdown
        if "berth delays" in q and ("weather" in q or "crane" in q or "slowdown" in q):
            return {
                "sql": """
                SELECT
                    b.call_id,
                    b.arrival_delay_hours,
                    b.berth_productivity_mph,
                    c.crane_productivity_mph,
                    c.breakdown_minutes
                FROM berth_operations b
                LEFT JOIN crane_operations c
                    ON b.call_id = c.call_id
                LIMIT 20
                """,
                "used_tables": ["berth_operations", "crane_operations"],
            }

        # 6. wave height + berth productivity
        if "wave height" in q and "berth productivity" in q:
            sql = """
            SELECT
                b.call_id,
                b.berth_productivity_mph,
                b.arrival_delay_hours
            FROM berth_operations b
            LIMIT 20
            """
            # environment is not naturally joinable here without a temporal alignment strategy.
            # keep it as supporting analytics rather than fake join logic.
            return {
                "sql": sql,
                "used_tables": ["berth_operations"],
            }

        # 7. wind conditions / vessel entry restriction support query
        if "wind" in q and ("vessel entry" in q or "entry" in q):
            return {
                "sql": """
                SELECT
                    timestamp,
                    wind_speed_ms,
                    wind_gust_ms,
                    wave_height_m
                FROM environment
                ORDER BY wind_speed_ms DESC
                LIMIT 20
                """,
                "used_tables": ["environment"],
            }

        # 8. generic environment analytics
        if "wind" in q or "weather" in q or "wave" in q:
            return {
                "sql": "SELECT * FROM environment LIMIT 20",
                "used_tables": ["environment"],
            }

        # 9. generic crane analytics
        if "crane" in q:
            return {
                "sql": "SELECT * FROM crane_operations LIMIT 20",
                "used_tables": ["crane_operations"],
            }

        # 10. generic berth analytics
        if "berth" in q:
            return {
                "sql": "SELECT * FROM berth_operations LIMIT 20",
                "used_tables": ["berth_operations"],
            }

        if "yard" in q:
            return {
                "sql": "SELECT * FROM yard_operations LIMIT 20",
                "used_tables": ["yard_operations"],
            }

        if "gate" in q:
            return {
                "sql": "SELECT * FROM gate_operations LIMIT 20",
                "used_tables": ["gate_operations"],
            }

        return {
            "sql": "SELECT * FROM vessel_calls LIMIT 20",
            "used_tables": ["vessel_calls"],
        }

    # =========================================================
    # Helpers
    # =========================================================

    def _get_time_col(self, table_name: str) -> Optional[str]:
        meta = self.schema_info.get(table_name, {})
        return meta.get("time_column")

    @staticmethod
    def _extract_top_k(query: str, default: int = 5) -> int:
        match = re.search(r"\btop\s+(\d+)\b", query)
        if match:
            return int(match.group(1))
        return default

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        return text.strip()

    @staticmethod
    def _looks_like_sql(text: str) -> bool:
        lowered = text.lower().strip()
        return lowered.startswith("select") or lowered.startswith("with")

    def _extract_table_names_from_sql(self, sql: str) -> List[str]:
        lowered = sql.lower()
        matches = re.findall(r"(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)", lowered)
        candidates = []
        for m in matches:
            if m in self.schema_info:
                candidates.append(m)
        return sorted(list(set(candidates)))

    @staticmethod
    def _infer_aggregation_from_sql(sql: str) -> Optional[str]:
        lowered = sql.lower()
        if "avg(" in lowered:
            return "mean"
        if "sum(" in lowered:
            return "sum"
        if "count(" in lowered:
            return "count"
        if "max(" in lowered:
            return "max"
        if "min(" in lowered:
            return "min"
        return None


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    db_path = project_root / "storage" / "sql" / "port_ops.duckdb"

    agent = SQLAgentV2(
        db_path=db_path,
        use_llm_sql=True,
    )

    demo_queries = [
        "What was the average crane productivity in 2015?",
        "What was the average wave height in 2015?",
        "What was the average turn time in 2015?",
        "What are the top 5 highest arrival delay cases?",
        "Why might berth delays be related to weather conditions and crane slowdown?",
    ]

    for q in demo_queries:
        print("=" * 100)
        print("QUERY:", q)
        print(agent.run(q))