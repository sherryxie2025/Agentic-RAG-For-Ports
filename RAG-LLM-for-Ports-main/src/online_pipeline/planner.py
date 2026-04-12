# src/online_pipeline/planner.py
"""
Query Planner — generates source-specific, schema-aware sub-queries.

Strategy:
1. LLM-based planning (with full schema context injected into prompt)
2. Rule-based fallback: keyword → column/variable mapping, never returns
   the raw original_query unchanged.

Every sub-query should carry *information gain* over the user query:
- SQL sub-queries mention exact table/column names
- Rule sub-queries mention target variables and threshold domains
- Graph sub-queries list entity pairs for path search
- Document sub-queries add domain synonyms for retrieval recall
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from .llm_client import llm_chat_json

logger = logging.getLogger("online_pipeline.planner")

# ── Schema context (compact, injected into LLM prompt) ─────────────────────

_SQL_SCHEMA = """SQL Tables:
- environment(timestamp, wave_height_m, wind_speed_ms, wind_gust_ms, tide_ft, air_temp_c, pressure_hpa, event_storm, event_high_tide)
- berth_operations(call_id, terminal_code, eta, ata, etb, atb, etd, atd, arrival_delay_hours, berth_delay_hours, berth_productivity_mph, containers_actual, containers_planned)
- crane_operations(call_id, crane_id, crane_productivity_mph, breakdown_minutes, waiting_time_minutes, utilization_pct, total_moves)
- yard_operations(yard_block, teu_received, teu_delivered, average_dwell_days, peak_occupancy_pct, reefer_containers, hazmat_containers)
- gate_operations(gate_date, total_transactions, average_turn_time_minutes, peak_hour_volume, appointment_transactions, rejected_trucks)
- vessel_calls(visit_id, vessel_name, vessel_capacity_teu, vessel_loa_meters, size_category, shipping_line)"""

_GRAPH_NODES = """Neo4j Nodes:
Metrics: wind_speed_ms, wave_height_m, berth_productivity_mph, arrival_delay_hours, crane_productivity_mph, breakdown_minutes, average_dwell_days, average_turn_time_minutes, yard_occupancy_pct, teu_received
Operations: vessel_entry, navigation, berth_operations, crane_operations, yard_operations, gate_operations, operational_pause, vessel_scheduling
Concepts: weather_conditions, crane_slowdown, operational_disruption, congestion, safety, compliance, gate_congestion, yard_overflow, storm_event
Relations: AFFECTS, CONTRIBUTES_TO, CAN_LEAD_TO, CAN_TRIGGER, MEASURES, DISRUPTS"""

_RULE_DOMAINS = """Rule Domains:
- Weather thresholds: wind_speed_ms, wind_gust_ms, wave_height_m, tide_ft, visibility, fog
- Vessel entry: pilotage, tug requirement, vessel length limits, speed restrictions
- Crane operations: crane_wind_limit, breakdown, visibility_work_stop
- Safety: hazardous_cargo, fire_safety, mooring_safety, emergency_response
- Compliance: isps_compliance, security_screening, inspection requirements
- Scheduling: truck_appointment, berth_priority, tidal_windows"""

# ── LLM planner system prompt (schema-enriched) ────────────────────────────

_PLANNER_SYSTEM = f"""You are a query planner for a port operations RAG system.
Given a user query and required sources, generate ONE optimized sub-query per source.

{_SQL_SCHEMA}

{_GRAPH_NODES}

{_RULE_DOMAINS}

Return ONLY a JSON object:
{{
  "sub_queries": [
    {{"source": "documents|sql|rules|graph", "query": "optimized sub-query", "purpose": "brief purpose"}}
  ],
  "note": "sub_queries only, no reasoning_goal needed"
}}

Critical rules:
- SQL sub-queries MUST name exact table(s) and column(s) from the schema.
- Rule sub-queries MUST name the target variable domain (e.g. "wind_speed_ms threshold").
- Graph sub-queries MUST list entity pairs (e.g. "weather_conditions → crane_slowdown → berth_operations").
- Document sub-queries should add domain synonyms to improve retrieval recall.
- NEVER return the user query verbatim — always add information gain.

Return ONLY valid JSON."""


# ── Keyword → column/table mapping for rule-based fallback ──────────────────

_SQL_KEYWORD_MAP: Dict[str, Dict[str, Any]] = {
    # Weather / environment
    "wind": {"table": "environment", "cols": ["wind_speed_ms", "wind_gust_ms"], "synonyms": ["wind speed", "gust"]},
    "wave": {"table": "environment", "cols": ["wave_height_m", "dominant_period_s"], "synonyms": ["wave height", "swell"]},
    "tide": {"table": "environment", "cols": ["tide_ft"], "synonyms": ["tidal level"]},
    "temperature": {"table": "environment", "cols": ["air_temp_c", "water_temp_c"], "synonyms": ["temp"]},
    "pressure": {"table": "environment", "cols": ["pressure_hpa"], "synonyms": ["atmospheric pressure"]},
    "storm": {"table": "environment", "cols": ["event_storm"], "synonyms": ["storm event"]},
    "weather": {"table": "environment", "cols": ["wind_speed_ms", "wave_height_m", "tide_ft"], "synonyms": ["environmental conditions"]},
    # Berth
    "berth productivity": {"table": "berth_operations", "cols": ["berth_productivity_mph"], "synonyms": ["moves per hour"]},
    "berth delay": {"table": "berth_operations", "cols": ["arrival_delay_hours", "berth_delay_hours"], "synonyms": ["waiting time"]},
    "arrival delay": {"table": "berth_operations", "cols": ["arrival_delay_hours"], "synonyms": ["late arrival"]},
    "containers": {"table": "berth_operations", "cols": ["containers_actual", "containers_planned"], "synonyms": ["container volume"]},
    # Crane
    "crane productivity": {"table": "crane_operations", "cols": ["crane_productivity_mph"], "synonyms": ["crane moves per hour"]},
    "crane": {"table": "crane_operations", "cols": ["crane_productivity_mph", "breakdown_minutes", "utilization_pct"], "synonyms": ["quay crane", "STS"]},
    "breakdown": {"table": "crane_operations", "cols": ["breakdown_minutes"], "synonyms": ["equipment failure", "downtime"]},
    # Yard
    "yard": {"table": "yard_operations", "cols": ["teu_received", "average_dwell_days", "peak_occupancy_pct"], "synonyms": ["container yard"]},
    "dwell": {"table": "yard_operations", "cols": ["average_dwell_days"], "synonyms": ["storage time", "dwell time"]},
    "occupancy": {"table": "yard_operations", "cols": ["peak_occupancy_pct"], "synonyms": ["yard utilization"]},
    # Gate
    "gate": {"table": "gate_operations", "cols": ["total_transactions", "average_turn_time_minutes"], "synonyms": ["truck gate"]},
    "turn time": {"table": "gate_operations", "cols": ["average_turn_time_minutes"], "synonyms": ["truck turnaround"]},
    "transaction": {"table": "gate_operations", "cols": ["total_transactions", "peak_hour_volume"], "synonyms": ["gate volume"]},
    # Vessel
    "vessel": {"table": "vessel_calls", "cols": ["vessel_name", "vessel_capacity_teu", "vessel_loa_meters"], "synonyms": ["ship"]},
    "loa": {"table": "vessel_calls", "cols": ["vessel_loa_meters"], "synonyms": ["length overall", "vessel length"]},
    "teu": {"table": "vessel_calls", "cols": ["vessel_capacity_teu"], "synonyms": ["twenty-foot equivalent"]},
}

_RULE_KEYWORD_MAP: Dict[str, Dict[str, Any]] = {
    "wind": {"variables": ["wind_speed_ms", "wind_gust_ms", "crane_wind_limit"], "domain": "weather thresholds"},
    "wave": {"variables": ["wave_height_m"], "domain": "weather thresholds"},
    "tide": {"variables": ["tide_ft", "tidal_windows"], "domain": "weather thresholds"},
    "fog": {"variables": ["visibility_work_stop", "fog_transit"], "domain": "visibility restrictions"},
    "visibility": {"variables": ["visibility_work_stop"], "domain": "visibility restrictions"},
    "vessel entry": {"variables": ["speed_restrictions", "pilotage_mandate", "vessel_loa_meters"], "domain": "vessel entry requirements"},
    "pilotage": {"variables": ["pilotage_mandate"], "domain": "vessel entry requirements"},
    "tug": {"variables": ["tug_assistance"], "domain": "vessel entry requirements"},
    "crane": {"variables": ["crane_wind_limit", "crane_inspection"], "domain": "crane operation limits"},
    "hazardous": {"variables": ["hazardous_cargo_handling", "hazardous_segregation"], "domain": "hazmat safety"},
    "hazmat": {"variables": ["hazardous_cargo_handling", "hazardous_segregation"], "domain": "hazmat safety"},
    "security": {"variables": ["isps_compliance", "security_screening"], "domain": "security compliance"},
    "isps": {"variables": ["isps_compliance"], "domain": "security compliance"},
    "fire": {"variables": ["fire_safety_terminal"], "domain": "safety"},
    "mooring": {"variables": ["mooring_safety"], "domain": "berth safety"},
    "reefer": {"variables": ["reefer_temp_log"], "domain": "reefer management"},
    "appointment": {"variables": ["truck_appointment"], "domain": "gate scheduling"},
    "berth priority": {"variables": ["berth_priority"], "domain": "berth scheduling"},
    "pause": {"variables": ["operational_threshold", "crane_wind_limit", "visibility_work_stop"], "domain": "operational restrictions"},
    "restrict": {"variables": ["speed_restrictions", "wind_restriction", "navigation_restriction"], "domain": "operational restrictions"},
    "suspend": {"variables": ["operational_threshold", "crane_wind_limit"], "domain": "operational restrictions"},
    "emergency": {"variables": ["emergency_response_protocols"], "domain": "emergency procedures"},
    "pollution": {"variables": ["pollution_prevention_measures"], "domain": "environmental compliance"},
    "maintenance": {"variables": ["berth_maintenance", "infrastructure_maintenance", "maintenance_safety"], "domain": "maintenance"},
}

_GRAPH_ENTITY_MAP: Dict[str, List[str]] = {
    "berth delay": ["arrival_delay_hours", "berth_operations"],
    "berth productivity": ["berth_productivity_mph", "berth_operations"],
    "crane slowdown": ["crane_slowdown", "crane_operations"],
    "crane productivity": ["crane_productivity_mph", "crane_operations"],
    "breakdown": ["breakdown_minutes", "crane_operations"],
    "weather": ["weather_conditions", "wind_speed_ms", "wave_height_m"],
    "wind": ["wind_speed_ms", "weather_conditions"],
    "wave": ["wave_height_m", "weather_conditions"],
    "storm": ["storm_event", "weather_conditions"],
    "congestion": ["congestion", "gate_congestion", "yard_overflow"],
    "yard": ["yard_operations", "yard_occupancy_pct", "yard_overflow"],
    "gate": ["gate_operations", "gate_congestion"],
    "vessel entry": ["vessel_entry", "navigation"],
    "safety": ["safety", "compliance"],
    "disruption": ["operational_disruption", "operational_pause"],
    "delay": ["arrival_delay_hours", "berth_operations", "operational_disruption"],
    "navigation": ["navigation", "vessel_entry", "navigation_restriction"],
}

# ── Document topic enrichment ───────────────────────────────────────────────

_DOC_SYNONYM_MAP: Dict[str, List[str]] = {
    "sustainability": ["green port", "emissions", "decarbonization", "GHG", "net-zero", "environmental"],
    "safety": ["risk", "resilience", "emergency", "hazard", "ISPS", "security"],
    "crane": ["quay crane", "STS crane", "gantry crane", "container handling"],
    "berth": ["wharf", "quay", "pier", "mooring", "berthing"],
    "vessel entry": ["pilotage", "harbour master", "navigation", "channel", "transit"],
    "environment": ["weather", "wind", "wave", "tide", "climate", "storm"],
    "gate": ["truck", "appointment", "turn time", "transaction", "drayage"],
    "yard": ["container yard", "stacking", "dwell time", "TEU", "reefer"],
    "financial": ["revenue", "budget", "expenditure", "fiscal", "annual report"],
    "infrastructure": ["design", "construction", "maintenance", "modernization"],
    "cybersecurity": ["cyber", "OT security", "network", "digital", "IT security"],
    "compliance": ["regulation", "ISPS", "IMO", "inspection", "audit", "certification"],
    "handbook": ["operating handbook", "port manual", "procedures", "operating rules"],
}


class QueryPlanner:
    """
    Schema-aware query planner for the AI Port Decision-Support System.

    Generates source-specific sub-queries with real information gain:
    - SQL: table + column names from schema
    - Rules: target variable names + threshold domain
    - Graph: entity pairs for Neo4j traversal
    - Documents: domain synonyms for retrieval recall boost
    """

    def plan(
        self,
        user_query: str,
        router_decision: Dict[str, Any],
    ) -> Dict[str, Any]:
        query = self._normalize(user_query)

        needs_vector = router_decision.get("needs_vector", False)
        needs_sql = router_decision.get("needs_sql", False)
        needs_rules = router_decision.get("needs_rules", False)
        needs_graph = router_decision.get("needs_graph_reasoning", False)

        source_plan: List[str] = []
        if needs_vector:
            source_plan.append("documents")
        if needs_sql:
            source_plan.append("sql")
        if needs_rules:
            source_plan.append("rules")
        if needs_graph:
            source_plan.append("graph")

        # ── v3: LLM-first for ALL queries, rule-based only as fallback ──
        # Sub-query generation is the most critical planning step: it
        # determines which exact table/column/variable/entity each
        # retriever targets. The LLM's schema-aware prompt (_PLANNER_SYSTEM)
        # produces far better sub-queries than keyword heuristics.
        import time as _time

        planning_method = "rule"
        _timings: Dict[str, float] = {}

        if source_plan:
            t_llm = _time.time()
            llm_plan = self._llm_plan(user_query, source_plan)
            _timings["sub_queries__llm_call"] = round(_time.time() - t_llm, 4)

            if llm_plan and llm_plan.get("sub_queries"):
                sub_queries = llm_plan["sub_queries"]
                planning_method = "llm"
            else:
                t_rule = _time.time()
                sub_queries = self._build_sub_queries(
                    query=query, original_query=user_query,
                    needs_vector=needs_vector, needs_sql=needs_sql,
                    needs_rules=needs_rules, needs_graph=needs_graph,
                )
                _timings["sub_queries__rule_fallback"] = round(_time.time() - t_rule, 4)
        else:
            sub_queries = []

        # Per-source timing: record which sources got sub-queries
        for sq in sub_queries:
            src = sq.get("source", "unknown")
            _timings[f"sub_query__{src}__method"] = 0.0  # placeholder for presence

        execution_strategy = self._infer_execution_strategy(
            needs_vector=needs_vector,
            needs_sql=needs_sql,
            needs_rules=needs_rules,
            needs_graph=needs_graph,
        )

        # ── Enhanced logging (INFO level) ──
        logger.info(
            "PLAN: sources=%s strategy=%s method=%s",
            source_plan, execution_strategy, planning_method,
        )
        for sq in sub_queries:
            is_passthrough = sq.get("query", "").strip() == user_query.strip()
            tag = " [PASSTHROUGH]" if is_passthrough else ""
            logger.info(
                "  sub_query[%s]: %s%s",
                sq.get("source"), sq.get("query", "")[:120], tag,
            )

        return {
            "source_plan": source_plan,
            "sub_queries": sub_queries,
            "execution_strategy": execution_strategy,
            "planning_method": planning_method,
            "_timings": _timings,
        }

    # ── LLM planner (schema-enriched prompt) ────────────────────────────────

    def _llm_plan(self, user_query: str, source_plan: List[str]) -> dict | None:
        """Use LLM to generate optimized sub-queries with schema context.

        v3: called for ALL queries (not just 3+ sources). This is the most
        critical planning step — better sub-queries directly improve SQL
        table/column targeting, rule variable matching, and graph entity
        selection. Timeout raised to 45s to give the LLM enough time for
        complex multi-source plans.
        """
        if not source_plan:
            return None
        try:
            result = llm_chat_json(
                messages=[
                    {"role": "system", "content": _PLANNER_SYSTEM},
                    {"role": "user", "content": (
                        f"Query: {user_query}\n"
                        f"Required sources: {', '.join(source_plan)}\n"
                        f"Generate one optimized sub-query per source."
                    )},
                ],
                temperature=0.1,
                timeout=45,
            )
            if isinstance(result, dict) and result.get("sub_queries"):
                logger.debug("LLM planner returned %d sub-queries", len(result["sub_queries"]))
                return result
        except Exception as e:
            logger.warning("LLM planner failed: %s => falling back to rule-based", e)
        return None

    # ── Rule-based fallback (schema-aware) ──────────────────────────────────

    def _build_sub_queries(
        self,
        query: str,
        original_query: str,
        needs_vector: bool,
        needs_sql: bool,
        needs_rules: bool,
        needs_graph: bool,
    ) -> List[Dict[str, str]]:
        sub_queries: List[Dict[str, str]] = []

        if needs_vector:
            sub_queries.append({
                "source": "documents",
                "query": self._build_doc_subquery(query, original_query),
                "purpose": "retrieve textual evidence from port documents",
            })

        if needs_sql:
            sub_queries.append({
                "source": "sql",
                "query": self._build_sql_subquery(query, original_query),
                "purpose": "retrieve structured operational data",
            })

        if needs_rules:
            sub_queries.append({
                "source": "rules",
                "query": self._build_rule_subquery(query, original_query),
                "purpose": "retrieve policy/rule thresholds",
            })

        if needs_graph:
            sub_queries.append({
                "source": "graph",
                "query": self._build_graph_subquery(query, original_query),
                "purpose": "multi-hop causal reasoning over entity graph",
            })

        return sub_queries

    def _build_sql_subquery(self, query: str, original_query: str) -> str:
        """Map query keywords to exact SQL table.columns, never return original_query raw."""
        matched_tables: Dict[str, List[str]] = {}  # table -> list of columns

        for keyword, info in _SQL_KEYWORD_MAP.items():
            if keyword in query:
                table = info["table"]
                if table not in matched_tables:
                    matched_tables[table] = []
                matched_tables[table].extend(info["cols"])

        if matched_tables:
            parts = []
            for table, cols in matched_tables.items():
                unique_cols = list(dict.fromkeys(cols))  # dedupe, preserve order
                parts.append(f"{table}({', '.join(unique_cols)})")
            schema_hint = "; ".join(parts)
            return f"{original_query} [Target: {schema_hint}]"

        # Generic fallback: infer from query structure, still add table hints
        table = self._guess_primary_table(query)
        return f"{original_query} [Target: {table}]"

    def _build_rule_subquery(self, query: str, original_query: str) -> str:
        """Map query keywords to rule variable domains, never return original_query raw."""
        matched_vars: List[str] = []
        matched_domains: List[str] = []

        for keyword, info in _RULE_KEYWORD_MAP.items():
            if keyword in query:
                matched_vars.extend(info["variables"])
                matched_domains.append(info["domain"])

        if matched_vars:
            unique_vars = list(dict.fromkeys(matched_vars))[:6]
            unique_domains = list(dict.fromkeys(matched_domains))[:3]
            return (
                f"Find rules for variables: {', '.join(unique_vars)} "
                f"(domain: {', '.join(unique_domains)})"
            )

        # Generic fallback: still enrich beyond raw query
        return f"{original_query} [Search all rule variables for matching thresholds and restrictions]"

    def _build_graph_subquery(self, query: str, original_query: str) -> str:
        """Map query keywords to Neo4j entity pairs, never return original_query raw."""
        matched_entities: List[str] = []

        for keyword, entities in _GRAPH_ENTITY_MAP.items():
            if keyword in query:
                matched_entities.extend(entities)

        if matched_entities:
            unique = list(dict.fromkeys(matched_entities))[:8]
            return f"Find causal paths between: {', '.join(unique)}"

        # Generic fallback
        return f"{original_query} [Explore causal relationships in the port operations knowledge graph]"

    def _build_doc_subquery(self, query: str, original_query: str) -> str:
        """Enrich document query with domain synonyms for better retrieval recall."""
        added_terms: List[str] = []

        for keyword, synonyms in _DOC_SYNONYM_MAP.items():
            if keyword in query:
                # Add 2-3 most relevant synonyms
                for syn in synonyms[:3]:
                    if syn.lower() not in query:
                        added_terms.append(syn)

        if added_terms:
            unique_terms = list(dict.fromkeys(added_terms))[:5]
            return f"{original_query} ({', '.join(unique_terms)})"

        return original_query  # Documents can use original query as-is (embedding handles semantics)

    # ── Helper: guess primary table from query ──────────────────────────────

    @staticmethod
    def _guess_primary_table(query: str) -> str:
        """When no specific keyword matched, guess the most likely SQL table."""
        table_signals = [
            ("berth", "berth_operations"),
            ("crane", "crane_operations"),
            ("yard", "yard_operations"),
            ("gate", "gate_operations"),
            ("vessel", "vessel_calls"),
            ("ship", "vessel_calls"),
            ("wind", "environment"),
            ("wave", "environment"),
            ("weather", "environment"),
            ("temperature", "environment"),
        ]
        for signal, table in table_signals:
            if signal in query:
                return table
        return "berth_operations, crane_operations, environment"

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def _infer_execution_strategy(
        self,
        needs_vector: bool,
        needs_sql: bool,
        needs_rules: bool,
        needs_graph: bool,
    ) -> str:
        source_count = sum([needs_vector, needs_sql, needs_rules, needs_graph])

        if needs_graph:
            return "parallel_then_graph_merge"
        if source_count >= 2:
            return "parallel_then_merge"
        return "single_source"
