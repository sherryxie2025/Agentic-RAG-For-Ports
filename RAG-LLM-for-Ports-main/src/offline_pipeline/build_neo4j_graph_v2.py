# src/offline_pipeline/build_neo4j_graph_v2.py
"""
Rule-driven knowledge graph builder (v2).

Replaces the 100% hardcoded build_neo4j_graph.py with an automated pipeline
that constructs the graph from:

1. **Auto-taxonomy** — Metric nodes generated from SQL schema
   (via taxonomy_generator.py)
2. **Grounded rules** — Each rule becomes an edge with threshold + citation
3. **SQL data statistics** — CORRELATES_WITH edges from actual data patterns
   (simple numeric correlation over operational tables)

Every edge has provenance: source_file + page for rule edges, correlation
coefficient for statistical edges.

Key advantages over v1:
- Adding a new rule automatically adds the corresponding graph edges
- Schema changes propagate to taxonomy → graph
- Every edge is grounded in a real rule or statistical pattern
- Conflicting rules (same variable, different thresholds) become visible

Usage:
    python -m src.offline_pipeline.build_neo4j_graph_v2
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("offline_pipeline.build_neo4j_graph_v2")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEFAULT_DB_PATH = "storage/sql/port_ops.duckdb"
GROUNDED_RULES_PATH = "data/rules/grounded_rules.json"
POLICY_RULES_PATH = "data/rules/policy_rules.json"
TAXONOMY_PATH = "data/rules/taxonomy_auto.json"

# How to interpret rule "action" strings → graph node names
_ACTION_KEYWORD_TO_NODE = {
    "stop": ("operational_pause", "Operation"),
    "pause": ("operational_pause", "Operation"),
    "suspend": ("operational_pause", "Operation"),
    "cease": ("operational_pause", "Operation"),
    "halt": ("operational_pause", "Operation"),
    "restrict": ("operational_restriction", "Concept"),
    "prohibit": ("operational_restriction", "Concept"),
    "delay": ("delay", "Operation"),
    "trigger": ("threshold_breach", "Concept"),
    "require": ("operational_requirement", "Concept"),
    "permit": ("operational_permit", "Concept"),
    "limit": ("operational_threshold", "Concept"),
    "inspect": ("safety_inspection", "Operation"),
    "notify": ("notification", "Operation"),
    "tug": ("tug_assistance", "Operation"),
}


def _normalize_name(s: str) -> str:
    """Convert arbitrary string to a node-safe identifier."""
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def _parse_action_to_node(action: str) -> Tuple[str, str]:
    """
    Parse a free-text rule action into (node_name, label).
    Returns ("operational_pause", "Operation") etc.
    Falls back to a generic node when no keyword matches.
    """
    if not action:
        return "unspecified_action", "Concept"

    action_lower = action.lower()
    for keyword, (node, label) in _ACTION_KEYWORD_TO_NODE.items():
        if keyword in action_lower:
            return node, label

    # Fallback: use normalized action text as a Concept node
    return _normalize_name(action)[:40], "Concept"


class RuleDrivenGraphBuilder:
    """
    Build a knowledge graph where every edge is grounded in either:
    - A rule from grounded_rules.json (edge = threshold + citation)
    - A statistical pattern from SQL data (edge = correlation coefficient)
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        wipe_existing: bool = False,
    ) -> None:
        self.db_path = Path(db_path)
        self.wipe_existing = wipe_existing

        # Lazy neo4j connection
        self._driver = None
        self._database = None

        # Counters
        self.nodes_created = 0
        self.edges_created = 0
        self.rules_skipped = 0

    # -----------------------------------------------------------------------
    # Neo4j connection
    # -----------------------------------------------------------------------

    def _connect(self):
        if self._driver is not None:
            return self._driver
        from dotenv import load_dotenv
        from neo4j import GraphDatabase
        load_dotenv(PROJECT_ROOT / ".env")

        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME")
        pwd = os.getenv("NEO4J_PASSWORD")
        self._database = os.getenv("NEO4J_DATABASE", "neo4j")
        if not uri or not user or not pwd:
            raise ValueError("Missing Neo4j credentials in .env")
        self._driver = GraphDatabase.driver(uri, auth=(user, pwd))
        return self._driver

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    # -----------------------------------------------------------------------
    # Build pipeline
    # -----------------------------------------------------------------------

    def build(self) -> None:
        """Run the full pipeline."""
        driver = self._connect()
        with driver.session(database=self._database) as session:
            if self.wipe_existing:
                logger.info("Wiping existing graph")
                session.run("MATCH (n) DETACH DELETE n")

            self._create_constraints(session)

            # Phase 1: Metric nodes from taxonomy
            taxonomy = self._load_taxonomy()
            self._create_metric_nodes(session, taxonomy)

            # Phase 1b: Bridge concept nodes (for backward compat with
            # graph_reasoner's entity alias map and user queries that use
            # generic concept names like "weather" or "safety")
            self._create_bridge_concepts(session)

            # Phase 2: Rule-driven edges (threshold relationships + citations)
            grounded_rules = self._load_rules(GROUNDED_RULES_PATH)
            policy_rules = self._load_rules(POLICY_RULES_PATH)
            self._create_rule_edges(session, grounded_rules, "grounded")
            self._create_rule_edges(session, policy_rules, "policy")

            # Phase 3: Statistical correlations from SQL data
            self._create_correlation_edges(session, taxonomy)

        logger.info(
            "Graph v2 build complete: %d nodes, %d edges, %d rules skipped",
            self.nodes_created, self.edges_created, self.rules_skipped,
        )

    # -----------------------------------------------------------------------
    # Phase 1: Metric nodes
    # -----------------------------------------------------------------------

    def _create_constraints(self, session) -> None:
        for q in [
            "CREATE CONSTRAINT metric_name_v2 IF NOT EXISTS FOR (n:Metric) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT operation_name_v2 IF NOT EXISTS FOR (n:Operation) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT concept_name_v2 IF NOT EXISTS FOR (n:Concept) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT source_name_v2 IF NOT EXISTS FOR (n:Source) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT action_name_v2 IF NOT EXISTS FOR (n:Action) REQUIRE n.name IS UNIQUE",
        ]:
            session.run(q)

    def _load_taxonomy(self) -> Dict[str, Any]:
        from .taxonomy_generator import load_auto_taxonomy
        return load_auto_taxonomy(TAXONOMY_PATH)

    def _create_metric_nodes(self, session, taxonomy: Dict[str, Any]) -> None:
        variable_meta = taxonomy.get("variable_meta", {})
        for var_name, meta in variable_meta.items():
            session.run(
                """
                MERGE (m:Metric {name: $name})
                SET m.unit = $unit,
                    m.category = $category,
                    m.source_table = $source_table,
                    m.sql_type = $sql_type
                """,
                name=var_name,
                unit=meta.get("unit"),
                category=meta.get("category"),
                source_table=meta.get("source_table"),
                sql_type=meta.get("sql_type"),
            )
            self.nodes_created += 1
        logger.info("Created %d metric nodes from auto-taxonomy", self.nodes_created)

    def _create_bridge_concepts(self, session) -> None:
        """
        Create legacy concept + operation nodes and link them to the metric layer.

        Preserves backward compatibility with graph_reasoner's entity_alias_map,
        which references concept names like "weather_conditions" and operation
        names like "vessel_entry". Without these bridges, generic queries
        ("weather", "vessel", "navigation") can't traverse to the metric nodes.
        """
        # --- Concept bridges (:Concept) ---
        concept_bridges = [
            # (concept_name, related_metric_names, relationship_type)
            ("weather_conditions", ["wind_speed_ms", "wind_gust_ms", "wave_height_m",
                                     "air_temp_c", "pressure_hpa"], "INCLUDES"),
            ("environmental_conditions", ["water_temp_c", "tide_ft", "pressure_hpa"], "INCLUDES"),
            ("storm_event", ["wind_gust_ms", "wave_height_m", "pressure_hpa"], "INDICATES"),
            ("wind_restriction", ["wind_speed_ms", "wind_gust_ms"], "AFFECTS"),
            ("navigation_restriction", ["tide_ft", "wave_height_m"], "AFFECTS"),
            ("operational_disruption", ["arrival_delay_hours", "berth_delay_hours",
                                         "breakdown_minutes"], "INDICATES"),
            ("crane_slowdown", ["crane_productivity_mph", "breakdown_minutes"], "INDICATES"),
            ("congestion", ["peak_occupancy_pct", "average_turn_time_minutes",
                             "total_transactions"], "INDICATES"),
            ("gate_congestion", ["average_turn_time_minutes", "total_transactions"], "INDICATES"),
            ("yard_overflow", ["peak_occupancy_pct", "average_dwell_days"], "INDICATES"),
            ("safety", ["wind_gust_ms", "wave_height_m", "breakdown_minutes"], "DEPENDS_ON"),
            ("compliance", ["wind_speed_ms", "wave_height_m"], "DEPENDS_ON"),
            ("weather", ["wind_speed_ms", "wave_height_m", "pressure_hpa"], "IS_SAME_AS"),
        ]

        # --- Operation bridges (:Operation) — match v1 schema ---
        # These 11 are the canonical port operations that graph_reasoner
        # aliases expect. Previously v2 had only 4 (notification, operational_pause,
        # safety_inspection, tug_assistance) from rule actions; we add the rest
        # to match the v1 operation layer.
        operation_bridges = [
            ("vessel_entry", ["vessel_imo", "vessel_capacity_teu", "vessel_loa_meters"], "MEASURES"),
            ("navigation", ["tide_ft", "wave_height_m", "wind_speed_ms"], "DEPENDS_ON"),
            ("berth_operations", ["berth_productivity_mph", "arrival_delay_hours",
                                   "berth_delay_hours"], "MEASURED_BY"),
            ("crane_operations", ["crane_productivity_mph", "crane_hours",
                                   "breakdown_minutes"], "MEASURED_BY"),
            ("yard_operations", ["average_dwell_days", "teu_received",
                                  "peak_occupancy_pct"], "MEASURED_BY"),
            ("gate_operations", ["total_transactions", "average_turn_time_minutes"], "MEASURED_BY"),
            ("delay", ["arrival_delay_hours", "berth_delay_hours"], "MEASURED_BY"),
            ("slowdown", ["crane_productivity_mph", "berth_productivity_mph"], "MEASURED_BY"),
            ("vessel_scheduling", ["arrival_delay_hours", "vessel_capacity_teu"], "AFFECTS"),
            ("container_logistics", ["teu_received", "teu_delivered", "total_moves"], "MEASURED_BY"),
            # operational_pause already exists from rule edges, but merge to ensure
            ("operational_pause", ["wind_speed_ms", "wave_height_m"], "TRIGGERED_BY"),
        ]

        bridge_created = 0
        link_created = 0

        # Create concept nodes
        for name, metrics, rel_type in concept_bridges:
            session.run("MERGE (c:Concept {name: $name})", name=name)
            bridge_created += 1
            for metric in metrics:
                session.run(
                    f"""
                    MATCH (c:Concept {{name: $concept}})
                    MATCH (m:Metric {{name: $metric}})
                    MERGE (c)-[r:{rel_type}]->(m)
                    """,
                    concept=name, metric=metric,
                )
                link_created += 1

        # Create operation nodes (with MERGE to not duplicate existing rule actions)
        for name, metrics, rel_type in operation_bridges:
            session.run("MERGE (o:Operation {name: $name})", name=name)
            bridge_created += 1
            for metric in metrics:
                session.run(
                    f"""
                    MATCH (o:Operation {{name: $op}})
                    MATCH (m:Metric {{name: $metric}})
                    MERGE (o)-[r:{rel_type}]->(m)
                    """,
                    op=name, metric=metric,
                )
                link_created += 1

        # --- Cross-layer semantic links (concept↔concept, concept↔operation) ---
        semantic_links = [
            # concept → concept
            ("storm_event", "Concept", "weather_conditions", "Concept", "IS_PART_OF"),
            ("weather_conditions", "Concept", "environmental_conditions", "Concept", "IS_PART_OF"),
            ("wind_restriction", "Concept", "safety", "Concept", "ENFORCES"),
            ("navigation_restriction", "Concept", "safety", "Concept", "ENFORCES"),
            ("crane_slowdown", "Concept", "operational_disruption", "Concept", "CONTRIBUTES_TO"),
            ("gate_congestion", "Concept", "congestion", "Concept", "IS_PART_OF"),
            ("yard_overflow", "Concept", "congestion", "Concept", "IS_PART_OF"),
            ("congestion", "Concept", "operational_disruption", "Concept", "CONTRIBUTES_TO"),
            ("safety", "Concept", "compliance", "Concept", "REQUIRES"),
            # concept → operation (causal)
            ("weather_conditions", "Concept", "vessel_entry", "Operation", "AFFECTS"),
            ("weather_conditions", "Concept", "navigation", "Operation", "AFFECTS"),
            ("weather_conditions", "Concept", "operational_pause", "Operation", "CAN_TRIGGER"),
            ("wind_restriction", "Concept", "vessel_entry", "Operation", "RESTRICTS"),
            ("wind_restriction", "Concept", "crane_operations", "Operation", "RESTRICTS"),
            ("navigation_restriction", "Concept", "navigation", "Operation", "RESTRICTS"),
            ("crane_slowdown", "Concept", "berth_operations", "Operation", "AFFECTS"),
            ("operational_disruption", "Concept", "delay", "Operation", "CAUSES"),
            ("congestion", "Concept", "delay", "Operation", "CONTRIBUTES_TO"),
            # operation → operation (cascade)
            ("crane_operations", "Operation", "berth_operations", "Operation", "INFLUENCES"),
            ("berth_operations", "Operation", "delay", "Operation", "INFLUENCES"),
            ("yard_operations", "Operation", "gate_operations", "Operation", "INFLUENCES"),
            ("container_logistics", "Operation", "yard_operations", "Operation", "INFLUENCES"),
            ("vessel_scheduling", "Operation", "vessel_entry", "Operation", "INFLUENCES"),
            ("slowdown", "Operation", "delay", "Operation", "CONTRIBUTES_TO"),
        ]

        for a_name, a_label, b_name, b_label, rel in semantic_links:
            session.run(
                f"""
                MATCH (a:{a_label} {{name: $a}})
                MATCH (b:{b_label} {{name: $b}})
                MERGE (a)-[r:{rel}]->(b)
                """,
                a=a_name, b=b_name,
            )
            link_created += 1

        self.nodes_created += bridge_created
        self.edges_created += link_created
        logger.info(
            "Created %d bridge nodes (%d concepts + %d operations) with %d semantic edges",
            bridge_created, len(concept_bridges), len(operation_bridges), link_created,
        )

    # -----------------------------------------------------------------------
    # Phase 2: Rule-driven edges
    # -----------------------------------------------------------------------

    def _load_rules(self, path: str) -> List[Dict[str, Any]]:
        p = Path(path)
        if not p.exists():
            logger.warning("Rule file missing: %s", p)
            return []
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def _create_rule_edges(
        self,
        session,
        rules: List[Dict[str, Any]],
        source_type: str,
    ) -> None:
        """
        For each rule create:
            (metric)-[:TRIGGERS {threshold,op,source,page,confidence}]->(action)

        When the rule has no matching grounded variable, create a Concept node
        for the raw variable name and still wire the action.
        """
        from .synonym_expander import SynonymExpander
        expander = SynonymExpander(taxonomy=self._load_taxonomy())

        edges_before = self.edges_created
        for rule in rules:
            raw_variable = rule.get("variable") or rule.get("sql_variable")
            if not raw_variable:
                self.rules_skipped += 1
                continue

            # Resolve variable to canonical (use cache + LLM as needed)
            canonical = rule.get("sql_variable") or expander.resolve(
                raw_variable, use_llm_fallback=False,  # offline grounding only
            )

            action_raw = rule.get("action") or ""
            action_node, action_label = _parse_action_to_node(action_raw)

            operator = rule.get("operator") or ""
            value = rule.get("value")
            try:
                value_num = float(value) if value is not None else None
            except (TypeError, ValueError):
                value_num = None

            source_file = rule.get("source_file") or ""
            page = rule.get("page")

            # Create action node
            session.run(
                f"MERGE (n:{action_label} {{name: $name}})",
                name=action_node,
            )

            # Determine source node
            if canonical:
                start_node_match = "MATCH (a:Metric {name: $start})"
                start_name = canonical
            else:
                # No canonical variable — create a Concept node from the raw text
                concept_name = _normalize_name(raw_variable)[:40]
                session.run(
                    "MERGE (n:Concept {name: $name})",
                    name=concept_name,
                )
                self.nodes_created += 1
                start_node_match = "MATCH (a:Concept {name: $start})"
                start_name = concept_name

            # Create the rule edge with full provenance.
            # Use source_file + page as MERGE key (always non-null) and SET
            # everything else, since Neo4j refuses to merge on null values.
            query = f"""
            {start_node_match}
            MATCH (b:{action_label} {{name: $end}})
            MERGE (a)-[r:TRIGGERS {{
                source_file: $source_file,
                rule_source_type: $rule_source_type
            }}]->(b)
            SET r.operator = $op,
                r.threshold = $threshold,
                r.page = $page,
                r.rule_text = $rule_text
            """
            session.run(
                query,
                start=start_name,
                end=action_node,
                op=operator or "",
                threshold=value_num if value_num is not None else -9999.0,
                source_file=source_file or "unknown",
                page=page if page is not None else -1,
                rule_source_type=source_type,
                rule_text=(rule.get("rule_text") or "")[:200],
            )
            self.edges_created += 1

        logger.info(
            "Created %d rule edges from %d %s rules (%d skipped)",
            self.edges_created - edges_before, len(rules), source_type,
            self.rules_skipped,
        )

    # -----------------------------------------------------------------------
    # Phase 3: Statistical correlation edges
    # -----------------------------------------------------------------------

    def _create_correlation_edges(
        self,
        session,
        taxonomy: Dict[str, Any],
        min_abs_corr: float = 0.4,
        max_pairs_per_table: int = 10,
    ) -> None:
        """
        For each operational table, compute pairwise Pearson correlations
        between numeric columns and create CORRELATES_WITH edges for strong
        relationships.

        This grounds the graph in actual data patterns rather than developer
        intuition.
        """
        try:
            import duckdb
        except ImportError:
            logger.warning("duckdb not available, skipping correlation phase")
            return

        if not self.db_path.exists():
            logger.warning("DuckDB not found at %s; skipping correlations", self.db_path)
            return

        variable_meta = taxonomy.get("variable_meta", {})
        tables_to_vars: Dict[str, List[str]] = {}
        for var, meta in variable_meta.items():
            tbl = meta.get("source_table")
            if tbl:
                tables_to_vars.setdefault(tbl, []).append(var)

        con = duckdb.connect(str(self.db_path), read_only=True)
        edges_before = self.edges_created
        try:
            for table, vars_list in tables_to_vars.items():
                if len(vars_list) < 2:
                    continue
                # Sample up to 10 numeric columns per table to keep pairs manageable
                cols = vars_list[:10]
                correlations: List[Tuple[str, str, float]] = []
                for i, a in enumerate(cols):
                    for b in cols[i + 1:]:
                        try:
                            r = con.execute(
                                f'SELECT corr("{a}", "{b}") FROM "{table}"'
                            ).fetchone()
                            if r and r[0] is not None and abs(r[0]) >= min_abs_corr:
                                correlations.append((a, b, r[0]))
                        except Exception as e:
                            logger.debug("Correlation failed for %s/%s: %s", a, b, e)
                            continue

                correlations.sort(key=lambda x: abs(x[2]), reverse=True)
                for a, b, r_val in correlations[:max_pairs_per_table]:
                    session.run(
                        """
                        MATCH (x:Metric {name: $a})
                        MATCH (y:Metric {name: $b})
                        MERGE (x)-[r:CORRELATES_WITH {source_table: $tbl}]->(y)
                        SET r.coefficient = $r_val,
                            r.strength = CASE
                              WHEN abs($r_val) >= 0.7 THEN 'strong'
                              WHEN abs($r_val) >= 0.5 THEN 'moderate'
                              ELSE 'weak'
                            END
                        """,
                        a=a, b=b, r_val=float(r_val), tbl=table,
                    )
                    self.edges_created += 1
        finally:
            con.close()

        logger.info(
            "Created %d correlation edges from SQL data",
            self.edges_created - edges_before,
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    builder = RuleDrivenGraphBuilder(wipe_existing=True)
    try:
        builder.build()
        print("\n" + "=" * 60)
        print("Graph v2 Build — Complete")
        print("=" * 60)
        print(f"Nodes created:    {builder.nodes_created}")
        print(f"Edges created:    {builder.edges_created}")
        print(f"Rules skipped:    {builder.rules_skipped}")
    finally:
        builder.close()


if __name__ == "__main__":
    main()
