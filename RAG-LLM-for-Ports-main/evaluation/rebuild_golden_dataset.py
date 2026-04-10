"""
Rebuild golden dataset with per-source golden metadata.

For each query, annotates:
- vector: relevant chunk_ids, source_files, pages (via embedding + keyword matching)
- sql: target tables, columns, expected aggregation type
- rules: relevant rule variables, operators, values
- graph: relevant entity pairs, relationship types

Also removes reasoning_goal field (unused) and keeps answer_mode + needs_*.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ── Load data sources ──────────────────────────────────────────────────────

def load_chunks():
    path = PROJECT_ROOT / "data" / "chunks" / "chunks_v1.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_rules():
    rules = []
    for name in ["grounded_rules.json", "policy_rules.json"]:
        path = PROJECT_ROOT / "data" / "rules" / name
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for r in data:
                r["_rule_source"] = name
            rules.extend(data)
    return rules


def load_sql_schema():
    import duckdb
    db_path = PROJECT_ROOT / "storage" / "sql" / "port_ops.duckdb"
    conn = duckdb.connect(str(db_path), read_only=True)
    schema = {}
    tables = [t[0] for t in conn.execute("SHOW TABLES").fetchall()]
    for t in tables:
        if t.startswith("v_"):  # skip views
            continue
        cols = [r[0] for r in conn.execute(f"DESCRIBE {t}").fetchall()]
        schema[t] = cols
    conn.close()
    return schema


# ── Graph node/relationship definitions ────────────────────────────────────

GRAPH_NODES = {
    "metrics": [
        "wind_speed_ms", "wind_gust_ms", "wave_height_m", "tide_level_m",
        "atmospheric_pressure_hpa", "berth_productivity_mph", "arrival_delay_hours",
        "crane_productivity_mph", "breakdown_minutes", "average_dwell_days",
        "average_turn_time_minutes", "teu_received", "containers_actual",
        "vessel_capacity", "total_transactions", "yard_occupancy_pct",
    ],
    "operations": [
        "vessel_entry", "navigation", "berth_operations", "crane_operations",
        "yard_operations", "gate_operations", "operational_pause", "delay",
        "slowdown", "vessel_scheduling", "container_logistics",
    ],
    "concepts": [
        "weather_conditions", "environmental_conditions", "wind_restriction",
        "navigation_restriction", "operational_threshold", "crane_slowdown",
        "operational_disruption", "congestion", "safety", "compliance",
        "gate_congestion", "yard_overflow", "storm_event",
    ],
}

ALL_GRAPH_NODES = (
    GRAPH_NODES["metrics"] + GRAPH_NODES["operations"] + GRAPH_NODES["concepts"]
)

GRAPH_RELATIONSHIPS = [
    "AFFECTS", "CAN_TRIGGER", "MEASURES", "CONTRIBUTES_TO",
    "CAN_LEAD_TO", "INFLUENCES", "INTENSIFIES", "DISRUPTS",
    "IS_PART_OF", "ENFORCES", "REQUIRES", "INDICATES",
]


# ── Annotators ─────────────────────────────────────────────────────────────

def annotate_vector(query: str, keywords: List[str], chunks: list,
                    chunk_embeddings: np.ndarray, model) -> Dict[str, Any]:
    """Find relevant chunks via embedding similarity + keyword matching."""
    query_emb = model.encode([query], normalize_embeddings=True)[0]
    emb_scores = chunk_embeddings @ query_emb

    kw_scores = np.zeros(len(chunks))
    for i, c in enumerate(chunks):
        text_lower = c["text"].lower()
        hits = sum(1 for kw in keywords if kw.lower() in text_lower)
        kw_scores[i] = hits / max(len(keywords), 1)

    combined = 0.7 * emb_scores + 0.3 * kw_scores
    top_indices = np.argsort(combined)[::-1][:30]

    relevant = []
    for idx in top_indices:
        score = float(combined[idx])
        if score < 0.30:
            break
        relevant.append({
            "chunk_id": chunks[idx]["chunk_id"],
            "source_file": chunks[idx]["source_file"],
            "page": chunks[idx].get("page"),
            "score": round(score, 4),
        })

    # Summarize source files
    src_counter = Counter(r["source_file"] for r in relevant)
    top_sources = [sf for sf, _ in src_counter.most_common(5)]

    return {
        "relevant_chunk_ids": [r["chunk_id"] for r in relevant],
        "relevant_source_files": top_sources,
        "relevant_pages": list(set(r["page"] for r in relevant if r["page"] is not None))[:10],
        "top_chunk_count": len(relevant),
    }


# ── SQL keyword→table/column mapping ──────────────────────────────────────

SQL_KEYWORD_MAP = {
    "wind": {"table": "environment", "cols": ["wind_speed_ms", "wind_gust_ms"]},
    "wave": {"table": "environment", "cols": ["wave_height_m"]},
    "wave height": {"table": "environment", "cols": ["wave_height_m"]},
    "tide": {"table": "environment", "cols": ["tide_ft"]},
    "temperature": {"table": "environment", "cols": ["air_temp_c", "water_temp_c"]},
    "pressure": {"table": "environment", "cols": ["pressure_hpa"]},
    "storm": {"table": "environment", "cols": ["event_storm"]},
    "weather": {"table": "environment", "cols": ["wind_speed_ms", "wave_height_m", "tide_ft"]},
    "berth productivity": {"table": "berth_operations", "cols": ["berth_productivity_mph"]},
    "berth delay": {"table": "berth_operations", "cols": ["arrival_delay_hours", "berth_delay_hours"]},
    "arrival delay": {"table": "berth_operations", "cols": ["arrival_delay_hours"]},
    "containers": {"table": "berth_operations", "cols": ["containers_actual", "containers_planned"]},
    "crane productivity": {"table": "crane_operations", "cols": ["crane_productivity_mph"]},
    "crane": {"table": "crane_operations", "cols": ["crane_productivity_mph", "breakdown_minutes"]},
    "breakdown": {"table": "crane_operations", "cols": ["breakdown_minutes"]},
    "yard": {"table": "yard_operations", "cols": ["teu_received", "average_dwell_days", "peak_occupancy_pct"]},
    "dwell": {"table": "yard_operations", "cols": ["average_dwell_days"]},
    "occupancy": {"table": "yard_operations", "cols": ["peak_occupancy_pct"]},
    "gate": {"table": "gate_operations", "cols": ["total_transactions", "average_turn_time_minutes"]},
    "turn time": {"table": "gate_operations", "cols": ["average_turn_time_minutes"]},
    "transaction": {"table": "gate_operations", "cols": ["total_transactions"]},
    "vessel": {"table": "vessel_calls", "cols": ["vessel_name", "vessel_capacity_teu", "vessel_loa_meters"]},
    "loa": {"table": "vessel_calls", "cols": ["vessel_loa_meters"]},
    "teu": {"table": "vessel_calls", "cols": ["vessel_capacity_teu"]},
    "berth": {"table": "berth_operations", "cols": ["berth_productivity_mph", "arrival_delay_hours"]},
}


def annotate_sql(query: str, keywords: List[str], sql_schema: dict) -> Dict[str, Any]:
    """Map query keywords to SQL tables and columns."""
    q = query.lower()
    matched_tables = {}
    for keyword, info in SQL_KEYWORD_MAP.items():
        if keyword in q:
            t = info["table"]
            if t not in matched_tables:
                matched_tables[t] = []
            matched_tables[t].extend(info["cols"])

    # Dedupe columns per table
    for t in matched_tables:
        matched_tables[t] = list(dict.fromkeys(matched_tables[t]))

    # Infer aggregation type
    agg = None
    if any(w in q for w in ["average", "avg", "mean"]):
        agg = "AVG"
    elif any(w in q for w in ["total", "sum"]):
        agg = "SUM"
    elif any(w in q for w in ["count", "how many", "number of"]):
        agg = "COUNT"
    elif any(w in q for w in ["maximum", "max", "highest", "top"]):
        agg = "MAX"
    elif any(w in q for w in ["minimum", "min", "lowest"]):
        agg = "MIN"
    elif any(w in q for w in ["compare", "versus", "vs"]):
        agg = "COMPARE"

    return {
        "expected_tables": matched_tables,
        "expected_aggregation": agg,
    }


# ── Rule matching ─────────────────────────────────────────────────────────

def annotate_rules(query: str, keywords: List[str], all_rules: list) -> Dict[str, Any]:
    """Find relevant rules by keyword matching."""
    q = query.lower()
    scored = []
    for rule in all_rules:
        search_text = " ".join([
            str(rule.get("rule_text", "")),
            str(rule.get("variable", "")),
            str(rule.get("condition", "")),
            str(rule.get("action", "")),
        ]).lower()

        score = 0
        for kw in keywords:
            if kw.lower() in search_text:
                score += 1
        # Also match query words
        for word in q.split():
            if len(word) > 3 and word in search_text:
                score += 0.3

        if score >= 1.0:
            scored.append((score, rule))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_rules = scored[:5]

    relevant = []
    for score, rule in top_rules:
        relevant.append({
            "variable": rule.get("variable"),
            "operator": rule.get("operator"),
            "value": rule.get("value"),
            "threshold": rule.get("threshold"),
            "rule_text": (rule.get("rule_text") or "")[:120],
            "source_file": rule.get("source_file"),
            "page": rule.get("page"),
            "score": round(score, 2),
        })

    return {
        "expected_rule_variables": list(dict.fromkeys(
            r["variable"] for r in relevant if r["variable"]
        )),
        "expected_rules": relevant,
    }


# ── Graph entity matching ─────────────────────────────────────────────────

GRAPH_KEYWORD_MAP = {
    "berth delay": ["arrival_delay_hours", "berth_operations"],
    "berth productivity": ["berth_productivity_mph", "berth_operations"],
    "crane slowdown": ["crane_slowdown", "crane_operations"],
    "crane productivity": ["crane_productivity_mph", "crane_operations"],
    "breakdown": ["breakdown_minutes", "crane_operations"],
    "weather": ["weather_conditions", "wind_speed_ms", "wave_height_m"],
    "wind": ["wind_speed_ms", "weather_conditions", "wind_restriction"],
    "wave": ["wave_height_m", "weather_conditions"],
    "storm": ["storm_event", "weather_conditions"],
    "congestion": ["congestion", "gate_congestion", "yard_overflow"],
    "yard": ["yard_operations", "yard_occupancy_pct"],
    "gate": ["gate_operations", "gate_congestion"],
    "vessel entry": ["vessel_entry", "navigation"],
    "safety": ["safety", "compliance"],
    "disruption": ["operational_disruption", "operational_pause"],
    "delay": ["arrival_delay_hours", "delay", "berth_operations"],
    "navigation": ["navigation", "vessel_entry", "navigation_restriction"],
    "dwell": ["average_dwell_days", "yard_operations"],
}


def annotate_graph(query: str, keywords: List[str]) -> Dict[str, Any]:
    """Map query to expected graph entities and relationship types."""
    q = query.lower()
    matched_entities = []

    for keyword, entities in GRAPH_KEYWORD_MAP.items():
        if keyword in q:
            matched_entities.extend(entities)

    unique = list(dict.fromkeys(matched_entities))[:10]

    # Infer expected relationships
    expected_rels = []
    if any(w in q for w in ["cause", "why", "factor", "led to", "contribute"]):
        expected_rels.extend(["CONTRIBUTES_TO", "CAN_LEAD_TO", "AFFECTS"])
    if any(w in q for w in ["impact", "affect", "influence"]):
        expected_rels.extend(["AFFECTS", "INFLUENCES"])
    if any(w in q for w in ["trigger", "disrupt"]):
        expected_rels.extend(["CAN_TRIGGER", "DISRUPTS"])
    if not expected_rels:
        expected_rels = ["AFFECTS", "CONTRIBUTES_TO"]

    return {
        "expected_entities": unique,
        "expected_relationships": list(dict.fromkeys(expected_rels)),
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("REBUILDING GOLDEN DATASET WITH PER-SOURCE METADATA")
    print("=" * 80)

    # Load golden dataset
    golden_path = PROJECT_ROOT / "evaluation" / "golden_dataset.json"
    with open(golden_path, "r", encoding="utf-8") as f:
        golden = json.load(f)
    print(f"Loaded {len(golden)} golden queries")

    # Load data sources
    chunks = load_chunks()
    rules = load_rules()
    sql_schema = load_sql_schema()
    print(f"Chunks: {len(chunks)}, Rules: {len(rules)}, SQL tables: {list(sql_schema.keys())}")

    # Check which queries need vector annotation
    vector_queries = [g for g in golden if "vector" in g.get("expected_sources", [])]
    print(f"Queries needing vector annotation: {len(vector_queries)}")

    # Load embedding model and encode chunks (only if vector queries exist)
    chunk_embeddings = None
    model = None
    if vector_queries:
        print("Loading embedding model ...")
        model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")
        print("Encoding chunks ...")
        chunk_texts = [c["text"] for c in chunks]
        chunk_embeddings = model.encode(
            chunk_texts, batch_size=512,
            normalize_embeddings=True, show_progress_bar=True,
        )
        print(f"  Shape: {chunk_embeddings.shape}")

    # Annotate each query
    for i, item in enumerate(golden):
        query = item["query"]
        keywords = item.get("expected_evidence_keywords", [])
        sources = item.get("expected_sources", [])

        # Remove reasoning_goal if present (Task 2a: unused field)
        item.pop("reasoning_goal", None)

        # Ensure needs_* flags exist (derived from expected_sources)
        item["needs_vector"] = "vector" in sources
        item["needs_sql"] = "sql" in sources
        item["needs_rules"] = "rules" in sources
        item["needs_graph"] = "graph" in sources

        # Per-source golden metadata
        if "vector" in sources and chunk_embeddings is not None:
            item["golden_vector"] = annotate_vector(
                query, keywords, chunks, chunk_embeddings, model
            )
        else:
            item["golden_vector"] = None

        if "sql" in sources:
            item["golden_sql"] = annotate_sql(query, keywords, sql_schema)
        else:
            item["golden_sql"] = None

        if "rules" in sources:
            item["golden_rules"] = annotate_rules(query, keywords, rules)
        else:
            item["golden_rules"] = None

        if "graph" in sources:
            item["golden_graph"] = annotate_graph(query, keywords)
        else:
            item["golden_graph"] = None

        # Progress
        src_tags = []
        if item["golden_vector"]:
            src_tags.append(f"vec:{item['golden_vector']['top_chunk_count']}chunks")
        if item["golden_sql"]:
            tables = list(item["golden_sql"]["expected_tables"].keys())
            src_tags.append(f"sql:{','.join(tables)}")
        if item["golden_rules"]:
            nvars = len(item["golden_rules"]["expected_rule_variables"])
            src_tags.append(f"rules:{nvars}vars")
        if item["golden_graph"]:
            nents = len(item["golden_graph"]["expected_entities"])
            src_tags.append(f"graph:{nents}ents")

        print(f"  [{i+1:2d}/{len(golden)}] {item['id']:12s} {' | '.join(src_tags)}")

    # Save
    out_path = PROJECT_ROOT / "evaluation" / "golden_dataset.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(golden, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")

    # Summary stats
    print("\n--- Annotation Summary ---")
    for src in ["vector", "sql", "rules", "graph"]:
        key = f"golden_{src}"
        annotated = sum(1 for g in golden if g.get(key) is not None)
        print(f"  {src}: {annotated} queries annotated")


if __name__ == "__main__":
    main()
