"""
Expand golden dataset v3: fill all 16 source combos (including empty)
and add guardrail test queries.

Missing combos:
- () empty — guardrail: no sources at all
- (graph,) — pure graph only
- (graph, rules) — graph + rules
- (graph, vector) — graph + vector
- (graph, rules, vector) — graph + rules + vector

Guardrail test scenarios:
- G1: decision_support WITHOUT rules → hard_recommendation_allowed=False
- G2: decision_support WITH rules → hard_recommendation_allowed=True
- G3: lookup with NO evidence at all → knowledge_fallback triggered
- G4: diagnostic WITHOUT graph → knowledge_fallback triggered
- G5: descriptive WITHOUT docs → knowledge_fallback triggered
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def find_chunks(chunks_by_file, source_file, keywords, max_results=10):
    if source_file not in chunks_by_file:
        return []
    results = []
    for c in chunks_by_file[source_file]:
        text_lower = c["text"].lower()
        hits = sum(1 for kw in keywords if kw.lower() in text_lower)
        if hits >= 1:
            results.append((hits, c))
    results.sort(key=lambda x: -x[0])
    return [c for _, c in results[:max_results]]


def vec_meta(chunks, files):
    return {
        "relevant_chunk_ids": [c["chunk_id"] for c in chunks],
        "relevant_source_files": files,
        "relevant_pages": list(set(c["page"] for c in chunks)),
    }


def sql_meta(tables, agg=None):
    return {"expected_tables": tables, "expected_aggregation": agg}


def rule_meta(variables):
    return {"expected_rule_variables": variables, "expected_rules": []}


def graph_meta(entities, rels):
    return {"expected_entities": entities, "expected_relationships": rels}


def main():
    golden_path = PROJECT_ROOT / "evaluation" / "golden_dataset.json"
    with open(golden_path, "r", encoding="utf-8") as f:
        golden = json.load(f)
    print(f"Existing: {len(golden)} queries")

    with open(PROJECT_ROOT / "data" / "chunks" / "chunks_v1.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    by_src = defaultdict(list)
    for c in chunks:
        by_src[c["source_file"]].append(c)

    new_queries = []

    # ================================================================
    # MISSING COMBO 1: (graph,) — pure graph only
    # ================================================================

    new_queries.append({
        "id": "GRAPH_ONLY_001", "difficulty": "hard",
        "query": "What are the causal relationships between weather conditions, crane slowdown, and berth delays in the knowledge graph?",
        "answer_mode": "diagnostic",
        "expected_sources": ["graph"],
        "needs_vector": False, "needs_sql": False, "needs_rules": False, "needs_graph": True,
        "expected_evidence_keywords": ["weather", "crane", "slowdown", "berth", "delay", "causal"],
        "reference_answer": "The knowledge graph shows: weather_conditions → crane_slowdown → berth_operations, with AFFECTS and CONTRIBUTES_TO relationships.",
        "golden_vector": None, "golden_sql": None, "golden_rules": None,
        "golden_graph": graph_meta(
            ["weather_conditions", "crane_slowdown", "berth_operations", "arrival_delay_hours"],
            ["AFFECTS", "CONTRIBUTES_TO", "CAN_LEAD_TO"],
        ),
    })

    new_queries.append({
        "id": "GRAPH_ONLY_002", "difficulty": "medium",
        "query": "What entities in the port knowledge graph are connected to operational disruption?",
        "answer_mode": "descriptive",
        "expected_sources": ["graph"],
        "needs_vector": False, "needs_sql": False, "needs_rules": False, "needs_graph": True,
        "expected_evidence_keywords": ["operational_disruption", "connected", "graph", "entities"],
        "reference_answer": "Operational_disruption is connected to storm_event, crane_slowdown, congestion, and operational_pause via CAN_LEAD_TO and DISRUPTS relationships.",
        "golden_vector": None, "golden_sql": None, "golden_rules": None,
        "golden_graph": graph_meta(
            ["operational_disruption", "storm_event", "crane_slowdown", "congestion", "operational_pause"],
            ["CAN_LEAD_TO", "DISRUPTS", "CAN_TRIGGER"],
        ),
    })

    # ================================================================
    # MISSING COMBO 2: (graph, rules) — graph + rules
    # ================================================================

    new_queries.append({
        "id": "MIX_GR_001", "difficulty": "hard",
        "query": "What causal chain links wind conditions to crane shutdown, and what are the specific wind speed thresholds in port rules?",
        "answer_mode": "diagnostic",
        "expected_sources": ["graph", "rules"],
        "needs_vector": False, "needs_sql": False, "needs_rules": True, "needs_graph": True,
        "expected_evidence_keywords": ["wind", "crane", "shutdown", "threshold", "knots", "causal"],
        "reference_answer": "Graph: wind_speed_ms → weather_conditions → crane_slowdown. Rules: wind speed limit 30-35 knots for entry, gust > 25 m/s triggers boom-down.",
        "golden_vector": None, "golden_sql": None,
        "golden_rules": rule_meta(["Wind Speed", "wind_gust_ms", "crane_wind_limit"]),
        "golden_graph": graph_meta(
            ["wind_speed_ms", "weather_conditions", "crane_slowdown", "wind_restriction"],
            ["AFFECTS", "CONTRIBUTES_TO", "CAN_TRIGGER"],
        ),
    })

    new_queries.append({
        "id": "MIX_GR_002", "difficulty": "hard",
        "query": "What is the causal path from yard overflow to gate congestion, and what occupancy threshold triggers overflow protocol?",
        "answer_mode": "decision_support",
        "expected_sources": ["graph", "rules"],
        "needs_vector": False, "needs_sql": False, "needs_rules": True, "needs_graph": True,
        "expected_evidence_keywords": ["yard", "overflow", "gate", "congestion", "85", "threshold"],
        "reference_answer": "Graph: yard_overflow → congestion → gate_congestion. Rules: peak_occupancy_pct > 85% triggers overflow protocol.",
        "golden_vector": None, "golden_sql": None,
        "golden_rules": rule_meta(["peak_occupancy_pct"]),
        "golden_graph": graph_meta(
            ["yard_overflow", "congestion", "gate_congestion", "yard_operations"],
            ["CAN_LEAD_TO", "CONTRIBUTES_TO"],
        ),
    })

    # ================================================================
    # MISSING COMBO 3: (graph, vector) — graph + vector
    # ================================================================

    matched = find_chunks(by_src, "IAPH-Risk-and-Resilience-Guidelines-for-Ports-BD.pdf",
                          ["disruption", "cascade", "resilience", "weather"])
    new_queries.append({
        "id": "MIX_GV_001", "difficulty": "hard",
        "query": "What do IAPH risk guidelines say about cascading disruptions, and what does the knowledge graph show about disruption pathways?",
        "answer_mode": "diagnostic",
        "expected_sources": ["graph", "vector"],
        "needs_vector": True, "needs_sql": False, "needs_rules": False, "needs_graph": True,
        "expected_evidence_keywords": ["disruption", "cascade", "resilience", "pathway", "IAPH"],
        "reference_answer": "IAPH guidelines describe cascading risks; the knowledge graph shows disruption pathways: storm → operational_disruption → delay.",
        "golden_vector": vec_meta(matched, ["IAPH-Risk-and-Resilience-Guidelines-for-Ports-BD.pdf"]),
        "golden_sql": None, "golden_rules": None,
        "golden_graph": graph_meta(
            ["storm_event", "operational_disruption", "delay", "weather_conditions"],
            ["CAN_TRIGGER", "CAN_LEAD_TO"],
        ),
    })

    matched = find_chunks(by_src, "Maritime-Singapore-Decarbonisation-blueprint.pdf",
                          ["emission", "fuel", "carbon", "impact"])
    new_queries.append({
        "id": "MIX_GV_002", "difficulty": "hard",
        "query": "What does Singapore's decarbonisation blueprint say about environmental impacts, and how do environmental conditions relate to port operations in the knowledge graph?",
        "answer_mode": "descriptive",
        "expected_sources": ["graph", "vector"],
        "needs_vector": True, "needs_sql": False, "needs_rules": False, "needs_graph": True,
        "expected_evidence_keywords": ["decarbonisation", "environmental", "Singapore", "conditions", "operations"],
        "reference_answer": "The blueprint describes emission reduction pathways; the knowledge graph shows environmental_conditions influencing berth_operations and crane_operations.",
        "golden_vector": vec_meta(matched, ["Maritime-Singapore-Decarbonisation-blueprint.pdf"]),
        "golden_sql": None, "golden_rules": None,
        "golden_graph": graph_meta(
            ["environmental_conditions", "weather_conditions", "berth_operations", "crane_operations"],
            ["AFFECTS", "INFLUENCES"],
        ),
    })

    # ================================================================
    # MISSING COMBO 4: (graph, rules, vector) — graph + rules + vector
    # ================================================================

    matched = find_chunks(by_src, "Port Information Manual 1.4.5.pdf",
                          ["weather", "vessel", "restriction", "navigation", "safety"])
    new_queries.append({
        "id": "MIX_GRV_001", "difficulty": "hard",
        "query": "What does the Port Information Manual say about weather-related navigation restrictions, what rules define the thresholds, and how does the causal graph link weather to vessel entry?",
        "answer_mode": "decision_support",
        "expected_sources": ["graph", "rules", "vector"],
        "needs_vector": True, "needs_sql": False, "needs_rules": True, "needs_graph": True,
        "expected_evidence_keywords": ["weather", "navigation", "restriction", "vessel", "entry", "wind", "threshold"],
        "reference_answer": "Port manual describes weather procedures; rules define wind speed thresholds; graph shows weather_conditions → wind_restriction → vessel_entry.",
        "golden_vector": vec_meta(matched, ["Port Information Manual 1.4.5.pdf"]),
        "golden_sql": None,
        "golden_rules": rule_meta(["Wind Speed", "wind_gust_ms"]),
        "golden_graph": graph_meta(
            ["weather_conditions", "wind_restriction", "navigation_restriction", "vessel_entry"],
            ["CAN_TRIGGER", "AFFECTS"],
        ),
    })

    matched = find_chunks(by_src, "IAPH-Risk-and-Resilience-Guidelines-for-Ports-BD.pdf",
                          ["storm", "emergency", "risk", "safety"])
    new_queries.append({
        "id": "MIX_GRV_002", "difficulty": "hard",
        "query": "Based on IAPH risk guidelines, emergency response rules, and the causal graph, what should happen when a storm event is detected?",
        "answer_mode": "decision_support",
        "expected_sources": ["graph", "rules", "vector"],
        "needs_vector": True, "needs_sql": False, "needs_rules": True, "needs_graph": True,
        "expected_evidence_keywords": ["storm", "emergency", "response", "risk", "disruption", "protocol"],
        "reference_answer": "IAPH guidelines describe storm risk scenarios; rules define pressure < 980 hPa threshold; graph shows storm_event → operational_disruption cascade.",
        "golden_vector": vec_meta(matched, ["IAPH-Risk-and-Resilience-Guidelines-for-Ports-BD.pdf"]),
        "golden_sql": None,
        "golden_rules": rule_meta(["pressure_hpa", "operational_threshold"]),
        "golden_graph": graph_meta(
            ["storm_event", "weather_conditions", "operational_disruption", "operational_pause", "safety"],
            ["CAN_TRIGGER", "CAN_LEAD_TO", "ENFORCES"],
        ),
    })

    # ================================================================
    # MISSING COMBO 5: () empty — NO sources, guardrail test
    # ================================================================

    new_queries.append({
        "id": "GUARD_EMPTY_001", "difficulty": "hard",
        "query": "What is the capital of France?",
        "answer_mode": "lookup",
        "expected_sources": [],
        "needs_vector": False, "needs_sql": False, "needs_rules": False, "needs_graph": False,
        "expected_evidence_keywords": [],
        "reference_answer": "This query is out of domain for a port operations system. No evidence should be retrieved.",
        "expected_guardrail": "no_source_fallback",
        "golden_vector": None, "golden_sql": None, "golden_rules": None, "golden_graph": None,
    })

    new_queries.append({
        "id": "GUARD_EMPTY_002", "difficulty": "hard",
        "query": "Tell me a joke about shipping containers.",
        "answer_mode": "lookup",
        "expected_sources": [],
        "needs_vector": False, "needs_sql": False, "needs_rules": False, "needs_graph": False,
        "expected_evidence_keywords": [],
        "reference_answer": "Off-topic query. The system should produce a weak/no answer or redirect to port operations topics.",
        "expected_guardrail": "no_source_fallback",
        "golden_vector": None, "golden_sql": None, "golden_rules": None, "golden_graph": None,
    })

    # ================================================================
    # GUARDRAIL TEST QUERIES
    # ================================================================

    # G1: decision_support WITHOUT rules → should trigger guardrail
    new_queries.append({
        "id": "GUARD_DS_NORULE_001", "difficulty": "hard",
        "query": "Based on current wind speed data only, should we restrict vessel entry?",
        "answer_mode": "decision_support",
        "expected_sources": ["sql"],
        "needs_vector": False, "needs_sql": True, "needs_rules": False, "needs_graph": False,
        "expected_evidence_keywords": ["wind_speed_ms", "vessel", "entry"],
        "reference_answer": "Without rule evidence, the system should NOT issue a hard stop/go recommendation. It should provide a bounded assessment and recommend manual verification.",
        "expected_guardrail": "decision_support_no_rules",
        "golden_vector": None,
        "golden_sql": sql_meta({"environment": ["wind_speed_ms"]}, "MAX"),
        "golden_rules": None, "golden_graph": None,
    })

    new_queries.append({
        "id": "GUARD_DS_NORULE_002", "difficulty": "hard",
        "query": "Given crane productivity data, should crane operations be paused?",
        "answer_mode": "decision_support",
        "expected_sources": ["sql"],
        "needs_vector": False, "needs_sql": True, "needs_rules": False, "needs_graph": False,
        "expected_evidence_keywords": ["crane_productivity_mph", "pause", "operations"],
        "reference_answer": "Without threshold rules, the system should not issue a definitive pause/continue recommendation.",
        "expected_guardrail": "decision_support_no_rules",
        "golden_vector": None,
        "golden_sql": sql_meta({"crane_operations": ["crane_productivity_mph"]}, "AVG"),
        "golden_rules": None, "golden_graph": None,
    })

    # G2: decision_support WITH rules → guardrail should NOT trigger
    new_queries.append({
        "id": "GUARD_DS_WITHRULE_001", "difficulty": "hard",
        "query": "Based on current wind speed data AND port wind restriction rules, should vessel entry be restricted?",
        "answer_mode": "decision_support",
        "expected_sources": ["sql", "rules"],
        "needs_vector": False, "needs_sql": True, "needs_rules": True, "needs_graph": False,
        "expected_evidence_keywords": ["wind_speed_ms", "vessel", "entry", "restriction", "knots", "threshold"],
        "reference_answer": "With both data and rule evidence, the system may provide an evidence-grounded recommendation comparing current wind against the 30-35 knot threshold.",
        "expected_guardrail": "decision_support_with_rules",
        "golden_vector": None,
        "golden_sql": sql_meta({"environment": ["wind_speed_ms"]}, "MAX"),
        "golden_rules": rule_meta(["Wind Speed"]),
        "golden_graph": None,
    })

    # G3: diagnostic WITHOUT graph → fallback triggered
    new_queries.append({
        "id": "GUARD_DIAG_NOGRAPH_001", "difficulty": "hard",
        "query": "Why do berth delays increase during certain weather patterns?",
        "answer_mode": "diagnostic",
        "expected_sources": ["sql"],
        "needs_vector": False, "needs_sql": True, "needs_rules": False, "needs_graph": False,
        "expected_evidence_keywords": ["berth", "delay", "weather", "pattern"],
        "reference_answer": "Without graph evidence, the system should provide a weaker explanation based only on SQL data, noting the absence of causal reasoning.",
        "expected_guardrail": "diagnostic_without_graph",
        "golden_vector": None,
        "golden_sql": sql_meta({"berth_operations": ["arrival_delay_hours"],
                                "environment": ["wind_speed_ms"]}, None),
        "golden_rules": None, "golden_graph": None,
    })

    # G4: descriptive WITHOUT docs → fallback triggered
    new_queries.append({
        "id": "GUARD_DESC_NODOC_001", "difficulty": "hard",
        "query": "Summarize the port's strategic plan for the next decade.",
        "answer_mode": "descriptive",
        "expected_sources": ["vector"],
        "needs_vector": True, "needs_sql": False, "needs_rules": False, "needs_graph": False,
        "expected_evidence_keywords": ["strategic", "plan", "decade", "vision"],
        "reference_answer": "If no relevant documents are retrieved, the system should acknowledge the gap and provide only cautious general-domain background.",
        "expected_guardrail": "descriptive_without_docs",
        "golden_vector": {
            "relevant_chunk_ids": [],  # intentionally empty — tests miss scenario
            "relevant_source_files": [],
            "relevant_pages": [],
        },
        "golden_sql": None, "golden_rules": None, "golden_graph": None,
    })

    # G5: knowledge_fallback — no evidence at all for a legitimate query
    new_queries.append({
        "id": "GUARD_FALLBACK_001", "difficulty": "hard",
        "query": "What is the optimal container dwell time to minimize yard congestion?",
        "answer_mode": "lookup",
        "expected_sources": ["sql", "rules"],
        "needs_vector": False, "needs_sql": True, "needs_rules": True, "needs_graph": False,
        "expected_evidence_keywords": ["dwell", "congestion", "optimal", "yard"],
        "reference_answer": "The answer should combine dwell time data (~4.74 days average) with threshold rules (85% occupancy). If rules are missing, fallback is triggered.",
        "expected_guardrail": "knowledge_fallback_test",
        "golden_vector": None,
        "golden_sql": sql_meta({"yard_operations": ["average_dwell_days", "peak_occupancy_pct"]}, "AVG"),
        "golden_rules": rule_meta(["peak_occupancy_pct"]),
        "golden_graph": None,
    })

    # ================================================================
    # Merge and save
    # ================================================================

    golden.extend(new_queries)

    out_path = PROJECT_ROOT / "evaluation" / "golden_dataset.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(golden, f, indent=2, ensure_ascii=False)

    print(f"\nExpanded: {len(golden) - len(new_queries)} + {len(new_queries)} = {len(golden)} queries")

    # Verify ALL 16 combos covered
    import itertools
    from collections import Counter
    sources = ['graph', 'rules', 'sql', 'vector']
    all_combos = [()]  # include empty
    for r in range(1, 5):
        for combo in itertools.combinations(sources, r):
            all_combos.append(tuple(sorted(combo)))

    existing = Counter(tuple(sorted(g['expected_sources'])) for g in golden)
    print(f"\n=== ALL 16 SOURCE COMBOS ===")
    for combo in all_combos:
        n = existing.get(combo, 0)
        tag = '' if n > 0 else '  ** STILL MISSING **'
        print(f"  {str(combo):50s} {n:3d}{tag}")

    missing = [c for c in all_combos if existing.get(c, 0) == 0]
    print(f"\nCovered: {16-len(missing)}/16")

    # Guardrail queries
    guard_queries = [g for g in golden if g.get("expected_guardrail")]
    print(f"\nGuardrail test queries: {len(guard_queries)}")
    for g in guard_queries:
        print(f"  {g['id']:30s} guardrail={g['expected_guardrail']}")


if __name__ == "__main__":
    main()
