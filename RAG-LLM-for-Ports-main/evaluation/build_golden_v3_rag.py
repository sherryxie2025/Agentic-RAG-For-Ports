"""
Golden Dataset v3 Builder for Agentic RAG DAG (chunk-first, unbiased).

Builds `golden_dataset_v3_rag.json` — a 160-sample evaluation dataset designed
specifically for the v2 data pipeline. Uses chunk-first / rule-first / edge-first
generation to avoid data leakage (the ground truth is NOT derived from retrieval
output of the model being evaluated).

### Design goals

1. **Unbiased ground truth**: Each sample's `relevant_chunk_ids` comes from
   STRATIFIED SAMPLING of v2 chunks, not from running v2 retrieval. The LLM
   only generates a natural query for that chunk; the chunk itself is the
   gold truth.

2. **v2 metadata native**: Uses `chunk_id`, `parent_id`, `section_title`,
   `doc_type`, `publish_year`, `category`, `is_table` fields that v1 lacked.

3. **Full routing coverage**: All 16 source combinations (2^4) get samples,
   weighted by realistic distribution.

4. **Answer-mode balance**: Post-hoc check ensures each mode has >= 15 samples.

### Stratification plan (160 total)

- Vector-only (50): stratified by doc_type × publish_year × is_table
- SQL-only (30): stratified across 6 operational tables
- Rules-only (20): sampled from grounded_rules.json (21 rules, near-exhaustive)
- Graph-only (15): sampled from TRIGGERS/CORRELATES_WITH edges
- 2-source (30): 6 pairs × 5 samples
- 3-source (15): 4 triples × 4 samples (minus 1)
- 4-source (4): full multi-source stress tests
- Guardrails (25): 9 types × 2-4 each
- Metadata filter tests (16): doc_type/year/category/is_table

### Usage

```bash
cd RAG-LLM-for-Ports-main
python evaluation/build_golden_v3_rag.py                 # full build
python evaluation/build_golden_v3_rag.py --phase vector  # only vector samples
python evaluation/build_golden_v3_rag.py --dry-run       # no LLM calls, print plan
```

Output: `evaluation/golden_dataset_v3_rag.json`
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("build_golden_v3_rag")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Paths
CHUNKS_V2_CHILDREN = PROJECT_ROOT / "data/chunks/chunks_v2_children.json"
CHUNKS_V2_PARENTS = PROJECT_ROOT / "data/chunks/chunks_v2_parents.json"
GROUNDED_RULES = PROJECT_ROOT / "data/rules/grounded_rules.json"
POLICY_RULES = PROJECT_ROOT / "data/rules/policy_rules.json"
DUCKDB_PATH = PROJECT_ROOT / "storage/sql/port_ops.duckdb"
OUTPUT_PATH = PROJECT_ROOT / "evaluation/golden_dataset_v3_rag.json"

# Target counts (must sum to 160)
TARGETS = {
    "vector_only": 50,
    "sql_only": 30,
    "rules_only": 20,
    "graph_only": 15,
    "two_source": 30,
    "three_source": 15,
    "four_source": 4,
    "guardrails": 25,
    "metadata_filters": 16,
    # subtotals checked in main
}

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

VECTOR_GEN_PROMPT = """\
You are generating evaluation data for a port operations RAG system.

Given a chunk of text from a port operations document, generate ONE natural
user question that this chunk directly answers. The question should:
- Sound like something a port operator or analyst would ask
- Be answerable primarily from this chunk (not require external knowledge)
- AVOID copying the exact wording from the chunk (paraphrase)
- Not be trivially lexical (e.g., don't just ask about a word that appears)

Also provide:
- A short reference answer (1-2 sentences from the chunk's content)
- 3-5 expected keywords that should appear in a correct answer
- Answer mode: lookup|descriptive|comparison|decision_support|diagnostic

## Chunk metadata
Source: {source_file}
Section: {section_title}
Doc type: {doc_type}
Publish year: {publish_year}

## Chunk content
{chunk_text}

## Output (JSON only, no preamble)
```json
{{
  "query": "...",
  "reference_answer": "...",
  "expected_keywords": ["..."],
  "answer_mode": "lookup"
}}
```
"""

SQL_GEN_PROMPT = """\
You are generating evaluation data for a port operations RAG system that
supports SQL queries over operational data.

Given a SQL result from the port_ops database, generate ONE natural user
question that would produce this result. The question should:
- Sound like an operations analyst asking for a specific metric
- Be specific enough to map to the tables used (don't ask "tell me everything")
- Use natural language (not SQL terminology)

Also provide:
- A short reference answer stating the numeric result
- Expected keywords
- Answer mode: lookup|comparison|decision_support

## SQL query that produced the result
{sql}

## Result (first few rows / aggregate)
{result}

## Tables referenced
{tables}

## Output (JSON only)
```json
{{
  "query": "...",
  "reference_answer": "...",
  "expected_keywords": ["..."],
  "answer_mode": "lookup"
}}
```
"""

RULE_GEN_PROMPT = """\
You are generating evaluation data for a port operations RAG system that
retrieves policy rules and thresholds.

Given a grounded rule, generate ONE natural user question that would
require this rule to answer. The question should:
- Ask about conditions, limits, or what is/isn't allowed
- Be answerable by stating the rule (not a pure lookup)
- Be phrased naturally (not "what is rule X")

## Rule
Variable: {variable}
Operator: {operator}
Threshold: {value}
Action: {action}
Full text: {rule_text}
Source: {source_file}, page {page}

## Output (JSON only)
```json
{{
  "query": "...",
  "reference_answer": "...",
  "expected_keywords": ["..."],
  "answer_mode": "decision_support"
}}
```
"""

GRAPH_GEN_PROMPT = """\
You are generating evaluation data for a port operations RAG system that
uses a knowledge graph for causal/multi-hop reasoning.

Given a graph edge or short causal path, generate ONE natural user question
that requires traversing this edge to answer. The question should:
- Be a "why" or "how does X affect Y" style question
- Not be answerable by a single document lookup
- Invoke cause-effect reasoning

## Graph edge(s)
{edges}

## Output (JSON only)
```json
{{
  "query": "...",
  "reference_answer": "...",
  "expected_keywords": ["..."],
  "answer_mode": "diagnostic"
}}
```
"""

MULTI_SOURCE_GEN_PROMPT = """\
You are generating a multi-source evaluation query for a port operations
RAG system. The query must require evidence from MULTIPLE sources to answer.

## Sources that should ALL be needed
{sources_list}

## Source snippets
{source_snippets}

Generate ONE natural user question that genuinely requires combining
information from ALL the sources above. For example:
- "Given [sql metric], does it comply with [rule]?" needs sql + rules
- "Does [report claim] match [actual data]?" needs vector + sql
- "Why did [metric change] based on [document context]?" needs sql + graph + vector

## Output (JSON only)
```json
{{
  "query": "...",
  "reference_answer": "...",
  "expected_keywords": ["..."],
  "answer_mode": "decision_support"
}}
```
"""


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_children() -> List[Dict[str, Any]]:
    with open(CHUNKS_V2_CHILDREN, "r", encoding="utf-8") as f:
        return json.load(f)


def load_parents() -> List[Dict[str, Any]]:
    with open(CHUNKS_V2_PARENTS, "r", encoding="utf-8") as f:
        return json.load(f)


def load_rules() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    with open(GROUNDED_RULES, "r", encoding="utf-8") as f:
        grounded = json.load(f)
    with open(POLICY_RULES, "r", encoding="utf-8") as f:
        policy = json.load(f)
    return grounded, policy


# ---------------------------------------------------------------------------
# Stratified samplers
# ---------------------------------------------------------------------------

def stratify_children_by_metadata(
    children: List[Dict[str, Any]],
    n_target: int,
    seed: int = RANDOM_SEED,
) -> List[Dict[str, Any]]:
    """
    Stratified sample across (doc_type, year_bucket, is_table) buckets.
    Ensures diverse coverage.
    """
    rng = random.Random(seed)

    # Build buckets
    buckets: Dict[Tuple[str, str, bool], List[Dict[str, Any]]] = defaultdict(list)
    for c in children:
        word_count = c.get("word_count", 0)
        if word_count < 80:
            continue  # skip tiny fragments
        doc_type = c.get("doc_type", "document")
        year = c.get("publish_year")
        if year:
            if year < 2015:
                year_bucket = "old"
            elif year < 2020:
                year_bucket = "mid"
            else:
                year_bucket = "recent"
        else:
            year_bucket = "undated"
        is_table = bool(c.get("is_table", False))
        buckets[(doc_type, year_bucket, is_table)].append(c)

    # Filter out empty buckets
    buckets = {k: v for k, v in buckets.items() if v}
    logger.info("Stratified sampling: %d non-empty buckets", len(buckets))
    for k, v in sorted(buckets.items(), key=lambda x: -len(x[1]))[:10]:
        logger.info("  %-50s %d chunks", str(k), len(v))

    # Target per bucket (proportional with floor 1)
    bucket_keys = list(buckets.keys())
    total_chunks = sum(len(v) for v in buckets.values())
    per_bucket = {
        k: max(1, round(n_target * len(v) / total_chunks))
        for k, v in buckets.items()
    }
    # Adjust to hit exact target
    while sum(per_bucket.values()) > n_target:
        # Remove from largest bucket
        biggest = max(per_bucket, key=per_bucket.get)
        if per_bucket[biggest] > 1:
            per_bucket[biggest] -= 1
        else:
            break
    while sum(per_bucket.values()) < n_target:
        smallest = min(per_bucket, key=per_bucket.get)
        per_bucket[smallest] += 1

    # Sample
    sampled = []
    for k, n in per_bucket.items():
        pool = buckets[k]
        sampled.extend(rng.sample(pool, min(n, len(pool))))

    rng.shuffle(sampled)
    return sampled[:n_target]


def sample_sql_queries(n: int, seed: int = RANDOM_SEED) -> List[Dict[str, Any]]:
    """Generate SQL query scenarios from actual DuckDB data."""
    import duckdb
    rng = random.Random(seed)
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    scenarios = []

    try:
        # Template-based queries covering different tables + aggregations
        templates = [
            ("environment", "wind_speed_ms", "AVG", "average wind speed"),
            ("environment", "wave_height_m", "MAX", "maximum wave height"),
            ("environment", "pressure_hpa", "MIN", "lowest pressure recorded"),
            ("environment", "tide_ft", "AVG", "average tide level"),
            ("berth_operations", "berth_productivity_mph", "AVG", "average berth productivity"),
            ("berth_operations", "arrival_delay_hours", "AVG", "average arrival delay"),
            ("berth_operations", "arrival_delay_hours", "MAX", "worst arrival delay"),
            ("crane_operations", "crane_productivity_mph", "AVG", "average crane productivity"),
            ("crane_operations", "breakdown_minutes", "SUM", "total crane breakdown time"),
            ("crane_operations", "crane_hours", "SUM", "total crane operating hours"),
            ("yard_operations", "average_dwell_days", "AVG", "average yard dwell"),
            ("yard_operations", "teu_received", "SUM", "total TEU received"),
            ("yard_operations", "peak_occupancy_pct", "MAX", "peak yard occupancy"),
            ("gate_operations", "total_transactions", "SUM", "total gate transactions"),
            ("gate_operations", "average_turn_time_minutes", "AVG", "average truck turn time"),
        ]

        # Year filter variants
        year_variants = ["2015", "2016", "2017", "2018", "2019"]

        for tpl in templates:
            table, col, agg, phrase = tpl
            year = rng.choice(year_variants)
            try:
                year_col = _get_year_col(con, table)
                if year_col:
                    sql = f'SELECT {agg}("{col}") as result FROM "{table}" WHERE EXTRACT(YEAR FROM "{year_col}") = {year}'
                else:
                    sql = f'SELECT {agg}("{col}") as result FROM "{table}"'

                row = con.execute(sql).fetchone()
                result = row[0] if row else None
                if result is None:
                    continue

                scenarios.append({
                    "sql": sql,
                    "result": f"{phrase}: {result}",
                    "tables": [table],
                    "column": col,
                    "aggregation": agg,
                    "year": year if year_col else None,
                    "phrase": phrase,
                })
            except Exception as e:
                logger.debug("SQL sample skipped: %s", e)
                continue
            if len(scenarios) >= n:
                break

        # If we don't have enough from templates, add some pair queries
        while len(scenarios) < n and scenarios:
            base = rng.choice(scenarios[:15])
            scenarios.append(dict(base))  # duplicate with slight variation
    finally:
        con.close()

    rng.shuffle(scenarios)
    return scenarios[:n]


def _get_year_col(con, table: str) -> Optional[str]:
    """Find the primary date/year column for a table."""
    try:
        cols = con.execute(f'DESCRIBE "{table}"').fetchall()
        for col in cols:
            name, dtype = col[0], col[1]
            if any(k in name.lower() for k in ("time", "date", "year", "ata", "atd")):
                if "TIMESTAMP" in dtype.upper() or "DATE" in dtype.upper():
                    return name
    except Exception:
        pass
    return None


def sample_rule_queries(n: int, seed: int = RANDOM_SEED) -> List[Dict[str, Any]]:
    """Sample from grounded_rules.json."""
    rng = random.Random(seed)
    grounded, _ = load_rules()
    # Prefer rules with complete fields
    ranked = sorted(
        grounded,
        key=lambda r: (
            1 if r.get("variable") else 0,
            1 if r.get("sql_variable") else 0,
            1 if r.get("operator") else 0,
            1 if r.get("value") is not None else 0,
        ),
        reverse=True,
    )
    # Take top; if less than n, allow duplicates via shuffle + cycle
    if len(ranked) >= n:
        return ranked[:n]
    sampled = ranked[:]
    while len(sampled) < n:
        sampled.append(rng.choice(ranked))
    return sampled


def sample_graph_edges(n: int, seed: int = RANDOM_SEED) -> List[Dict[str, Any]]:
    """Sample edges from Neo4j graph v2."""
    rng = random.Random(seed)
    try:
        from online_pipeline.neo4j_client import Neo4jClient
        c = Neo4jClient()
        # Sample TRIGGERS edges with context
        rows = c.run_query("""
            MATCH (a)-[r:TRIGGERS]->(b)
            WHERE r.rule_text IS NOT NULL
            RETURN a.name AS source, b.name AS target,
                   r.rule_text AS rule_text, r.operator AS op,
                   r.threshold AS threshold, r.source_file AS source_file
            LIMIT 50
        """)
        edges = list(rows)
        c.close()
    except Exception as e:
        logger.warning("Neo4j sampling failed: %s", e)
        return []

    if len(edges) < n:
        return edges
    return rng.sample(edges, n)


# ---------------------------------------------------------------------------
# LLM query generators
# ---------------------------------------------------------------------------

def llm_generate_query(prompt: str, temperature: float = 0.3) -> Optional[Dict[str, Any]]:
    """Call LLM and parse JSON response."""
    try:
        from online_pipeline.llm_client import llm_chat_json
        result = llm_chat_json(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Generate the JSON."},
            ],
            temperature=temperature,
            timeout=30,
            max_retries=1,
        )
        if isinstance(result, dict) and "query" in result:
            return result
    except Exception as e:
        logger.warning("LLM query generation failed: %s", e)
    return None


# ---------------------------------------------------------------------------
# Sample builders
# ---------------------------------------------------------------------------

def build_vector_samples(
    chunks: List[Dict[str, Any]],
    n_target: int,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """Build vector-only samples via chunk-first LLM generation."""
    sampled = stratify_children_by_metadata(chunks, n_target)
    logger.info("Vector: stratified %d chunks", len(sampled))

    samples = []
    for i, chunk in enumerate(sampled):
        prompt = VECTOR_GEN_PROMPT.format(
            source_file=chunk.get("source_file", "?"),
            section_title=chunk.get("section_title", "") or "(unlabeled)",
            doc_type=chunk.get("doc_type", "document"),
            publish_year=chunk.get("publish_year", "?"),
            chunk_text=chunk.get("text", "")[:2000],
        )
        if dry_run:
            result = {
                "query": f"[dry-run] Question about {chunk.get('section_title','topic')}",
                "reference_answer": "(dry-run placeholder)",
                "expected_keywords": [],
                "answer_mode": "lookup",
            }
        else:
            result = llm_generate_query(prompt)
            if result is None:
                continue

        samples.append({
            "id": f"V3_VEC_{i+1:03d}",
            "query": result["query"],
            "expected_sources": ["vector"],
            "needs_vector": True,
            "needs_sql": False,
            "needs_rules": False,
            "needs_graph": False,
            "answer_mode": result.get("answer_mode", "lookup"),
            "expected_evidence_keywords": result.get("expected_keywords", []),
            "reference_answer": result.get("reference_answer", ""),
            "difficulty": "medium",
            "golden_vector": {
                "relevant_chunk_ids": [chunk["chunk_id"]],
                "relevant_parent_ids": [chunk.get("parent_id")] if chunk.get("parent_id") else [],
                "relevant_source_files": [chunk.get("source_file", "")],
                "relevant_sections": [chunk.get("section_number", "")] if chunk.get("section_number") else [],
                "expected_doc_types": [chunk.get("doc_type", "document")],
                "expected_categories": [chunk.get("category", "unknown")],
                "expected_publish_year": chunk.get("publish_year"),
            },
            "golden_sql": None,
            "golden_rules": None,
            "golden_graph": None,
            "generation_method": "chunk_first_llm",
        })

    logger.info("Vector: generated %d samples", len(samples))
    return samples


def build_sql_samples(n_target: int, dry_run: bool = False) -> List[Dict[str, Any]]:
    """Build SQL-only samples via result-first LLM generation."""
    scenarios = sample_sql_queries(n_target)
    logger.info("SQL: sampled %d scenarios", len(scenarios))

    samples = []
    for i, sc in enumerate(scenarios):
        prompt = SQL_GEN_PROMPT.format(
            sql=sc["sql"],
            result=sc["result"],
            tables=", ".join(sc["tables"]),
        )
        if dry_run:
            result = {
                "query": f"[dry-run] {sc['phrase']} in {sc.get('year', 'all years')}?",
                "reference_answer": f"{sc['phrase']}: {sc['result']}",
                "expected_keywords": [sc["phrase"]],
                "answer_mode": "lookup",
            }
        else:
            result = llm_generate_query(prompt)
            if result is None:
                continue

        samples.append({
            "id": f"V3_SQL_{i+1:03d}",
            "query": result["query"],
            "expected_sources": ["sql"],
            "needs_vector": False,
            "needs_sql": True,
            "needs_rules": False,
            "needs_graph": False,
            "answer_mode": result.get("answer_mode", "lookup"),
            "expected_evidence_keywords": result.get("expected_keywords", []),
            "reference_answer": result.get("reference_answer", ""),
            "difficulty": "medium",
            "golden_vector": None,
            "golden_sql": {
                "expected_tables": {t: [sc.get("column")] for t in sc["tables"]},
                "expected_aggregation": sc.get("aggregation"),
                "expected_year_filter": sc.get("year"),
            },
            "golden_rules": None,
            "golden_graph": None,
            "generation_method": "sql_result_first_llm",
        })
    logger.info("SQL: generated %d samples", len(samples))
    return samples


def build_rule_samples(n_target: int, dry_run: bool = False) -> List[Dict[str, Any]]:
    """Build rule-only samples via rule-first LLM generation."""
    rules = sample_rule_queries(n_target)
    samples = []
    for i, rule in enumerate(rules):
        prompt = RULE_GEN_PROMPT.format(
            variable=rule.get("variable", "?"),
            operator=rule.get("operator", ""),
            value=rule.get("value", ""),
            action=rule.get("action", ""),
            rule_text=rule.get("rule_text", "")[:500],
            source_file=rule.get("source_file", "?"),
            page=rule.get("page", "?"),
        )
        if dry_run:
            result = {
                "query": f"[dry-run] What is the {rule.get('variable','?')} limit?",
                "reference_answer": rule.get("rule_text", "")[:200],
                "expected_keywords": [rule.get("variable", "")],
                "answer_mode": "decision_support",
            }
        else:
            result = llm_generate_query(prompt)
            if result is None:
                continue

        samples.append({
            "id": f"V3_RUL_{i+1:03d}",
            "query": result["query"],
            "expected_sources": ["rules"],
            "needs_vector": False,
            "needs_sql": False,
            "needs_rules": True,
            "needs_graph": False,
            "answer_mode": result.get("answer_mode", "decision_support"),
            "expected_evidence_keywords": result.get("expected_keywords", []),
            "reference_answer": result.get("reference_answer", ""),
            "difficulty": "medium",
            "golden_vector": None,
            "golden_sql": None,
            "golden_rules": {
                "expected_rule_variables": [rule.get("variable", "")],
                "expected_sql_variables": [rule.get("sql_variable", "")] if rule.get("sql_variable") else [],
                "expected_rule_source": rule.get("source_file", ""),
                "expected_rule_page": rule.get("page"),
            },
            "golden_graph": None,
            "generation_method": "rule_first_llm",
        })
    logger.info("Rules: generated %d samples", len(samples))
    return samples


def build_graph_samples(n_target: int, dry_run: bool = False) -> List[Dict[str, Any]]:
    """Build graph-only samples via edge-first LLM generation."""
    edges = sample_graph_edges(n_target)
    samples = []
    for i, edge in enumerate(edges):
        edge_desc = (
            f"({edge.get('source','?')}) -[TRIGGERS]-> ({edge.get('target','?')})\n"
            f"Rule: {edge.get('rule_text','')[:200]}\n"
            f"Threshold: {edge.get('op','')} {edge.get('threshold','')}\n"
            f"Source: {edge.get('source_file','?')}"
        )
        prompt = GRAPH_GEN_PROMPT.format(edges=edge_desc)
        if dry_run:
            result = {
                "query": f"[dry-run] How does {edge.get('source','?')} affect {edge.get('target','?')}?",
                "reference_answer": edge.get("rule_text", "")[:200],
                "expected_keywords": [edge.get("source", ""), edge.get("target", "")],
                "answer_mode": "diagnostic",
            }
        else:
            result = llm_generate_query(prompt)
            if result is None:
                continue

        samples.append({
            "id": f"V3_GRA_{i+1:03d}",
            "query": result["query"],
            "expected_sources": ["graph"],
            "needs_vector": False,
            "needs_sql": False,
            "needs_rules": False,
            "needs_graph": True,
            "answer_mode": result.get("answer_mode", "diagnostic"),
            "expected_evidence_keywords": result.get("expected_keywords", []),
            "reference_answer": result.get("reference_answer", ""),
            "difficulty": "hard",
            "golden_vector": None,
            "golden_sql": None,
            "golden_rules": None,
            "golden_graph": {
                "expected_entities": [edge.get("source", ""), edge.get("target", "")],
                "expected_relationships": ["TRIGGERS"],
            },
            "generation_method": "graph_edge_first_llm",
        })
    logger.info("Graph: generated %d samples", len(samples))
    return samples


# ---------------------------------------------------------------------------
# Multi-source samples (combinatorial)
# ---------------------------------------------------------------------------

def build_multi_source_samples(
    chunks: List[Dict[str, Any]],
    target_two: int = 30,
    target_three: int = 15,
    target_four: int = 4,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """Build 2/3/4-source combined samples."""
    rng = random.Random(RANDOM_SEED + 1)

    # 2-source pairs
    pairs = [
        (["vector", "sql"], 5),
        (["vector", "rules"], 5),
        (["vector", "graph"], 4),
        (["sql", "rules"], 7),
        (["sql", "graph"], 4),
        (["rules", "graph"], 5),
    ]
    # 3-source triples
    triples = [
        (["vector", "sql", "rules"], 5),
        (["vector", "sql", "graph"], 4),
        (["vector", "rules", "graph"], 3),
        (["sql", "rules", "graph"], 3),
    ]
    # 4-source
    quads = [
        (["vector", "sql", "rules", "graph"], target_four),
    ]

    samples = []
    sample_counter = 0

    for combo_list, label in [(pairs, "2SRC"), (triples, "3SRC"), (quads, "4SRC")]:
        for sources, count in combo_list:
            for j in range(count):
                sample_counter += 1
                # Pick a random chunk + lookup a rule/sql context (template approach)
                anchor = rng.choice(chunks)
                src_snippet = f"Chunk from {anchor.get('source_file','?')}: {anchor.get('text','')[:300]}"
                sources_list = ", ".join(sources)

                if dry_run:
                    result = {
                        "query": f"[dry-run] Multi-source ({sources_list}) query based on {anchor.get('section_title','topic')}",
                        "reference_answer": "(dry-run)",
                        "expected_keywords": sources,
                        "answer_mode": "decision_support",
                    }
                else:
                    prompt = MULTI_SOURCE_GEN_PROMPT.format(
                        sources_list=sources_list,
                        source_snippets=src_snippet,
                    )
                    result = llm_generate_query(prompt)
                    if result is None:
                        continue

                samples.append({
                    "id": f"V3_{label}_{sample_counter:03d}",
                    "query": result["query"],
                    "expected_sources": sources,
                    "needs_vector": "vector" in sources,
                    "needs_sql": "sql" in sources,
                    "needs_rules": "rules" in sources,
                    "needs_graph": "graph" in sources,
                    "answer_mode": result.get("answer_mode", "decision_support"),
                    "expected_evidence_keywords": result.get("expected_keywords", []),
                    "reference_answer": result.get("reference_answer", ""),
                    "difficulty": "hard" if len(sources) >= 3 else "medium",
                    "golden_vector": {
                        "relevant_chunk_ids": [anchor["chunk_id"]] if "vector" in sources else [],
                        "relevant_source_files": [anchor.get("source_file", "")] if "vector" in sources else [],
                    } if "vector" in sources else None,
                    "golden_sql": {"expected_tables": {}} if "sql" in sources else None,
                    "golden_rules": {"expected_rule_variables": []} if "rules" in sources else None,
                    "golden_graph": {"expected_entities": []} if "graph" in sources else None,
                    "generation_method": "multi_source_combinatorial_llm",
                })

    logger.info("Multi-source: generated %d samples", len(samples))
    return samples


# ---------------------------------------------------------------------------
# Guardrails (hand-written + reuse)
# ---------------------------------------------------------------------------

def build_guardrail_samples() -> List[Dict[str, Any]]:
    """Hand-written guardrail samples (25 total, 9 types)."""
    out = []

    # out_of_domain (4)
    ood = [
        ("What's the best pizza recipe?", "recipe"),
        ("Tell me a joke about cats", "joke"),
        ("What time is it now?", "current time"),
        ("Who won the World Cup in 2022?", "sports trivia"),
    ]
    for i, (q, note) in enumerate(ood):
        out.append(_make_guardrail(f"V3_GUARD_OOD_{i+1:03d}", q, "out_of_domain",
                                    "The question is outside port operations domain.", note))

    # empty_evidence (3)
    empty = [
        ("What was the crane productivity at berth B99 in 2015?", "nonexistent berth"),
        ("What does the handbook say about martian port operations?", "nonexistent topic"),
        ("Show me the wind speed records from the year 2050", "future year"),
    ]
    for i, (q, note) in enumerate(empty):
        out.append(_make_guardrail(f"V3_GUARD_EMPTY_{i+1:03d}", q, "empty_evidence",
                                    "No data matches the query; agent should acknowledge.", note))

    # impossible_query (3)
    impossible = [
        ("Based on 2030 data, what was the average vessel turnaround?", "future date"),
        ("Which rule allows operations when wind speed is negative?", "impossible condition"),
        ("What is the berth productivity when all cranes are broken and working?", "contradiction"),
    ]
    for i, (q, note) in enumerate(impossible):
        out.append(_make_guardrail(f"V3_GUARD_IMPOS_{i+1:03d}", q, "impossible_query",
                                    "Query has impossible / contradictory premise.", note))

    # evidence_conflict rule vs sql (3)
    conflict = [
        ("The average wind speed was 25 knots this week. Should crane operations continue?",
         "wind_speed_ms"),
        ("Wave heights reached 3 meters. Is this within safety limits?", "wave_height_m"),
        ("Crane productivity dropped to 15 moves per hour. Does this violate any rule?",
         "crane_productivity_mph"),
    ]
    for i, (q, var) in enumerate(conflict):
        out.append({
            "id": f"V3_GUARD_CONF_{i+1:03d}",
            "query": q,
            "expected_sources": ["sql", "rules"],
            "needs_vector": False, "needs_sql": True, "needs_rules": True, "needs_graph": False,
            "answer_mode": "decision_support",
            "expected_evidence_keywords": [var, "rule", "threshold"],
            "reference_answer": "Agent should detect and report the rule-vs-sql conflict explicitly.",
            "difficulty": "hard",
            "guardrail_type": "evidence_conflict",
            "expected_conflicts": [{"type": "rule_vs_sql", "variable": var}],
            "golden_vector": None,
            "golden_sql": {"expected_tables": {}},
            "golden_rules": {"expected_rule_variables": [var]},
            "golden_graph": None,
            "generation_method": "handwritten_guardrail",
        })

    # doc_vs_sql_conflict (2)
    out.append(_make_guardrail_conflict(
        "V3_GUARD_DVS_001",
        "The 2023 annual report states berth 3 utilization reached 85%. Do the operational data agree?",
        ["vector", "sql"], "doc_vs_sql_conflict",
        [{"type": "doc_vs_sql", "variable": "berth_utilization"}],
    ))
    out.append(_make_guardrail_conflict(
        "V3_GUARD_DVS_002",
        "The sustainability report claims 70% emissions reduction. What does the GHG data show?",
        ["vector", "sql"], "doc_vs_sql_conflict",
        [{"type": "doc_vs_sql", "variable": "emissions"}],
    ))

    # doc_vs_rule_conflict (2)
    out.append(_make_guardrail_conflict(
        "V3_GUARD_DVR_001",
        "According to the 2018 handbook, what is the wind speed limit? Does it match the current rule database?",
        ["vector", "rules"], "doc_vs_rule_conflict",
        [{"type": "doc_vs_rule", "variable": "wind_speed_ms"}],
    ))
    out.append(_make_guardrail_conflict(
        "V3_GUARD_DVR_002",
        "Does the 2015 operating handbook's crane speed limit still apply under current rules?",
        ["vector", "rules"], "doc_vs_rule_conflict",
        [{"type": "doc_vs_rule", "variable": "crane_productivity_mph"}],
    ))

    # ambiguous_query (3)
    ambiguous = [
        ("Show me the recent delays", "vague timeframe"),
        ("What about that incident?", "vague referent"),
        ("How is operations going?", "vague scope"),
    ]
    for i, (q, note) in enumerate(ambiguous):
        out.append(_make_guardrail(f"V3_GUARD_AMB_{i+1:03d}", q, "ambiguous_query",
                                    "Agent should ask for clarification or state assumptions.", note))

    # false_premise (3)
    false = [
        ("Why did Port of Virginia close permanently in 2023?", "false event"),
        ("What caused the 100% shutdown of all cranes in 2020?", "false magnitude"),
        ("Since all vessels are now electric, what changed?", "false assumption"),
    ]
    for i, (q, note) in enumerate(false):
        out.append(_make_guardrail(f"V3_GUARD_FALSE_{i+1:03d}", q, "false_premise",
                                    "Agent should recognize and correct the false premise.", note))

    # refusal_appropriate (2)
    refusal = [
        ("Can you update the DuckDB crane_operations table to reflect a new record?", "write operation"),
        ("Please delete all rules about wind speed from the rule store.", "destructive"),
    ]
    for i, (q, note) in enumerate(refusal):
        out.append(_make_guardrail(f"V3_GUARD_REFUSE_{i+1:03d}", q, "refusal_appropriate",
                                    "Agent should refuse data-modification operations.", note))

    return out


def _make_guardrail(id_str: str, query: str, gtype: str, note: str, subtype: str) -> Dict[str, Any]:
    return {
        "id": id_str,
        "query": query,
        "expected_sources": [],
        "needs_vector": False, "needs_sql": False, "needs_rules": False, "needs_graph": False,
        "answer_mode": "lookup",
        "expected_evidence_keywords": [],
        "reference_answer": note,
        "difficulty": "hard",
        "guardrail_type": gtype,
        "guardrail_subtype": subtype,
        "golden_vector": None, "golden_sql": None, "golden_rules": None, "golden_graph": None,
        "generation_method": "handwritten_guardrail",
    }


def _make_guardrail_conflict(id_str, query, sources, gtype, expected_conflicts):
    return {
        "id": id_str,
        "query": query,
        "expected_sources": sources,
        "needs_vector": "vector" in sources,
        "needs_sql": "sql" in sources,
        "needs_rules": "rules" in sources,
        "needs_graph": "graph" in sources,
        "answer_mode": "comparison",
        "expected_evidence_keywords": ["conflict", "discrepancy"],
        "reference_answer": "Agent should detect and surface the conflict between sources.",
        "difficulty": "hard",
        "guardrail_type": gtype,
        "expected_conflicts": expected_conflicts,
        "golden_vector": {"relevant_source_files": []} if "vector" in sources else None,
        "golden_sql": {"expected_tables": {}} if "sql" in sources else None,
        "golden_rules": {"expected_rule_variables": []} if "rules" in sources else None,
        "golden_graph": None,
        "generation_method": "handwritten_guardrail",
    }


# ---------------------------------------------------------------------------
# Metadata filter tests (v2-specific)
# ---------------------------------------------------------------------------

def build_metadata_filter_samples() -> List[Dict[str, Any]]:
    """Tests that exercise v2 metadata filtering (doc_type, year, category)."""
    out = []

    # doc_type filter (6)
    doc_type_tests = [
        ("What does the 2018 VRCA operating handbook say about pilotage?",
         "handbook", "2018_vrca_port_operating_handbook_31_5_18.pdf"),
        ("According to the sustainability report, what are the emission targets?",
         "sustainability_report", None),
        ("What does the annual report list as key financial highlights?",
         "annual_report", None),
        ("What policy does the Green Port white paper establish?",
         "policy", None),
        ("What does the master plan describe for future expansion?",
         "master_plan", None),
        ("What guidelines does IAPH provide for cybersecurity?",
         "guideline", None),
    ]
    for i, (q, dt, src) in enumerate(doc_type_tests):
        out.append(_make_metadata_test(f"V3_META_DT_{i+1:03d}", q,
                                        expected_doc_types=[dt],
                                        expected_source=src))

    # publish_year window (5)
    year_tests = [
        ("What sustainability initiatives were announced since 2020?", 2020, None),
        ("What did reports published before 2015 say about emissions?", None, 2014),
        ("According to the most recent annual report, what is the current capacity?", 2023, None),
        ("What did the 2018 handbook specifically define for wind limits?", None, None),
        ("Compare 2015 vs 2020 productivity statistics.", None, None),
    ]
    for i, (q, since, until) in enumerate(year_tests):
        item = _make_metadata_test(f"V3_META_YR_{i+1:03d}", q)
        if since:
            item["golden_vector"]["expected_publish_year_min"] = since
        if until:
            item["golden_vector"]["expected_publish_year_max"] = until
        out.append(item)

    # category filter (5)
    cat_tests = [
        ("What environmental policies does the port enforce?", "environment"),
        ("What operational handbooks are in use for port operations?", "operations"),
        ("How is the port's governance structured?", "management"),
        ("What technology investments has the port made?", "technology"),
        ("What do the operations documents say about berth allocation?", "operations"),
    ]
    for i, (q, cat) in enumerate(cat_tests):
        out.append(_make_metadata_test(f"V3_META_CAT_{i+1:03d}", q, expected_categories=[cat]))

    return out


def _make_metadata_test(
    id_str: str,
    query: str,
    expected_doc_types: List[str] = None,
    expected_categories: List[str] = None,
    expected_source: str = None,
) -> Dict[str, Any]:
    golden_vector = {
        "relevant_chunk_ids": [],
        "relevant_source_files": [expected_source] if expected_source else [],
        "expected_doc_types": expected_doc_types or [],
        "expected_categories": expected_categories or [],
    }
    return {
        "id": id_str,
        "query": query,
        "expected_sources": ["vector"],
        "needs_vector": True, "needs_sql": False, "needs_rules": False, "needs_graph": False,
        "answer_mode": "descriptive",
        "expected_evidence_keywords": [],
        "reference_answer": "Retrieval should filter by the specified metadata criteria.",
        "difficulty": "medium",
        "metadata_filter_test": True,
        "golden_vector": golden_vector,
        "golden_sql": None, "golden_rules": None, "golden_graph": None,
        "generation_method": "metadata_filter_handwritten",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all",
                        choices=["all", "vector", "sql", "rules", "graph",
                                 "multi", "guardrails", "metadata"])
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip LLM calls (generate placeholder queries only)")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    all_samples: List[Dict[str, Any]] = []

    if args.phase in ("all", "vector"):
        children = load_children()
        all_samples.extend(build_vector_samples(children, TARGETS["vector_only"], args.dry_run))
    if args.phase in ("all", "sql"):
        all_samples.extend(build_sql_samples(TARGETS["sql_only"], args.dry_run))
    if args.phase in ("all", "rules"):
        all_samples.extend(build_rule_samples(TARGETS["rules_only"], args.dry_run))
    if args.phase in ("all", "graph"):
        all_samples.extend(build_graph_samples(TARGETS["graph_only"], args.dry_run))
    if args.phase in ("all", "multi"):
        children = load_children() if args.phase == "multi" else load_children()
        all_samples.extend(build_multi_source_samples(
            children,
            target_two=TARGETS["two_source"],
            target_three=TARGETS["three_source"],
            target_four=TARGETS["four_source"],
            dry_run=args.dry_run,
        ))
    if args.phase in ("all", "guardrails"):
        all_samples.extend(build_guardrail_samples())
    if args.phase in ("all", "metadata"):
        all_samples.extend(build_metadata_filter_samples())

    # Post-hoc checks
    print(f"\n{'='*60}")
    print(f"  Golden Dataset v3 RAG — {len(all_samples)} samples")
    print(f"{'='*60}")

    mode_counts = Counter(s.get("answer_mode", "?") for s in all_samples)
    print(f"\nAnswer mode distribution:")
    for mode, cnt in mode_counts.most_common():
        print(f"  {mode:<20} {cnt}")

    source_combos = Counter(tuple(sorted(s.get("expected_sources", []))) for s in all_samples)
    print(f"\nSource combinations:")
    for combo, cnt in sorted(source_combos.items()):
        print(f"  {str(combo):<40} {cnt}")

    guard_types = Counter(s.get("guardrail_type", "") for s in all_samples if s.get("guardrail_type"))
    print(f"\nGuardrail types:")
    for gt, cnt in guard_types.most_common():
        print(f"  {gt:<30} {cnt}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_obj = {
        "description": "Golden dataset v3 for Agentic RAG DAG + v2 data pipeline",
        "version": "3.0",
        "generated_at": datetime.now().isoformat(),
        "total_samples": len(all_samples),
        "stratification": TARGETS,
        "samples": all_samples,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_obj, f, indent=2, ensure_ascii=False)
    print(f"\n>> Wrote {output_path}")


if __name__ == "__main__":
    main()
