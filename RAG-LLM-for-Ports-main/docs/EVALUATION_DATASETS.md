# Evaluation Datasets — Construction Methods

> Covers all three evaluation layers: single-turn, multi-turn (single-session), and cross-session.
> For the full 863-line methodology of the 205-sample golden set, see `evaluation/GOLDEN_DATASET_V3_METHODOLOGY_CN.md`.

---

## Overview

| Layer | File | Size | Tests what | How built |
|---|---|---|---|---|
| **Single-turn** | `golden_dataset_v3_rag.json` | 205 samples | Router / retrieval / answer / guardrails | Reverse generation by Claude Opus from sampled chunks |
| **Multi-turn** | `golden_dataset_v3_multi_turn.json` | 10 conv / 31 turns | Short-term memory (co-ref, entity tracking, topic switch) | Hand-written templates derived from the 205 base samples |
| **Cross-session** | `golden_dataset_v4_cross_session.json` | 35 conv / 70 turns | Long-term memory (cross-session recall, discrimination) | Claude subagent generates session-2 queries from real retrieval data |

The three layers are additive — each one tests a capability the previous layer cannot:
- Single-turn: can the pipeline answer one question correctly?
- Multi-turn: can it maintain context within a conversation?
- Cross-session: can it recall knowledge from a previous conversation days later?

---

## Layer 1: Single-Turn Golden Dataset (205 samples)

### Core design principle: reverse generation (no data leakage)

Standard approach (leaky):
```
query → retriever finds chunks → LLM judges relevance → ground truth
  ↑                                                        ↓
  └──────────── evaluated with the same retriever ─────────┘  ⚠️ circular
```

Our approach (unbiased):
```
v2 chunk (stratified sample) → Claude Opus generates a query answerable by this chunk
                                → the chunk IS the ground truth (no retriever involved)
```

The ground truth is **independent of any retriever's output**, so it can fairly evaluate any system variant (v1 RAG, agent v1, agent v2, final DAG).

Additionally, the **question-generating model (Claude Opus) is different from the answer-generating model (Qwen 3.5 Flash)** — eliminating model self-bias.

### Coverage dimensions (6 orthogonal axes)

| Dimension | Values | How ensured |
|---|---|---|
| **Data source combos** | All 2^4 = 16 (including 0-source guardrails) | Explicit target counts per combo; rare combos intentionally oversampled |
| **Answer mode** | 5: lookup / descriptive / comparison / decision_support / diagnostic | Each ≥20 samples; Opus prompt specifies mode |
| **Guardrail types** | 9: OOD, empty evidence, impossible, evidence conflict (3 types), ambiguous, false premise, refusal | 25 total guardrail samples, ≥2 per type |
| **Document type** | handbook / policy / sustainability report / annual report / master plan | Stratified sampling from v2 chunks by `doc_type` |
| **Time window** | old (<2015) / mid (2015–2019) / recent (≥2020) / undated | Stratified by `publish_year` bucket |
| **Chunk properties** | Short/mid/long word count; table vs prose | Stratified by `is_table` and `word_count` |

### Sample size justification

Using binomial CI formula `CI = z × sqrt(p(1-p)/n)` with z=1.96 (95%):

| Stratum | Target CI | Required n | Actual n |
|---|---|---|---|
| Small (graph-only, rule-only) | ±25% | ≥15 | 15–20 |
| Medium (vector, sql) | ±15% | ≥40 | 30–66 |
| Overall (Micro-F1) | ±8% | ≥150 | 205 |

205 is the minimum that satisfies all stratum constraints simultaneously.

### Sampling strategy per source type

| Source | Sampling method | Ground truth |
|---|---|---|
| **Vector** | Stratified from 16,124 v2 child chunks by (doc_type × year_bucket × is_table) → 18 buckets → proportional allocation | The sampled chunk itself |
| **SQL** | 15 (table, column, aggregation) templates run against DuckDB → actual query results as golden numbers | Real SQL output (e.g., `AVG(wind_speed_ms)=5.14`) |
| **Rules** | Top-20 from grounded_rules.json ranked by completeness (has variable + operator + threshold + sql_variable) | The rule's text, variable, threshold |
| **Graph** | Random 15 TRIGGERS edges from Neo4j v2 with rule_text annotations | Edge path + rule citation |
| **Multi-source** | 11 combo types × N each, anchor chunk randomly selected, Opus generates a query requiring ALL listed sources | Union of per-source golden fields |

### Generation pipeline

```
1. Stratified sampling → scaffold files (per source type)
2. Claude Opus subagents (parallel, one per source type):
   - Input: sampled chunk/rule/edge/SQL template + target answer_mode + target difficulty
   - Output: query, expected_evidence_keywords, reference_answer, golden per-source fields
3. Merge + deduplicate + validate (all expected_sources covered, no trivial queries)
4. Output: golden_dataset_v3_rag.json (205 samples)
```

**Script**: `evaluation/build_golden_v3_rag.py`

---

## Layer 2: Multi-Turn Single-Session Dataset (10 conv / 31 turns)

### Purpose

Tests **short-term memory** capabilities that single-turn evaluation cannot measure:
- Co-reference resolution ("And the rules for that?")
- Entity tracking across turns (berth B3 mentioned in turn 1, referenced as "it" in turn 3)
- Topic-switch detection (turn 4 pivots to unrelated topic — memory must NOT carry old context)
- Long-range summarisation (6+ turns trigger automatic compression)
- Guardrail persistence (OOD query mid-conversation, then recovery)

### Construction method: derived from the 205 base samples

Every turn carries `derived_from_sample_id` linking back to a specific base sample. This means:
- The turn inherits all golden fields (`expected_sources`, `needs_*`, `answer_mode`, `expected_evidence_keywords`, `reference_answer`, `golden_vector/sql/rules/graph`)
- **No new ground truth needs to be created** — evaluation reuses the single-turn scoring modules
- Source/mode/guardrail coverage is inherited from the 205-sample design

### Conversation design: 6 patterns

Each pattern targets a specific memory capability:

| Pattern | Count | Turns | What it tests |
|---|---|---|---|
| `entity_anchored` | 2 | 3 | Same entity across turns; pronoun resolution |
| `mode_progression` | 2 | 3 | lookup → comparison → decision_support mode escalation |
| `cross_source_verification` | 2 | 2–3 | SQL fact → rules check → vector explanation |
| `topic_switch` | 2 | 2–3 | **Negative test**: memory must NOT carry old context |
| `long_summarisation` | 1 | 6 | Triggers `_summarise_oldest_half` + key_facts extraction |
| `guardrail_in_conversation` | 1 | 3 | OOD mid-chat → next turn must recover |

### Follow-up turn annotation

Turns after turn 1 carry additional fields for memory-specific evaluation:
- `rephrase_as`: the follow-up phrasing (e.g., "And what are the policy rules around that?")
- `expected_resolved_query_contains`: keywords the co-ref resolver should inject (e.g., `["tide", "rule"]`)
- `expected_resolved_query_should_not_contain`: keywords that should NOT appear after a topic switch
- `expected_memory_recall`: `{from_turn: 1, key_fact: "tide"}` — what the memory should remember
- `evaluation_focus`: tag for the specific capability being tested

### Cross-source within a conversation

A single conversation deliberately mixes different data sources across turns:

```
Example (MT3_001 entity_anchored):
  T1: SQL / lookup     — "What was the average tide level in 2016?"
  T2: rules / decision — "And what are the policy rules around that?"
  T3: graph / diagnostic — "If that threshold is breached, which operations are affected?"
```

This mirrors how real port managers investigate a topic: start with data, check policy, then trace impact.

### Script and generation

Templates are hand-written in `evaluation/build_multi_turn_v3.py` as a `CONVERSATIONS` list. Each template specifies `from_sample_id` references + optional `rephrase_as` overrides. Running the script materialises the templates into `golden_dataset_v3_multi_turn.json`.

**Why hand-written, not LLM-generated**: the multi-turn dataset tests *structural* memory capabilities (co-reference, topic switch, summarisation trigger). These need precise control over turn sequence and evaluation_focus tags. LLM generation would add noise without adding value.

---

## Layer 3: Cross-Session Dataset (35 conv / 70 turns)

### Purpose

Tests **long-term memory** — the ability to recall facts from a previous session (different `session_id`) in a new session. This is the capability that neither Layer 1 nor Layer 2 can evaluate.

### Construction method: Claude subagent from real data

Unlike Layer 2 (hand-written templates), Layer 3 session-2 queries are **generated by Claude subagent** reading actual sample data. Reason: cross-session recall requires **diverse, natural-language queries** that don't share sentence templates — otherwise BGE embeddings cluster and the experiment cannot discriminate between keyword and vector retrieval.

**Process:**
1. Claude subagent reads `golden_dataset_v3_rag.json` (205 samples)
2. Selects 25 diverse samples covering all 4 data sources × 5 answer modes
3. For each selected sample, reads the `query` + `reference_answer` to understand what facts session 1 would establish
4. Generates a session-2 query in **free-form natural language** — varying style across the dataset:
   - Precise recall: "What was the exact tide reading we found?"
   - Casual: "Any equipment issues we looked at before?"
   - Applied decision: "Given last week's wind data, should we halt crane ops?"
   - Comparative: "How does this compare to what we found before?"
   - Instruction: "Pull up the gate throughput numbers we reviewed"
5. Also generates 10 **negative** conversations (session 1 about topic A, session 2 about unrelated topic B)
6. Every session-2 query includes `expected_cross_session_hit: true/false` and `expected_memory_recall.key_fact`

**Constraint**: no two session-2 queries share the same sentence structure. This prevents embedding clustering.

### Session structure

```
For each conversation:
  Session 1 (session_order=1):
    - from_sample_id = "V3_SQL_001"  ← borrows the base sample's query
    - DAG runs real retrieval → real answer
    - end_session() writes summary + atomic key_facts to long-term DuckDB

  Session 2 (session_order=2, fresh session_id):
    - raw_query = "What was the sea-level measurement we pulled last time?"  ← Claude-generated
    - build_context() → long_term.retrieve() → should find session 1's content
    - Scoring: did the retrieved session_ids include session 1's sid?
```

Session 1 uses `from_sample_id` only to inherit the query text and golden fields. **The actual memory content is created at runtime** — the DAG runs, produces an answer, and `end_session` writes it to the DuckDB long-term store.

### Coverage

| Dimension | Values | Count |
|---|---|---|
| Data source | sql / vector / rules / graph / multi | 7 / 6 / 5 / 4 / 3 |
| Answer mode | lookup / comparison / decision_support / diagnostic / descriptive | 5 / 5 / 6 / 5 / 4 |
| Query style | precise / casual / applied / comparative / instruction / paraphrase | Distributed across 25 |
| Negative | topic_drift (truly unrelated topics) | 10 |
| **Total** | | **35 conv / 70 turns** |

### Evaluation metrics (cross-session specific)

| Metric | Definition |
|---|---|
| `cross_session_hit_rate` | Fraction of positive turns where `long_term.retrieve()` top-5 contains the correct prior session_id |
| `correct_session_recall_rate` | Among hits, fraction that matched the specific expected session |
| `cross_session_leak_rate` | Fraction of negative turns that wrongly returned prior-session content |
| `score_gap (pos − neg)` | Avg top-score for positive turns minus avg top-score for negative turns (discrimination power) |

### Iteration history: why v3 → v4

| Version | Conversations | hit_rate (Phase B) | Problem |
|---|---|---|---|
| v3 (5 conv) | 5 | 75% | Small sample; optimistic |
| v3 (31 conv) | 31 | **15%** | Template-style queries → embeddings clustered; narrative summaries indistinguishable |
| **v4 (35 conv)** | 35 | **76%** | Claude-generated diverse queries + conclusion-oriented summaries + atomic key_facts |

The 15% → 76% improvement came from fixing **both the evaluation data AND the system**:
- **Data side**: diverse query styles prevent embedding clustering
- **System side**: atomic key_facts as separate long-term entries + conclusion-oriented summary prompt

### Script

- **Generator**: `evaluation/build_cross_session_v4.py` (Claude subagent output)
- **Evaluation driver**: `evaluation/run_cross_session_evaluation.py`
- **Output**: `evaluation/golden_dataset_v4_cross_session.json`

---

## Latency Evaluation & Instrumentation

### Pipeline-level timing (built into the DAG)

Every DAG node is wrapped by `NodeFactory._timed()` (`langgraph_nodes.py:61–76`):

```python
def _timed(self, node_name, func, state):
    t0 = time.time()
    result = func(state)
    elapsed = round(time.time() - t0, 4)
    result["_node_timings"][node_name] = elapsed   # injected into LangGraph state
    return result
```

All 9 nodes are instrumented: `route_query`, `planner` (with sub-stages `planner__query_rewrite`, `planner__plan_total`, `planner__sub_queries__llm_call`), `retrieve_documents`, `rerank_documents`, `retrieve_rules`, `run_sql`, `run_graph_reasoner`, `merge_evidence`, `synthesize_answer`.

The timing dict flows through the LangGraph state and is extracted by the eval driver in `per_sample_results[i]["stage_timings"]`.

### Latency evaluation module (`eval_latency.py`)

Computes **p50 / p95 / p99 / mean / max** for each stage across all samples:

```
Stage                   mean    p50     p95     p99     max   (seconds)
-----------------------------------------------------------------------
route_query            21.46   21.49   28.08   29.62   30.01
planner                27.89   28.70   42.81   44.57   45.01
retrieve_documents      0.55    0.38    1.27    1.54    1.61
rerank_documents        0.48    0.51    0.59    0.61    0.61
run_sql                 6.88    6.88    6.88    6.88    6.88
merge_evidence          0.00    0.00    0.00    0.00    0.00
synthesize_answer      40.96   40.05   53.43   55.61   56.15
end_to_end             91.72   92.11  121.29  126.74  128.10
```

(Numbers from 10-sample smoke on the current DAG. Full 205-sample v3 report: e2e p50 = 72s.)

Also tracks:
- **Iteration distribution**: `{1: N}` (DAG always 1 iteration, no replan)
- **Re-plan trigger rate**: 0% for DAG (vs 66% for the retired ReAct agent)
- **ReAct observation stats**: 0 for DAG (validates no unintended agent behaviour)

### Memory-specific latency

Two wall-clock measurements per turn in `run_multi_turn_evaluation.py` and `run_cross_session_evaluation.py`:

| Operation | What it measures | Where timed | Typical value |
|---|---|---|---|
| `resolve_followup_ms` | LLM call to rewrite follow-up queries into standalone form | `memory_manager.resolve_followup()` | ~1.7–3.3s (LLM-bound) |
| `build_context_ms` | Short-term format + long-term retrieve (BGE embed_query + DuckDB vss) | `memory_manager.build_context()` | ~4ms (no embedder) / ~140ms (with BGE) |

These are aggregated in `eval_memory.py` as `avg_resolve_followup_ms` and `avg_build_context_ms` across all turns.

### Latency comparison: v1 Agent → v2 Agent → v3 DAG

From `AGENT_FINAL_COMPARISON.md` and `rag_v3_n205.json`:

| Stage | v1 Agent | v2 Agent | **v3 DAG (current)** |
|---|---|---|---|
| End-to-end p50 | 117.8s | 61.8s | **72s** |
| End-to-end p95 | 253.1s | 91.9s | **121s** |
| Re-plan rate | 66% | 10% | **0%** |

v3 DAG is slightly slower than v2 Agent's p50 (72 vs 62s) because the DAG always runs all routed sources in parallel (no early-stop), but **p95 is much more predictable** (no replan tail).

### Latency overhead of memory

Measured on the v4 cross-session dataset (35 conversations / 70 turns):

| Config | resolve_followup | build_context | Total overhead per turn |
|---|---|---|---|
| Phase A (no embedder) | 1.9s | 11ms | **~2s** |
| Phase B (BGE + vss) | 1.8s | 143ms | **~2s** |

The `resolve_followup` LLM call dominates (~95% of overhead). `build_context` is negligible even with BGE embedding. **Total memory overhead adds ~2s per turn** on top of the 72s DAG latency — a 2.8% increase.

### Where latency data lives

| File | Content |
|---|---|
| `evaluation/agent/eval_latency.py` | Percentile computation + stage breakdown |
| `evaluation/agent/eval_memory.py` | Memory-specific latency aggregation |
| `evaluation/agent/reports/rag_v3_n205.json` → `single_turn.latency` | Full 205-sample per-stage stats |
| `evaluation/agent/reports/rag_v3_n10_post_cleanup_smoke.json` | 10-sample smoke after memory integration (regression test) |
| `evaluation/agent/reports/rag_v4_cross_session_phase_*.json` | Memory latency per turn in cross-session runs |
| `src/online_pipeline/langgraph_nodes.py:55–76` | Instrumentation source code |

---

## Summary: Design Philosophy Across All Three Layers

| Principle | Layer 1 (single-turn) | Layer 2 (multi-turn) | Layer 3 (cross-session) |
|---|---|---|---|
| **Ground truth source** | Reverse generation from sampled chunks | Inherited from Layer 1 via `derived_from_sample_id` | Session 1 runs real DAG; session 2 checks recall |
| **Question generator** | Claude Opus (different from answer model) | Hand-written templates | Claude subagent (diverse NL) |
| **Why that method** | Avoids data leakage; unbiased | Structural memory tests need precise turn control | Cross-session recall needs diverse vocabulary to avoid embedding clustering |
| **Coverage guarantee** | 2^4 source combos × 5 modes × 9 guardrails | 6 memory-stress patterns | 4 sources × 5 modes × 6 query styles + 10 negatives |
| **Anti-pattern avoided** | Circular: retriever generates own gold | Monotone templates → embedding clustering | Same as Layer 2 (v3 failed, v4 fixed) |
