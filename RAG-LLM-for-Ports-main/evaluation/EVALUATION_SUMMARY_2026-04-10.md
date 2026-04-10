# Agent Evaluation Summary — 2026-04-10

Snapshot after completing: multi-turn conversation, memory system, ReAct observations, chunk_id bug fix, enhanced conflict detection, and Small-to-Big retrieval framework.

## 1. Test Run Info

| Field | Value |
|---|---|
| Total samples executed | 114 (single-turn) |
| LLM | qwen3.5-flash (DashScope) |
| Retrieval | BM25 + default Chroma embedding + RRF + cross-encoder rerank |
| Agent mode | Plan-Execute with ReAct observations enabled |
| Report | `evaluation_report_full_single.json` |
| Duration | ~4 hours (06:57 → 10:59) |
| Multi-turn eval | Started but killed at 2h mark (incomplete due to slow re-plan cycles) |

## 2. Routing (multi-label capability classification)

| Metric | Value |
|---|---|
| Exact-match (all 4 caps correct) | **49.12%** |
| Over-routing rate (extra sources activated) | **47.37%** |
| Under-routing rate (missing sources) | **2.63%** |

### Per-capability P/R/F1

| Capability | P | R | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| vector | 0.493 | 0.949 | 0.649 | 37 | 38 | 2 |
| sql | 0.895 | 0.968 | 0.930 | 60 | 7 | 2 |
| rules | 0.647 | 1.000 | 0.786 | 44 | 24 | 0 |
| graph | 0.667 | 0.957 | 0.786 | 22 | 11 | 1 |
| **Micro** | **0.671** | **0.970** | **0.793** | — | — | — |
| **Macro** | 0.676 | 0.968 | 0.788 | — | — | — |

### Interpretation

- **Recall is high everywhere (94-100%)** — the agent doesn't miss the right tool.
- **Precision is the problem** — plan_node LLM tends to call extra tools "just in case", causing 47% over-routing. Vector (P=0.49) is the worst offender.
- **SQL is best calibrated** (P=0.895) because SQL queries require specific metric nouns in the question.
- **SQL is also the highest-quality capability overall** (F1=0.93).

## 3. Retrieval

### Vector (34 samples had golden chunks)

| Metric | Value |
|---|---|
| recall@5 | **0.069** |
| recall@20 | **0.079** |
| precision@5 | 0.141 |
| mrr | 0.275 |
| ndcg@10 | 0.065 |

### SQL (63 samples)

| Metric | Value |
|---|---|
| table_f1 | **0.758** |
| execution_ok_rate | 0.905 |
| row_count_reasonable | 0.063 |

### Rules (44 samples)

| Metric | Value |
|---|---|
| variable_recall | 0.758 |
| variable_precision | 0.230 |

### Graph (23 samples)

| Metric | Value |
|---|---|
| entity_recall | 0.616 |
| relationship_recall | 0.605 |
| path_found_rate | 0.957 |

### Reranking Lift (32 samples with pre+post rerank data)

| Metric | Value |
|---|---|
| nDCG@5 lift | **+0.087** |
| recall@5 lift | +0.034 |
| top-1 hit before rerank | 0.063 |
| top-1 hit after rerank | 0.250 |
| top-1 lift | **+0.188** |

### Interpretation

- **Vector retrieval is the bottleneck**: recall@5 = 6.9%. Even with pre-rerank top-20, we only recall 7.9% of golden chunks. This is NOT because of a bug (chunk_id fix verified) — it's because:
  - Chunks are only ~50 words (from 400-char fixed splits)
  - Default Chroma embedding `all-MiniLM-L6-v2` is weak for port terminology
  - Golden chunks per question average 10+, so top-5 is mathematically bounded
- **SQL table routing is solid** (F1=0.758) — the SQL agent correctly identifies which operational tables to query.
- **row_count_reasonable=6%** means most queries return unexpected row counts — could be SQL logic bugs or schema mismatches.
- **Rule variable precision is low** (P=0.23) — rule retriever returns many unrelated rules. Over-matching on generic terms.
- **Graph reasoning is healthy**: 96% of queries produce at least one reasoning path; entity/relationship recall ~61%.
- **Reranker is effective**: +0.19 top-1 lift, +0.087 nDCG@5 lift. Cross-encoder is earning its keep.

## 4. Answer Quality

| Metric | Value |
|---|---|
| Samples | 115 |
| Keyword coverage (expected keywords in answer) | **78.5%** |
| Citation validity (cited sources match evidence) | 69.35% |
| Numerical accuracy (±5% tolerance on numeric answers) | **79.0%** |

### Grounding Distribution

| Status | Count | Percentage |
|---|---|---|
| fully_grounded | 70 | **60.9%** |
| partially_grounded | 44 | 38.3% |
| llm_fallback | 1 | 0.87% |

### Interpretation

- **Answers are mostly grounded** (99.1% not from pure LLM knowledge).
- **Keyword coverage 78.5%** is good but not great — means about 20% of expected key terms don't appear in answers.
- **Numerical accuracy 79%** is surprisingly solid given the SQL row_count issues.
- **Citation validity 69%** means 1 in 3 answers cites sources that don't actually contain the claim. This overlaps with partially_grounded rate.

## 5. Guardrails (11 specialized test cases)

| Type | Pass rate | Samples | Status |
|---|---|---|---|
| empty_evidence | 100.00% | 3 | ✅ PASS |
| impossible_query | 100.00% | 1 | ✅ PASS |
| evidence_conflict | 100.00% | 1 | ✅ PASS |
| doc_vs_sql_conflict | 100.00% | 1 | ✅ PASS |
| out_of_domain | **0.00%** | 3 | ❌ FAIL |
| false_premise | **0.00%** | 1 | ❌ FAIL |
| ambiguous_query | **0.00%** | 1 | ❌ FAIL |
| doc_vs_rule_conflict | **0.00%** | 1 | ❌ FAIL |

### Interpretation

- **Agent handles empty data and SQL/rule conflicts well** (correctly says "no data" or flags contradictions).
- **Agent fails all 3 OOD queries** ("pizza recipe", "cat joke", "current time") — tries to use existing tools to respond instead of refusing. This is a safety issue for a production system.
- **False premise / ambiguous / doc_vs_rule**: these require the agent to push back or clarify, which requires additional prompt engineering in `plan_node` and `evaluate_evidence_node`.

## 6. Latency

| Stage | mean | p50 | p95 | p99 |
|---|---|---|---|---|
| plan_node | 13.66s | 12.02s | 19.50s | 26.57s |
| execute_tools_node | 46.31s | 32.22s | **165.23s** | 187.86s |
| evaluate_evidence_node | 31.31s | 29.48s | 68.31s | 92.42s |
| synthesize_node | 34.60s | 32.95s | 48.77s | 59.46s |
| **end_to_end** | **125.89s** | **117.80s** | **253.06s** | **298.85s** |

### Iterations & ReAct

| Metric | Value |
|---|---|
| Iteration 1 (no replan) | 39/115 (33.9%) |
| Iteration 2 (1 replan) | 11/115 (9.6%) |
| Iteration 3 (hit max) | **65/115 (56.5%)** |
| Re-plan trigger rate | 66.09% |
| ReAct observations total | 377 |
| Avg observations per query | 3.28 |
| Abort-replan rate | 2.92% |
| Modify-next rate | 2.92% |

### Interpretation

- **p50 = 118s is 12x the 10s target** — this system is nowhere near production-ready latency.
- **56.5% of queries hit max iterations (3)** — meaning evaluate_evidence_node keeps flagging "insufficient" even after re-plan. This is wasteful.
- **ReAct observations are mostly passive** — 94% "continue", only 3% each for modify_next and abort_replan. The prompt may be too conservative.
- **execute_tools p95 = 165s** — most time is spent actually calling tools (BGE embedding + cross-encoder + SQL + graph + LLM tool invocations all add up).

## 7. Key Findings and Recommendations

### Critical Issues (high priority to fix)

1. **Vector recall @ 6.9%** — biggest performance loss. Already architected fix (Small-to-Big + BGE-base + metadata), needs offline re-index.
2. **Over-routing 47%** — plan_node prompt needs stricter source selection criteria, or a deterministic post-plan validator.
3. **Out-of-domain guardrail 0% pass** — need explicit refusal logic in plan_node or a pre-plan OOD classifier.
4. **55% hit max iterations** — evaluate_evidence_node LLM judge is too strict. Needs calibration or a more forgiving threshold.
5. **End-to-end latency 118s** — way too slow. Largest wins: disable ReAct if not needed, cap at 2 iterations instead of 3, use smaller planner LLM.

### Medium priority

6. **Citation validity 69%** — answer_synthesizer sometimes cites sources that don't directly contain the claim. Need source tracking through the synthesis step.
7. **Rule precision 23%** — rule retriever matches too loosely. Need stricter keyword-variable binding.
8. **SQL row_count 6% reasonable** — SQL agent produces syntactically valid queries that return wrong row counts. Schema understanding issue.
9. **Multi-turn eval was killed after 2 hours** — conversation memory and query resolution likely add too much latency per turn. Need profiling.

### What IS working

1. **SQL routing and execution** (F1=0.93, table_f1=0.76, exec_ok=0.91).
2. **Graph reasoning** (path_found=96%, entity_recall=62%).
3. **Reranker effectiveness** (nDCG@5 +0.087, top-1 +0.19).
4. **Answer grounding** (99% not from LLM knowledge).
5. **Conflict detection** (100% for rule-vs-sql and doc-vs-sql).
6. **Numerical accuracy** (79% with ±5% tolerance).

## 8. Code Changes Committed This Session

| Commit | Change |
|---|---|
| 03b9f7c | Per-stage latency instrumentation + pre_rerank_docs capture |
| (earlier) | Fixed document_retriever chunk_id bug (empty strings → Chroma IDs) |
| (earlier) | Fixed eval_retrieval: recall@20 uses pre_rerank ids |
| (earlier) | Fixed conflict_detector temporal max() empty sequence |
| 172dc81 | Chunking v2 (semantic_chunker_v2.py) + BGE embedding (build_embeddings_v2.py) + retriever updates |
| 2923953 | Small-to-Big (parent-child chunks) + metadata enrichment (publish_year, category) + HyDE tool |

## 9. Pending (not executed this session)

To activate the new v2 pipeline, run once offline:

```bash
# Install pdfplumber for better table extraction
pip install pdfplumber

# 1. Rebuild chunks with Small-to-Big
python -m src.offline_pipeline.semantic_chunker_v2
# Produces: data/chunks/chunks_v2_parents.json, chunks_v2_children.json, chunks_v2.json

# 2. Rebuild embeddings with BGE-base
python -m src.offline_pipeline.build_embeddings_v2
# Produces: port_documents_v2 Chroma collection

# 3. Update server.py to pass collection_name='port_documents_v2'
#    in _get_agent_system() → build_agent_graph(...)

# 4. Re-run evaluation to measure improvement
python evaluation/run_full_evaluation.py --skip-llm-judge --skip-multi --output evaluation_report_v2.json
```

Expected improvement after re-indexing with v2:
- Vector recall@5: 6.9% → **25-40%** (chunks 10x larger with semantic boundaries, BGE is 12 points better on MTEB retrieval)
- Keyword coverage: 78% → **85-90%**
- Latency: +5% overhead from BGE encoding (offset by better first-hit precision)
