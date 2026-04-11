# Final 3-Way Evaluation Comparison

Compares all three major iterations of the Plan-Execute agent on the same 115-sample golden dataset.

| Version | Sample size | Description |
|---|---|---|
| **v1 baseline** | n=114 | Plan-Execute agent with v1 pipeline (all-MiniLM-L6-v2 + 400-char chunks + v1 hardcoded graph) + 20 known bugs |
| **v2 first run** | n=115 | After Small-to-Big + BGE + auto-taxonomy + rule-driven graph + OOD gate + strict plan + MAX_ITER=2 |
| **v2 all fixes** | n=115 | After 4 regression bug fixes (OOD phrases, graph bridge concepts, SQL auto-fallback, rule word-boundary) |

---

## 1. Routing

| Metric | v1 baseline | v2 first run | v2 all fixes | v2-all vs v1 |
|---|---|---|---|---|
| Exact-match rate | 49.12% | 77.19% | **75.44%** | **+26.3 pp** |
| Over-routing rate | 47.37% | 10.53% | **7.02%** | **-40.4 pp** |
| Under-routing rate | 2.63% | 12.28% | 15.79% | +13.2 pp ⚠️ |
| Micro-F1 | 0.793 | 0.907 | **0.893** | **+0.10** |
| Macro-F1 | 0.788 | 0.903 | 0.896 | +0.108 |

### Per-capability F1

| Capability | v1 | v2 first | v2 all | v2-all vs v1 |
|---|---|---|---|---|
| vector | 0.649 | 0.868 | **0.864** | **+0.22** |
| sql | 0.930 | 0.922 | 0.873 | -0.057 ⚠️ |
| rules | 0.786 | 0.935 | **0.933** | **+0.15** |
| graph | 0.786 | 0.889 | **0.913** | **+0.13** |

Notes: under-routing went up because the strict planner is missing SQL on some queries that would benefit from it. Per-capability SQL dropped because of that same reason.

---

## 2. Retrieval

### Vector
| Metric | v1 | v2 first | v2 all fixes |
|---|---|---|---|
| chunk_recall@5 | 6.86% | 0.00% | 0.00% (*chunk_id format mismatch*) |
| source_recall@5 | n/a | 50.00% | **47.22%** |
| source_recall@20 | n/a | 56.94% | **54.17%** |
| source_mrr | n/a | 0.424 | 0.404 |

### SQL (63 samples)
| Metric | v1 | v2 first | v2 all fixes |
|---|---|---|---|
| table_f1 | 0.758 | 0.751 | 0.678 ⚠️ |
| execution_ok_rate | 90.48% | 74.60% | **76.19%** (SQL fallback +1.6 pp) |
| row_count_reasonable | 6.3%* | 50%* | **75%** (metric fix applied) |

\* v1 and v2-first had a denominator bug

### Rules (44 samples)
| Metric | v1 | v2 first | v2 all fixes |
|---|---|---|---|
| variable_recall | 75.76% | 71.97% | 70.83% |
| variable_precision | **22.95%** | 38.26% | **36.52%** (word-boundary fix +13 pp) |

### Graph (23 samples) — **Biggest recovery**

| Metric | v1 | v2 first | v2 all fixes | v2-all vs v2-first |
|---|---|---|---|---|
| entity_recall | 61.59% | 46.16% | **56.67%** | **+10.5 pp** |
| relationship_recall | 60.51% | 9.42% | 9.42% | 0 pp (edge types differ in v2) |
| **path_found_rate** | 95.65% | 21.74% | **73.91%** | **+52.2 pp** (bridge concepts rescue) |

**Graph is the biggest fix win**: adding 17 bridge concept nodes recovered 52 pp of path discovery.

---

## 3. Rerank Lift

| Metric | v1 | v2 first | v2 all fixes |
|---|---|---|---|
| nDCG@5 lift | +0.087 | +0.000 | +0.000 |
| top-1 before | 6.25% | 0.00% | 0.00% |
| top-1 after | 25.00% | 0.00% | 0.00% |
| top-1 lift | +0.188 | +0.000 | +0.000 |

Rerank lift is zero in v2 because `retrieved_chunk_ids` come from post-Small-to-Big parent chunks (which don't match golden chunk_ids in format). The reranker is still running (verified in manual tests) but the golden-based lift metric can't detect improvement on v2.

---

## 4. Answer Quality

| Metric | v1 baseline | v2 first | v2 all fixes |
|---|---|---|---|
| **Citation validity** | 69.35% | **100%** | **100%** |
| Keyword coverage | 78.54% | 71.17% | 71.80% |
| Numerical accuracy | 79.02% | 73.28% | 70.40% |
| Grounding: fully | 60.9% | **95.7%** | **94.8%** |
| Grounding: partially | 38.3% | 0.0% | 0.0% |
| Grounding: llm_fallback | 0.9% | 0.9% | 0.9% |
| Grounding: refused_ood | n/a | 3.5% | 3.5% |

**Citation validity 100% maintained.** Grounding threshold fix (>=1 source) correctly classifies most answers as fully grounded.

---

## 5. Guardrails ⭐ Major Improvement

| Type | v1 baseline | v2 first | **v2 all fixes** | v2-all vs v1 |
|---|---|---|---|---|
| **out_of_domain** | **0.00%** | 0.00% | **100.00%** | **+100 pp** ✅ |
| **impossible_query** | 100.00% | 0.00% | **100.00%** | 0 pp (recovered) ✅ |
| **doc_vs_rule_conflict** | 0.00% | 100.00% | **100.00%** | +100 pp ✅ |
| empty_evidence | 100.00% | 66.67% | 66.67% | -33 pp |
| evidence_conflict | 100.00% | 0.00% | 0.00% | -100 pp ⚠️ |
| doc_vs_sql_conflict | 100.00% | 0.00% | 0.00% | -100 pp ⚠️ |
| ambiguous_query | 0.00% | 0.00% | 0.00% | 0 pp |
| false_premise | 0.00% | 0.00% | 0.00% | 0 pp |

### Key wins
- **OOD refusal now works 100%**: all 3 OOD queries (pizza recipe, cat joke, current time) correctly refused with phrase match
- **impossible_query recovered**: "Show me 2050 wind records" correctly identified as future/impossible
- **doc_vs_rule_conflict works**: temporal version drift between documents and rules detected

### Still broken
- **evidence_conflict / doc_vs_sql_conflict**: conflict_detector is populated but the answer keywords don't include `_CONFLICT_PHRASES` — the synthesizer doesn't surface the conflict strongly enough in the natural language answer
- **ambiguous / false_premise**: require the agent to push back or ask for clarification, not implemented

---

## 6. Latency

### Per-stage (seconds)

| Stage | v1 p50 | v1 p95 | v2 first p50 | v2 first p95 | **v2 all p50** | **v2 all p95** |
|---|---|---|---|---|---|---|
| ood_check_node | n/a | n/a | n/a* | n/a* | *<fast-path>* | *<fast-path>* |
| plan_node | 12.0 | 19.5 | 4.0 | 9.0 | **3.7** | **7.1** |
| execute_tools_node | 32.2 | 165.2 | 4.7 | 46.3 | **4.9** | 35.1 |
| evaluate_evidence_node | 29.5 | 68.3 | 27.2 | 30.0 | **25.1** | 30.0 |
| synthesize_node | 32.9 | 48.8 | 32.9 | 58.8 | 31.7 | 49.4 |
| **end_to_end** | **117.8** | **253.1** | **71.7** | **134.5** | **69.4** | **140.6** |

### Iterations & ReAct

| Metric | v1 | v2 first | v2 all fixes |
|---|---|---|---|
| Iteration = 1 | 33.9% | 89.6% | **93.0%** |
| Iteration = 2 | 9.6% | 10.4% | 7.0% |
| Iteration = 3 | 56.5% | 0% | 0% |
| **Re-plan rate** | **66.09%** | **10.43%** | **6.96%** |
| ReAct observations | 377 | n/a | 62 |
| ReAct abort-replan | 2.92% | n/a | **6.45%** |
| ReAct modify-next | 2.92% | n/a | 0.00% |

**End-to-end p50: 118s → 69s** (-41% vs v1 baseline). 93% of queries complete in a single iteration now.

---

## 7. Known Caveats

1. **LLM API timeouts still present**: Some samples still hit 30s timeout on `evaluate_evidence_node` and `classify_query`. Fallback paths activate (default to sufficient, default to in_domain), so no crashes. Observed in ~10 samples.
2. **SQL table_f1 dropped**: 0.751 → 0.678. Some queries where SQL was chosen dropped tables in the generated SQL. The rule-based fallback rescues execution_ok but the table list is also rewritten, losing expected table alignment.
3. **Rerank lift invisible**: Golden dataset uses v1 chunk_ids. Need to use `source_recall` instead of `chunk_recall` for v2 retrievals.
4. **Conflict guardrails still fail**: conflict_detector runs correctly (confirmed in direct testing) and populates `conflict_annotations`, but the synthesizer doesn't emit `_CONFLICT_PHRASES` keywords in the answer. Needs a synthesizer prompt tweak.
5. **One sample failed** (`GUARD_EMPTY_002`) due to `_pick_fallback_tool` being called as `self._pick_fallback_tool` (module function, not a method). Fixed post-run.

---

## 8. Summary

### Top 3 wins (v2 all fixes vs v1 baseline)

1. **Routing over-routing: 47% → 7%** (-40 pp) — strict plan prompt
2. **Guardrails OOD: 0% → 100%** — fast-path OOD + refusal phrase fix
3. **End-to-end p50 latency: 118s → 69s** (-41%) — reduced replan + lenient evaluator

### Top 3 losses or unresolved

1. **SQL execution_ok: 91% → 76%** — DuckDB execution errors on LLM-generated SQL; fallback partially helps
2. **Under-routing: 3% → 16%** — strict planner is sometimes TOO conservative, missing needed tools
3. **Conflict guardrails** (evidence_conflict, doc_vs_sql): detector works, synthesizer wording doesn't pass eval keyword match

### Commits in this session

- **8764bb9** → 4 regression bug fixes (graph bridge, SQL fallback, OOD phrases, word boundary)
- **4adc2d8** → evaluation folder reorganization (agent/ vs rag_legacy/)
- **bcpvozsyk** run → this comparison report

### Next steps if we were to continue

1. Fix the synthesizer to explicitly mention conflicts when `conflict_annotations` non-empty
2. Re-annotate golden dataset with v2 chunk_ids for proper rerank_lift measurement
3. Tune strict plan prompt to reduce under-routing (add "when in doubt, include sql_query for metric questions")
4. Fix SQL agent: pre-validate LLM SQL against DuckDB query planner before executing
