# Evaluation Directory

Contains the evaluation framework for both the legacy RAG-DAG system and the new Plan-Execute Agent.

## Directory Structure

```
evaluation/
├── README.md                                # This file
├── golden_dataset.json                      # Shared: 101 base test samples
├── golden_dataset_v3_extras.json            # Shared: 12 guardrails + 5 multi-turn + 3 gap-fills
│
├── agent/                                   # Plan-Execute Agent evaluation
│   ├── run_full_evaluation.py              # Unified driver
│   ├── compare_agent_v1_v2.py              # v1/v2 agent comparison
│   ├── eval_routing.py                     # Multi-label routing P/R/F1
│   ├── eval_retrieval.py                   # Per-source IR metrics + rerank lift
│   ├── eval_answer_quality.py              # Objective + LLM-judge
│   ├── eval_multi_turn.py                  # Resolution / coherence / memory
│   ├── eval_guardrails.py                  # OOD / conflict / ambiguous / false premise
│   ├── eval_latency.py                     # Per-stage p50/p95/p99, iteration dist, ReAct
│   ├── AGENT_v1_BASELINE_REPORT.md         # v1 agent baseline write-up
│   ├── AGENT_v2_FINAL_REPORT.md            # v2 agent final write-up
│   └── reports/
│       ├── agent_v1_n114_baseline.json     # v1 agent, 114 samples (baseline)
│       ├── agent_v1_n20_buggy.json         # intermediate: chunk_id bug era
│       ├── agent_v1_n20_chunkid_fixed.json # intermediate: after chunk_id fix
│       ├── agent_v1_n3_smoke_early.json    # early smoke test
│       ├── agent_v2_n10_smoke.json         # v2 smoke (10 samples)
│       ├── agent_v2_n30_intermediate.json  # v2 intermediate (30 samples)
│       └── agent_v2_n115_full.json         # v2 FULL (115 samples)
│
└── rag_legacy/                              # Pre-agent DAG-based RAG evaluation
    ├── run_evaluation.py                    # Old DAG runner
    ├── rebuild_golden_dataset.py           # Dataset builders
    ├── expand_golden_v2.py
    ├── expand_golden_v3.py
    ├── build_golden_v2.py
    ├── expand_rules.py
    ├── annotate_relevant_chunks.py
    ├── train_intent_classifier.py          # Legacy MLP classifier training
    ├── streaming_benchmark.py              # Streaming output benchmarks
    ├── streaming_benchmark.log
    ├── streaming_benchmark_results.json
    ├── augmented_intent_data.json          # Training data for intent classifier
    ├── dashboard.html                      # Old HTML dashboard
    ├── CHANGELOG.md                        # Pre-agent changelog
    ├── EVALUATION_REPORT.md                # Pre-agent report
    ├── R5_postmortem.md                    # R5 run postmortem
    ├── reports/
    │   ├── rag_baseline.json               # DAG baseline report
    │   └── rag_r5b.json                    # R5b run report
    └── logs/                                # All old .log files
        ├── eval_20q.log
        ├── eval_ablation_output.log
        ├── eval_full_output.log
        ├── eval_full_v2.log
        ├── eval_output.log
        ├── eval_v3_qwen35b.log
        ├── eval_v4_full.log
        ├── eval_v5_final.log
        ├── eval_v5b_final.log
        ├── eval_v6_r4dict.log
        ├── evaluation_run.log
        └── pipeline_debug.log
```

## Two Evaluation Eras

### 1. Legacy RAG-DAG (`rag_legacy/`)

The **pre-agent era** — a fixed LangGraph DAG workflow
(`src/online_pipeline/langgraph_workflow.py`) running a hardcoded
pipeline of `route_query → planner → retrievers → merge → synthesize`.

- No agent behavior (no planning, no re-plan, no tool selection)
- Manual rule/graph curation
- Intent classification via MLP + rule-based fallback
- Historical reports in `rag_legacy/reports/`

### 2. Plan-Execute Agent (`agent/`)

The **current agent era** — a true Plan-Execute LangGraph agent
(`src/online_pipeline/agent_graph.py`) with:

- LLM-driven planning with strict tool selection
- ReAct observation loop within `execute_tools_node`
- Small-to-Big parent-child retrieval
- BGE-base embeddings + auto-taxonomy + rule-driven knowledge graph
- OOD gate (fast-path + LLM fallback)
- Multi-turn conversation with session memory
- Enhanced conflict detection (Rule↔SQL + Doc↔SQL + Doc↔Rule + temporal)

Reports are in `agent/reports/` and tracked across two pipeline versions:

| Version | Data pipeline | Reports |
|---|---|---|
| **agent v1** | Original chunking (v1 chunks) + `all-MiniLM-L6-v2` embeddings | `agent_v1_n*.json` |
| **agent v2** | Small-to-Big chunks + `BGE-base-en-v1.5` + auto-taxonomy + v2 graph | `agent_v2_n*.json` |

## Running Evaluations

### Full v2 agent evaluation (recommended)

```bash
cd RAG-LLM-for-Ports-main
python evaluation/agent/run_full_evaluation.py
```

### Quick options

```bash
# Smoke test (3 samples, no LLM judge)
python evaluation/agent/run_full_evaluation.py --limit 3 --skip-llm-judge --skip-multi

# Single-turn only, no LLM judge
python evaluation/agent/run_full_evaluation.py --skip-llm-judge --skip-multi

# Specific number of samples
python evaluation/agent/run_full_evaluation.py --limit 30

# Custom output path
python evaluation/agent/run_full_evaluation.py --output evaluation/agent/reports/my_run.json
```

### Compare agent v1 vs v2

```bash
python evaluation/agent/compare_agent_v1_v2.py
```

### Legacy DAG evaluation (pre-agent baseline)

```bash
python evaluation/rag_legacy/run_evaluation.py
```

## Metric Dimensions (7)

Implemented in `agent/eval_*.py`:

1. **Routing** (`eval_routing.py`) — Multi-label capability classification
   - Per-capability P/R/F1, Micro/Macro F1
   - Exact-match rate, over/under-routing rates
   - question_type + answer_mode accuracy

2. **Retrieval** (`eval_retrieval.py`) — Per-source metrics
   - Vector: chunk_recall@5/20 + source_recall@5/20 (cross-format compat)
   - SQL: table_f1, execution_ok_rate, row_count_reasonable
   - Rules: variable_recall / variable_precision
   - Graph: entity_recall, relationship_recall, path_found_rate

3. **Rerank Lift** (`eval_retrieval.py → evaluate_reranking_lift`)
   - nDCG@5 lift, recall@5 lift, top-1 lift

4. **Answer Quality** (`eval_answer_quality.py`)
   - Objective: keyword coverage, citation validity, numerical accuracy, grounding distribution
   - LLM-as-Judge: faithfulness / relevance / completeness (1-5, capped to 20 samples)

5. **Multi-turn** (`eval_multi_turn.py`)
   - Query resolution quality
   - Entity tracking recall + persistence
   - LLM-judge coherence (consistency / context_use / reference_resolution / topic_handling)

6. **Guardrails** (`eval_guardrails.py`)
   - OOD refusal, empty evidence, evidence conflict (Rule↔SQL / Doc↔SQL / Doc↔Rule),
     ambiguous query, false premise, impossible query

7. **Latency** (`eval_latency.py`)
   - Per-stage p50/p95/p99/max/mean (ood_check, plan, execute_tools, evaluate, synthesize, end_to_end)
   - Iteration distribution + re-plan trigger rate
   - ReAct stats: observations count, abort rate, modify rate

## Latest Results Summary

See `agent/AGENT_v2_FINAL_REPORT.md` for the full write-up.

Key highlights (v1 n=114 → v2 n=115):
- Routing exact-match: **49% → 77%** (+28 pp)
- Over-routing: **47% → 11%** (-36 pp)
- Micro F1: **0.79 → 0.91**
- Citation validity: **69% → 100%**
- End-to-end p50 latency: **118s → 72s** (-39%)
- Re-plan rate: **66% → 10%**
- SQL table F1: **0.76 → 0.75** (maintained)
- Rules variable precision: **23% → 38%** (+15 pp, word-boundary fix)
