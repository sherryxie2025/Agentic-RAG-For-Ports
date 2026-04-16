# Evaluation

The evaluation framework for the **agentic-RAG DAG**
(`src/online_pipeline/langgraph_workflow.py`).

The earlier ReAct/Plan-Execute agent is retired and lives under
`legacy/react_agent/` along with its v1/v2 reports.

## Directory layout

```
evaluation/
├── README.md                                # this file
│
│  --- single-turn (the canonical 205-sample benchmark) ---
├── golden_dataset.json                      # original 101-sample base
├── golden_dataset_v3_rag.json               # 205 samples, full 2^4 source coverage,
│                                            # 5 answer modes, 9 guardrail types
├── golden_dataset_v3_extras.json            # legacy: kept for back-compat
├── build_golden_v3_rag.py                   # builds golden_dataset_v3_rag.json
├── merge_golden_v3.py                       # merges base + extras
├── dump_golden_scaffolds.py                 # dumps scaffold task templates
├── run_rag_evaluation.py                    # main single-turn runner (DAG, 205 samples)
├── rerun_contaminated.py / rescore_answer_quality.py / render_eval_markdown.py
│
│  --- multi-turn (NEW: built on top of the 205 samples) ---
├── golden_dataset_v3_multi_turn.json        # generated; lists 10 conversations / ~31 turns
├── build_multi_turn_v3.py                   # generator: composes conversations from base 205
├── run_multi_turn_evaluation.py             # multi-turn runner (DAG with memory)
│
│  --- shared per-metric modules (single + multi turn reuse them) ---
├── agent/
│   ├── eval_routing.py                      # multi-label routing P/R/F1
│   ├── eval_retrieval.py                    # per-source IR metrics + rerank lift
│   ├── eval_answer_quality.py               # objective + LLM-judge
│   ├── eval_guardrails.py                   # OOD / conflict / ambiguous / false premise
│   ├── eval_latency.py                      # per-stage p50/p95/p99
│   ├── eval_multi_turn.py                   # legacy (was used for the ReAct agent)
│   ├── eval_memory.py                       # NEW: industry-aligned memory metrics
│   ├── reports/
│   │   ├── rag_v3_n205.json                 # canonical single-turn V3 report
│   │   ├── rag_v3_multi_turn.json           # written by run_multi_turn_evaluation.py
│   │   └── ...                              # historical agent_v1/v2 reports
│   └── scaffolds/                           # per-source scaffold tasks/results
│
└── rag_legacy/                              # pre-V3 DAG reports + scripts (read-only)
```

## Single-turn evaluation (canonical)

```bash
cd RAG-LLM-for-Ports-main
python evaluation/run_rag_evaluation.py [--limit N] [--skip-llm-judge]
```

Default output: `evaluation/agent/reports/rag_v3_n205.json`.

## Multi-turn evaluation (NEW)

The dataset is composed *from* the same 205 single-turn samples — every
turn carries `derived_from_sample_id` plus the inherited golden
retrieval/answer fields, so a single conversation spans multiple source
combos and answer modes (one user can mix vector / SQL / rules / graph
across turns the way real users do).

### Patterns covered (10 conversations, ~31 turns)
| pattern | count | what it tests |
|---|---|---|
| entity_anchored | 2 | same berth/crane/vessel across turns; pronoun resolution |
| mode_progression | 2 | lookup → comparison → decision_support → diagnostic |
| cross_source_verification | 2 | SQL fact then policy/graph follow-up about it |
| topic_switch | 2 | turn N pivots; old context must NOT carry |
| long_summarisation | 1 | 6 turns to trigger short-term auto-summary |
| guardrail_in_conversation | 1 | OOD mid-chat; agent must recover next turn |

### How to extend the dataset
Edit `CONVERSATIONS` in `build_multi_turn_v3.py` — each conversation is a
hand-written list of `from_sample_id` references plus optional
`rephrase_as` follow-up text. Then:
```bash
python evaluation/build_multi_turn_v3.py
```

### How to run it
```bash
python evaluation/run_multi_turn_evaluation.py [--limit N] [--skip-llm-judge]
```

Default output: `evaluation/agent/reports/rag_v3_multi_turn.json`.

## Memory metrics (in `agent/eval_memory.py`)

Industry-aligned set, each tied to a published benchmark / library:

| Metric | Inspired by | What it measures |
|---|---|---|
| `coref_resolution_contains` | LangChain conv-eval, MT-Bench-Conv | After follow-up rewrite, did the standalone query include the expected referents? |
| `coref_resolution_exclusion` | LongChat / TopicSwitch | After a topic switch, did the rewrite NOT carry over old entities? |
| `memory_recall@k` | MemGPT, ChatRAG-Bench | Of the gold "must-recall" facts at turn N, how many appear in `memory_context`? |
| `memory_precision` (LLM judge) | RAGAS context_precision | Of items in `memory_context`, how many are actually relevant to the current turn? |
| `answer_faithfulness_to_mem` | RAGAS faithfulness | Are answer claims about earlier turns consistent with what was actually said? |
| `temporal_recall_decay` | LongMemEval | Recall@k bucketed by fact age — the forgetting curve. |
| `entity_persistence` | DialDoc, Multi-Doc QA | Fraction of prior-turn entities still in `active_entities` at turn N. |
| `topic_shift_detected_rate` | TIAGE, TopiOCQA | Heuristic: did the system NOT carry old entities when the gold says topic switched? |
| `cross_session_hit_rate` | LangChain VectorStoreRetrieverMemory | When a new session asks about a prior session's topic, does long-term retrieve it? |
| `context_token_overhead` | MemGPT efficiency table | char(memory_context) / char(base_query) — proxy for token cost. |
| `latency_overhead_ms` | Production observability | Wall time of `resolve_followup` + `build_context` per turn. |
```
