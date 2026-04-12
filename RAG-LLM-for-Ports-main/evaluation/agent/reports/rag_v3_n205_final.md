# RAG 评测报告 — Agentic RAG LangGraph DAG

- 生成时间: 2026-04-12T05:29:40.873962
- 数据管线: v2 (Small-to-Big + BGE + auto-taxonomy + rule-driven graph)
- Golden 数据集: golden_dataset_v3_rag.json (Opus 4.1 generated, 205 samples)
- 样本数: 205 / 205
- 并发 workers: 3
- 原始 JSON: `rag_v3_n205_final.json`
- per_sample_results 保存: 是 (205 条)

## 1. 路由 (Routing)

| 指标 | 值 |
|---|---|
| Micro F1 | 0.8351 |
| Micro Precision | 0.7690 |
| Micro Recall | 0.9135 |
| Macro F1 | 0.8395 |
| Exact match rate | 68.29% |
| Over-routing rate | 20.98% |
| Under-routing rate | 3.90% |

### Per-capability F1

| Capability | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| vector | 0.9659 | 0.8500 | 0.9043 | 85 | 3 | 15 |
| sql | 0.5833 | 0.9403 | 0.7200 | 63 | 45 | 4 |
| rules | 0.7703 | 1.0000 | 0.8702 | 57 | 17 | 0 |
| graph | 0.8261 | 0.9048 | 0.8636 | 38 | 8 | 4 |

## 2. 检索 (Retrieval)

### 2.1 Vector

| 指标 | chunk 级 | source 级 |
|---|---|---|
| Recall@5 | 0.5750 | 0.7284 |
| Recall@20 | 0.6000 | 0.7778 |
| Precision@5 | 0.1150 | 0.4459 |
| MRR | 0.4775 | 0.6877 |
| NDCG@10 | 0.3551 | 1.8063 |
| Samples | 80 | 81 |

### 2.2 SQL

| 指标 | 值 |
|---|---|
| Table F1 | 0.9889 |
| Execution OK rate | 94.03% |
| Samples | 67 |

### 2.3 Rules

| 指标 | 值 |
|---|---|
| Variable Precision | 0.1884 |
| Variable Recall | 0.3043 |
| Samples | 23 |

### 2.4 Graph

| 指标 | 值 |
|---|---|
| path_found_rate | 85.71% |
| Samples | 42 |

### 2.5 Rerank

| 指标 | 值 |
|---|---|
| NDCG@5 lift | 0.2024 |
| Recall@5 lift | 0.1667 |
| Top-1 hit before | 27.27% |
| Top-1 hit after | 50.00% |
| Top-1 lift | 22.73% |
| Samples | 66 |

## 3. 答案质量 (Answer Quality)

| 指标 | 值 |
|---|---|
| Keyword coverage | 55.19% |
| Citation validity | 100.00% |
| Numerical accuracy | 52.20% |
| Embedding cosine similarity (BGE) | 0.7258 |
| ROUGE-L F1 | 0.0836 |
| Similarity 样本数 | 205 |

### LLM Judge

| 维度 (1-5) | 值 |
|---|---|
| Faithfulness | 0.00 |
| Relevance | 0.00 |
| Completeness | 0.00 |
| LLM judged samples | 0 |

### Grounding distribution

- `fully_grounded`: 154
- `llm_fallback`: 50
- `weakly_grounded`: 1

## 4. 护栏 (Guardrails)

| 类型 | Pass Rate | 样本数 |
|---|---|---|
| ambiguous_query | 33.33% | 3 |
| doc_vs_rule_conflict | 50.00% | 2 |
| doc_vs_sql_conflict | 100.00% | 2 |
| empty_evidence | 100.00% | 3 |
| evidence_conflict | 33.33% | 3 |
| false_premise | 0.00% | 3 |
| impossible_query | 100.00% | 3 |
| out_of_domain | 75.00% | 4 |
| refusal_appropriate | 0.00% | 2 |

## 5. 延迟 (Latency)

### 节点级延迟

| 阶段 | count | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| planner | 205 | 25.0300 | 21.6710 | 45.0130 | 45.7810 | 46.1220 |
| merge_evidence | 205 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0010 |
| retrieve_rules | 74 | 0.0020 | 0.0010 | 0.0100 | 0.0130 | 0.0140 |
| retrieve_documents | 88 | 0.4570 | 0.3540 | 0.9590 | 2.2160 | 2.8450 |
| end_to_end | 205 | 89.3990 | 81.7550 | 146.0820 | 160.4560 | 189.1450 |
| synthesize_answer | 205 | 34.1580 | 33.5900 | 49.9190 | 60.0150 | 60.0160 |
| route_query | 205 | 18.6440 | 17.7120 | 30.0180 | 30.0310 | 30.7590 |
| run_sql | 108 | 4.3620 | 3.5230 | 9.2090 | 11.2340 | 12.0310 |
| run_graph_reasoner | 46 | 23.1110 | 28.1180 | 30.9680 | 31.6800 | 31.9820 |
| rerank_documents | 88 | 0.4670 | 0.4680 | 0.5690 | 0.9480 | 0.9660 |

### Planner 子阶段延迟

| 子阶段 | count | mean | p50 | p95 | max |
|---|---|---|---|---|---|
| planner__sub_queries__llm_call | 202 | 25.4010 | 22.0540 | 45.0090 | 46.1220 |
| planner__plan_total | 205 | 25.0290 | 21.6710 | 45.0130 | 46.1220 |
| planner__sub_query__graph__method | 46 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| planner__query_rewrite | 205 | 0.0000 | 0.0000 | 0.0010 | 0.0150 |
| planner__sub_queries__rule_fallback | 18 | 0.0010 | 0.0000 | 0.0070 | 0.0120 |
| planner__sub_query__sql__method | 108 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| planner__sub_query__rules__method | 74 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| planner__sub_query__documents__method | 88 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
