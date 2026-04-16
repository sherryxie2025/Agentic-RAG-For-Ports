# RAG 评测报告 — Agentic RAG LangGraph DAG

- 生成时间: 2026-04-15T02:26:01.262412
- 数据管线: v2 (Small-to-Big + BGE + auto-taxonomy + rule-driven graph)
- Golden 数据集: golden_dataset_v3_rag.json (Opus 4.1 generated, 205 samples)
- 样本数: 10 / 205
- 并发 workers: 1
- 原始 JSON: `rag_v3_n10_post_cleanup_smoke.json`
- per_sample_results 保存: 是 (10 条)

## 1. 路由 (Routing)

| 指标 | 值 |
|---|---|
| Micro F1 | 0.6957 |
| Micro Precision | 0.6154 |
| Micro Recall | 0.8000 |
| Macro F1 | 0.2222 |
| Exact match rate | 60.00% |
| Over-routing rate | 20.00% |
| Under-routing rate | 0.00% |

### Per-capability F1

| Capability | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| vector | 1.0000 | 0.8000 | 0.8889 | 8 | 0 | 2 |
| sql | 0.0000 | 0.0000 | 0.0000 | 0 | 3 | 0 |
| rules | 0.0000 | 0.0000 | 0.0000 | 0 | 2 | 0 |
| graph | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0 |

## 2. 检索 (Retrieval)

### 2.1 Vector

| 指标 | chunk 级 | source 级 |
|---|---|---|
| Recall@5 | 0.8000 | 0.8000 |
| Recall@20 | 0.8000 | 0.8000 |
| Precision@5 | 0.1600 | 0.7067 |
| MRR | 0.6583 | 0.8000 |
| NDCG@10 | 0.6078 | 2.9502 |
| Samples | 10 | 10 |

### 2.2 SQL

| 指标 | 值 |
|---|---|
| Table F1 | — |
| Execution OK rate | — |
| Samples | 0 |

### 2.3 Rules

| 指标 | 值 |
|---|---|
| Variable Precision | — |
| Variable Recall | — |
| Samples | 0 |

### 2.4 Graph

| 指标 | 值 |
|---|---|
| path_found_rate | — |
| Samples | 0 |

### 2.5 Rerank

| 指标 | 值 |
|---|---|
| NDCG@5 lift | 0.1788 |
| Recall@5 lift | 0.2500 |
| Top-1 hit before | 62.50% |
| Top-1 hit after | 75.00% |
| Top-1 lift | 12.50% |
| Samples | 8 |

## 3. 答案质量 (Answer Quality)

| 指标 | 值 |
|---|---|
| Keyword coverage | 28.00% |
| Citation validity | 100.00% |
| Numerical accuracy | 36.31% |
| Embedding cosine similarity (BGE) | 0.7406 |
| ROUGE-L F1 | 0.0908 |
| Similarity 样本数 | 10 |

### LLM Judge

| 维度 (1-5) | 值 |
|---|---|
| Faithfulness | 0.00 |
| Relevance | 0.00 |
| Completeness | 0.00 |
| LLM judged samples | 0 |

### Grounding distribution

- `fully_grounded`: 10

## 4. 护栏 (Guardrails)

| 类型 | Pass Rate | 样本数 |
|---|---|---|

## 5. 延迟 (Latency)

### 节点级延迟

| 阶段 | count | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| planner | 10 | 30.0530 | 26.1180 | 46.9140 | 46.9440 | 46.9520 |
| merge_evidence | 10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| retrieve_rules | 2 | 0.0060 | 0.0060 | 0.0100 | 0.0110 | 0.0110 |
| rerank_documents | 8 | 0.5140 | 0.5200 | 0.6580 | 0.6950 | 0.7050 |
| synthesize_answer | 10 | 32.8210 | 31.4630 | 49.9210 | 57.9980 | 60.0170 |
| retrieve_documents | 8 | 0.6820 | 0.5550 | 1.5880 | 2.0100 | 2.1160 |
| end_to_end | 10 | 90.1630 | 87.4610 | 135.4920 | 151.3080 | 155.2620 |
| run_sql | 3 | 5.7190 | 5.5070 | 8.7250 | 9.0110 | 9.0830 |
| route_query | 10 | 19.8120 | 18.8180 | 30.0110 | 30.0120 | 30.0130 |

### Planner 子阶段延迟

| 子阶段 | count | mean | p50 | p95 | max |
|---|---|---|---|---|---|
| planner__query_rewrite | 10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| planner__sub_queries__llm_call | 10 | 30.0530 | 26.1180 | 46.9140 | 46.9520 |
| planner__sub_query__sql__method | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| planner__plan_total | 10 | 30.0530 | 26.1180 | 46.9140 | 46.9520 |
| planner__sub_queries__rule_fallback | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| planner__sub_query__documents__method | 8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| planner__sub_query__rules__method | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
