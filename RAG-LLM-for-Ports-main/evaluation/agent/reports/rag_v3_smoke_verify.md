# RAG 评测报告 — Agentic RAG LangGraph DAG

- 生成时间: 2026-04-12T03:42:18.844858
- 数据管线: v2 (Small-to-Big + BGE + auto-taxonomy + rule-driven graph)
- Golden 数据集: golden_dataset_v3_rag.json (Opus 4.1 generated, 205 samples)
- 样本数: 3 / 205
- 并发 workers: 1
- 原始 JSON: `rag_v3_smoke_verify.json`
- per_sample_results 保存: 是 (3 条)

## 1. 路由 (Routing)

| 指标 | 值 |
|---|---|
| Micro F1 | 0.5000 |
| Micro Precision | 0.4000 |
| Micro Recall | 0.6667 |
| Macro F1 | 0.2000 |
| Exact match rate | 0.00% |
| Over-routing rate | 66.67% |
| Under-routing rate | 33.33% |

### Per-capability F1

| Capability | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| vector | 1.0000 | 0.6667 | 0.8000 | 2 | 0 | 1 |
| sql | 0.0000 | 0.0000 | 0.0000 | 0 | 2 | 0 |
| rules | 0.0000 | 0.0000 | 0.0000 | 0 | 1 | 0 |
| graph | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0 |

## 2. 检索 (Retrieval)

### 2.1 Vector

| 指标 | chunk 级 | source 级 |
|---|---|---|
| Recall@5 | 0.6667 | 0.6667 |
| Recall@20 | 0.6667 | 0.6667 |
| Precision@5 | 0.1333 | 0.4667 |
| MRR | 0.4167 | 0.6667 |
| NDCG@10 | 0.4297 | 2.2112 |
| Samples | 3 | 3 |

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
| NDCG@5 lift | 0.2153 |
| Recall@5 lift | 0.5000 |
| Top-1 hit before | 50.00% |
| Top-1 hit after | 50.00% |
| Top-1 lift | 0.00% |
| Samples | 2 |

## 3. 答案质量 (Answer Quality)

| 指标 | 值 |
|---|---|
| Keyword coverage | 0.00% |
| Citation validity | 100.00% |
| Numerical accuracy | 0.00% |
| Embedding cosine similarity (BGE) | 0.5776 |
| ROUGE-L F1 | 0.0605 |
| Similarity 样本数 | 3 |

### LLM Judge

| 维度 (1-5) | 值 |
|---|---|
| Faithfulness | 0.00 |
| Relevance | 0.00 |
| Completeness | 0.00 |
| LLM judged samples | 0 |

### Grounding distribution

- `llm_fallback`: 1
- `fully_grounded`: 2

## 4. 护栏 (Guardrails)

| 类型 | Pass Rate | 样本数 |
|---|---|---|

## 5. 延迟 (Latency)

### 节点级延迟

| 阶段 | count | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| rerank_documents | 2 | 0.3090 | 0.3090 | 0.3420 | 0.3450 | 0.3460 |
| end_to_end | 3 | 2.8500 | 2.6250 | 3.4260 | 3.4970 | 3.5140 |
| retrieve_rules | 1 | 0.0010 | 0.0010 | 0.0010 | 0.0010 | 0.0010 |
| merge_evidence | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| retrieve_documents | 2 | 0.8080 | 0.8080 | 1.2640 | 1.3050 | 1.3150 |
| planner | 3 | 0.3030 | 0.2740 | 0.6000 | 0.6290 | 0.6360 |
| synthesize_answer | 3 | 0.4260 | 0.2730 | 0.6870 | 0.7240 | 0.7340 |
| route_query | 3 | 0.8890 | 0.2730 | 1.9360 | 2.0840 | 2.1210 |
| run_sql | 2 | 0.3180 | 0.3180 | 0.3310 | 0.3320 | 0.3320 |

### Planner 子阶段延迟

| 子阶段 | count | mean | p50 | p95 | max |
|---|---|---|---|---|---|
| planner__sub_queries__llm_call | 2 | 0.4540 | 0.4540 | 0.6180 | 0.6360 |
| planner__sub_query__rules__method | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| planner__query_rewrite | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| planner__plan_total | 3 | 0.3030 | 0.2730 | 0.6000 | 0.6360 |
| planner__sub_query__sql__method | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| planner__sub_queries__rule_fallback | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| planner__sub_query__documents__method | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
