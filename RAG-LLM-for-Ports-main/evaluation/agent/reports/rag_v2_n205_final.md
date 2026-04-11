# RAG 评测报告 — Agentic RAG LangGraph DAG

- 生成时间: 2026-04-11T14:54:59.646448
- 数据管线: v2 (Small-to-Big + BGE + auto-taxonomy + rule-driven graph)
- Golden 数据集: golden_dataset_v3_rag.json (Opus 4.1 generated, 205 samples)
- 样本数: 205 / 205
- 原始 JSON: `rag_v2_n205_final.json`
- per_sample_results 保存: 否 (历史报告)

## 1. 路由 (Routing)

| 指标 | 值 |
|---|---|
| Micro F1 | 0.7659 |
| Micro Precision | 0.6517 |
| Micro Recall | 0.9286 |
| Macro F1 | 0.7750 |
| Exact match rate | 49.76% |
| Over-routing rate | 43.41% |
| Under-routing rate | 1.46% |

### Per-capability F1

| Capability | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| vector | 0.6642 | 0.8900 | 0.7607 | 89 | 45 | 11 |
| sql | 0.5780 | 0.9403 | 0.7159 | 63 | 46 | 4 |
| rules | 0.6628 | 1.0000 | 0.7972 | 57 | 29 | 0 |
| graph | 0.7600 | 0.9048 | 0.8261 | 38 | 12 | 4 |

## 2. 检索 (Retrieval)

### 2.1 Vector

| 指标 | chunk 级 | source 级 |
|---|---|---|
| Recall@5 | 0.6125 | 0.8272 |
| Recall@20 | 0.6500 | 0.8395 |
| Precision@5 | 0.1225 | 0.5037 |
| MRR | 0.4715 | 0.7412 |
| NDCG@10 | 0.4658 | 2.0157 |
| Samples | 80 | 81 |

### 2.2 SQL

| 指标 | 值 |
|---|---|
| Table F1 | 1.0000 |
| Execution OK rate | 94.03% |
| Samples | 67 |

### 2.3 Rules

| 指标 | 值 |
|---|---|
| Variable Precision | 0.3406 |
| Variable Recall | 0.3913 |
| Samples | 23 |

### 2.4 Graph

| 指标 | 值 |
|---|---|
| path_found_rate | 73.81% |
| Samples | 42 |

### 2.5 Rerank

| 指标 | 值 |
|---|---|
| NDCG@5 lift | 0.0797 |
| Recall@5 lift | 0.1286 |
| Top-1 hit before | 38.57% |
| Top-1 hit after | 44.29% |
| Top-1 lift | 5.71% |
| Samples | 70 |

## 3. 答案质量 (Answer Quality)

| 指标 | 值 |
|---|---|
| Keyword coverage | 59.91% |
| Citation validity | 100.00% |
| Numerical accuracy | 59.33% |
| Embedding cosine similarity (BGE) | — |
| ROUGE-L F1 | — |
| Similarity 样本数 | 0 |

### LLM Judge

| 维度 (1-5) | 值 |
|---|---|
| Faithfulness | 0.00 |
| Relevance | 0.00 |
| Completeness | 0.00 |
| LLM judged samples | 0 |

### Grounding distribution

- `fully_grounded`: 154
- `llm_fallback`: 51

## 4. 护栏 (Guardrails)

| 类型 | Pass Rate | 样本数 |
|---|---|---|
| ambiguous_query | 0.00% | 3 |
| doc_vs_rule_conflict | 50.00% | 2 |
| doc_vs_sql_conflict | 0.00% | 2 |
| empty_evidence | 33.33% | 3 |
| evidence_conflict | 100.00% | 3 |
| false_premise | 0.00% | 3 |
| impossible_query | 100.00% | 3 |
| out_of_domain | 100.00% | 4 |
| refusal_appropriate | 0.00% | 2 |

## 5. 延迟 (Latency)

| 阶段 | count | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| end_to_end | 205 | 113.4530 | 108.5490 | 183.2490 | 193.0000 | 202.6320 |
