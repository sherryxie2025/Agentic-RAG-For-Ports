# RAG 评测报告 — Agentic RAG LangGraph DAG

- 生成时间: 2026-04-12T03:25:23.459470
- 数据管线: v2 (Small-to-Big + BGE + auto-taxonomy + rule-driven graph)
- Golden 数据集: golden_dataset_v3_rag.json (Opus 4.1 generated, 205 samples)
- 样本数: 205 / 205
- 并发 workers: 3
- 原始 JSON: `rag_v3_n205.json`
- per_sample_results 保存: 是 (205 条)

## 1. 路由 (Routing)

| 指标 | 值 |
|---|---|
| Micro F1 | 0.8288 |
| Micro Precision | 0.7610 |
| Micro Recall | 0.9098 |
| Macro F1 | 0.8313 |
| Exact match rate | 67.32% |
| Over-routing rate | 21.46% |
| Under-routing rate | 4.88% |

### Per-capability F1

| Capability | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| vector | 0.9663 | 0.8600 | 0.9101 | 86 | 3 | 14 |
| sql | 0.5794 | 0.9254 | 0.7126 | 62 | 45 | 5 |
| rules | 0.7467 | 0.9825 | 0.8485 | 56 | 19 | 1 |
| graph | 0.8085 | 0.9048 | 0.8539 | 38 | 9 | 4 |

## 2. 检索 (Retrieval)

### 2.1 Vector

| 指标 | chunk 级 | source 级 |
|---|---|---|
| Recall@5 | 0.5250 | 0.8025 |
| Recall@20 | 0.6250 | 0.8395 |
| Precision@5 | 0.1050 | 0.4539 |
| MRR | 0.4323 | 0.7082 |
| NDCG@10 | 0.4325 | 1.8929 |
| Samples | 80 | 81 |

### 2.2 SQL

| 指标 | 值 |
|---|---|
| Table F1 | 1.0000 |
| Execution OK rate | 92.54% |
| Samples | 67 |

### 2.3 Rules

| 指标 | 值 |
|---|---|
| Variable Precision | 0.1739 |
| Variable Recall | 0.4348 |
| Samples | 23 |

### 2.4 Graph

| 指标 | 值 |
|---|---|
| path_found_rate | 85.71% |
| Samples | 42 |

### 2.5 Rerank

| 指标 | 值 |
|---|---|
| NDCG@5 lift | 0.0532 |
| Recall@5 lift | 0.0299 |
| Top-1 hit before | 34.33% |
| Top-1 hit after | 43.28% |
| Top-1 lift | 8.96% |
| Samples | 67 |

## 3. 答案质量 (Answer Quality)

| 指标 | 值 |
|---|---|
| Keyword coverage | 53.50% |
| Citation validity | 100.00% |
| Numerical accuracy | 53.70% |
| Embedding cosine similarity (BGE) | 0.7211 |
| ROUGE-L F1 | 0.0798 |
| Similarity 样本数 | 205 |

### LLM Judge

| 维度 (1-5) | 值 |
|---|---|
| Faithfulness | 0.00 |
| Relevance | 0.00 |
| Completeness | 0.00 |
| LLM judged samples | 0 |

### Grounding distribution

- `llm_fallback`: 40
- `fully_grounded`: 165

## 4. 护栏 (Guardrails)

| 类型 | Pass Rate | 样本数 |
|---|---|---|
| ambiguous_query | 33.33% | 3 |
| doc_vs_rule_conflict | 0.00% | 2 |
| doc_vs_sql_conflict | 50.00% | 2 |
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
| end_to_end | 205 | 105.8680 | 96.3520 | 173.0470 | 194.0480 | 219.4190 |
| merge_evidence | 205 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0010 |
| retrieve_rules | 75 | 0.0030 | 0.0020 | 0.0080 | 0.0190 | 0.0220 |
| synthesize_answer | 205 | 34.1390 | 32.9860 | 51.3610 | 60.0190 | 60.7550 |
| route_query | 205 | 19.2080 | 18.7370 | 30.0180 | 30.0310 | 32.6510 |
| run_sql | 107 | 6.1750 | 6.0180 | 10.1940 | 12.0070 | 13.7530 |
| run_graph_reasoner | 47 | 26.9370 | 30.0390 | 30.9560 | 32.2330 | 32.9070 |
| retrieve_documents | 89 | 0.6120 | 0.5870 | 1.0750 | 1.7120 | 3.3240 |
| planner | 205 | 38.3330 | 34.4070 | 71.0980 | 91.5080 | 92.1420 |
| rerank_documents | 89 | 0.4360 | 0.4480 | 0.5610 | 0.5970 | 0.6000 |

### Planner 子阶段延迟

| 子阶段 | count | mean | p50 | p95 | max |
|---|---|---|---|---|---|
| planner__plan_and_subqueries | 205 | 5.1950 | 0.0000 | 30.8150 | 32.3430 |
| planner__query_rewrite | 205 | 33.1380 | 33.0830 | 60.7680 | 60.8580 |
