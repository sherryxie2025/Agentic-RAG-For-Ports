# RAG 评测报告 — Agentic RAG LangGraph DAG

- 生成时间: 2026-04-11T23:57:28.434662
- 数据管线: v2 (Small-to-Big + BGE + auto-taxonomy + rule-driven graph)
- Golden 数据集: golden_dataset_v3_rag.json (Opus 4.1 generated, 205 samples)
- 样本数: 3 / 205
- 并发 workers: 1
- 原始 JSON: `rag_v3_smoke_n3.json`
- per_sample_results 保存: 是 (3 条)

## 1. 路由 (Routing)

| 指标 | 值 |
|---|---|
| Micro F1 | 0.8571 |
| Micro Precision | 0.7500 |
| Micro Recall | 1.0000 |
| Macro F1 | 0.2500 |
| Exact match rate | 66.67% |
| Over-routing rate | 33.33% |
| Under-routing rate | 0.00% |

### Per-capability F1

| Capability | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| vector | 1.0000 | 1.0000 | 1.0000 | 3 | 0 | 0 |
| sql | 0.0000 | 0.0000 | 0.0000 | 0 | 1 | 0 |
| rules | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0 |
| graph | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0 |

## 2. 检索 (Retrieval)

### 2.1 Vector

| 指标 | chunk 级 | source 级 |
|---|---|---|
| Recall@5 | 1.0000 | 1.0000 |
| Recall@20 | 1.0000 | 1.0000 |
| Precision@5 | 0.2000 | 0.8000 |
| MRR | 0.5278 | 0.7778 |
| NDCG@10 | 0.7854 | 3.2589 |
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
| NDCG@5 lift | -0.0231 |
| Recall@5 lift | 0.3333 |
| Top-1 hit before | 66.67% |
| Top-1 hit after | 33.33% |
| Top-1 lift | -33.33% |
| Samples | 3 |

## 3. 答案质量 (Answer Quality)

| 指标 | 值 |
|---|---|
| Keyword coverage | 33.33% |
| Citation validity | 100.00% |
| Numerical accuracy | 21.43% |
| Embedding cosine similarity (BGE) | 0.7438 |
| ROUGE-L F1 | 0.0873 |
| Similarity 样本数 | 3 |

### LLM Judge

| 维度 (1-5) | 值 |
|---|---|
| Faithfulness | 0.00 |
| Relevance | 0.00 |
| Completeness | 0.00 |
| LLM judged samples | 0 |

### Grounding distribution

- `fully_grounded`: 3

## 4. 护栏 (Guardrails)

| 类型 | Pass Rate | 样本数 |
|---|---|---|

## 5. 延迟 (Latency)

| 阶段 | count | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| retrieve_documents | 3 | 1.3840 | 0.8120 | 2.5500 | 2.7040 | 2.7430 |
| route_query | 3 | 24.4910 | 26.4660 | 29.6610 | 29.9450 | 30.0160 |
| synthesize_answer | 3 | 35.1490 | 39.4670 | 40.8200 | 40.9410 | 40.9710 |
| end_to_end | 3 | 121.7390 | 106.3600 | 150.9880 | 154.9540 | 155.9460 |
| rerank_documents | 3 | 0.5350 | 0.3930 | 0.7880 | 0.8230 | 0.8320 |
| run_sql | 1 | 6.1300 | 6.1300 | 6.1300 | 6.1300 | 6.1300 |
| merge_evidence | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| planner | 3 | 45.6340 | 40.3630 | 58.7240 | 60.3560 | 60.7640 |
