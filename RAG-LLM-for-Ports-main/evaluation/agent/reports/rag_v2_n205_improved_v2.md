# RAG 评测报告 — Agentic RAG LangGraph DAG

- 生成时间: 2026-04-11T22:14:44.036317
- 数据管线: v2 (Small-to-Big + BGE + auto-taxonomy + rule-driven graph)
- Golden 数据集: golden_dataset_v3_rag.json (Opus 4.1 generated, 205 samples)
- 样本数: 205 / 205
- 并发 workers: 3
- 原始 JSON: `rag_v2_n205_improved_v2.json`
- per_sample_results 保存: 是 (205 条)

## 1. 路由 (Routing)

| 指标 | 值 |
|---|---|
| Micro F1 | 0.8229 |
| Micro Precision | 0.7645 |
| Micro Recall | 0.8910 |
| Macro F1 | 0.8243 |
| Exact match rate | 67.32% |
| Over-routing rate | 19.51% |
| Under-routing rate | 4.88% |

### Per-capability F1

| Capability | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| vector | 0.9231 | 0.8400 | 0.8796 | 84 | 7 | 16 |
| sql | 0.6200 | 0.9254 | 0.7425 | 62 | 38 | 5 |
| rules | 0.7297 | 0.9474 | 0.8244 | 54 | 20 | 3 |
| graph | 0.8222 | 0.8810 | 0.8506 | 37 | 8 | 5 |

## 2. 检索 (Retrieval)

### 2.1 Vector

| 指标 | chunk 级 | source 级 |
|---|---|---|
| Recall@5 | 0.5625 | 0.7407 |
| Recall@20 | 0.6500 | 0.8025 |
| Precision@5 | 0.1125 | 0.4840 |
| MRR | 0.4760 | 0.7078 |
| NDCG@10 | 0.4091 | 1.9546 |
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
| Variable Precision | 0.2029 |
| Variable Recall | 0.4783 |
| Samples | 23 |

### 2.4 Graph

| 指标 | 值 |
|---|---|
| path_found_rate | 88.10% |
| Samples | 42 |

### 2.5 Rerank

| 指标 | 值 |
|---|---|
| NDCG@5 lift | 0.1403 |
| Recall@5 lift | 0.0758 |
| Top-1 hit before | 30.30% |
| Top-1 hit after | 48.48% |
| Top-1 lift | 18.18% |
| Samples | 66 |

## 3. 答案质量 (Answer Quality)

| 指标 | 值 |
|---|---|
| Keyword coverage | 59.69% |
| Citation validity | 100.00% |
| Numerical accuracy | 58.00% |
| Embedding cosine similarity (BGE) | 0.7306 |
| ROUGE-L F1 | 0.0813 |
| Similarity 样本数 | 205 |

### LLM Judge

| 维度 (1-5) | 值 |
|---|---|
| Faithfulness | 0.00 |
| Relevance | 0.00 |
| Completeness | 0.00 |
| LLM judged samples | 0 |

### Grounding distribution

- `fully_grounded`: 170
- `llm_fallback`: 35

## 4. 护栏 (Guardrails)

| 类型 | Pass Rate | 样本数 |
|---|---|---|
| ambiguous_query | 33.33% | 3 |
| doc_vs_rule_conflict | 100.00% | 2 |
| doc_vs_sql_conflict | 100.00% | 2 |
| empty_evidence | 100.00% | 3 |
| evidence_conflict | 33.33% | 3 |
| false_premise | 33.33% | 3 |
| impossible_query | 100.00% | 3 |
| out_of_domain | 75.00% | 4 |
| refusal_appropriate | 0.00% | 2 |

## 5. 延迟 (Latency)

| 阶段 | count | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| end_to_end | 205 | 98.5440 | 91.2450 | 161.8770 | 179.5770 | 193.7350 |
