# Port RAG System - Evaluation Report

**Date:** 2026-04-09T02:35:56
**Model:** qwen3.5-35b-a3b (Alibaba DashScope)
**Embeddings:** BAAI/bge-small-en (384-dim)
**Reranker:** cross-encoder/ms-marco-MiniLM-L-6-v2
**Total Queries:** 101
**Answer Rate:** 100.0%

## Aggregate Metrics

| Metric | Score |
|--------|-------|
| Evidence Keyword Recall | 0.6331 |
| Source Routing Precision | 0.7814 |
| Source Routing Recall | 0.9035 |
| Source Routing F1 | 0.8151 |
| Answer Faithfulness | 0.5110 |
| Semantic Similarity | 0.8672 |
| MRR (doc queries only) | 0.6262 |
| NDCG@5 (doc queries only) | 0.8808 |
| SQL Result Accuracy | 0.6519 |
| Claim Citation Rate | 0.6535 |
| Claim Grounding Rate | 0.5782 |
| Answer Confidence | 0.5617 |

## Per-Source Retrieval Stats

| Source | Expected | Actual | TP | Precision | Recall |
|--------|----------|--------|----|-----------|--------|
| documents | 35 | 64 | 34 | 0.531 | 0.971 |
| sql | 54 | 43 | 41 | 0.954 | 0.759 |
| rules | 39 | 49 | 38 | 0.775 | 0.974 |
| graph | 20 | 23 | 20 | 0.870 | 1.000 |

## Per Intent Type Breakdown

| Intent Type | Count | Evidence Recall | Routing F1 | Faithfulness | Sem. Sim. | MRR | NDCG@5 | Confidence |
|-------------|-------|-----------------|------------|--------------|-----------|-----|--------|------------|
| document_lookup | 19 | 0.647 | 0.912 | 0.646 | 0.853 | 0.000 | 0.000 | 0.420 |
| structured_data | 23 | 0.480 | 0.887 | 0.361 | 0.873 | 0.000 | 0.000 | 0.490 |
| policy_rule | 15 | 0.703 | 0.667 | 0.671 | 0.876 | 0.000 | 0.000 | 0.580 |
| hybrid_reasoning | 22 | 0.626 | 0.903 | 0.497 | 0.862 | 0.000 | 0.000 | 0.644 |
| causal_multihop | 20 | 0.715 | 0.736 | 0.502 | 0.872 | 0.000 | 0.000 | 0.702 |
| out_of_domain | 2 | 1.000 | 0.000 | 0.000 | 0.870 | 0.000 | 0.000 | 0.275 |

## Per-Query Details

| ID | Intent | Mode | Difficulty | Confidence | Routing F1 | Evidence Recall | Faithfulness | Time (s) |
|----|--------|------|------------|------------|------------|-----------------|--------------|----------|
| VEC_001 | document_lookup | descriptive | easy | 0.40 | 1.00 | 1.00 | 0.73 | 85.3 |
| VEC_002 | document_lookup | descriptive | easy | 0.40 | 1.00 | 0.75 | 0.67 | 83.1 |
| VEC_003 | document_lookup | descriptive | easy | 0.58 | 0.67 | 0.25 | 0.86 | 141.3 |
| VEC_004 | document_lookup | descriptive | easy | 0.40 | 1.00 | 0.80 | 0.69 | 74.1 |
| VEC_005 | document_lookup | descriptive | easy | 0.55 | 0.67 | 0.60 | 0.63 | 123.5 |
| VEC_006 | document_lookup | descriptive | easy | 0.40 | 1.00 | 0.50 | 0.47 | 66.0 |
| VEC_007 | document_lookup | descriptive | easy | 0.40 | 1.00 | 0.80 | 0.80 | 63.3 |
| VEC_008 | document_lookup | descriptive | easy | 0.40 | 1.00 | 0.80 | 0.92 | 88.8 |
| VEC_009 | document_lookup | descriptive | easy | 0.40 | 1.00 | 0.80 | 0.93 | 79.5 |
| VEC_010 | document_lookup | descriptive | easy | 0.40 | 1.00 | 1.00 | 0.25 | 108.2 |
| SQL_001 | structured_data | lookup | easy | 0.45 | 1.00 | 0.33 | 0.22 | 71.5 |
| SQL_002 | structured_data | lookup | easy | 0.45 | 1.00 | 0.50 | 0.22 | 96.1 |
| SQL_003 | structured_data | lookup | easy | 0.45 | 1.00 | 0.50 | 0.60 | 122.0 |
| SQL_004 | structured_data | lookup | easy | 0.45 | 1.00 | 0.67 | 0.38 | 86.7 |
| SQL_005 | structured_data | lookup | easy | 0.45 | 1.00 | 0.00 | 0.71 | 64.0 |
| SQL_006 | structured_data | lookup | medium | 0.45 | 1.00 | 0.33 | 0.29 | 74.9 |
| SQL_007 | structured_data | lookup | medium | 0.45 | 1.00 | 0.67 | 0.14 | 85.3 |
| SQL_008 | structured_data | comparison | medium | 0.45 | 1.00 | 0.67 | 0.60 | 85.3 |
| SQL_009 | structured_data | lookup | medium | 0.45 | 1.00 | 0.67 | 0.50 | 81.8 |
| SQL_010 | structured_data | lookup | medium | 0.45 | 1.00 | 0.33 | 0.06 | 100.2 |
| SQL_011 | structured_data | comparison | medium | 0.45 | 1.00 | 0.00 | 0.80 | 73.0 |
| SQL_012 | structured_data | lookup | easy | 0.45 | 1.00 | 0.50 | 0.75 | 94.8 |
| SQL_013 | structured_data | lookup | medium | 0.45 | 1.00 | 0.33 | 0.10 | 97.3 |
| SQL_014 | structured_data | diagnostic | hard | 0.62 | 0.67 | 0.67 | 0.43 | 120.6 |
| SQL_015 | structured_data | lookup | medium | 0.45 | 1.00 | 0.00 | 0.25 | 113.6 |
| RULE_001 | policy_rule | lookup | easy | 0.58 | 0.67 | 1.00 | 0.73 | 198.3 |
| RULE_002 | policy_rule | lookup | easy | 0.58 | 0.67 | 0.75 | 0.46 | 92.3 |
| RULE_003 | policy_rule | lookup | easy | 0.58 | 0.67 | 0.50 | 0.60 | 118.0 |
| RULE_004 | policy_rule | descriptive | medium | 0.58 | 0.67 | 1.00 | 0.74 | 108.3 |
| RULE_005 | policy_rule | lookup | easy | 0.58 | 0.67 | 1.00 | 0.56 | 125.0 |
| RULE_006 | policy_rule | descriptive | medium | 0.58 | 0.67 | 0.75 | 0.88 | 60.0 |
| RULE_007 | policy_rule | descriptive | medium | 0.58 | 0.67 | 0.60 | 0.79 | 91.8 |
| RULE_008 | policy_rule | descriptive | medium | 0.58 | 0.67 | 1.00 | 0.77 | 94.4 |
| RULE_009 | policy_rule | lookup | easy | 0.58 | 0.67 | 0.75 | 0.69 | 106.0 |
| RULE_010 | policy_rule | descriptive | medium | 0.58 | 0.67 | 0.80 | 0.75 | 112.2 |
| MIX_SR_001 | hybrid_reasoning | decision_support | hard | 0.63 | 0.67 | 0.60 | 0.17 | 86.1 |
| MIX_SR_002 | hybrid_reasoning | decision_support | hard | 0.73 | 0.80 | 0.40 | 0.33 | 123.0 |
| MIX_SR_003 | hybrid_reasoning | decision_support | hard | 0.73 | 0.80 | 0.75 | 0.41 | 102.3 |
| MIX_SR_004 | hybrid_reasoning | decision_support | hard | 0.73 | 0.80 | 1.00 | 0.36 | 115.1 |
| MIX_SR_005 | hybrid_reasoning | descriptive | medium | 0.73 | 0.80 | 0.60 | 0.47 | 127.0 |
| MIX_VS_001 | hybrid_reasoning | descriptive | hard | 0.55 | 1.00 | 0.40 | 0.35 | 92.0 |
| MIX_VS_002 | hybrid_reasoning | comparison | medium | 0.55 | 1.00 | 0.75 | 0.61 | 133.5 |
| MIX_VS_003 | hybrid_reasoning | comparison | medium | 0.55 | 1.00 | 0.50 | 0.46 | 82.8 |
| MIX_VS_004 | hybrid_reasoning | diagnostic | hard | 0.55 | 1.00 | 0.25 | 0.59 | 112.1 |
| MIX_VS_005 | hybrid_reasoning | diagnostic | hard | 0.55 | 1.00 | 0.80 | 0.53 | 106.8 |
| GRAPH_001 | causal_multihop | diagnostic | hard | 0.75 | 0.40 | 0.83 | 0.65 | 156.0 |
| GRAPH_002 | causal_multihop | diagnostic | hard | 0.90 | 0.67 | 1.00 | 0.69 | 211.4 |
| GRAPH_003 | causal_multihop | diagnostic | hard | 0.57 | 0.50 | 0.60 | 0.65 | 120.6 |
| GRAPH_004 | causal_multihop | diagnostic | hard | 0.75 | 0.40 | 0.60 | 0.71 | 115.7 |
| GRAPH_005 | causal_multihop | diagnostic | hard | 0.75 | 0.40 | 0.60 | 0.65 | 153.0 |
| COMPLEX_001 | hybrid_reasoning | decision_support | hard | 0.73 | 1.00 | 0.67 | 0.62 | 138.8 |
| COMPLEX_002 | causal_multihop | diagnostic | hard | 0.90 | 0.67 | 0.71 | 0.52 | 300.6 |
| COMPLEX_003 | hybrid_reasoning | descriptive | hard | 0.73 | 1.00 | 0.80 | 0.72 | 151.4 |
| COMPLEX_004 | causal_multihop | decision_support | hard | 0.80 | 0.80 | 0.80 | 0.14 | 280.2 |
| COMPLEX_005 | causal_multihop | diagnostic | hard | 0.72 | 0.86 | 0.57 | 0.78 | 318.4 |
| VEC_011 | document_lookup | descriptive | easy | 0.40 | 1.00 | 1.00 | 0.71 | 128.3 |
| VEC_012 | document_lookup | descriptive | medium | 0.40 | 1.00 | 0.80 | 0.54 | 86.9 |
| VEC_013 | document_lookup | descriptive | easy | 0.40 | 1.00 | 0.40 | 0.50 | 64.2 |
| VEC_014 | document_lookup | descriptive | medium | 0.40 | 1.00 | 0.20 | 0.77 | 66.5 |
| VEC_015 | document_lookup | lookup | medium | 0.40 | 1.00 | 0.40 | 0.36 | 79.2 |
| VEC_016 | document_lookup | descriptive | easy | 0.45 | 0.00 | 0.00 | 0.50 | 83.8 |
| VEC_017 | document_lookup | descriptive | medium | 0.40 | 1.00 | 0.40 | 0.45 | 54.5 |
| VEC_018 | document_lookup | descriptive | medium | 0.40 | 1.00 | 0.80 | 0.88 | 72.2 |
| SQL_016 | structured_data | lookup | easy | 0.45 | 0.00 | 0.00 | 0.00 | 206.1 |
| SQL_017 | structured_data | comparison | medium | 0.45 | 1.00 | 0.80 | 0.05 | 32.7 |
| SQL_018 | structured_data | comparison | hard | 0.45 | 1.00 | 0.40 | 0.83 | 77.0 |
| SQL_019 | structured_data | lookup | medium | 0.45 | 1.00 | 0.50 | 0.00 | 104.1 |
| SQL_020 | structured_data | comparison | hard | 0.45 | 1.00 | 0.75 | 0.12 | 100.6 |
| RULE_011 | policy_rule | lookup | easy | 0.58 | 0.67 | 0.50 | 0.67 | 93.5 |
| RULE_012 | policy_rule | lookup | easy | 0.58 | 0.67 | 0.00 | 0.39 | 108.5 |
| RULE_013 | policy_rule | lookup | easy | 0.58 | 0.67 | 0.75 | 0.78 | 101.1 |
| RULE_014 | policy_rule | lookup | easy | 0.58 | 0.67 | 0.40 | 0.77 | 68.2 |
| RULE_015 | policy_rule | lookup | easy | 0.58 | 0.67 | 0.75 | 0.50 | 119.3 |
| MIX_SR_006 | hybrid_reasoning | decision_support | hard | 0.63 | 1.00 | 0.75 | 0.71 | 69.8 |
| MIX_SR_007 | hybrid_reasoning | decision_support | hard | 0.63 | 1.00 | 1.00 | 0.62 | 81.7 |
| MIX_SR_008 | hybrid_reasoning | decision_support | hard | 0.63 | 0.67 | 0.40 | 0.40 | 48.9 |
| MIX_SR_009 | hybrid_reasoning | decision_support | hard | 0.63 | 1.00 | 0.60 | 0.40 | 84.5 |
| MIX_SR_010 | hybrid_reasoning | decision_support | hard | 0.63 | 1.00 | 0.25 | 0.19 | 65.0 |
| MIX_VR_001 | hybrid_reasoning | descriptive | medium | 0.58 | 1.00 | 0.80 | 0.63 | 186.0 |
| MIX_VR_002 | hybrid_reasoning | descriptive | medium | 0.58 | 1.00 | 0.60 | 0.82 | 85.7 |
| MIX_VR_003 | hybrid_reasoning | descriptive | hard | 0.58 | 1.00 | 0.60 | 0.77 | 94.6 |
| GRAPH_006 | causal_multihop | diagnostic | hard | 0.57 | 0.50 | 0.80 | 0.43 | 288.7 |
| GRAPH_007 | causal_multihop | diagnostic | hard | 0.57 | 0.67 | 0.60 | 0.10 | 153.4 |
| GRAPH_008 | causal_multihop | diagnostic | hard | 0.57 | 1.00 | 1.00 | 0.25 | 122.7 |
| COMPLEX_006 | causal_multihop | diagnostic | hard | 0.75 | 0.67 | 0.83 | 0.65 | 145.1 |
| GRAPH_ONLY_001 | causal_multihop | diagnostic | hard | 0.47 | 1.00 | 0.50 | 0.38 | 123.2 |
| GRAPH_ONLY_002 | causal_multihop | descriptive | medium | 0.47 | 1.00 | 0.25 | 0.07 | 79.0 |
| MIX_GR_001 | causal_multihop | diagnostic | hard | 0.75 | 0.80 | 0.50 | 0.28 | 179.1 |
| MIX_GR_002 | causal_multihop | decision_support | hard | 0.75 | 0.80 | 0.83 | 0.39 | 181.8 |
| MIX_GV_001 | causal_multihop | diagnostic | hard | 0.75 | 0.80 | 0.60 | 0.47 | 159.1 |
| MIX_GV_002 | causal_multihop | descriptive | hard | 0.75 | 0.80 | 0.80 | 0.67 | 147.0 |
| MIX_GRV_001 | causal_multihop | decision_support | hard | 0.75 | 1.00 | 0.86 | 0.76 | 148.3 |
| MIX_GRV_002 | causal_multihop | decision_support | hard | 0.75 | 1.00 | 1.00 | 0.81 | 192.2 |
| GUARD_EMPTY_001 | out_of_domain | lookup | hard | 0.25 | 0.00 | 1.00 | 0.00 | 68.7 |
| GUARD_EMPTY_002 | out_of_domain | lookup | hard | 0.30 | 0.00 | 1.00 | 0.00 | 78.5 |
| GUARD_DS_NORULE_001 | structured_data | decision_support | hard | 0.63 | 0.67 | 1.00 | 0.21 | 79.8 |
| GUARD_DS_NORULE_002 | structured_data | decision_support | hard | 0.63 | 0.67 | 0.67 | 0.44 | 96.1 |
| GUARD_DS_WITHRULE_001 | hybrid_reasoning | decision_support | hard | 0.63 | 0.67 | 0.50 | 0.11 | 83.2 |
| GUARD_DIAG_NOGRAPH_001 | structured_data | diagnostic | hard | 0.85 | 0.40 | 0.75 | 0.61 | 137.5 |
| GUARD_DESC_NODOC_001 | document_lookup | descriptive | hard | 0.40 | 1.00 | 1.00 | 0.62 | 83.2 |
| GUARD_FALLBACK_001 | hybrid_reasoning | lookup | hard | 0.90 | 0.67 | 0.75 | 0.68 | 221.1 |

## Metric Definitions

- **Evidence Keyword Recall**: Fraction of expected keywords found in retrieved evidence
- **Source Routing Precision**: Fraction of selected sources that were expected
- **Source Routing Recall**: Fraction of expected sources that were selected
- **Source Routing F1**: Harmonic mean of routing precision and recall
- **Answer Faithfulness**: Fraction of answer sentences grounded in retrieved evidence
- **Answer Confidence**: System-reported confidence score (0-1)
