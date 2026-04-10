# Round 5 Post-Mortem: MLP Classifier & Embedding Matching Experiment

**Date:** 2026-04-06
**Author:** Sherry (with Claude Code)
**Status:** Experiment concluded. R4 remains best configuration.

---

## 1. Experiment Objective

Replace expensive LLM API calls in two pipeline stages with lightweight local models:

| Stage | R4 (baseline) | R5 (experiment) | Goal |
|-------|:------------:|:---------------:|:----:|
| Intent Routing fallback | LLM zero-shot (2.4s P50) | MLP classifier (0.01s P50) | 240x latency reduction |
| Graph Entity Extraction | LLM few-shot (21.7s P50) | BGE embedding cosine (2.1s P50) | 10x latency reduction |
| Query Abbreviation Expand | LLM rewrite (27.4s P50) | Dictionary lookup (0ms) | Eliminate API call |

---

## 2. Experimental Design

### 2a. MLP Intent Classifier

**Data Pipeline:**
1. Seed: 80 golden dataset queries with expected_sources labels
2. Augmentation: LLM generates 5 paraphrases per query -> 480 samples total
3. Features: BGE-small-en embeddings (384-dim)
4. Model: sklearn MLPClassifier(256, 128), ReLU, early stopping
5. Evaluation: 5-fold cross-validation per label

**Cross-Validation Results:**

| Label | Train Samples | CV F1 | Assessment |
|-------|:------------:|:-----:|:----------:|
| sql | 324 (67.5%) | **0.930** | Excellent |
| documents | 192 (40.0%) | 0.739 | Moderate |
| rules | 168 (35.0%) | 0.577 | Weak |
| graph | 60 (12.5%) | 0.000 | Failed |

### 2b. Embedding Entity Matching

**Index Construction:**
- 40 Neo4j node names, each with 2-4 human-readable aliases
- Total: 135 alias strings encoded with BGE-small-en
- Matching: query embedding vs alias embeddings, cosine similarity, top-3

**Offline Validation (manual spot-check):**

| Query | Top-1 Match | Cosine | Correct? |
|-------|-------------|:------:|:--------:|
| "berth delays + weather" | arrival_delay_hours | 0.946 | Yes |
| "crane breakdowns + logistics" | breakdown_minutes | 0.924 | Yes |
| "high tide + vessel entry" | vessel_entry | 0.907 | Yes |

Embeddings looked excellent in isolation.

### 2c. Dictionary Query Rewrite

**Design:** 85 port/maritime abbreviations in JSON lookup table.
**Rule:** If any abbreviation found in query -> expand inline, skip LLM entirely.

---

## 3. Results

### 3a. R5a: MLP Override Mode (first attempt)

MLP predictions **replaced** rule-based routing decisions entirely when confidence < 0.50.

| Metric | R4 | R5a | Delta |
|--------|:--:|:---:|:-----:|
| Routing F1 | 0.864 | **0.593** | -31% |
| Evidence Recall | 0.643 | **0.325** | -49% |
| Faithfulness | 0.513 | **0.317** | -38% |
| Graph recall | 0.900 | **0.000** | -100% |

**Diagnosis:** MLP's sql bias (F1=0.93) overwhelmed all other labels. Graph class (F1=0.00) was completely suppressed. Document routing collapsed from 0.906 to 0.135 F1.

### 3b. R5b: MLP Additive Mode (mitigation attempt)

Changed MLP to **only add** sources (never remove rule-based decisions).

| Metric | R4 | R5b | Delta |
|--------|:--:|:---:|:-----:|
| Routing F1 | 0.864 | 0.714 | -17% |
| Evidence Recall | 0.643 | 0.470 | -27% |
| Graph recall | 0.900 | 0.000 | -100% |
| Claim Citation | 0.620 | **0.656** | +6% |
| route_query P50 | 2.4s | **0.01s** | -99.6% |

**Partial recovery:** Routing F1 improved from 0.593 to 0.714, but still -17% vs R4. Graph remained zero because embedding matching returned high-similarity entities (>0.85) that didn't trigger LLM fallback, yet those entities had no Neo4j edges.

---

## 4. Root Cause Analysis

### RCA-1: Insufficient Training Data for Multi-Label Classification

| Factor | Impact | Evidence |
|--------|--------|----------|
| 480 total samples | High | Industry best practice: 2000+ for multi-label |
| Graph: only 60 samples (12.5%) | Critical | CV F1 = 0.000 |
| Class imbalance: sql 67% vs graph 12% | High | MLP over-predicts sql, never predicts graph |
| LLM paraphrase quality | Medium | Paraphrases preserve intent but not source diversity |

**Counterfactual:** With 2000+ balanced samples (500 per label), MLP would likely reach F1 > 0.85 for all labels based on the strong BGE embedding space.

### RCA-2: Embedding Similarity != Graph Traversability

| Factor | Impact | Evidence |
|--------|--------|----------|
| BGE cosine scores universally high (0.85+) | Critical | Threshold 0.5 never triggers LLM fallback |
| Matched entities may lack Neo4j edges | Critical | Top-3 entities often had no connecting paths |
| Entity relevance != graph structure coverage | High | "vessel_capacity" matched but had no path to "delay" |

**Lesson:** Semantic similarity measures text relevance, not graph connectivity. A graph-aware scoring function (considering node degree, path existence) would be needed to make embedding matching work for graph traversal.

### RCA-3: Additive MLP Creates Over-Triggering

| Factor | Impact | Evidence |
|--------|--------|----------|
| MLP sql recall = 93% | High | Almost all queries get needs_sql=True added |
| SQL source over-fire: 54 expected -> 69 actual (28% excess) | Medium | Precision dropped from 0.925 to 0.681 |
| No confidence filtering on MLP predictions | High | Binary predictions without probability threshold |

**Improvement path:** Use `clf.predict_proba()` with per-label thresholds (e.g., only add sql if P(sql) > 0.8) instead of binary predict.

---

## 5. What Worked

| Component | Latency Improvement | Accuracy Impact | Keep? |
|-----------|:------------------:|:---------------:|:-----:|
| Dictionary query rewrite | 35s -> 0ms (for dict hits) | Neutral | **Yes** |
| MLP routing inference | 2.4s -> 0.01s | -17% routing F1 | No (needs more data) |
| Embedding entity matching | 21.7s -> 2.1s | Graph recall -> 0 | No (needs graph-aware scoring) |

**Decision:** Only dictionary rewrite is merged into final R4 configuration. MLP and embedding code remain in codebase as reference implementations for future scaling.

---

## 6. Lessons Learned

### For Interview Discussion

1. **Low-resource ML vs LLM zero-shot:** In domains with < 500 labeled samples, LLM zero-shot classification outperforms fine-tuned classifiers. The crossover point is approximately 2000+ balanced samples in our port operations domain.

2. **Accuracy-latency tradeoff is not linear:** Replacing a 2.4s LLM call with a 0.01s MLP gave 240x speedup but cost 17% accuracy. In a RAG pipeline where routing errors cascade through all downstream stages, the accuracy cost is amplified beyond the single-stage metric.

3. **Embedding similarity is a necessary but insufficient condition for graph reasoning:** High cosine similarity (0.94) between query and node name does not guarantee useful graph traversal paths. Future work should combine embedding matching with graph structure priors (node connectivity, path existence scores).

4. **Additive fusion needs confidence gating:** Blindly adding classifier predictions to rule-based decisions creates over-triggering. Per-label probability thresholds (from predict_proba) are essential for effective fusion.

### For Future Iterations

- Collect 2000+ labeled routing samples from production queries
- Implement `predict_proba` thresholding for MLP fusion
- Build graph-aware entity scoring that combines embedding similarity with Neo4j degree/path metrics
- Consider LoRA fine-tuning of BGE for port-domain entity matching

---

## 7. File Inventory

| File | Description | Status |
|------|-------------|--------|
| `data/abbreviation_dict.json` | 85 abbreviation mappings | **Active in R4+dict** |
| `src/online_pipeline/query_rewriter.py` | Dict-first + LLM fallback | **Active in R4+dict** |
| `src/online_pipeline/graph_entity_index.py` | Embedding entity index | Archived (R5 only) |
| `evaluation/train_intent_classifier.py` | MLP training pipeline | Archived (R5 only) |
| `evaluation/augmented_intent_data.json` | 480 augmented samples | Archived (R5 only) |
| `storage/models/intent_classifier.pkl` | Trained MLP artifact | Archived (R5 only) |
| `evaluation/eval_v5_final.log` | R5a raw log | Reference |
| `evaluation/eval_v5b_final.log` | R5b raw log | Reference |
| `evaluation/evaluation_report_r5b.json` | R5b full JSON report | Reference |
