"""
Answer Quality Evaluation: Single-turn answer grading.

Combines objective metrics (keyword coverage, citation validity,
numerical accuracy) with LLM-as-judge scoring for:
- Faithfulness (no hallucination)
- Relevance (answers the question)
- Completeness (covers reference answer)

Requires an LLM for the judge scores. Uses the project's llm_client.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make project imports work when run standalone
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from online_pipeline.llm_client import llm_chat_json
except ImportError:
    llm_chat_json = None

# BGE model for embedding similarity. Loaded lazily on first use so
# this module stays import-safe for hosts that don't have torch.
_BGE_MODEL = None
_BGE_TOKENIZER = None


def _get_bge():
    """Lazy-load BGE-base-en-v1.5 model + tokenizer for similarity scoring."""
    global _BGE_MODEL, _BGE_TOKENIZER
    if _BGE_MODEL is not None:
        return _BGE_MODEL, _BGE_TOKENIZER
    try:
        from transformers import AutoModel, AutoTokenizer
        import torch  # noqa: F401
        _BGE_TOKENIZER = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
        _BGE_MODEL = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5")
        _BGE_MODEL.eval()
    except Exception as e:
        print(f"[warn] BGE unavailable, similarity disabled: {e}")
        _BGE_MODEL = False
        _BGE_TOKENIZER = False
    return _BGE_MODEL, _BGE_TOKENIZER


# ---------------------------------------------------------------------------
# LLM judge prompt
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """\
You are an evaluator for a port decision-support system's answer quality.
Score the ANSWER against the REFERENCE on three dimensions, each 1-5.

## Scoring Rubric
- **Faithfulness**: Is the answer grounded in the evidence (1=hallucinated, 5=fully grounded)
- **Relevance**: Does the answer address the question (1=off-topic, 5=exactly on-topic)
- **Completeness**: Does it cover the reference answer key points (1=missing all, 5=covers all)

## Question
{question}

## Reference Answer
{reference}

## Evidence Summary
{evidence}

## Candidate Answer
{answer}

Return ONLY a JSON object:
```json
{{
  "faithfulness": 1-5,
  "relevance": 1-5,
  "completeness": 1-5,
  "rationale": "<brief explanation>"
}}
```
"""


# ---------------------------------------------------------------------------
# Objective metrics
# ---------------------------------------------------------------------------

def keyword_coverage(expected_keywords: List[str], answer_text: str) -> float:
    """Fraction of expected keywords present in the answer."""
    if not expected_keywords:
        return 1.0
    text_lower = answer_text.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in text_lower)
    return hits / len(expected_keywords)


def citation_validity(sources_used: List[str], evidence_bundle: Dict[str, Any]) -> float:
    """
    Fraction of cited sources that actually exist in the evidence bundle.
    A citation is valid if it references a source category that has evidence.

    Handles multiple naming conventions:
      - documents / vector / doc
      - sql / sql_results / structured_operational_data / data
      - rules / rule
      - graph / graph_reasoning / knowledge_graph
    """
    if not sources_used:
        return 1.0

    # Normalize a source string to its canonical category
    def _normalize(src: str) -> str:
        s = src.lower().strip().replace(" ", "_")
        if any(k in s for k in ("document", "vector", "handbook", "report", "doc")):
            return "documents"
        if any(k in s for k in ("sql", "structured", "operational_data",
                                 "database", "table")):
            return "sql"
        if "rule" in s or "polic" in s or "threshold" in s:
            return "rules"
        if "graph" in s or "reasoning_path" in s or "knowledge_graph" in s:
            return "graph"
        return s  # unknown — pass through, will be counted as invalid

    available = set()
    if evidence_bundle.get("documents"):
        available.add("documents")
    sql_results = evidence_bundle.get("sql_results") or []
    if any(isinstance(r, dict) and r.get("execution_ok") for r in sql_results) or sql_results:
        available.add("sql")
    rules = evidence_bundle.get("rules", {}) or {}
    if rules.get("matched_rules"):
        available.add("rules")
    graph = evidence_bundle.get("graph", {}) or {}
    if graph.get("reasoning_paths"):
        available.add("graph")

    normalized_citations = [_normalize(s) for s in sources_used]
    valid = sum(1 for s in normalized_citations if s in available)
    return valid / len(normalized_citations) if normalized_citations else 1.0


def numerical_accuracy(
    reference_answer: str, candidate_answer: str, tolerance: float = 0.05
) -> Optional[float]:
    """
    Compare numbers found in reference vs candidate.
    Returns fraction of reference numbers matched within tolerance, or None.
    """
    num_re = re.compile(r"-?\d+\.?\d*")
    ref_nums = [float(x) for x in num_re.findall(reference_answer)]
    cand_nums = [float(x) for x in num_re.findall(candidate_answer)]

    if not ref_nums:
        return None

    matched = 0
    for rn in ref_nums:
        for cn in cand_nums:
            if rn == 0:
                if abs(cn) < tolerance:
                    matched += 1
                    break
            elif abs(cn - rn) / abs(rn) <= tolerance:
                matched += 1
                break

    return matched / len(ref_nums)


def grounding_flag(final_answer: Dict[str, Any]) -> str:
    """Extract grounding status from final_answer dict."""
    if not isinstance(final_answer, dict):
        return "unknown"
    if final_answer.get("knowledge_fallback_used"):
        return "llm_fallback"
    return final_answer.get("grounding_status", "unknown")


# ---------------------------------------------------------------------------
# Semantic similarity (BGE cosine) + ROUGE-L
#
# These are local-only metrics (no API calls). They complement the surface
# keyword_coverage metric by capturing:
#   - embedding_similarity: semantic closeness (paraphrase tolerant)
#   - rougeL_f1:            longest-common-subsequence lexical overlap
# ---------------------------------------------------------------------------

def _bge_encode(text: str):
    """Encode a single string to a BGE embedding vector. Returns None on error."""
    model, tok = _get_bge()
    if model is False:
        return None
    try:
        import torch
        # BGE query prefix improves semantic matching for short texts.
        # For eval we use no prefix — comparing paragraph-to-paragraph.
        enc = tok(
            text if text else " ",
            padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        )
        with torch.no_grad():
            out = model(**enc)
        # Mean pooling over token embeddings (mask-aware)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        summed = (out.last_hidden_state * mask).sum(dim=1)
        lens = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / lens
        # L2 normalize
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled[0].cpu().numpy()
    except Exception:
        return None


def embedding_similarity(reference: str, candidate: str) -> Optional[float]:
    """
    Cosine similarity between BGE embeddings of reference and candidate.
    Returns float in [-1, 1], or None if embeddings unavailable.
    """
    if not reference or not candidate:
        return None
    v_ref = _bge_encode(reference)
    v_cand = _bge_encode(candidate)
    if v_ref is None or v_cand is None:
        return None
    # Vectors are L2-normalized, so dot == cosine
    return float((v_ref * v_cand).sum())


def _tokenize_for_rouge(text: str) -> List[str]:
    """Lowercase word tokens, dropping punctuation for ROUGE scoring."""
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _lcs_length(a: List[str], b: List[str]) -> int:
    """Classic DP longest common subsequence length."""
    if not a or not b:
        return 0
    la, lb = len(a), len(b)
    # Use rolling array to keep memory O(min(la, lb))
    if la < lb:
        a, b = b, a
        la, lb = lb, la
    prev = [0] * (lb + 1)
    curr = [0] * (lb + 1)
    for i in range(1, la + 1):
        ai = a[i - 1]
        for j in range(1, lb + 1):
            if ai == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev
    return prev[lb]


def rouge_l_f1(reference: str, candidate: str) -> Optional[float]:
    """
    ROUGE-L F1 based on longest common subsequence.
    Returns float in [0, 1], or None if inputs are empty.
    """
    ref_tokens = _tokenize_for_rouge(reference)
    cand_tokens = _tokenize_for_rouge(candidate)
    if not ref_tokens or not cand_tokens:
        return None
    lcs = _lcs_length(ref_tokens, cand_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(cand_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

def llm_judge(
    question: str,
    reference: str,
    evidence_summary: str,
    candidate_answer: str,
) -> Dict[str, Any]:
    """Call LLM judge to score the answer. Returns dict with scores."""
    if llm_chat_json is None:
        return {"error": "llm_client not available"}

    prompt = JUDGE_PROMPT.format(
        question=question,
        reference=reference,
        evidence=evidence_summary[:2000],
        answer=candidate_answer[:2000],
    )
    result = llm_chat_json(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Evaluate and return JSON."},
        ],
        temperature=0.0,
        timeout=60,
    )
    if not isinstance(result, dict):
        return {"error": "llm judge failed"}
    return {
        "faithfulness": int(result.get("faithfulness", 0)),
        "relevance": int(result.get("relevance", 0)),
        "completeness": int(result.get("completeness", 0)),
        "rationale": result.get("rationale", ""),
    }


# ---------------------------------------------------------------------------
# Main evaluation entry
# ---------------------------------------------------------------------------

@dataclass
class AnswerQualityMetrics:
    avg_keyword_coverage: float = 0.0
    avg_citation_validity: float = 0.0
    avg_numerical_accuracy: Optional[float] = None
    grounding_distribution: Dict[str, int] = field(default_factory=dict)

    # Local semantic/lexical overlap (no API cost)
    avg_embedding_similarity: Optional[float] = None
    avg_rougeL_f1: Optional[float] = None
    samples_similarity_scored: int = 0

    # LLM judge scores
    avg_faithfulness: float = 0.0
    avg_relevance: float = 0.0
    avg_completeness: float = 0.0

    samples_total: int = 0
    samples_llm_judged: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "avg_keyword_coverage": round(self.avg_keyword_coverage, 4),
            "avg_citation_validity": round(self.avg_citation_validity, 4),
            "avg_numerical_accuracy": (
                round(self.avg_numerical_accuracy, 4) if self.avg_numerical_accuracy is not None else None
            ),
            "avg_embedding_similarity": (
                round(self.avg_embedding_similarity, 4) if self.avg_embedding_similarity is not None else None
            ),
            "avg_rougeL_f1": (
                round(self.avg_rougeL_f1, 4) if self.avg_rougeL_f1 is not None else None
            ),
            "samples_similarity_scored": self.samples_similarity_scored,
            "grounding_distribution": self.grounding_distribution,
            "avg_faithfulness": round(self.avg_faithfulness, 3),
            "avg_relevance": round(self.avg_relevance, 3),
            "avg_completeness": round(self.avg_completeness, 3),
            "samples_total": self.samples_total,
            "samples_llm_judged": self.samples_llm_judged,
        }


def evaluate_answers(
    results: List[Dict[str, Any]],
    golden: List[Dict[str, Any]],
    use_llm_judge: bool = True,
    max_llm_samples: int = 30,
) -> AnswerQualityMetrics:
    """
    Evaluate answer quality for a list of agent results.

    Each result should contain:
        - id: matching golden id
        - answer_text: string
        - sources_used: list of strings
        - evidence_bundle: dict (documents, sql_results, rules, graph)
        - final_answer: full dict with grounding_status, etc.
    """
    golden_by_id = {g["id"]: g for g in golden}

    kw_total = 0.0
    cite_total = 0.0
    num_scores = []
    sim_scores: List[float] = []
    rouge_scores: List[float] = []
    grounding_dist: Dict[str, int] = {}

    judge_f, judge_r, judge_c = 0.0, 0.0, 0.0
    judged_count = 0

    for idx, r in enumerate(results):
        g = golden_by_id.get(r.get("id"))
        if not g:
            continue

        answer = r.get("answer_text", "") or ""
        sources = r.get("sources_used", []) or []
        bundle = r.get("evidence_bundle", {}) or {}
        final = r.get("final_answer", {}) or {}
        reference = g.get("reference_answer", "") or ""

        # Objective metrics
        kw_total += keyword_coverage(g.get("expected_evidence_keywords", []), answer)
        cite_total += citation_validity(sources, bundle)

        num_acc = numerical_accuracy(reference, answer)
        if num_acc is not None:
            num_scores.append(num_acc)

        # Semantic + lexical similarity (local, no API cost)
        if reference and answer:
            sim = embedding_similarity(reference, answer)
            if sim is not None:
                sim_scores.append(sim)
            rouge = rouge_l_f1(reference, answer)
            if rouge is not None:
                rouge_scores.append(rouge)

        grounding = grounding_flag(final)
        grounding_dist[grounding] = grounding_dist.get(grounding, 0) + 1

        # LLM judge (capped to control cost)
        if use_llm_judge and judged_count < max_llm_samples:
            evidence_summary = _build_evidence_summary(bundle)
            judge = llm_judge(
                question=g.get("query", ""),
                reference=reference,
                evidence_summary=evidence_summary,
                candidate_answer=answer,
            )
            if "error" not in judge:
                judge_f += judge["faithfulness"]
                judge_r += judge["relevance"]
                judge_c += judge["completeness"]
                judged_count += 1

    total = len(results)
    return AnswerQualityMetrics(
        avg_keyword_coverage=kw_total / total if total else 0.0,
        avg_citation_validity=cite_total / total if total else 0.0,
        avg_numerical_accuracy=sum(num_scores) / len(num_scores) if num_scores else None,
        avg_embedding_similarity=(sum(sim_scores) / len(sim_scores)) if sim_scores else None,
        avg_rougeL_f1=(sum(rouge_scores) / len(rouge_scores)) if rouge_scores else None,
        samples_similarity_scored=len(sim_scores),
        grounding_distribution=grounding_dist,
        avg_faithfulness=judge_f / judged_count if judged_count else 0.0,
        avg_relevance=judge_r / judged_count if judged_count else 0.0,
        avg_completeness=judge_c / judged_count if judged_count else 0.0,
        samples_total=total,
        samples_llm_judged=judged_count,
    )


def _build_evidence_summary(bundle: Dict[str, Any]) -> str:
    """Build a compact text summary of evidence for the LLM judge."""
    parts = []
    docs = bundle.get("documents", [])
    if docs:
        parts.append(f"Documents: {len(docs)} chunks")
        for d in docs[:3]:
            parts.append(f"  - {(d.get('text', '') or '')[:150]}")
    sql = bundle.get("sql_results", [])
    if sql:
        parts.append(f"SQL results: {len(sql)} queries")
        for r in sql[:2]:
            if isinstance(r, dict):
                parts.append(f"  - {json.dumps(r.get('rows', [])[:3], default=str)[:200]}")
    rules = bundle.get("rules", {})
    if rules and isinstance(rules, dict):
        matched = rules.get("matched_rules", [])
        if matched:
            parts.append(f"Rules: {len(matched)} matched")
            for rm in matched[:2]:
                parts.append(f"  - {(rm.get('rule_text', '') or '')[:120]}")
    return "\n".join(parts)


def print_answer_report(metrics: AnswerQualityMetrics) -> None:
    print("\n" + "=" * 70)
    print("  ANSWER QUALITY EVALUATION")
    print("=" * 70)
    print(f"  Samples: {metrics.samples_total}")
    print(f"\n  Objective metrics:")
    print(f"    Keyword coverage:    {metrics.avg_keyword_coverage:.2%}")
    print(f"    Citation validity:   {metrics.avg_citation_validity:.2%}")
    if metrics.avg_numerical_accuracy is not None:
        print(f"    Numerical accuracy:  {metrics.avg_numerical_accuracy:.2%}")
    if metrics.avg_embedding_similarity is not None:
        print(f"    Embedding cosine:    {metrics.avg_embedding_similarity:.4f}"
              f"  (n={metrics.samples_similarity_scored})")
    if metrics.avg_rougeL_f1 is not None:
        print(f"    ROUGE-L F1:          {metrics.avg_rougeL_f1:.4f}")
    print(f"\n  Grounding distribution: {metrics.grounding_distribution}")
    if metrics.samples_llm_judged > 0:
        print(f"\n  LLM judge scores ({metrics.samples_llm_judged} samples, 1-5 scale):")
        print(f"    Faithfulness:  {metrics.avg_faithfulness:.2f}")
        print(f"    Relevance:     {metrics.avg_relevance:.2f}")
        print(f"    Completeness:  {metrics.avg_completeness:.2f}")
