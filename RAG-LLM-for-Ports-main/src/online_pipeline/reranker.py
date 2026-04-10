# src/online_pipeline/reranker.py
"""
Cross-Encoder Reranker for retrieved documents.

Uses a pointwise cross-encoder model to re-score (query, document) pairs.
The cross-encoder jointly encodes query and document, producing a more
accurate relevance score than bi-encoder similarity.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (~80MB, fast inference)

Design rationale:
- Bi-encoder (embedding) retrieval is fast but approximate
- Cross-encoder is slower but significantly more accurate for top-k reranking
- We retrieve top-20 with hybrid search, then rerank to top-5

Future improvements:
- Fine-tune reranker on domain-specific port operations query-document pairs
  using hard negatives mined from the same collection
- Consider listwise reranking (e.g., LLM-based listwise scoring)
- Distillation from larger cross-encoder to smaller model for latency
"""

from __future__ import annotations

from typing import List

import logging

from sentence_transformers import CrossEncoder

from .state_schema import RetrievedDocument

logger = logging.getLogger("online_pipeline.reranker")


class CrossEncoderReranker:

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cuda",
        default_top_k: int = 5,
    ) -> None:
        self.model = CrossEncoder(model_name, device=device)
        self.default_top_k = default_top_k

    def rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_k: int | None = None,
    ) -> List[RetrievedDocument]:
        """
        Re-score documents using the cross-encoder and return top_k.

        Each (query, document_text) pair is scored jointly by the model.
        """
        if not documents:
            return []

        top_k = top_k or self.default_top_k

        # Build input pairs
        pairs = [(query, doc.get("text", "")) for doc in documents]

        # Score
        scores = self.model.predict(pairs)

        # Attach scores and sort
        scored_docs = []
        for doc, score in zip(documents, scores):
            reranked = doc.copy()
            reranked["score"] = round(float(score), 6)
            scored_docs.append(reranked)

        scored_docs.sort(key=lambda d: d["score"], reverse=True)

        result = scored_docs[:top_k]
        if result:
            score_before = [round(float(s), 4) for s in scores[:3]]
            score_after = [d["score"] for d in result[:3]]
            logger.info(
                "RERANK: %d -> %d docs | before_top3=%s after_top3=%s",
                len(documents), len(result), score_before, score_after,
            )
        else:
            logger.info("RERANK: no documents to rerank")

        return result
