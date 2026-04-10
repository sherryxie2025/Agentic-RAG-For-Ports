# src/online_pipeline/hybrid_retriever.py
"""
Hybrid Retriever: BM25 (sparse) + Dense (ChromaDB) + Reciprocal Rank Fusion.

Design rationale:
- Dense retrieval captures semantic similarity but misses exact keyword matches
- BM25 captures exact term matching (acronyms like LOA, TEU, ISPS)
- RRF fuses both ranked lists without requiring score normalization

Future improvements:
- Learned sparse retrieval (SPLADE) to replace BM25
- Query-adaptive weight between dense and sparse
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import logging

from rank_bm25 import BM25Okapi

from .document_retriever import ChromaDocumentRetriever
from .source_registry import SourceRegistry
from .state_schema import RetrievedDocument

logger = logging.getLogger("online_pipeline.hybrid_retriever")


class HybridDocumentRetriever:

    def __init__(
        self,
        registry: SourceRegistry,
        collection_name: Optional[str] = None,
        rrf_k: int = 60,
    ) -> None:
        self.registry = registry
        self.rrf_k = rrf_k

        # Dense retriever (existing)
        self.dense_retriever = ChromaDocumentRetriever(
            registry=registry,
            collection_name=collection_name,
        )

        # Build BM25 index from chunk texts
        self.chunks, self.bm25 = self._build_bm25_index()

    def _build_bm25_index(self):
        chunks_path = self.registry.chunks_file
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        # Tokenize for BM25
        tokenized = [c["text"].lower().split() for c in chunks]
        bm25 = BM25Okapi(tokenized)
        return chunks, bm25

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[RetrievedDocument]:
        """Hybrid retrieve: dense + BM25 with RRF fusion."""

        # 1. Dense retrieval
        dense_results = self.dense_retriever.retrieve(query=query, top_k=top_k)

        # 2. BM25 retrieval
        bm25_results = self._bm25_retrieve(query, top_k=top_k)

        # 3. Reciprocal Rank Fusion
        fused = self._rrf_fuse(dense_results, bm25_results)

        dense_top = dense_results[0]["score"] if dense_results else 0
        bm25_top = bm25_results[0]["score"] if bm25_results else 0
        fused_top = fused[0]["score"] if fused else 0
        logger.info(
            "RETRIEVE: dense=%d (top=%.4f) bm25=%d (top=%.4f) fused=%d (top=%.6f)",
            len(dense_results), dense_top,
            len(bm25_results), bm25_top,
            len(fused[:top_k]), fused_top,
        )
        if fused:
            logger.debug("Top doc: %s (score=%.6f)", fused[0].get("source_file", "?"), fused_top)

        return fused[:top_k]

    def _bm25_retrieve(self, query: str, top_k: int = 10) -> List[RetrievedDocument]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top_k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] <= 0:
                continue
            chunk = self.chunks[idx]
            doc: RetrievedDocument = {
                "chunk_id": str(chunk.get("chunk_id", "")),
                "doc_id": chunk.get("doc_id"),
                "source_file": str(chunk.get("source_file", "")),
                "page": int(chunk.get("page", -1)),
                "text": chunk["text"],
                "score": float(scores[idx]),
            }
            results.append(doc)

        return results

    def _rrf_fuse(
        self,
        dense_results: List[RetrievedDocument],
        bm25_results: List[RetrievedDocument],
    ) -> List[RetrievedDocument]:
        """Reciprocal Rank Fusion: score = sum(1 / (k + rank)) across lists."""

        doc_scores: Dict[str, float] = {}
        doc_map: Dict[str, RetrievedDocument] = {}

        # Score from dense list
        for rank, doc in enumerate(dense_results):
            key = doc.get("chunk_id", "") or doc.get("text", "")[:50]
            rrf_score = 1.0 / (self.rrf_k + rank + 1)
            doc_scores[key] = doc_scores.get(key, 0) + rrf_score
            doc_map[key] = doc

        # Score from BM25 list
        for rank, doc in enumerate(bm25_results):
            key = doc.get("chunk_id", "") or doc.get("text", "")[:50]
            rrf_score = 1.0 / (self.rrf_k + rank + 1)
            doc_scores[key] = doc_scores.get(key, 0) + rrf_score
            if key not in doc_map:
                doc_map[key] = doc

        # Sort by fused score
        sorted_keys = sorted(doc_scores.keys(), key=lambda k: doc_scores[k], reverse=True)

        fused = []
        for key in sorted_keys:
            doc = doc_map[key].copy()
            doc["score"] = round(doc_scores[key], 6)
            fused.append(doc)

        return fused

    def list_collection_names(self) -> List[str]:
        return self.dense_retriever.list_collection_names()
