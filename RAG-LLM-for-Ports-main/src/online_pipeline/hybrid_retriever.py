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
        dense_weight: float = 1.0,
        bm25_weight: float = 1.2,
        enable_small_to_big: bool = True,
    ) -> None:
        self.registry = registry
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

        # Dense retriever (existing)
        self.dense_retriever = ChromaDocumentRetriever(
            registry=registry,
            collection_name=collection_name,
        )

        # Build BM25 index from chunk texts
        self.chunks, self.bm25 = self._build_bm25_index()

        # Small-to-Big: load parent store if parents file exists
        self.parent_store = None
        if enable_small_to_big:
            try:
                from .parent_store import ParentChunkStore
                parents_path = self.registry.chunks_dir / "chunks_v2_parents.json"
                if parents_path.exists():
                    self.parent_store = ParentChunkStore(parents_path)
                    logger.info(
                        "Small-to-Big enabled: %d parents loaded",
                        len(self.parent_store),
                    )
                else:
                    logger.info("No parents file; Small-to-Big disabled")
            except Exception as e:
                logger.warning("Failed to load parent store: %s", e)

    def _build_bm25_index(self):
        # Prefer v2 chunks if they exist (matches new BGE Chroma collection)
        chunks_v2_path = self.registry.chunks_dir / "chunks_v2.json"
        chunks_path = chunks_v2_path if chunks_v2_path.exists() else self.registry.chunks_file
        logger.info("BM25 index source: %s", chunks_path.name)

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
        return_parents: bool = False,
    ) -> List[RetrievedDocument]:
        """
        Hybrid retrieve: dense + BM25 with RRF fusion.

        Args:
            query: natural language query
            top_k: number of children (or parents if return_parents=True) to return
            return_parents: if True and parent_store is loaded, replace children
                with their parent chunks (deduped, preserving rank order).
                Use this for generation context. Use False for metric evaluation
                (since golden datasets reference child chunk_ids).
        """
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

        top_children = fused[:top_k]

        # Small-to-Big: replace children with parents if requested
        if return_parents and self.parent_store is not None:
            return self._children_to_parents(top_children)

        return top_children

    def _children_to_parents(
        self, children: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """
        Replace each retrieved child with its parent chunk (deduped).
        Preserves rank order and propagates scores from the best matching child.
        """
        parent_best_score: Dict[str, float] = {}
        parent_best_child: Dict[str, RetrievedDocument] = {}

        for child in children:
            # Look up parent_id from metadata (set by document_retriever for v2 collections)
            parent_id = (
                child.get("parent_id")
                or (child.get("metadata") or {}).get("parent_id")
            )
            if not parent_id:
                # No parent link — keep the child as-is (fallback)
                pid = f"self::{child.get('chunk_id', '')}"
                parent_best_score.setdefault(pid, child.get("score") or 0)
                parent_best_child.setdefault(pid, child)
                continue

            score = child.get("score") or 0
            if parent_id not in parent_best_score or score > parent_best_score[parent_id]:
                parent_best_score[parent_id] = score
                parent_best_child[parent_id] = child

        # Sort parent IDs by best score and fetch parent chunks
        sorted_pids = sorted(
            parent_best_score.keys(),
            key=lambda p: parent_best_score[p],
            reverse=True,
        )

        result: List[RetrievedDocument] = []
        for pid in sorted_pids:
            if pid.startswith("self::"):
                result.append(parent_best_child[pid])
                continue
            parent = self.parent_store.get(pid)
            if parent is None:
                # Parent missing — fall back to child
                result.append(parent_best_child[pid])
                continue
            # Build a RetrievedDocument from the parent, inheriting the score
            parent_doc: RetrievedDocument = {
                "chunk_id": parent.get("chunk_id", ""),
                "doc_id": parent.get("doc_id"),
                "source_file": parent.get("source_file", ""),
                "page": parent.get("page", -1),
                "text": parent.get("text", ""),
                "score": parent_best_score[pid],
            }
            # Carry metadata forward
            for key in (
                "section_number", "section_title", "doc_type",
                "category", "publish_year", "is_table", "word_count",
            ):
                if key in parent:
                    parent_doc[key] = parent[key]
            result.append(parent_doc)

        logger.info(
            "SMALL_TO_BIG: %d children -> %d unique parents",
            len(children), len(result),
        )
        return result

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

        # Score from dense list (weighted)
        for rank, doc in enumerate(dense_results):
            key = doc.get("chunk_id", "") or doc.get("text", "")[:50]
            rrf_score = self.dense_weight / (self.rrf_k + rank + 1)
            doc_scores[key] = doc_scores.get(key, 0) + rrf_score
            doc_map[key] = doc

        # Score from BM25 list (weighted — slightly higher default for
        # port domain which has many acronyms: LOA, TEU, ISPS, etc.)
        for rank, doc in enumerate(bm25_results):
            key = doc.get("chunk_id", "") or doc.get("text", "")[:50]
            rrf_score = self.bm25_weight / (self.rrf_k + rank + 1)
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
