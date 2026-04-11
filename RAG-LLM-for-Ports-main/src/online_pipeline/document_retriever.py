# src/online_pipeline/document_retriever.py

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .source_registry import SourceRegistry
from .state_schema import PortQAState, RetrievedDocument

try:
    import chromadb
except ImportError as e:
    raise ImportError(
        "chromadb is required for document retrieval. "
        "Please install it with: pip install chromadb"
    ) from e

logger = logging.getLogger("online_pipeline.document_retriever")

# BGE query prefix — only used when v2 collection is active
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Collections that use BGE (and therefore need the query prefix)
_BGE_COLLECTIONS = {"port_documents_v2"}


class ChromaDocumentRetriever:
    """
    Chroma-backed retriever for document chunks.

    Supports two collection schemas:
    - `port_documents` (v1): default Chroma embedding (all-MiniLM-L6-v2),
      chunk_id stored as Chroma ID, minimal metadata.
    - `port_documents_v2` (v2): BGE-base embeddings (externally encoded),
      rich metadata (section_title, doc_type, is_table, etc.), BGE query prefix.

    The retriever auto-detects which mode to use based on collection name
    and embedding model availability.
    """

    def __init__(
        self,
        registry: SourceRegistry,
        collection_name: Optional[str] = None,
    ) -> None:
        self.registry = registry
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=str(self.registry.chroma_dir))
        self.collection = self._load_collection()

        # If the collection is BGE-backed, load the model for query encoding
        self._bge_model = None
        active_name = getattr(self.collection, "name", "")
        if active_name in _BGE_COLLECTIONS:
            try:
                from sentence_transformers import SentenceTransformer
                self._bge_model = SentenceTransformer(
                    "BAAI/bge-base-en-v1.5",
                    device="cuda",
                )
                self._bge_model.max_seq_length = 512
                logger.info("Loaded BGE-base model for collection %s", active_name)
            except Exception as e:
                logger.warning(
                    "BGE collection %s selected but model failed to load: %s. "
                    "Falling back to Chroma default embedding.",
                    active_name, e,
                )

    def _load_collection(self):
        collections = self.client.list_collections()
        if not collections:
            raise ValueError(
                f"No Chroma collections found under: {self.registry.chroma_dir}"
            )

        if self.collection_name is not None:
            return self.client.get_collection(name=self.collection_name)

        # fallback: use the first available collection
        first_collection = collections[0]
        name = getattr(first_collection, "name", None) or str(first_collection)
        return self.client.get_collection(name=name)

    def list_collection_names(self) -> List[str]:
        collections = self.client.list_collections()
        names: List[str] = []
        for c in collections:
            names.append(getattr(c, "name", str(c)))
        return names

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[RetrievedDocument]:
        # If BGE collection is active, encode query with prefix and pass embeddings
        if self._bge_model is not None:
            prefixed = BGE_QUERY_PREFIX + query
            query_emb = self._bge_model.encode(
                [prefixed],
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).tolist()
            results = self.collection.query(
                query_embeddings=query_emb,
                n_results=top_k,
            )
        else:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
            )

        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved: List[RetrievedDocument] = []
        for chroma_id, text, metadata, distance in zip(ids, docs, metadatas, distances):
            metadata = metadata or {}

            # Chroma ID is the canonical chunk_id (format: docid_section_part).
            # Fall back to metadata.chunk_id if present, else use the Chroma ID.
            chunk_id = str(metadata.get("chunk_id") or chroma_id or "")

            item: RetrievedDocument = {
                "chunk_id": chunk_id,
                "doc_id": metadata.get("doc_id"),
                "source_file": str(metadata.get("source_file", "")),
                "page": int(metadata.get("page", -1)) if metadata.get("page") is not None else -1,
                "text": text,
                "score": self._distance_to_score(distance),
            }
            # Propagate v2 metadata if present (section info, doc_type, is_table,
            # parent_id for small-to-big lookup, category + publish_year for filtering)
            for key in (
                "parent_id", "section_number", "section_title",
                "doc_type", "category", "publish_year",
                "is_table", "word_count",
            ):
                if key in metadata:
                    item[key] = metadata[key]
            retrieved.append(item)

        return retrieved

    @staticmethod
    def _distance_to_score(distance: Any) -> Optional[float]:
        """
        Convert Chroma distance to a similarity-like score.

        This is heuristic because the exact meaning depends on the embedding space/index config.
        We keep it simple for MVP:
            score = 1 / (1 + distance)
        """
        try:
            d = float(distance)
            return round(1.0 / (1.0 + d), 4)
        except (TypeError, ValueError):
            return None

    def update_state(
        self,
        state: PortQAState,
        top_k: int = 5,
    ) -> PortQAState:
        query = state.get("user_query", "")
        retrieved_docs = self.retrieve(query=query, top_k=top_k)

        state["retrieved_docs"] = retrieved_docs

        trace = state.get("reasoning_trace", [])
        trace.append(
            f"ChromaDocumentRetriever => retrieved {len(retrieved_docs)} document chunks from Chroma."
        )
        state["reasoning_trace"] = trace

        return state


def retrieve_documents(
    project_root: str | Path,
    query: str,
    top_k: int = 5,
    collection_name: Optional[str] = None,
) -> List[RetrievedDocument]:
    registry = SourceRegistry.from_project_root(project_root)
    retriever = ChromaDocumentRetriever(registry=registry, collection_name=collection_name)
    return retriever.retrieve(query=query, top_k=top_k)


if __name__ == "__main__":
    # Example usage
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    registry = SourceRegistry.from_project_root(PROJECT_ROOT)
    retriever = ChromaDocumentRetriever(registry=registry)

    print("Available collections:", retriever.list_collection_names())

    query = "What does the handbook say about restricted night navigation?"
    docs = retriever.retrieve(query, top_k=3)

    print("=" * 80)
    print("QUERY:", query)
    for i, doc in enumerate(docs, start=1):
        print(f"[{i}] source={doc.get('source_file')} page={doc.get('page')} score={doc.get('score')}")
        print(doc.get("text", "")[:400])
        print("-" * 80)