# src/online_pipeline/parent_store.py
"""
Parent chunk store for Small-to-Big retrieval.

In the Small-to-Big pattern:
- Children chunks (~250 words) live in the vector DB, used for precise retrieval
- Parent chunks (~1500 words) live in this store, fetched by parent_id to
  supply rich context to the LLM during answer generation

The store is a simple in-memory dict loaded from chunks_v2_parents.json.
For very large collections, this could be backed by SQLite or a KV store.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("online_pipeline.parent_store")


class ParentChunkStore:
    """
    In-memory key-value store mapping parent_id -> parent chunk dict.

    Loaded once at startup from chunks_v2_parents.json. Thread-safe for reads
    (dict lookups are atomic in CPython).
    """

    def __init__(self, parents_file: Optional[str | Path] = None) -> None:
        self._store: Dict[str, Dict[str, Any]] = {}
        self._loaded = False
        if parents_file:
            self.load(parents_file)

    def load(self, parents_file: str | Path) -> None:
        """Load parents from a JSON file."""
        path = Path(parents_file)
        if not path.exists():
            logger.warning("Parent file %s does not exist — store will be empty", path)
            return

        with open(path, "r", encoding="utf-8") as f:
            parents = json.load(f)

        self._store = {p["chunk_id"]: p for p in parents if p.get("chunk_id")}
        self._loaded = True
        logger.info("ParentChunkStore loaded %d parents from %s", len(self._store), path)

    def get(self, parent_id: str) -> Optional[Dict[str, Any]]:
        """Get a single parent by ID."""
        return self._store.get(parent_id)

    def get_many(self, parent_ids: List[str]) -> List[Dict[str, Any]]:
        """Get multiple parents, preserving order and skipping misses."""
        result = []
        seen = set()  # dedupe — multiple children may point to same parent
        for pid in parent_ids:
            if pid and pid not in seen and pid in self._store:
                result.append(self._store[pid])
                seen.add(pid)
        return result

    def is_loaded(self) -> bool:
        return self._loaded

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, parent_id: str) -> bool:
        return parent_id in self._store
