# src/online_pipeline/source_registry.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class SourceRegistry:
    """
    Centralized file/path registry for the online pipeline.

    Assumes the following project structure:

    RAG-LLM-for-Ports-main/
    ├─ data/
    │  ├─ chunks/
    │  ├─ rules/
    │  └─ sql_data/
    ├─ src/
    │  ├─ offline_pipeline/
    │  └─ online_pipeline/
    └─ storage/
       └─ chroma/
    """

    project_root: Path
    data_dir: Path
    chunks_dir: Path
    rules_dir: Path
    sql_data_dir: Path
    storage_dir: Path
    chroma_dir: Path

    chunks_file: Path
    chunks_with_embeddings_file: Path

    rule_candidate_chunks_file: Path
    raw_rules_file: Path
    grounded_rules_file: Path
    policy_rules_file: Path

    @classmethod
    def from_project_root(cls, project_root: str | Path) -> "SourceRegistry":
        root = Path(project_root).resolve()

        data_dir = root / "data"
        chunks_dir = data_dir / "chunks"
        rules_dir = data_dir / "rules"
        sql_data_dir = data_dir / "sql_data"
        storage_dir = root / "storage"
        chroma_dir = storage_dir / "chroma"

        return cls(
            project_root=root,
            data_dir=data_dir,
            chunks_dir=chunks_dir,
            rules_dir=rules_dir,
            sql_data_dir=sql_data_dir,
            storage_dir=storage_dir,
            chroma_dir=chroma_dir,
            chunks_file=chunks_dir / "chunks_v1.json",
            chunks_with_embeddings_file=chunks_dir / "chunks_with_embeddings_v1.json",
            rule_candidate_chunks_file=rules_dir / "rule_candidate_chunks_v1.json",
            raw_rules_file=rules_dir / "raw_rules.json",
            grounded_rules_file=rules_dir / "grounded_rules.json",
            policy_rules_file=rules_dir / "policy_rules.json",
        )

    def as_dict(self) -> Dict[str, str]:
        return {
            "project_root": str(self.project_root),
            "data_dir": str(self.data_dir),
            "chunks_dir": str(self.chunks_dir),
            "rules_dir": str(self.rules_dir),
            "sql_data_dir": str(self.sql_data_dir),
            "storage_dir": str(self.storage_dir),
            "chroma_dir": str(self.chroma_dir),
            "chunks_file": str(self.chunks_file),
            "chunks_with_embeddings_file": str(self.chunks_with_embeddings_file),
            "rule_candidate_chunks_file": str(self.rule_candidate_chunks_file),
            "raw_rules_file": str(self.raw_rules_file),
            "grounded_rules_file": str(self.grounded_rules_file),
            "policy_rules_file": str(self.policy_rules_file),
        }

    def validate_basic_paths(self) -> Dict[str, bool]:
        return {
            "project_root_exists": self.project_root.exists(),
            "data_dir_exists": self.data_dir.exists(),
            "chunks_dir_exists": self.chunks_dir.exists(),
            "rules_dir_exists": self.rules_dir.exists(),
            "sql_data_dir_exists": self.sql_data_dir.exists(),
            "storage_dir_exists": self.storage_dir.exists(),
            "chroma_dir_exists": self.chroma_dir.exists(),
            "chunks_file_exists": self.chunks_file.exists(),
            "chunks_with_embeddings_file_exists": self.chunks_with_embeddings_file.exists(),
            "rule_candidate_chunks_file_exists": self.rule_candidate_chunks_file.exists(),
            "raw_rules_file_exists": self.raw_rules_file.exists(),
            "grounded_rules_file_exists": self.grounded_rules_file.exists(),
            "policy_rules_file_exists": self.policy_rules_file.exists(),
        }