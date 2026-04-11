# src/offline_pipeline/synonym_expander.py
"""
LLM-powered synonym expansion with persistent cache.

When the rule grounder encounters a variable name it doesn't recognize
(e.g., "average wind velocity" from the LLM extractor), this module asks
an LLM to pick the closest canonical variable from the auto-generated
taxonomy, then caches the mapping so future queries are free.

Design:
- Cache stored as JSON at data/rules/synonym_cache.json
- Cache survives across runs, grows monotonically
- Each mapping has a confidence score (LLM self-rated)
- Low-confidence mappings (<0.5) are NOT cached (to avoid poisoning)
- Simple thread-unsafe singleton — this is an offline/batch step

Usage:
    from .synonym_expander import SynonymExpander
    expander = SynonymExpander()
    canonical = expander.resolve("average wind velocity")  # -> "wind_speed_ms"
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("offline_pipeline.synonym_expander")

CACHE_PATH = "data/rules/synonym_cache.json"

# Prompt template for LLM synonym resolution
RESOLVE_PROMPT = """\
You are matching a rule variable to a canonical variable in a port
operations database. Given the user's variable phrase and the list of
available canonical variables, return the closest match.

## Available Canonical Variables
{canonical_list}

## Input Variable (to match)
"{variable}"

## Instructions
- Return the canonical variable that most likely refers to the same concept.
- If no good match exists (similarity < 50%), return "null".
- Also return a confidence score (0.0 to 1.0).

## Output Format
Return ONLY JSON:
```json
{{
  "canonical": "<canonical_name_or_null>",
  "confidence": 0.0-1.0,
  "reasoning": "<one sentence>"
}}
```
"""


class SynonymExpander:
    """Persistent LLM-backed synonym cache."""

    def __init__(
        self,
        taxonomy: Optional[Dict[str, Any]] = None,
        cache_path: str = CACHE_PATH,
        confidence_threshold: float = 0.5,
    ) -> None:
        # Lazy-load llm_client to avoid import cycles
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(PROJECT_ROOT / "src"))

        self.cache_path = Path(cache_path)
        self.confidence_threshold = confidence_threshold
        self.cache: Dict[str, Dict[str, Any]] = self._load_cache()
        self.taxonomy = taxonomy
        self._canonical_list: Optional[List[str]] = None
        self._llm_calls = 0
        self._cache_hits = 0
        self._resolved = 0

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Failed to load synonym cache: %s", e)
        return {}

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)

    def _get_canonical_list(self) -> List[str]:
        if self._canonical_list is None:
            if self.taxonomy is None:
                from .taxonomy_generator import load_auto_taxonomy
                self.taxonomy = load_auto_taxonomy()
            self._canonical_list = sorted(self.taxonomy.get("variable_meta", {}).keys())
        return self._canonical_list

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def resolve(
        self,
        variable: str,
        use_llm_fallback: bool = True,
    ) -> Optional[str]:
        """
        Resolve a variable string to a canonical name.

        Resolution order:
        1. Exact match in cache
        2. Substring/normalized lookup in taxonomy synonym_map
        3. LLM fallback (if enabled) — cached on high confidence
        """
        if not variable:
            return None

        key = variable.lower().strip()

        # 1. Cache lookup
        if key in self.cache:
            self._cache_hits += 1
            entry = self.cache[key]
            if entry.get("confidence", 0) >= self.confidence_threshold:
                self._resolved += 1
                return entry["canonical"]
            return None

        # 2. Static synonym map (from taxonomy)
        if self.taxonomy is None:
            from .taxonomy_generator import load_auto_taxonomy
            self.taxonomy = load_auto_taxonomy()

        synonym_map = self.taxonomy.get("synonym_map", {})
        if key in synonym_map:
            canonical = synonym_map[key]
            self.cache[key] = {
                "canonical": canonical,
                "confidence": 1.0,
                "source": "taxonomy_synonym_map",
            }
            self._save_cache()
            self._resolved += 1
            return canonical

        # Token-level match: split query into tokens; if ALL tokens of a
        # seed synonym appear in the query (whole words), it's a good match.
        # This handles cases like "berth productivity" -> berth_productivity_mph
        # without over-matching "berth moves per hour" to random things.
        query_tokens = set(key.replace("_", " ").split())
        if query_tokens:
            for syn_key, syn_val in synonym_map.items():
                syn_tokens = set(syn_key.replace("_", " ").split())
                if syn_tokens and syn_tokens.issubset(query_tokens) and len(syn_tokens) >= 2:
                    # Multi-token syn fully contained in query
                    self.cache[key] = {
                        "canonical": syn_val,
                        "confidence": 0.8,
                        "source": "taxonomy_token_subset",
                    }
                    self._save_cache()
                    self._resolved += 1
                    return syn_val

        # 3. LLM fallback
        if not use_llm_fallback:
            return None

        return self._llm_resolve(variable)

    def _llm_resolve(self, variable: str) -> Optional[str]:
        """Ask LLM to pick the closest canonical variable."""
        from online_pipeline.llm_client import llm_chat_json

        canonical_list = self._get_canonical_list()
        canonical_str = "\n".join(f"- {v}" for v in canonical_list[:80])

        prompt = RESOLVE_PROMPT.format(
            canonical_list=canonical_str,
            variable=variable,
        )
        self._llm_calls += 1
        result = llm_chat_json(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Match: {variable}"},
            ],
            temperature=0.0,
            timeout=20,
            max_retries=0,
        )

        if not isinstance(result, dict):
            logger.warning("SYNONYM LLM failed for: %s", variable)
            return None

        canonical = result.get("canonical")
        confidence = float(result.get("confidence", 0))

        # Validate: canonical must exist in taxonomy
        if canonical == "null" or canonical is None:
            self.cache[variable.lower().strip()] = {
                "canonical": None,
                "confidence": confidence,
                "source": "llm_rejected",
                "reasoning": result.get("reasoning", ""),
            }
            self._save_cache()
            return None

        if canonical not in canonical_list:
            logger.warning(
                "SYNONYM LLM returned unknown variable: %s (expected one of %d)",
                canonical, len(canonical_list),
            )
            return None

        # Cache only if confident
        key = variable.lower().strip()
        self.cache[key] = {
            "canonical": canonical,
            "confidence": confidence,
            "source": "llm",
            "reasoning": result.get("reasoning", ""),
        }
        self._save_cache()

        if confidence >= self.confidence_threshold:
            self._resolved += 1
            return canonical
        return None

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        return {
            "cache_size": len(self.cache),
            "cache_hits": self._cache_hits,
            "llm_calls": self._llm_calls,
            "resolved_this_run": self._resolved,
        }

    def print_stats(self) -> None:
        s = self.stats()
        print(f"SynonymExpander stats:")
        print(f"  Cache entries:    {s['cache_size']}")
        print(f"  Cache hits:       {s['cache_hits']}")
        print(f"  LLM calls:        {s['llm_calls']}")
        print(f"  Resolved (run):   {s['resolved_this_run']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    expander = SynonymExpander()

    # Demo: resolve some rule-extracted variable names
    test_variables = [
        "wind speed",                  # direct
        "wave",                         # substring
        "wind velocity",                # LLM
        "crane breakdown time",         # LLM
        "vessel size",                  # LLM
        "berth moves per hour",         # substring
        "xyz123",                       # no match
    ]

    print("\n" + "=" * 60)
    print("Synonym Expansion Demo")
    print("=" * 60)
    for v in test_variables:
        result = expander.resolve(v)
        print(f"  {v:<30} -> {result}")

    print()
    expander.print_stats()
