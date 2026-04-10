# src/online_pipeline/query_rewriter.py
"""
Query Rewriter with dictionary-first abbreviation expansion.

Strategy:
1. Dictionary lookup: expand known abbreviations instantly (0ms, no API cost)
2. LLM fallback: only called when no dictionary match found

Dictionary: data/abbreviation_dict.json (85 port/maritime abbreviations)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import logging

from .llm_client import llm_chat_raw_post, get_model_name

logger = logging.getLogger("online_pipeline.query_rewriter")

# ---------------------------------------------------------------------------
# Load abbreviation dictionary once at import time
# ---------------------------------------------------------------------------

_DICT_PATH = Path(__file__).resolve().parents[2] / "data" / "abbreviation_dict.json"
_ABBREV_MAP: Dict[str, str] = {}

try:
    with open(_DICT_PATH, "r", encoding="utf-8") as _f:
        _raw = json.load(_f)
        _ABBREV_MAP = _raw.get("abbreviations", {})
except Exception:
    pass

# Pre-compile a regex that matches any abbreviation as a whole word (case-insensitive)
# Sort by length descending so "STS crane" matches before "STS"
_sorted_abbrevs = sorted(_ABBREV_MAP.keys(), key=len, reverse=True)
if _sorted_abbrevs:
    _pattern_parts = [re.escape(a) for a in _sorted_abbrevs]
    _ABBREV_REGEX = re.compile(r'\b(' + '|'.join(_pattern_parts) + r')\b', re.IGNORECASE)
else:
    _ABBREV_REGEX = None


REWRITE_SYSTEM_PROMPT = """You are a query rewriter for a port operations knowledge system.

Given a user query, output a JSON object with:
1. "rewritten_query": an improved version of the query that expands abbreviations and adds domain synonyms
2. "expanded_terms": list of key terms/synonyms added

Rules:
- Expand port/maritime abbreviations
- Add relevant synonyms: "berth" -> "berth wharf quay", "crane" -> "crane quay crane STS", "delay" -> "delay waiting time"
- Keep the original intent intact
- Do NOT add information not implied by the query
- Return ONLY valid JSON, no markdown"""


class QueryRewriter:

    def __init__(
        self,
        model_name: str | None = None,
    ) -> None:
        self.model_name = model_name or get_model_name()
        self.abbrev_map = _ABBREV_MAP

    def rewrite(self, query: str) -> Dict[str, Any]:
        """
        Dictionary-first rewrite. LLM fallback only when no abbreviation matched.
        """
        # Step 1: dictionary expansion
        expanded_query, expanded_terms = self._dict_expand(query)

        if expanded_terms:
            logger.info("DICT rewrite: %d expansions %s", len(expanded_terms), expanded_terms)
            logger.debug("Rewritten query: %s", expanded_query)
            return {"rewritten_query": expanded_query, "expanded_terms": expanded_terms}

        # Step 2: no dictionary match — try LLM
        try:
            logger.info("No dict match => calling LLM rewriter")
            response = self._call_llm(query)
            result = self._parse_response(response)
            if result and result.get("rewritten_query"):
                logger.info("LLM rewrite: expanded_terms=%s", result.get("expanded_terms", []))
                logger.debug("LLM rewritten query: %s", result.get("rewritten_query"))
                return result
            else:
                logger.warning("LLM rewrite returned empty or unparseable response")
        except Exception as e:
            logger.error("LLM rewrite failed: %s", e)

        # Graceful fallback: return original query
        logger.info("Rewrite fallback: returning original query unchanged")
        return {"rewritten_query": query, "expanded_terms": []}

    def _dict_expand(self, query: str) -> tuple[str, List[str]]:
        """
        Expand abbreviations using the dictionary.
        Returns (expanded_query, list_of_expanded_terms).
        """
        if not _ABBREV_REGEX:
            return query, []

        expanded_terms = []
        found_positions = []

        for m in _ABBREV_REGEX.finditer(query):
            abbr = m.group(0)
            # Look up case-insensitive: try exact, then upper
            expansion = self.abbrev_map.get(abbr) or self.abbrev_map.get(abbr.upper()) or self.abbrev_map.get(abbr.upper().replace('-', ''))
            if expansion:
                found_positions.append((m.start(), m.end(), abbr, expansion))
                expanded_terms.append(f"{abbr} ({expansion})")

        if not found_positions:
            return query, []

        # Build expanded query: replace abbreviation with "ABBR (Expansion)"
        result = query
        # Process from end to start to keep positions valid
        for start, end, abbr, expansion in reversed(found_positions):
            result = result[:start] + f"{abbr} ({expansion})" + result[end:]

        return result, expanded_terms

    def _call_llm(self, query: str) -> str:
        return llm_chat_raw_post(
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Rewrite this query:\n{query}"},
            ],
            temperature=0.1,
            model=self.model_name,
            timeout=(10, 60),
        )

    @staticmethod
    def _parse_response(text: str) -> Dict[str, Any] | None:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return None
