# src/online_pipeline/conversation_memory.py
"""
Conversation memory for the agentic-RAG DAG (langgraph_workflow).

Designed lean — only what the DAG pipeline actually needs, none of the
ReAct-agent baggage. Two tiers:

- **ShortTermMemory** (in-memory, per session)
  Tracks recent turns, active port-domain entities, and a per-source
  evidence digest from the last assistant turn. Auto-summarises older
  turns via the same LLM client used elsewhere.

- **LongTermMemory** (SQLite, cross-session)
  Stores end-of-session summaries plus per-user "sticky" preferences
  ("always cite sources", "prefer SI units", ...). Retrieved by entity
  overlap + token overlap with the current query.

`MemoryManager` is the only object the workflow needs to talk to.
It is **optional**: if `build_langgraph_workflow` is called without
a memory manager, the DAG behaves exactly as the single-turn 205-sample
evaluation expects — no behaviour change to the existing baseline.

Industry-aligned design choices (parallels to LangChain `ConversationBufferMemory`,
LlamaIndex `ChatMemoryBuffer`, MemGPT recall vs core memory):
- Recent buffer (k turns) + LLM-summarised episodic store
- Long-term semantic store with relevance ranking
- Explicit entity/slot tracking for co-reference resolution
- Session boundary writes summary to long-term (consolidation)
"""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import uuid
from collections import OrderedDict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import duckdb

from .llm_client import llm_chat, llm_chat_json
from .state_schema import ConversationSummary, ConversationTurn, KeyFactRecord

logger = logging.getLogger("online_pipeline.memory")


# ---------------------------------------------------------------------------
# Lightweight entity extraction (port domain, no LLM)
# ---------------------------------------------------------------------------

_BERTH_RE = re.compile(r"\b[Bb](?:erth\s*)?(\d{1,2})\b")
_CRANE_RE = re.compile(r"\b[Cc]rane\s*(\d{1,2})\b")
_VESSEL_RE = re.compile(r"\b[Vv]essel\s+([A-Z][A-Za-z0-9 ]{2,20})\b")
_DATE_RE = re.compile(r"\b(20\d{2})(?:-(\d{2}))?\b")

_METRIC_KEYWORDS = (
    "wind speed", "wave height", "tide level", "tidal", "pressure",
    "berth productivity", "crane rate", "moves per hour",
    "delay", "dwell time", "turn time", "transit time",
    "teu", "loa", "dwt", "draft", "draught",
    "ghg", "co2", "emission", "sustainability",
)


def extract_entities(text: str) -> List[str]:
    """Pull port-domain entities out of a free-text turn. Pure regex/dict."""
    if not text:
        return []
    out: List[str] = []
    text_lower = text.lower()

    for m in _BERTH_RE.finditer(text):
        out.append(f"berth_B{int(m.group(1))}")
    for m in _CRANE_RE.finditer(text):
        out.append(f"crane_{int(m.group(1))}")
    for m in _VESSEL_RE.finditer(text):
        out.append(f"vessel_{m.group(1).strip()}")
    for kw in _METRIC_KEYWORDS:
        if kw in text_lower:
            out.append(f"metric_{kw.replace(' ', '_')}")
    for m in _DATE_RE.finditer(text):
        if m.group(2):
            out.append(f"date_{m.group(1)}-{m.group(2)}")
        else:
            out.append(f"year_{m.group(1)}")

    # Dedupe but keep order
    seen = set()
    uniq = []
    for e in out:
        if e not in seen:
            seen.add(e)
            uniq.append(e)
    return uniq


# ---------------------------------------------------------------------------
# Key-fact extraction (Phase C — the 3rd layer of short-term memory)
# ---------------------------------------------------------------------------
#
# When turns get summarised, the summary compresses narrative but loses
# concrete facts. We extract those facts into a separate list that outlives
# individual summaries — the idea is that if turn 42 asks "what was that
# threshold we saw in turn 3?", even after turns 1-6 got summarised and
# that summary aged out, the extracted fact "wind > 25 m/s triggers
# restriction" is still retrievable.
#
# Extraction strategy: LLM-first (schema-constrained JSON) with a cheap
# regex fallback so the pipeline never blocks on LLM availability.

# Pattern matchers for the *content* of a candidate sentence (not used to
# split). Sentence splitting is handled separately by `_split_sentences`
# below so decimals like "4.0 hours" don't get cut in half by the period.

# Number + unit, e.g. "25 moves/hr", "1.4 m", "45 hours".
_HAS_NUMERIC_FACT_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:m/s|m|meters?|ft|feet|%|TEU|mph|moves/?hr?|"
    r"moves per hour|kW|MW|hours?|minutes?|kn|knots)\b",
    re.IGNORECASE,
)

# Port-domain entity ID. Case-insensitive for "berth/crane/vessel" keywords
# but **case-sensitive** for vessel proper nouns — otherwise patterns like
# "vessel arrival" (common English) would falsely match as "vessel <NAME>".
_HAS_ENTITY_RE = re.compile(
    r"\b(?:[Bb]erth\s*B?\d+"
    r"|[Cc]rane\s*\d+"
    r"|[Vv]essel\s+[A-Z][A-Z0-9]{2,}(?:\s+[A-Z][A-Z0-9]*)?)\b"
)

# Lines that are conversation labels, not standalone facts — skip them
# even if they contain a matched number/entity.
_ROLE_PREFIX_RE = re.compile(r"^\s*(?:user|assistant|system)\s*:", re.IGNORECASE)

# Citation tags the synthesiser inserts in answers (kept when presenting,
# stripped when deduplicating).
_SOURCE_TAG_RE = re.compile(r"\[(?:sql|rule|doc|graph|general\s*knowledge)\]",
                             re.IGNORECASE)

# Sentence terminators that also preserve decimals. Splits on ".!?" only
# when followed by whitespace + uppercase-or-end (so "4.0" stays intact
# because the "." is followed by "0", not whitespace).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\(]|$)")


_KEY_FACT_PROMPT = (
    "Extract 1 to 5 concrete KEY FACTS from this port-operations conversation "
    "summary. Each fact must be a SINGLE short sentence and must contain at "
    "least one of: a specific berth/crane/vessel ID, a numeric value with "
    "units, or a named threshold/policy.\n\n"
    "Reject vague or subjective claims. Reject user questions. Reject facts "
    "already obvious from the domain.\n\n"
    'Return ONLY a JSON object: {"facts": ["fact 1", "fact 2", ...]}'
)


def normalize_fact(text: str) -> str:
    """Canonical form used for deduping key facts.

    Applies:
    - lowercase
    - strip citation tags [sql]/[rule]/[doc]/[graph]
    - collapse whitespace
    - strip surrounding punctuation/quotes
    """
    if not text:
        return ""
    s = _SOURCE_TAG_RE.sub(" ", text.lower())
    # strip quotes and common punct at the boundary
    s = re.sub(r"[\"'`]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.strip(".,;:!?-() ")
    return s


def _split_sentences(text: str) -> List[str]:
    """Sentence splitter that preserves decimals (4.0 stays whole)."""
    if not text:
        return []
    # Split on newline first to keep "user: ..." / "assistant: ..." lines
    # separate from each other (many summaries are a conversation transcript).
    parts: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts.extend(_SENTENCE_SPLIT_RE.split(line))
    return [p.strip() for p in parts if p.strip()]


def _regex_extract_key_facts(text: str, max_facts: int = 5) -> List[str]:
    """Sentence-level fallback: keep sentences carrying a numeric value or
    entity ID, strip `user:`/`assistant:` prefixes, normalise-dedupe.

    A pure user question like "What is the tide?" has no fact pattern to
    match so it is naturally filtered; an assistant statement that happens
    to be prefixed with "assistant:" still survives after prefix strip.
    """
    if not text:
        return []

    seen = set()
    out: List[str] = []
    for sent in _split_sentences(text):
        # Strip role prefix; downstream pattern matching decides if what
        # remains is a fact.
        sent_clean = _ROLE_PREFIX_RE.sub("", sent, count=1).strip()
        if not sent_clean:
            continue
        if not (_HAS_NUMERIC_FACT_RE.search(sent_clean)
                or _HAS_ENTITY_RE.search(sent_clean)):
            continue

        key = normalize_fact(sent_clean)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(sent_clean[:200])
        if len(out) >= max_facts:
            break
    return out


def extract_key_facts(summary_or_turns_text: str, max_facts: int = 5) -> List[str]:
    """LLM-first extraction with a regex fallback. Returns short fact strings."""
    if not summary_or_turns_text.strip():
        return []
    try:
        out = llm_chat_json(
            messages=[
                {"role": "system", "content": _KEY_FACT_PROMPT},
                {"role": "user", "content": summary_or_turns_text[:2000]},
            ],
            temperature=0.0,
            timeout=20,
        )
        if isinstance(out, dict):
            facts = out.get("facts") or []
            if isinstance(facts, list):
                clean = [str(f).strip()[:200] for f in facts if str(f).strip()]
                if clean:
                    return clean[:max_facts]
    except Exception as exc:                           # pragma: no cover — defensive
        logger.warning("LLM key-fact extraction failed: %s", exc)

    return _regex_extract_key_facts(summary_or_turns_text, max_facts=max_facts)


# ---------------------------------------------------------------------------
# Short-term memory (per session, in-process)
# ---------------------------------------------------------------------------

class ShortTermMemory:
    """
    Per-session conversation memory with 3-layer lifecycle:

    | Layer         | Content                                  | Lifecycle                           |
    |---------------|------------------------------------------|-------------------------------------|
    | `turns`       | Last ~half of `max_raw_turns` verbatim   | Oldest half LLM-summarised on cap   |
    | `summaries`   | 2-3 sentence LLM compressions            | Grow ∞; key facts extracted each    |
    | `key_facts`   | Atomic fact strings (numbers + IDs)      | Persist; FIFO cap at max_key_facts  |

    Facts extracted from every new summary are stored separately so that
    even when a summary ages out or drifts, concrete numbers and entity IDs
    remain addressable by follow-ups like "what was that wind threshold
    again?".

    - `active_entities`: maps entity → last turn it appeared in.
    - `last_evidence_digest`: per-source one-line summaries from the most
      recent assistant turn.
    """

    def __init__(
        self,
        session_id: str,
        max_raw_turns: int = 10,
        max_key_facts: int = 40,
    ) -> None:
        self.session_id = session_id
        self.max_raw_turns = max_raw_turns
        self.max_key_facts = max_key_facts
        self.turns: List[ConversationTurn] = []
        self.summaries: List[ConversationSummary] = []
        self.key_facts: List[KeyFactRecord] = []
        self.active_entities: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.last_evidence_digest: Dict[str, str] = {}   # source -> short text
        self._turn_counter = 0

    # ------ writes ------

    def add_user_turn(self, content: str) -> ConversationTurn:
        return self._add_turn("user", content, tool_results_summary=None)

    def add_assistant_turn(
        self,
        answer_text: str,
        sources_used: Optional[List[str]] = None,
        evidence_digest: Optional[Dict[str, str]] = None,
    ) -> ConversationTurn:
        turn = self._add_turn(
            "assistant",
            answer_text,
            tool_results_summary=sources_used or [],
        )
        if evidence_digest:
            self.last_evidence_digest = dict(evidence_digest)
        return turn

    def _add_turn(
        self,
        role: str,
        content: str,
        tool_results_summary: Optional[List[str]],
    ) -> ConversationTurn:
        self._turn_counter += 1
        ents = extract_entities(content)
        turn: ConversationTurn = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "turn_id": self._turn_counter,
            "tool_results_summary": tool_results_summary,
            "entities_mentioned": ents,
        }
        self.turns.append(turn)
        for e in ents:
            self.active_entities[e] = {
                "last_turn": self._turn_counter,
                "context": content[:120],
            }
            # Keep entity LRU bounded
            self.active_entities.move_to_end(e)

        if len(self.turns) > self.max_raw_turns:
            self._summarise_oldest_half()
        return turn

    # ------ reads ------

    def recent_turns(self, k: int = 6) -> List[ConversationTurn]:
        return self.turns[-k:]

    def all_entities(self) -> List[str]:
        return list(self.active_entities.keys())

    def recent_key_facts(self, k: int = 8) -> List[KeyFactRecord]:
        """Most-recently-extracted key facts (by `from_turn_ids[-1]`)."""
        if not self.key_facts:
            return []
        ordered = sorted(
            self.key_facts,
            key=lambda f: max(f.get("from_turn_ids", [0])),
            reverse=True,
        )
        return ordered[:k]

    def format_for_prompt(self, max_chars: int = 2400) -> str:
        """Compact context block for injection into router/planner/synthesizer.

        Order is deliberate: key_facts FIRST so concrete numbers survive
        downstream prompt truncation even when the summary/turns blocks
        get long."""
        parts: List[str] = []

        # 1. Key facts (persist even after summaries age out)
        kf = self.recent_key_facts(8)
        if kf:
            lines = [f"  - {f.get('fact', '')}" for f in kf]
            parts.append("[Key facts]:\n" + "\n".join(lines))

        # 2. Summaries (2-3 sentences each)
        if self.summaries:
            joined = " ".join(s.get("summary_text", "") for s in self.summaries)
            parts.append(f"[Earlier conversation]: {joined[:600]}")

        # 3. Recent raw turns (most recent verbatim)
        recent = self.recent_turns(6)
        if recent:
            lines = []
            for t in recent:
                role = t.get("role", "?")
                content = (t.get("content") or "")[:280]
                lines.append(f"  {role}: {content}")
            parts.append("[Recent turns]:\n" + "\n".join(lines))

        # 4. Active entities (LRU)
        if self.active_entities:
            top = list(self.active_entities.keys())[-8:]
            parts.append(f"[Active entities]: {', '.join(top)}")

        # 5. Last assistant turn's per-source evidence digest
        if self.last_evidence_digest:
            digest = "; ".join(
                f"{k}: {v[:120]}" for k, v in self.last_evidence_digest.items()
            )
            parts.append(f"[Last evidence]: {digest}")

        text = "\n\n".join(parts)
        return text[:max_chars]

    # ------ summarisation ------

    def _summarise_oldest_half(self) -> None:
        half = max(1, len(self.turns) // 2)
        old, self.turns = self.turns[:half], self.turns[half:]
        body = "\n".join(
            f"{t.get('role', '?')}: {(t.get('content') or '')[:240]}"
            for t in old
        )
        ids = [t.get("turn_id", 0) for t in old]
        try:
            text = llm_chat(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Summarise this port-operations chat in 2-3 sentences. "
                            "Preserve berth/crane/vessel IDs, dates, numeric thresholds, "
                            "and the assistant's key conclusions."
                        ),
                    },
                    {"role": "user", "content": body},
                ],
                temperature=0.0,
                timeout=30,
            )
        except Exception as exc:                       # pragma: no cover — defensive
            logger.warning("Summarisation failed: %s", exc)
            text = ""
        if not text:
            text = f"(Compressed turns {ids[0]}-{ids[-1]})"

        ents: List[str] = []
        for t in old:
            ents.extend(t.get("entities_mentioned", []) or [])

        # Extract concrete key facts from the combined body (not just the
        # summary) so numbers and IDs in the raw turns survive even after
        # the summary itself drifts. Stored on both the summary (for
        # provenance) AND in self.key_facts (for long-lived retrieval).
        facts = extract_key_facts(body + "\n\n" + text, max_facts=5)

        self.summaries.append(ConversationSummary(
            summary_text=text,
            turns_covered=ids,
            key_entities=sorted(set(ents)),
            key_facts=facts,
        ))

        if facts:
            now_iso = datetime.now().isoformat(timespec="seconds")
            # Dedupe against what's already stored using normalize_fact so
            # that e.g. "tide was 4.0 m [sql]" and "tide was 4.0 m." collapse
            # to one entry. Keeps the older entry (stable turn_ids).
            seen = {normalize_fact(f.get("fact", "")) for f in self.key_facts}
            for f in facts:
                key = normalize_fact(f)
                if not key or key in seen:
                    continue
                seen.add(key)
                self.key_facts.append(KeyFactRecord(
                    fact=f,
                    from_turn_ids=ids,
                    entities=extract_entities(f),
                    extracted_at=now_iso,
                ))
            # FIFO cap: drop oldest extracted facts if over budget.
            if len(self.key_facts) > self.max_key_facts:
                overflow = len(self.key_facts) - self.max_key_facts
                del self.key_facts[:overflow]


# ---------------------------------------------------------------------------
# Long-term memory (DuckDB, cross-session)
# ---------------------------------------------------------------------------
#
# Phase A migration: moved from SQLite to DuckDB to unify with the project's
# existing DuckDB stack (`storage/sql/port_ops.duckdb`). Benefits:
#   - Single embedded OLAP engine for the whole project (consistency)
#   - Native JSON type for `entities` (no manual serialisation)
#   - Native TIMESTAMP type (was ISO-string, now sortable by SQL)
#   - FLOAT[768] column pre-allocated for Phase B vector retrieval
#   - Time-decay + per-entry-type importance built into the ranking formula
#
# On first run: if the old SQLite `storage/memory/long_term.db` exists,
# rows are copied into DuckDB and the type `conversation_summary` is renamed
# to the new canonical `session_summary`. The SQLite file is left untouched
# as a safety copy.

_ENTRY_TYPE_WEIGHTS: Dict[str, float] = {
    "user_preference": 1.2,
    "session_summary": 1.0,
    "faq_pattern": 0.8,
}
_DEFAULT_ENTRY_WEIGHT = 1.0

# BGE-base-en-v1.5 dimension (project's document retriever uses this model).
_EMBEDDING_DIM = 768

# Must match the prefix used by `document_retriever.py` and
# `build_embeddings_v2.py` so query / corpus vectors live in the same space.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Half-life for the recency bonus, in days. 30d = memories from earlier this
# month stay warm, year-old memories decay to near zero.
_TIME_DECAY_HALFLIFE_DAYS = 30.0


# ---------------------------------------------------------------------------
# BGE embedder wrapper (Phase B)
# ---------------------------------------------------------------------------
#
# A thin wrapper around sentence-transformers that:
#   - Loads BAAI/bge-base-en-v1.5 once per process (lazy).
#   - Uses the same query prefix convention as document_retriever, so memory
#     embeddings live in the same semantic space as document embeddings.
#   - Normalises output (so dot product == cosine similarity, matching the
#     convention used by DuckDB `vss` and Chroma-BGE integration).
#
# We deliberately do NOT inject this into LongTermMemory at construction time;
# `MemoryManager` passes it via constructor so unit tests can run without it
# and `LongTermMemory` falls back to the Phase-A keyword-only path when no
# embedder is configured.

class BGEEmbedder:
    """Singleton BGE-base-en-v1.5 embedder. Matches project-wide BGE usage."""

    _instance: Optional["BGEEmbedder"] = None

    def __init__(self, device: str = "cuda") -> None:
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)
        self.model.max_seq_length = 512
        self.dim = _EMBEDDING_DIM
        logger.info("BGEEmbedder loaded on device=%s", device)

    @classmethod
    def get(cls, device: str = "cuda") -> "BGEEmbedder":
        if cls._instance is None:
            try:
                cls._instance = cls(device=device)
            except Exception as exc:
                logger.warning("BGE load on %s failed (%s); falling back to cpu", device, exc)
                cls._instance = cls(device="cpu")
        return cls._instance

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed stored memory content (no prefix — corpus side)."""
        return self.model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        ).tolist()

    def embed_query(self, query: str) -> List[float]:
        """Embed a search query (with BGE retrieval prefix)."""
        return self.model.encode(
            [_BGE_QUERY_PREFIX + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0].tolist()

_DUCKDB_DDL = [
    """
    CREATE TABLE IF NOT EXISTS lt_memory (
        entry_id VARCHAR PRIMARY KEY,
        session_id VARCHAR,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        entry_type VARCHAR,                -- 'session_summary' | 'user_preference' | 'faq_pattern'
        content TEXT,
        entities JSON,                      -- native JSON list
        embedding FLOAT[768],                -- filled by Phase B (nullable for now)
        importance REAL DEFAULT 1.0,
        access_count INTEGER DEFAULT 0
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_lt_session ON lt_memory(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_lt_type ON lt_memory(entry_type)",
    "CREATE INDEX IF NOT EXISTS idx_lt_created ON lt_memory(created_at)",
]


def _time_decay(age_days: float) -> float:
    """Exponential decay with half-life `_TIME_DECAY_HALFLIFE_DAYS`. 0d -> 1.0."""
    if age_days <= 0:
        return 1.0
    return math.exp(-math.log(2) * age_days / _TIME_DECAY_HALFLIFE_DAYS)


class LongTermMemory:
    """
    DuckDB-backed cross-session memory.

    Scoring:
        Phase A (no embedder configured):
            score = word_overlap     * 0.35
                  + entity_overlap   * 0.45
                  + time_decay       * 0.12
                  + access_norm      * 0.03
                  + importance_delta * 0.05

        Phase B (embedder configured):
            BGE cosine similarity replaces word_overlap. Two-stage retrieval:
              1. vss vector recall against `embedding` column -> Top-20
              2. rerank by cosine * 0.55 + entity * 0.25 + decay * 0.12
                                + access * 0.03 + importance * 0.05
            Falls back to Phase-A path when an entry has no embedding yet
            (e.g. migrated legacy rows before backfill).

    The public API (`store`, `retrieve`, `store_session_summary`, `close`)
    is stable.
    """

    def __init__(
        self,
        db_path: str | Path,
        embedder: Optional["BGEEmbedder"] = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self.db_path))
        for stmt in _DUCKDB_DDL:
            self._conn.execute(stmt)
        # Try to load vss extension for native cosine-similarity on FLOAT[768].
        # Falls back to Python cosine if the extension is unavailable.
        self._vss_ok = False
        try:
            self._conn.execute("INSTALL vss")
            self._conn.execute("LOAD vss")
            self._vss_ok = True
            logger.info("DuckDB vss extension loaded")
        except Exception as exc:                       # pragma: no cover — defensive
            logger.warning("vss extension unavailable (%s); will use Python cosine", exc)
        self.embedder = embedder
        logger.info(
            "LongTermMemory (DuckDB) @ %s  [embedder=%s, vss=%s]",
            self.db_path, "BGE" if embedder else "none", self._vss_ok,
        )
        self._maybe_migrate_from_sqlite()

    # --------------------------------------------------------------
    # One-time SQLite -> DuckDB migration
    # --------------------------------------------------------------

    def _maybe_migrate_from_sqlite(self) -> None:
        count = self._conn.execute("SELECT COUNT(*) FROM lt_memory").fetchone()[0]
        if count > 0:
            return

        # Look for the legacy file in either the old default location or next
        # to the current db_path.
        legacy_paths = [
            self.db_path.parent.parent / "memory" / "long_term.db",
            self.db_path.parent / "long_term.db",
        ]
        legacy = next((p for p in legacy_paths if p.exists()), None)
        if legacy is None:
            return

        logger.info("Migrating legacy SQLite long-term memory: %s", legacy)
        try:
            sq = sqlite3.connect(str(legacy))
            try:
                rows = sq.execute(
                    "SELECT entry_id, session_id, timestamp, entry_type, "
                    "       content, entities, access_count "
                    "FROM long_term_memory"
                ).fetchall()
            finally:
                sq.close()
        except Exception as exc:                       # pragma: no cover — defensive
            logger.warning("SQLite migration read failed: %s", exc)
            return

        migrated = 0
        for r in rows:
            entry_id, sid, ts, etype, content, ents_json, acc = r
            if etype == "conversation_summary":
                etype = "session_summary"
            try:
                ents = json.loads(ents_json) if ents_json else []
            except json.JSONDecodeError:
                ents = []
            importance = _ENTRY_TYPE_WEIGHTS.get(etype, _DEFAULT_ENTRY_WEIGHT)
            try:
                ts_val = datetime.fromisoformat(ts) if ts else datetime.now()
            except Exception:
                ts_val = datetime.now()
            self._conn.execute(
                "INSERT INTO lt_memory "
                "(entry_id, session_id, created_at, entry_type, content, "
                " entities, embedding, importance, access_count) "
                "VALUES (?, ?, ?, ?, ?, ?::JSON, NULL, ?, ?)",
                (entry_id, sid, ts_val, etype, content, json.dumps(ents),
                 importance, acc or 0),
            )
            migrated += 1
        logger.info("Migrated %d legacy memory entries into DuckDB", migrated)

    # --------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------

    def store(
        self,
        entry_type: str,
        content: str,
        entities: List[str],
        session_id: str = "",
        embedding: Optional[List[float]] = None,
    ) -> str:
        """Insert a memory entry. If an embedder is configured and no explicit
        `embedding` was passed, the content is embedded automatically."""
        entry_id = uuid.uuid4().hex[:12]
        importance = _ENTRY_TYPE_WEIGHTS.get(entry_type, _DEFAULT_ENTRY_WEIGHT)

        if embedding is None and self.embedder is not None and content:
            try:
                embedding = self.embedder.embed_documents([content])[0]
            except Exception as exc:                   # pragma: no cover — defensive
                logger.warning("store(): BGE embed failed (%s); saving without embedding", exc)
                embedding = None

        if embedding is not None and len(embedding) != _EMBEDDING_DIM:
            raise ValueError(
                f"embedding must have length {_EMBEDDING_DIM}, got {len(embedding)}"
            )
        self._conn.execute(
            "INSERT INTO lt_memory "
            "(entry_id, session_id, created_at, entry_type, content, "
            " entities, embedding, importance, access_count) "
            "VALUES (?, ?, ?, ?, ?, ?::JSON, ?, ?, 0)",
            (
                entry_id,
                session_id,
                datetime.now(),
                entry_type,
                content,
                json.dumps(entities),
                embedding,
                importance,
            ),
        )
        return entry_id

    def backfill_embeddings(self, batch_size: int = 32) -> int:
        """Compute embeddings for every row where `embedding IS NULL`.

        Returns the number of rows updated. Useful after a Phase-A migration
        (rows were imported from SQLite without embeddings)."""
        if self.embedder is None:
            logger.warning("backfill_embeddings: no embedder configured, skipping")
            return 0

        rows = self._conn.execute(
            "SELECT entry_id, content FROM lt_memory WHERE embedding IS NULL"
        ).fetchall()
        if not rows:
            return 0

        updated = 0
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            texts = [r[1] or "" for r in batch]
            vectors = self.embedder.embed_documents(texts)
            for (entry_id, _), vec in zip(batch, vectors):
                self._conn.execute(
                    "UPDATE lt_memory SET embedding = ? WHERE entry_id = ?",
                    (vec, entry_id),
                )
                updated += 1
        logger.info("backfill_embeddings: updated %d row(s)", updated)
        return updated

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        entry_types: Optional[List[str]] = None,
        recall_pool: int = 20,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant memories.

        Two-stage when an embedder is configured:
          1. vss cosine recall on the `embedding` column -> `recall_pool`
             candidates (padded with recent non-embedded rows if needed, so
             unmigrated legacy rows are not silently invisible).
          2. Rerank by a blended score (cosine / entity / decay / access /
             importance) and return top_k.

        Phase-A fallback (no embedder): pure keyword + entity + decay scoring
        over the most recent 200 entries.
        """
        q_ents = set(extract_entities(query or ""))
        q_words = set((query or "").lower().split())
        now = datetime.now()

        vec_query: Optional[List[float]] = None
        if self.embedder is not None and query:
            try:
                vec_query = self.embedder.embed_query(query)
            except Exception as exc:                   # pragma: no cover — defensive
                logger.warning("retrieve(): BGE embed_query failed (%s)", exc)

        # ----- 1. Build candidate pool ----------------------------------
        filter_sql = ""
        params: List[Any] = []
        if entry_types:
            filter_sql = f"WHERE entry_type IN ({','.join('?' * len(entry_types))})"
            params.extend(entry_types)

        if vec_query is not None and self._vss_ok:
            # Vector recall: NULL-embedding rows drop out here, so we UNION
            # with a recency slice to keep them discoverable.
            vec_sql = (
                "SELECT entry_id, session_id, created_at, entry_type, content, "
                "       entities, importance, access_count, "
                "       array_cosine_similarity(embedding, ?::FLOAT[768]) AS cos_sim "
                "FROM lt_memory "
                f"{filter_sql}{' AND' if filter_sql else 'WHERE'} embedding IS NOT NULL "
                "ORDER BY cos_sim DESC LIMIT ?"
            )
            vec_params = [vec_query] + params + [recall_pool]
            vec_rows = self._conn.execute(vec_sql, vec_params).fetchall()

            # Recency top-up (catches legacy rows not yet backfilled)
            recency_sql = (
                "SELECT entry_id, session_id, created_at, entry_type, content, "
                "       entities, importance, access_count, 0.0 AS cos_sim "
                "FROM lt_memory "
                f"{filter_sql}{' AND' if filter_sql else 'WHERE'} embedding IS NULL "
                "ORDER BY created_at DESC LIMIT ?"
            )
            rec_rows = self._conn.execute(recency_sql, params + [recall_pool]).fetchall()

            # Dedupe by entry_id (vec_rows always wins since cos_sim is real)
            seen = set()
            rows = []
            for r in list(vec_rows) + list(rec_rows):
                if r[0] in seen:
                    continue
                seen.add(r[0])
                rows.append(r)
        else:
            # Phase-A path: recent 200, keyword + entity scoring
            recent_sql = (
                "SELECT entry_id, session_id, created_at, entry_type, content, "
                "       entities, importance, access_count, 0.0 AS cos_sim "
                "FROM lt_memory "
                f"{filter_sql} "
                "ORDER BY created_at DESC LIMIT 200"
            )
            rows = self._conn.execute(recent_sql, params).fetchall()

        # ----- 2. Rerank ------------------------------------------------
        scored: List[Dict[str, Any]] = []
        for r in rows:
            (entry_id, sid, created_at, etype, content, ents_json,
             importance, acc, cos_sim) = r
            cos_sim = float(cos_sim or 0.0)

            # JSON column can come back as list (new) or str (legacy paths)
            if isinstance(ents_json, str):
                try:
                    stored_ents = set(json.loads(ents_json))
                except json.JSONDecodeError:
                    stored_ents = set()
            elif isinstance(ents_json, list):
                stored_ents = set(ents_json)
            else:
                stored_ents = set()

            ent_overlap = (
                len(q_ents & stored_ents) / max(len(q_ents), 1) if q_ents else 0.0
            )
            age_days = max(0.0, (now - created_at).total_seconds() / 86400.0)
            decay = _time_decay(age_days)
            access_norm = min(acc or 0, 5) / 5.0
            imp = importance or _DEFAULT_ENTRY_WEIGHT
            importance_delta = imp - _DEFAULT_ENTRY_WEIGHT

            if vec_query is not None:
                # Phase-B blended score
                score = (
                    max(cos_sim, 0.0) * 0.55
                    + ent_overlap * 0.25
                    + decay * 0.12
                    + access_norm * 0.03
                    + importance_delta * 0.05
                )
                threshold = 0.18
            else:
                # Phase-A keyword fallback
                content_words = set((content or "").lower().split())
                word_overlap = (
                    len(q_words & content_words) / max(len(q_words), 1)
                    if q_words else 0.0
                )
                score = (
                    word_overlap * 0.35
                    + ent_overlap * 0.45
                    + decay * 0.12
                    + access_norm * 0.03
                    + importance_delta * 0.05
                )
                threshold = 0.08

            if score >= threshold:
                scored.append({
                    "entry_id": entry_id,
                    "session_id": sid,
                    "timestamp": created_at.isoformat(timespec="seconds"),
                    "age_days": round(age_days, 2),
                    "entry_type": etype,
                    "content": content,
                    "entities": list(stored_ents),
                    "importance": round(imp, 2),
                    "cosine": round(cos_sim, 4),
                    "score": round(score, 4),
                })

        scored.sort(key=lambda x: x["score"], reverse=True)
        out = scored[:top_k]

        if out:
            ids = [r["entry_id"] for r in out]
            placeholder = ",".join("?" * len(ids))
            self._conn.execute(
                f"UPDATE lt_memory SET access_count = access_count + 1 "
                f"WHERE entry_id IN ({placeholder})",
                ids,
            )
        return out

    def store_session_summary(
        self,
        session_id: str,
        summary: str,
        entities: List[str],
        embedding: Optional[List[float]] = None,
    ) -> str:
        return self.store(
            "session_summary", summary, entities, session_id, embedding=embedding
        )

    def get_frequent_topics(self, top_k: int = 5) -> List[str]:
        rows = self._conn.execute(
            "SELECT content FROM lt_memory "
            "ORDER BY access_count DESC LIMIT ?",
            (top_k,),
        ).fetchall()
        return [row[0][:100] for row in rows]

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# MemoryManager — single facade
# ---------------------------------------------------------------------------

class MemoryManager:
    """
    Facade that the LangGraph workflow and the eval driver talk to.

    Lifecycle in a multi-turn run:
        mgr = MemoryManager(project_root)
        sid = mgr.start_session()
        for raw_query in turns:
            ctx = mgr.build_context(sid, raw_query)              # for prompt injection
            mgr.record_user_turn(sid, raw_query)
            state = workflow.invoke({..., "session_id": sid, "memory_context": ctx, ...})
            mgr.record_assistant_turn(sid, state)                # auto-extracts evidence
        mgr.end_session(sid)                                     # writes long-term summary
    """

    def __init__(
        self,
        project_root: str | Path,
        db_path: Optional[str | Path] = None,
        max_raw_turns: int = 10,
        use_embeddings: bool = True,
        embedder_device: str = "cuda",
    ) -> None:
        self.project_root = Path(project_root)
        if db_path is None:
            # Default: DuckDB file alongside business SQL so we have one
            # embedded OLAP engine per project. First run auto-migrates the
            # legacy SQLite `storage/memory/long_term.db` if present.
            db_path = self.project_root / "storage" / "sql" / "memory.duckdb"

        embedder: Optional[BGEEmbedder] = None
        if use_embeddings:
            try:
                embedder = BGEEmbedder.get(device=embedder_device)
            except Exception as exc:
                logger.warning(
                    "BGE embedder init failed (%s); long-term memory "
                    "degrades to Phase-A keyword ranking", exc,
                )

        self.long_term = LongTermMemory(db_path, embedder=embedder)

        # Opportunistic backfill: legacy rows imported from SQLite have no
        # embeddings. Compute them once so vector retrieval works immediately.
        if embedder is not None:
            try:
                n = self.long_term.backfill_embeddings()
                if n:
                    logger.info("Backfilled %d legacy row(s) with embeddings", n)
            except Exception as exc:                   # pragma: no cover — defensive
                logger.warning("backfill_embeddings failed: %s", exc)

        self.max_raw_turns = max_raw_turns
        self._sessions: Dict[str, ShortTermMemory] = {}

    # ------ session lifecycle ------

    def start_session(self, session_id: Optional[str] = None) -> str:
        sid = session_id or uuid.uuid4().hex[:12]
        if sid not in self._sessions:
            self._sessions[sid] = ShortTermMemory(sid, max_raw_turns=self.max_raw_turns)
        return sid

    def get_session(self, session_id: str) -> ShortTermMemory:
        return self._sessions.setdefault(
            session_id, ShortTermMemory(session_id, max_raw_turns=self.max_raw_turns)
        )

    def end_session(self, session_id: str) -> Optional[str]:
        st = self._sessions.pop(session_id, None)
        if not st or not st.turns:
            return None
        body = "\n".join(
            f"{t.get('role', '?')}: {(t.get('content') or '')[:200]}"
            for t in st.turns[-12:]
        )
        try:
            summary = llm_chat(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Summarise this port decision-support session in 2-3 sentences. "
                            "Capture (a) what topics the user explored, (b) key answers/numbers "
                            "given, (c) any unresolved questions."
                        ),
                    },
                    {"role": "user", "content": body},
                ],
                temperature=0.0,
                timeout=30,
            )
        except Exception as exc:                       # pragma: no cover — defensive
            logger.warning("Session summary LLM failed: %s", exc)
            summary = ""
        if not summary:
            summary = f"Session {session_id}: {len(st.turns)} turns"
        return self.long_term.store_session_summary(
            session_id, summary, st.all_entities()
        )

    # ------ recording (called by workflow caller) ------

    def record_user_turn(self, session_id: str, raw_query: str) -> ConversationTurn:
        return self.get_session(session_id).add_user_turn(raw_query)

    def record_assistant_turn(
        self,
        session_id: str,
        graph_state: Dict[str, Any],
    ) -> ConversationTurn:
        """
        Pull the answer + per-source evidence digest straight out of the
        finished LangGraph state.
        """
        st = self.get_session(session_id)
        final = graph_state.get("final_answer") or {}
        answer_text = final.get("answer", "") if isinstance(final, dict) else str(final)
        sources_used = (
            final.get("sources_used", []) if isinstance(final, dict) else []
        )
        digest = self._evidence_digest(graph_state)
        return st.add_assistant_turn(
            answer_text=answer_text,
            sources_used=sources_used,
            evidence_digest=digest,
        )

    @staticmethod
    def _evidence_digest(state: Dict[str, Any]) -> Dict[str, str]:
        """One-line per-source summary from the merged evidence bundle."""
        out: Dict[str, str] = {}
        docs = state.get("retrieved_docs") or []
        if docs:
            top = docs[0]
            chunk_id = top.get("chunk_id") or top.get("doc_id") or "?"
            text = (top.get("text") or "")[:140]
            out["documents"] = f"top={chunk_id}: {text}"
        sql = state.get("sql_results") or []
        if sql:
            r = sql[0]
            plan = r.get("plan", {}) or {}
            out["sql"] = (
                f"tables={plan.get('target_tables', [])} "
                f"rows={r.get('row_count', 0)} ok={r.get('execution_ok')}"
            )
        rules = (state.get("rule_results") or {}).get("matched_rules") or []
        if rules:
            out["rules"] = "; ".join(
                (r.get("rule_text", "") or "")[:80] for r in rules[:2]
            )
        graph = state.get("graph_results") or {}
        paths = graph.get("reasoning_paths") or []
        if paths:
            out["graph"] = (paths[0].get("explanation", "") or "")[:140]
        return out

    # ------ context building (called before workflow.invoke) ------

    def build_context(
        self,
        session_id: str,
        current_query: str,
        max_chars: int = 3000,
    ) -> str:
        """
        Merged short-term + long-term context for prompt injection.
        Budget: ~70% short-term, ~30% long-term.
        """
        st = self._sessions.get(session_id)
        parts: List[str] = []
        if st and st.turns:
            short_ctx = st.format_for_prompt(max_chars=int(max_chars * 0.7))
            if short_ctx.strip():
                parts.append(short_ctx)

        lt_hits = self.long_term.retrieve(current_query, top_k=3)
        if lt_hits:
            lines = ["[Memories from prior sessions]:"]
            for m in lt_hits:
                lines.append(f"  - ({m['entry_type']}) {m['content'][:200]}")
            parts.append("\n".join(lines))

        return "\n\n".join(parts)[:max_chars] if parts else ""

    # ------ co-reference resolution ------

    def resolve_followup(
        self,
        session_id: str,
        raw_query: str,
    ) -> Tuple[str, bool]:
        """
        If the query looks like a follow-up (short, contains pronouns/conjunctions),
        rewrite it into a standalone question using recent context.

        Returns `(resolved_query, was_rewritten)`.
        """
        st = self._sessions.get(session_id)
        if not st or not st.turns:
            return raw_query, False
        if not _looks_like_followup(raw_query):
            return raw_query, False

        ctx = st.format_for_prompt(max_chars=1500)
        try:
            resolved = llm_chat(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You rewrite follow-up port-operations queries into self-contained "
                            "questions using the conversation context.\n"
                            "Rules:\n"
                            "1. Inline pronouns/references (\"that berth\", \"those rules\") "
                            "with their explicit referents.\n"
                            "2. Carry over numeric filters / berth IDs / time windows that "
                            "were established earlier and still apply.\n"
                            "3. If the new query is on a clearly different topic, do NOT carry "
                            "old context — just return it lightly cleaned.\n"
                            "4. Output ONLY the rewritten query, no preamble.\n\n"
                            f"--- Conversation ---\n{ctx}\n--- End ---"
                        ),
                    },
                    {"role": "user", "content": raw_query},
                ],
                temperature=0.0,
                timeout=20,
            )
        except Exception as exc:                       # pragma: no cover — defensive
            logger.warning("Co-ref LLM failed: %s", exc)
            return raw_query, False

        resolved = (resolved or "").strip().strip('"').strip("`")
        if not resolved or resolved.lower() == raw_query.lower():
            return raw_query, False
        logger.info("CO-REF: '%s' -> '%s'", raw_query[:60], resolved[:60])
        return resolved, True

    def close(self) -> None:
        self.long_term.close()


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

_FOLLOWUP_EN = (
    "what about", "how about", "and the", "and for", "same for", "also for",
    "what if", "instead", "those", "this one", "that one", "them",
)
_FOLLOWUP_ZH = ("那", "呢", "这个", "那个", "刚才", "上面", "还有", "另外")


def _looks_like_followup(q: str) -> bool:
    """Cheap heuristic: short query, pronouns, or co-reference markers."""
    if not q:
        return False
    s = q.strip().lower()
    if len(s.split()) <= 4:
        return True
    if any(p in s for p in _FOLLOWUP_EN):
        return True
    if any(p in q for p in _FOLLOWUP_ZH):
        return True
    return False
