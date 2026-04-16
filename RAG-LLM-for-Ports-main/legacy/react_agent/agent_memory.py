# src/online_pipeline/agent_memory.py
"""
Memory system for the Plan-and-Execute agent.

Two tiers:
- **Short-term memory** (ShortTermMemoryStore): in-memory, per-session.
  Tracks conversation turns, active entities, recent tool results.
  Auto-summarizes old turns to stay within context budget.

- **Long-term memory** (LongTermMemoryStore): SQLite-backed, cross-session.
  Stores conversation summaries, frequent topics, user preferences.
  Retrieved by keyword relevance for new sessions.

MemoryManager is the unified facade used by SessionManager and AgentNodes.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .llm_client import llm_chat, llm_chat_json
from .state_schema import ConversationSummary, ConversationTurn

logger = logging.getLogger("online_pipeline.agent_memory")

# ---------------------------------------------------------------------------
# Domain entity extraction (lightweight, no LLM needed)
# ---------------------------------------------------------------------------

# Port domain patterns
_BERTH_RE = re.compile(r"\b[Bb](?:erth\s*)?(\d{1,2})\b")
_CRANE_RE = re.compile(r"\b[Cc]rane\s*(\d{1,2})\b")
_VESSEL_RE = re.compile(r"\b[Vv]essel\s+([A-Z][A-Za-z\s]{2,20})\b")
_METRIC_KEYWORDS = {
    "wind speed", "wave height", "tide", "pressure", "productivity",
    "delay", "dwell", "turn time", "transactions", "moves per hour",
    "teu", "loa", "crane rate", "berth productivity", "breakdown",
}
_DATE_RE = re.compile(r"\b(20\d{2})\b")


def extract_entities(text: str) -> List[str]:
    """Extract port domain entities from text. Fast regex, no LLM."""
    entities = []
    text_lower = text.lower()

    for m in _BERTH_RE.finditer(text):
        entities.append(f"berth_B{m.group(1)}")
    for m in _CRANE_RE.finditer(text):
        entities.append(f"crane_{m.group(1)}")
    for m in _VESSEL_RE.finditer(text):
        entities.append(f"vessel_{m.group(1).strip()}")
    for kw in _METRIC_KEYWORDS:
        if kw in text_lower:
            entities.append(f"metric_{kw.replace(' ', '_')}")
    for m in _DATE_RE.finditer(text):
        entities.append(f"year_{m.group(1)}")

    return list(set(entities))


# ---------------------------------------------------------------------------
# Short-term memory (in-memory, per session)
# ---------------------------------------------------------------------------

class ShortTermMemoryStore:
    """
    Per-session conversation memory.

    Keeps the most recent `max_raw_turns` turns in full detail.
    Older turns are compressed into ConversationSummary objects via LLM.
    """

    def __init__(self, session_id: str, max_raw_turns: int = 10) -> None:
        self.session_id = session_id
        self.turns: List[ConversationTurn] = []
        self.summaries: List[ConversationSummary] = []
        self.active_entities: Dict[str, Any] = {}
        self.last_tool_results: Dict[str, str] = {}   # tool_name -> brief summary
        self.max_raw_turns = max_raw_turns
        self._turn_counter = 0

    def add_turn(
        self,
        role: str,
        content: str,
        tool_results_summary: Optional[List[str]] = None,
    ) -> ConversationTurn:
        """Add a new conversation turn."""
        self._turn_counter += 1
        entities = extract_entities(content)

        turn = ConversationTurn(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat(),
            turn_id=self._turn_counter,
            tool_results_summary=tool_results_summary,
            entities_mentioned=entities,
        )
        self.turns.append(turn)

        # Update active entities
        for e in entities:
            self.active_entities[e] = {
                "last_turn": self._turn_counter,
                "context": content[:100],
            }

        # Auto-summarize if too many raw turns
        if len(self.turns) > self.max_raw_turns:
            self._summarize_old_turns()

        return turn

    def update_tool_results(self, tool_name: str, summary: str) -> None:
        """Track the most recent result per tool."""
        self.last_tool_results[tool_name] = summary

    def get_context_for_prompt(self, max_chars: int = 3000) -> str:
        """
        Build a formatted context string for injection into LLM prompts.
        Budget: ~3000 chars ≈ ~750 tokens.
        """
        parts = []

        # 1. Summaries of older turns
        if self.summaries:
            summary_text = " ".join(s.get("summary_text", "") for s in self.summaries)
            parts.append(f"[Earlier conversation summary]: {summary_text[:800]}")

        # 2. Recent raw turns
        recent = self.turns[-6:]  # last 3 exchanges (user+assistant)
        if recent:
            turn_lines = []
            for t in recent:
                role = t.get("role", "?")
                content = t.get("content", "")[:300]
                turn_lines.append(f"  {role}: {content}")
            parts.append("[Recent conversation]:\n" + "\n".join(turn_lines))

        # 3. Active entities
        if self.active_entities:
            top_entities = sorted(
                self.active_entities.items(),
                key=lambda x: x[1].get("last_turn", 0),
                reverse=True,
            )[:8]
            ent_str = ", ".join(k for k, _ in top_entities)
            parts.append(f"[Active entities]: {ent_str}")

        # 4. Last tool results
        if self.last_tool_results:
            tool_str = "; ".join(
                f"{k}: {v[:100]}" for k, v in self.last_tool_results.items()
            )
            parts.append(f"[Recent tool results]: {tool_str}")

        result = "\n\n".join(parts)
        return result[:max_chars]

    def get_all_entities(self) -> List[str]:
        """Return all entities mentioned in the session."""
        return list(self.active_entities.keys())

    def _summarize_old_turns(self) -> None:
        """Compress the oldest half of turns into a summary."""
        half = len(self.turns) // 2
        old_turns = self.turns[:half]
        self.turns = self.turns[half:]

        # Build text to summarize
        turn_text = "\n".join(
            f"{t.get('role', '?')}: {t.get('content', '')[:200]}"
            for t in old_turns
        )
        turn_ids = [t.get("turn_id", 0) for t in old_turns]

        # LLM summarization
        summary_text = llm_chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Summarize this conversation about port operations in 2-3 sentences. "
                        "Preserve key entities (berth IDs, metrics, dates) and factual conclusions."
                    ),
                },
                {"role": "user", "content": turn_text},
            ],
            temperature=0.0,
            timeout=30,
        )

        if not summary_text:
            summary_text = f"(Summary of turns {turn_ids[0]}-{turn_ids[-1]})"

        # Extract key entities from all old turns
        all_entities = []
        for t in old_turns:
            all_entities.extend(t.get("entities_mentioned", []))

        self.summaries.append(ConversationSummary(
            summary_text=summary_text,
            turns_covered=turn_ids,
            key_entities=list(set(all_entities)),
            key_facts=[],
        ))

        logger.info(
            "Summarized %d old turns into summary (session=%s)",
            len(old_turns), self.session_id,
        )


# ---------------------------------------------------------------------------
# Long-term memory (SQLite, cross-session)
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS long_term_memory (
    entry_id TEXT PRIMARY KEY,
    session_id TEXT,
    timestamp TEXT,
    entry_type TEXT,
    content TEXT,
    entities TEXT,
    access_count INTEGER DEFAULT 0
)
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_ltm_entities ON long_term_memory(entities);
"""


class LongTermMemoryStore:
    """
    SQLite-backed long-term memory for cross-session context.

    Entry types:
    - "conversation_summary": end-of-session summary
    - "faq_pattern": frequently asked question patterns
    - "user_preference": user interaction preferences
    - "domain_fact": learned domain facts
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute(_CREATE_TABLE_SQL)
        self._conn.execute(_CREATE_INDEX_SQL)
        self._conn.commit()
        logger.info("LongTermMemoryStore initialized at %s", self.db_path)

    def store(
        self,
        entry_type: str,
        content: str,
        entities: List[str],
        session_id: str = "",
    ) -> str:
        """Store a memory entry. Returns entry_id."""
        entry_id = str(uuid.uuid4())[:12]
        self._conn.execute(
            "INSERT INTO long_term_memory (entry_id, session_id, timestamp, entry_type, content, entities) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (entry_id, session_id, datetime.now().isoformat(), entry_type,
             content, json.dumps(entities)),
        )
        self._conn.commit()
        return entry_id

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories by keyword overlap.

        Scores by: keyword matches in content + entity overlap + recency + access_count.
        """
        query_lower = query.lower()
        query_entities = set(extract_entities(query))
        query_words = set(query_lower.split())

        cursor = self._conn.execute(
            "SELECT entry_id, session_id, timestamp, entry_type, content, entities, access_count "
            "FROM long_term_memory ORDER BY timestamp DESC LIMIT 100"
        )

        scored = []
        for row in cursor:
            entry_id, session_id, ts, entry_type, content, entities_json, access_count = row
            content_lower = content.lower()
            stored_entities = set(json.loads(entities_json)) if entities_json else set()

            # Score: word overlap + entity overlap + access boost
            word_overlap = len(query_words & set(content_lower.split())) / max(len(query_words), 1)
            entity_overlap = len(query_entities & stored_entities) / max(len(query_entities), 1) if query_entities else 0
            score = word_overlap * 0.4 + entity_overlap * 0.5 + min(access_count, 5) * 0.02

            if score > 0.1:
                scored.append({
                    "entry_id": entry_id,
                    "session_id": session_id,
                    "entry_type": entry_type,
                    "content": content,
                    "entities": list(stored_entities),
                    "score": round(score, 3),
                })

        # Sort by score, return top_k
        scored.sort(key=lambda x: x["score"], reverse=True)
        results = scored[:top_k]

        # Increment access counts
        for r in results:
            self._conn.execute(
                "UPDATE long_term_memory SET access_count = access_count + 1 WHERE entry_id = ?",
                (r["entry_id"],),
            )
        if results:
            self._conn.commit()

        return results

    def store_session_summary(
        self, session_id: str, summary: str, entities: List[str]
    ) -> str:
        """Store a session-level summary when session ends."""
        return self.store("conversation_summary", summary, entities, session_id)

    def get_frequent_topics(self, top_k: int = 5) -> List[str]:
        """Get most accessed memory entries as frequent topics."""
        cursor = self._conn.execute(
            "SELECT content FROM long_term_memory "
            "ORDER BY access_count DESC LIMIT ?",
            (top_k,),
        )
        return [row[0][:100] for row in cursor]

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# MemoryManager — unified facade
# ---------------------------------------------------------------------------

class MemoryManager:
    """
    Coordinates short-term (per-session) and long-term (cross-session) memory.

    Used by SessionManager for conversation management and by AgentNodes
    for context injection.
    """

    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root)
        db_path = self.project_root / "storage" / "memory" / "long_term.db"
        self.long_term = LongTermMemoryStore(db_path)
        self._sessions: Dict[str, ShortTermMemoryStore] = {}

    def get_or_create_session(self, session_id: str) -> ShortTermMemoryStore:
        """Get or create a short-term memory store for the session."""
        if session_id not in self._sessions:
            self._sessions[session_id] = ShortTermMemoryStore(session_id)
            logger.info("Created new short-term memory for session=%s", session_id)
        return self._sessions[session_id]

    def build_memory_context(
        self, session_id: str, current_query: str, max_chars: int = 3000
    ) -> str:
        """
        Build combined memory context for prompt injection.

        Budget allocation: ~70% short-term, ~30% long-term.
        """
        parts = []

        # Short-term: conversation context
        short_term = self._sessions.get(session_id)
        if short_term:
            st_context = short_term.get_context_for_prompt(
                max_chars=int(max_chars * 0.7)
            )
            if st_context.strip():
                parts.append(st_context)

        # Long-term: relevant memories from previous sessions
        lt_results = self.long_term.retrieve(current_query, top_k=3)
        if lt_results:
            lt_lines = ["[Relevant memories from previous sessions]:"]
            for mem in lt_results:
                lt_lines.append(
                    f"  - ({mem['entry_type']}) {mem['content'][:200]}"
                )
            lt_text = "\n".join(lt_lines)
            remaining = max_chars - len("\n\n".join(parts)) if parts else max_chars
            parts.append(lt_text[:int(remaining)])

        return "\n\n".join(parts) if parts else ""

    def end_session(self, session_id: str) -> None:
        """
        End a session: summarize and store in long-term memory,
        then clean up short-term store.
        """
        short_term = self._sessions.get(session_id)
        if not short_term or not short_term.turns:
            self._sessions.pop(session_id, None)
            return

        # Build session summary
        all_content = "\n".join(
            f"{t.get('role', '?')}: {t.get('content', '')[:150]}"
            for t in short_term.turns[-10:]  # last 10 turns
        )
        summary = llm_chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Summarize this port decision-support conversation in 2-3 sentences. "
                        "Focus on what was asked, what was found, and key conclusions."
                    ),
                },
                {"role": "user", "content": all_content},
            ],
            temperature=0.0,
            timeout=30,
        )

        if not summary:
            summary = f"Session {session_id}: {len(short_term.turns)} turns"

        entities = short_term.get_all_entities()
        self.long_term.store_session_summary(session_id, summary, entities)
        logger.info(
            "Session %s ended: %d turns, %d entities stored to long-term",
            session_id, len(short_term.turns), len(entities),
        )

        self._sessions.pop(session_id, None)

    def close(self) -> None:
        """Clean up all resources."""
        self.long_term.close()
