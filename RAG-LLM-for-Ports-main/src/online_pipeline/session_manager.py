# src/online_pipeline/session_manager.py
"""
Session manager for multi-turn conversation support.

Handles:
- Session lifecycle (create, track, end)
- Follow-up query resolution (co-reference, ellipsis)
- Conversation turn recording
- Memory context injection for agent state
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .agent_memory import MemoryManager, ShortTermMemoryStore
from .llm_client import llm_chat
from .agent_prompts import QUERY_RESOLUTION_PROMPT

logger = logging.getLogger("online_pipeline.session_manager")


class SessionManager:
    """
    Manages multi-turn conversation sessions.

    Each session has:
    - A unique session_id
    - Short-term memory (conversation turns, entities, tool results)
    - Access to long-term memory for cross-session context

    Usage in /ask_agent endpoint:
        1. session_mgr.get_or_create(session_id) -> (session_id, short_term)
        2. session_mgr.resolve_query(session_id, raw_query) -> standalone query
        3. session_mgr.build_agent_state_extras(session_id, query) -> state dict
        4. ... invoke agent ...
        5. session_mgr.record_turn(session_id, "user", raw_query)
        6. session_mgr.record_turn(session_id, "assistant", answer, tool_summaries)
    """

    def __init__(self, memory_manager: MemoryManager) -> None:
        self.memory = memory_manager
        self._session_meta: Dict[str, Dict[str, Any]] = {}

    def get_or_create(
        self, session_id: Optional[str] = None
    ) -> Tuple[str, ShortTermMemoryStore]:
        """
        Get existing session or create a new one.
        Returns (session_id, short_term_memory_store).
        """
        if not session_id:
            session_id = str(uuid.uuid4())[:12]

        short_term = self.memory.get_or_create_session(session_id)

        if session_id not in self._session_meta:
            self._session_meta[session_id] = {
                "created_at": datetime.now().isoformat(),
                "last_active": datetime.now().isoformat(),
                "turn_count": 0,
            }

        self._session_meta[session_id]["last_active"] = datetime.now().isoformat()
        return session_id, short_term

    def resolve_query(self, session_id: str, raw_query: str) -> str:
        """
        Resolve a follow-up query into a standalone question.

        If the session has no history or the query is self-contained,
        returns the query unchanged.

        Examples:
        - "那 crane 呢？" + history about berth B3 → "What about crane operations at berth B3?"
        - "and the rules?" + history about wind speed → "What are the rules for wind speed limits?"
        """
        short_term = self.memory.get_or_create_session(session_id)

        # No history → query is already standalone
        if not short_term.turns:
            return raw_query

        # Check if query likely needs resolution (short, contains references)
        needs_resolution = self._likely_needs_resolution(raw_query)
        if not needs_resolution:
            return raw_query

        # Build conversation context for resolution
        conversation_context = short_term.get_context_for_prompt(max_chars=1500)

        resolved = llm_chat(
            messages=[
                {
                    "role": "system",
                    "content": QUERY_RESOLUTION_PROMPT.format(
                        conversation_context=conversation_context,
                        current_query=raw_query,
                    ),
                },
                {"role": "user", "content": raw_query},
            ],
            temperature=0.0,
            timeout=30,
        )

        if resolved and resolved.strip():
            resolved = resolved.strip()
            if resolved != raw_query:
                logger.info(
                    "Query resolved: '%s' → '%s' (session=%s)",
                    raw_query[:50], resolved[:50], session_id,
                )
            return resolved

        return raw_query

    def record_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_results_summary: Optional[List[str]] = None,
    ) -> None:
        """Record a conversation turn in short-term memory."""
        short_term = self.memory.get_or_create_session(session_id)
        short_term.add_turn(role, content, tool_results_summary)

        meta = self._session_meta.get(session_id)
        if meta:
            meta["turn_count"] = meta.get("turn_count", 0) + 1
            meta["last_active"] = datetime.now().isoformat()

    def build_agent_state_extras(
        self, session_id: str, current_query: str
    ) -> Dict[str, Any]:
        """
        Build the extra state fields to inject into AgentState
        for multi-turn and memory support.

        Returns a dict that can be merged with the base agent state.
        """
        short_term = self.memory.get_or_create_session(session_id)

        # Conversation history (recent turns as ConversationTurn list)
        recent_turns = short_term.turns[-6:]  # last 3 exchanges

        # Conversation summary (compressed older turns)
        summaries = short_term.summaries
        summary_text = " ".join(
            s.get("summary_text", "") for s in summaries
        ) if summaries else None

        # Combined memory context (short-term + long-term)
        memory_context = self.memory.build_memory_context(
            session_id, current_query, max_chars=3000
        )

        # Active entities
        active_entities = dict(short_term.active_entities)

        return {
            "session_id": session_id,
            "conversation_history": recent_turns,
            "conversation_summary": summary_text if summary_text else None,
            "memory_context": memory_context if memory_context else None,
            "active_entities": active_entities,
        }

    def end_session(self, session_id: str) -> None:
        """End a session and persist summary to long-term memory."""
        self.memory.end_session(session_id)
        self._session_meta.pop(session_id, None)
        logger.info("Session %s ended", session_id)

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session metadata (for debugging/API)."""
        meta = self._session_meta.get(session_id)
        if not meta:
            return None
        short_term = self.memory.get_or_create_session(session_id)
        return {
            **meta,
            "session_id": session_id,
            "total_turns": len(short_term.turns),
            "active_entities": list(short_term.active_entities.keys()),
            "summaries": len(short_term.summaries),
        }

    @staticmethod
    def _likely_needs_resolution(query: str) -> bool:
        """
        Heuristic: does this query likely reference prior context?

        Short queries, queries with pronouns/references, or queries
        starting with conjunctions are likely follow-ups.
        """
        q = query.lower().strip()

        # Very short queries are often follow-ups
        if len(q.split()) <= 4:
            return True

        # Chinese follow-up patterns
        chinese_patterns = [
            "那", "呢", "这个", "那个", "同样的", "上面的", "刚才",
            "还有", "另外", "以及", "关于",
        ]
        if any(p in q for p in chinese_patterns):
            return True

        # English follow-up patterns
        english_patterns = [
            "what about", "how about", "and the", "and for",
            "same for", "also for", "the same", "that one",
            "those", "this", "these", "it", "them",
            "what if", "instead",
        ]
        if any(q.startswith(p) or f" {p} " in f" {q} " for p in english_patterns):
            return True

        return False
