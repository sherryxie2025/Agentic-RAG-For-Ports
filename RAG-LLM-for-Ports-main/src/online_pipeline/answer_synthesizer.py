# src/online_pipeline/answer_synthesizer.py

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import logging

from .llm_client import get_openai_client, get_model_name

logger = logging.getLogger("online_pipeline.answer_synthesizer")


class AnswerSynthesizer:
    """
    Product-style, evidence-grounded, LLM-augmented answer synthesizer.

    Design goals:
    - Evidence first: SQL / docs / rules / graph are primary.
    - Controlled LLM fallback: allowed only as a supplement.
    - Product-style answer: not system logs, not raw branch concatenation.
    - Safe decision support: if policy/rule evidence is missing, do not fabricate hard thresholds.
    """

    def __init__(
        self,
        use_llm_fallback: bool = True,
        model_name: str | None = None,
    ) -> None:
        self.use_llm_fallback = use_llm_fallback
        self.model_name = model_name or get_model_name()
        self.client = get_openai_client() if self.use_llm_fallback else None

    # =========================================================
    # Public API
    # =========================================================

    def synthesize(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = state.get("user_query", "")
        answer_mode = state.get("answer_mode", "lookup")

        docs = state.get("retrieved_docs", []) or []
        sql_results = state.get("sql_results", []) or []
        rule_results = state.get("rule_results", {}) or {}
        graph_results = state.get("graph_results", {}) or {}

        doc_summary = self._summarize_docs(docs)
        sql_summary = self._summarize_sql(sql_results)
        rule_summary = self._summarize_rules(rule_results)
        graph_summary = self._summarize_graph(graph_results)

        sources_used = self._collect_sources_used(
            docs=docs,
            sql_results=sql_results,
            rule_results=rule_results,
            graph_results=graph_results,
        )

        reasoning_summary: List[str] = []
        caveats: List[str] = []
        knowledge_fallback_notes: List[str] = []

        if doc_summary:
            reasoning_summary.append("Document evidence available.")
        if sql_summary:
            reasoning_summary.append("Structured operational evidence available.")
        if rule_summary:
            reasoning_summary.append("Rule evidence available.")
        if graph_summary:
            reasoning_summary.append("Graph reasoning evidence available.")

        if state.get("needs_rules") and not rule_summary:
            caveats.append(
                "Requested rule evidence was not retrieved from the current rule store."
            )

        if state.get("needs_vector") and not doc_summary:
            caveats.append(
                "Requested document evidence was not retrieved from the current document store."
            )

        if state.get("needs_graph_reasoning") and not graph_summary:
            caveats.append(
                "Graph reasoning was requested but no strong graph path was found."
            )

        knowledge_fallback_used, fallback_reason = self._should_use_knowledge_fallback(
            answer_mode=answer_mode,
            needs_rules=state.get("needs_rules", False),
            needs_vector=state.get("needs_vector", False),
            needs_graph_reasoning=state.get("needs_graph_reasoning", False),
            has_doc_summary=bool(doc_summary),
            has_sql_summary=bool(sql_summary),
            has_rule_summary=bool(rule_summary),
            has_graph_summary=bool(graph_summary),
        )

        if knowledge_fallback_used:
            if state.get("needs_rules") and not rule_summary:
                knowledge_fallback_notes.append(
                    "Rule evidence was missing, so the answer may include cautious general-domain operational interpretation."
                )
            if state.get("needs_vector") and not doc_summary:
                knowledge_fallback_notes.append(
                    "Document evidence was missing, so the answer may include limited general-domain background context."
                )
            if answer_mode == "diagnostic":
                knowledge_fallback_notes.append(
                    "Diagnostic explanation may include plausible domain mechanisms beyond directly retrieved evidence."
                )

        llm_answer_used = False
        llm_error: Optional[str] = None
        answer_text: str

        evidence_packet = self._build_evidence_packet(
            query=query,
            answer_mode=answer_mode,
            doc_summary=doc_summary,
            sql_summary=sql_summary,
            rule_summary=rule_summary,
            graph_summary=graph_summary,
            caveats=caveats,
            knowledge_fallback_used=knowledge_fallback_used,
            fallback_reason=fallback_reason,
            sources_used=sources_used,
        )

        # Detect SQL-primary scenario for focused prompt
        is_sql_primary = (
            bool(sql_summary)
            and not doc_summary
            and not graph_summary
            and answer_mode in ("lookup", "comparison")
        )

        logger.info(
            "SYNTH: mode=%s sources=%s sql_primary=%s fallback=%s reason=%s",
            answer_mode, sources_used, is_sql_primary,
            knowledge_fallback_used, fallback_reason,
        )
        logger.info(
            "SYNTH evidence: doc=%s sql=%s rules=%s graph=%s",
            bool(doc_summary), bool(sql_summary), bool(rule_summary), bool(graph_summary),
        )
        if caveats:
            for c in caveats:
                logger.warning("SYNTH caveat: %s", c)

        if self.client is not None:
            llm_answer, llm_error = self._call_llm_answer(
                evidence_packet=evidence_packet,
                sql_primary=is_sql_primary,
            )
            if llm_answer:
                answer_text = llm_answer
                llm_answer_used = True
            else:
                answer_text = self._build_rule_based_answer(
                    query=query,
                    answer_mode=answer_mode,
                    doc_summary=doc_summary,
                    sql_summary=sql_summary,
                    rule_summary=rule_summary,
                    graph_summary=graph_summary,
                    knowledge_fallback_used=knowledge_fallback_used,
                    caveats=caveats,
                )
        else:
            answer_text = self._build_rule_based_answer(
                query=query,
                answer_mode=answer_mode,
                doc_summary=doc_summary,
                sql_summary=sql_summary,
                rule_summary=rule_summary,
                graph_summary=graph_summary,
                knowledge_fallback_used=knowledge_fallback_used,
                caveats=caveats,
            )

        grounding_status = self._infer_grounding_status(
            has_docs=bool(doc_summary),
            has_sql=bool(sql_summary),
            has_rules=bool(rule_summary),
            has_graph=bool(graph_summary),
            knowledge_fallback_used=knowledge_fallback_used,
        )

        confidence = self._estimate_confidence(
            answer_mode=answer_mode,
            has_docs=bool(doc_summary),
            has_sql=bool(sql_summary),
            has_rules=bool(rule_summary),
            has_graph=bool(graph_summary),
            knowledge_fallback_used=knowledge_fallback_used,
        )

        logger.info(
            "SYNTH result: confidence=%.2f grounding=%s llm_used=%s answer_len=%d",
            confidence, grounding_status, llm_answer_used, len(answer_text),
        )
        if llm_error:
            logger.error("SYNTH LLM error: %s", llm_error)

        return {
            "answer": answer_text,
            "confidence": confidence,
            "sources_used": sources_used,
            "reasoning_summary": reasoning_summary,
            "caveats": caveats,
            "grounding_status": grounding_status,
            "llm_answer_used": llm_answer_used,
            "knowledge_fallback_used": knowledge_fallback_used,
            "knowledge_fallback_notes": knowledge_fallback_notes,
            # debug-friendly metadata
            "fallback_reason": fallback_reason,
            "llm_error": llm_error,
            "evidence_snapshot": {
                "doc_summary": doc_summary,
                "sql_summary": sql_summary,
                "rule_summary": rule_summary,
                "graph_summary": graph_summary,
            },
        }

    # =========================================================
    # Evidence summarization
    # =========================================================

    def _summarize_docs(self, docs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not docs:
            return None

        top_docs = docs[:3]
        items = []
        for d in top_docs:
            items.append(
                {
                    "source_file": d.get("source_file"),
                    "page": d.get("page"),
                    "score": d.get("score"),
                    "snippet": (d.get("text") or "")[:500].replace("\n", " ").strip(),
                }
            )

        best = items[0]
        return {
            "top_documents": items,
            "best_source_file": best.get("source_file"),
            "best_page": best.get("page"),
            "best_snippet": best.get("snippet"),
        }

    def _summarize_sql(self, sql_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not sql_results:
            return None

        first = sql_results[0]
        if not first.get("execution_ok"):
            return {
                "execution_ok": False,
                "error": first.get("error"),
            }

        rows = first.get("rows", []) or []
        plan = first.get("plan", {}) or {}
        preview_rows = [r.get("data", {}) for r in rows[:5]]

        derived_insights = self._derive_sql_insights(
            preview_rows=preview_rows,
            aggregation=plan.get("aggregation"),
        )

        return {
            "execution_ok": True,
            "used_tables": plan.get("target_tables", []),
            "generated_sql": plan.get("generated_sql"),
            "aggregation": plan.get("aggregation"),
            "row_count": first.get("row_count", len(rows)),
            "preview_rows": preview_rows,
            "derived_insights": derived_insights,
        }

    def _summarize_rules(self, rule_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not rule_results:
            return None

        matched_rules = rule_results.get("matched_rules", []) or []
        if not matched_rules:
            return None

        top_rules = []
        for rule in matched_rules[:3]:
            top_rules.append(
                {
                    "rule_text": rule.get("rule_text"),
                    "variable": rule.get("variable"),
                    "threshold_raw": rule.get("threshold_raw"),
                    "operator": rule.get("operator"),
                    "value": rule.get("value"),
                    "canonical_unit": rule.get("canonical_unit"),
                    "source_file": rule.get("source_file"),
                    "page": rule.get("page"),
                    "triggered": rule.get("triggered"),
                    "trigger_explanation": rule.get("trigger_explanation"),
                }
            )

        top_rule = top_rules[0]
        return {
            "applicable_rule_count": rule_results.get("applicable_rule_count", len(matched_rules)),
            "triggered_rule_count": rule_results.get("triggered_rule_count", 0),
            "top_rule": top_rule,
            "top_rules": top_rules,
        }

    def _summarize_graph(self, graph_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not graph_results:
            return None

        paths = graph_results.get("reasoning_paths", []) or []
        if not paths:
            return None

        top_paths = []
        for p in paths[:4]:
            path_nodes = p.get("path_nodes", []) or []
            path_edges = p.get("path_edges", []) or []

            relation_chain = []
            for i in range(min(len(path_nodes) - 1, len(path_edges))):
                relation_chain.append(
                    {
                        "from": path_nodes[i],
                        "edge": path_edges[i],
                        "to": path_nodes[i + 1],
                    }
                )

            top_paths.append(
                {
                    "start_node": p.get("start_node"),
                    "end_node": p.get("end_node"),
                    "path_nodes": path_nodes,
                    "path_edges": path_edges,
                    "path_text": " -> ".join(path_nodes) if path_nodes else None,
                    "relation_chain": relation_chain,
                    "explanation": p.get("explanation"),
                }
            )

        return {
            "query_entities": graph_results.get("query_entities", []),
            "expanded_nodes": graph_results.get("expanded_nodes", []),
            "top_paths": top_paths,
            "graph_interpretation": self._derive_graph_insights(top_paths),
        }

    # =========================================================
    # Evidence packet for LLM
    # =========================================================

    def _build_evidence_packet(
        self,
        query: str,
        answer_mode: str,
        doc_summary: Optional[Dict[str, Any]],
        sql_summary: Optional[Dict[str, Any]],
        rule_summary: Optional[Dict[str, Any]],
        graph_summary: Optional[Dict[str, Any]],
        caveats: List[str],
        knowledge_fallback_used: bool,
        fallback_reason: Optional[str],
        sources_used: List[str],
    ) -> Dict[str, Any]:

        decision_guardrail = self._decision_guardrail(
            answer_mode=answer_mode,
            rule_summary=rule_summary,
        )

        fallback_policy = {
            "enabled": knowledge_fallback_used,
            "reason": fallback_reason,
            "instruction": (
                "You may use cautious general maritime/port operational knowledge ONLY as supplementary explanation. "
                "Do not invent retrieved rules, do not fabricate thresholds, and do not present general knowledge as database evidence."
            ),
        }

        response_contract = {
            "required_sections": self._required_sections(answer_mode),
            "style": "professional, product-style, evidence-grounded",
            "forbidden_behaviors": [
                "Do not mention internal variable names.",
                "Do not say 'SQL returned' or 'graph returned' in a log-like way.",
                "Do not fabricate policy thresholds.",
                "Do not claim proven causality unless evidence explicitly supports it.",
            ],
        }

        return {
            "query": query,
            "answer_mode": answer_mode,
            "sources_used": sources_used,
            "evidence": {
                "documents": doc_summary,
                "sql": sql_summary,
                "rules": rule_summary,
                "graph": graph_summary,
            },
            "caveats": caveats,
            "fallback_policy": fallback_policy,
            "decision_guardrail": decision_guardrail,
            "response_contract": response_contract,
        }


    def _decision_guardrail(
        self,
        answer_mode: str,
        rule_summary: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Extra safety layer for decision-support answers.

        If decision-support is requested but no rule evidence is available,
        the final answer must NOT issue a hard operational recommendation.
        """
        no_rule_decision_mode = answer_mode == "decision_support" and not rule_summary

        if no_rule_decision_mode:
            return {
                "hard_recommendation_allowed": False,
                "allowed_recommendation_style": [
                    "situational assessment",
                    "risk signal description",
                    "manual verification recommendation",
                    "policy-check recommendation",
                ],
                "forbidden_recommendation_style": [
                    "Proceed with operations",
                    "Pause operations now",
                    "Suspend operations",
                    "Restrict vessel entry above X",
                    "Safe to continue",
                    "No immediate hazard requiring suspension",
                ],
                "instruction": (
                    "Because no rule evidence was retrieved, you must NOT issue a hard stop/go "
                    "or restrict/allow decision. You may only provide a bounded operational "
                    "assessment, explain what the current data suggests, and recommend manual "
                    "verification against terminal or port policy."
                ),
            }

        return {
            "hard_recommendation_allowed": True,
            "allowed_recommendation_style": [
                "evidence-grounded recommendation"
            ],
            "forbidden_recommendation_style": [],
            "instruction": (
                "If sufficient rule evidence exists, you may provide an evidence-grounded recommendation."
            ),
        }


    # =========================================================
    # Fallback logic
    # =========================================================

    def _should_use_knowledge_fallback(
        self,
        answer_mode: str,
        needs_rules: bool,
        needs_vector: bool,
        needs_graph_reasoning: bool,
        has_doc_summary: bool,
        has_sql_summary: bool,
        has_rule_summary: bool,
        has_graph_summary: bool,
    ) -> Tuple[bool, Optional[str]]:
        if answer_mode == "decision_support":
            if needs_rules and not has_rule_summary:
                return True, "decision_support_without_rule_evidence"
            if not has_sql_summary and not has_rule_summary:
                return True, "decision_support_with_weak_grounding"

        if answer_mode == "diagnostic":
            if needs_graph_reasoning and not has_graph_summary:
                return True, "diagnostic_without_graph_evidence"
            if has_sql_summary and has_graph_summary:
                return True, "diagnostic_mechanism_enrichment"

        if answer_mode == "descriptive":
            if needs_vector and not has_doc_summary:
                return True, "descriptive_without_docs"

        if answer_mode == "lookup":
            if not any([has_doc_summary, has_sql_summary, has_rule_summary, has_graph_summary]):
                return True, "lookup_without_grounded_evidence"

        return False, None

    # =========================================================
    # LLM call
    # =========================================================

    def _call_llm_answer(
        self,
        evidence_packet: Dict[str, Any],
        sql_primary: bool = False,
    ) -> Tuple[Optional[str], Optional[str]]:
        try:
            if sql_primary:
                return self._call_llm_sql_focused(evidence_packet)

            system_prompt = """
You are the final decision-support layer of an AI Port Operations System.

Your role:
- Produce a polished, product-style answer for an end user.
- Prioritize retrieved evidence.
- Use general operational knowledge only as a controlled supplement when the fallback policy allows it.

Core rules:
1. Evidence first. Prefer SQL, rules, graph, and documents over general knowledge.
2. Never present general knowledge as if it were retrieved evidence.
3. If rule evidence is missing, do NOT issue a hard policy ruling, do NOT invent a threshold, and do NOT produce a stop/go recommendation.
4. For diagnostic questions, you may explain plausible mechanisms and contributing pathways.
5. Keep the response professional and well-structured.
6. Do not write like a system log.
7. Do not mention internal keys such as 'sql_summary', 'graph_summary', or 'fallback_policy'.

Inline citation requirement (MANDATORY):
- Every factual claim MUST be annotated with its source tag:
  [doc] for document evidence, [sql] for structured data, [rule] for policy/rule evidence, [graph] for graph reasoning.
- If a statement is based on general domain knowledge (not from retrieved evidence), tag it as [general knowledge].
- Example: "Wind speed averaged 5.2 m/s in 2015 [sql]. Port regulations require vessels to reduce speed in high-wind conditions [rule]."
- Do NOT make untagged factual claims. Every sentence with a factual assertion must have at least one source tag.

Critical guardrail for decision-support:
- If the decision_guardrail says hard_recommendation_allowed = false,
  you MUST NOT tell the user to proceed, pause, suspend, restrict, allow, or continue operations.
- In that case, the Recommendation section must be framed as:
  - bounded assessment,
  - uncertainty statement,
  - manual verification against policy,
  - request for explicit rule/policy confirmation.
- You must not transform general operational norms into actionable policy instructions.

Output requirements:
- Use markdown section headers.
- Be concise but sufficiently explanatory for a decision-support context.
- Synthesize the evidence into a coherent answer instead of listing raw outputs.

Section guidance by mode:
- diagnostic:
  ### Key Insight
  ### Evidence-Based Analysis
  ### Mechanism / Interpretation
  ### Notes

- decision_support:
  ### Current Assessment
  ### Evidence-Based Analysis
  ### Recommendation
  ### Notes

- fact_lookup / descriptive:
  ### Answer
  ### Supporting Evidence
  ### Notes
""".strip()

            user_prompt = (
                "Generate the final user-facing answer from the structured evidence below.\n\n"
                "Important:\n"
                "- If fallback is enabled, you may add cautious general-domain interpretation.\n"
                "- If rules are missing, you must explicitly avoid definitive policy claims.\n"
                "- You MUST obey decision_guardrail.\n"
                "- If decision_guardrail.hard_recommendation_allowed is false, "
                "the Recommendation section must NOT contain a stop/go instruction.\n"
                "- Use the required section structure implied by the answer mode.\n"
                "- MANDATORY: Tag EVERY factual claim with [doc], [sql], [rule], [graph], or [general knowledge]. "
                "No untagged factual claims are allowed.\n\n"
                f"{json.dumps(evidence_packet, ensure_ascii=False, indent=2, default=str)}"
            )


            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.35,
                timeout=120,
            )

            text = response.choices[0].message.content
            if text and text.strip():
                return text.strip(), None
            return None, "empty_llm_response"

        except Exception as e:
            return None, f"{type(e).__name__}: {e}"

    def _call_llm_answer_stream(
        self, evidence_packet: Dict[str, Any], sql_primary: bool = False,
    ):
        """
        Streaming variant of _call_llm_answer. Yields text chunks as they arrive.
        First token latency ~2-3s vs full response ~30s.

        Usage:
            for chunk in synthesizer.synthesize_stream(state):
                print(chunk, end="", flush=True)
        """
        if sql_primary:
            system_prompt = (
                "You summarize SQL query results for a port operations system.\n"
                "Rules: State ONLY numbers and facts. Tag every claim with [sql]. Be concise."
            )
            user_prompt = (
                "Summarize these SQL results. Tag every number with [sql].\n\n"
                f"{json.dumps(evidence_packet, ensure_ascii=False, indent=2, default=str)}"
            )
            temperature = 0.1
        else:
            system_prompt = self._get_general_system_prompt()
            user_prompt = (
                "Generate the final user-facing answer from the structured evidence below.\n\n"
                "- MANDATORY: Tag EVERY factual claim with [doc], [sql], [rule], [graph], or [general knowledge].\n\n"
                f"{json.dumps(evidence_packet, ensure_ascii=False, indent=2, default=str)}"
            )
            temperature = 0.35

        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                timeout=120,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except Exception as e:
            yield f"\n[Streaming error: {type(e).__name__}: {e}]"

    def synthesize_stream(self, state: Dict[str, Any]):
        """
        Streaming version of synthesize(). Yields answer text chunks.
        For demo/UI use — perceived latency drops from ~30s to ~2s first token.
        """
        query = state.get("user_query", "")
        answer_mode = state.get("answer_mode", "lookup")

        docs = state.get("retrieved_docs", []) or []
        sql_results = state.get("sql_results", []) or []
        rule_results = state.get("rule_results", {}) or {}
        graph_results = state.get("graph_results", {}) or {}

        doc_summary = self._summarize_docs(docs)
        sql_summary = self._summarize_sql(sql_results)
        rule_summary = self._summarize_rules(rule_results)
        graph_summary = self._summarize_graph(graph_results)

        sources_used = self._collect_sources_used(
            docs=docs, sql_results=sql_results,
            rule_results=rule_results, graph_results=graph_results,
        )

        evidence_packet = self._build_evidence_packet(
            query=query, answer_mode=answer_mode,
            doc_summary=doc_summary, sql_summary=sql_summary,
            rule_summary=rule_summary, graph_summary=graph_summary,
            caveats=[], knowledge_fallback_used=False,
            fallback_reason=None, sources_used=sources_used,
        )

        is_sql_primary = (
            bool(sql_summary) and not doc_summary
            and not graph_summary
            and answer_mode in ("lookup", "comparison")
        )

        if self.client is not None:
            yield from self._call_llm_answer_stream(evidence_packet, sql_primary=is_sql_primary)
        else:
            yield self._build_rule_based_answer(
                query=query, answer_mode=answer_mode,
                doc_summary=doc_summary, sql_summary=sql_summary,
                rule_summary=rule_summary, graph_summary=graph_summary,
                knowledge_fallback_used=False, caveats=[],
            )

    def _get_general_system_prompt(self) -> str:
        """Extract the general system prompt (shared between blocking and streaming)."""
        return """You are the final decision-support layer of an AI Port Operations System.
Core rules: Evidence first. Never present general knowledge as retrieved evidence.
Inline citation requirement (MANDATORY): Tag EVERY factual claim with [doc], [sql], [rule], [graph], or [general knowledge].
Use markdown section headers. Be concise but explanatory."""

    def _call_llm_sql_focused(
        self, evidence_packet: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Focused prompt for SQL-primary queries to maximize faithfulness."""
        system_prompt = """You summarize SQL query results for a port operations system.

Rules:
1. State ONLY the numbers and facts from the SQL results. Do not add interpretive analysis.
2. Tag every factual claim with [sql]. If rules are present, use [rule].
3. Structure: ### Answer (the direct answer) ### Data Details (key numbers) ### Notes (caveats only if needed).
4. Be concise. A lookup query needs 2-4 sentences, not paragraphs.
5. If the SQL query returned no rows or an error, say so clearly.
6. Never invent data. Only report what the SQL results contain."""

        user_prompt = (
            "Summarize these SQL results into a direct answer. "
            "Tag every number with [sql]. Be factual and concise.\n\n"
            f"{json.dumps(evidence_packet, ensure_ascii=False, indent=2, default=str)}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                timeout=120,
            )
            text = response.choices[0].message.content
            if text and text.strip():
                return text.strip(), None
            return None, "empty_llm_response"
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"

    # =========================================================
    # Rule-based fallback answer
    # =========================================================

    def _build_rule_based_answer(
        self,
        query: str,
        answer_mode: str,
        doc_summary: Optional[Dict[str, Any]],
        sql_summary: Optional[Dict[str, Any]],
        rule_summary: Optional[Dict[str, Any]],
        graph_summary: Optional[Dict[str, Any]],
        knowledge_fallback_used: bool,
        caveats: List[str],
    ) -> str:
        if answer_mode == "decision_support":
            parts: List[str] = ["### Current Assessment"]

            if sql_summary and sql_summary.get("derived_insights"):
                parts.append(
                    "Available operational data provides supporting context for the question."
                )
            else:
                parts.append(
                    "Only limited grounded evidence is currently available for this assessment."
                )

            parts.append("\n### Evidence-Based Analysis")
            if sql_summary and sql_summary.get("preview_rows"):
                parts.append(
                    f"- Structured data is available from: {', '.join(sql_summary.get('used_tables', []))}."
                )
            if rule_summary and rule_summary.get("top_rule"):
                top_rule = rule_summary["top_rule"]
                parts.append(
                    f"- A relevant rule was retrieved: {top_rule.get('rule_text') or 'rule available'}."
                )
            else:
                parts.append(
                    "- No matching rule threshold was retrieved from the current rule store."
                )

            parts.append("\n### Recommendation")
            if rule_summary:
                parts.append(
                    "A grounded recommendation should be based primarily on the retrieved rule together with the current operational evidence."
                )
            else:
                parts.append(
                    "A definitive stop / continue recommendation cannot be issued automatically because the required policy threshold was not retrieved."
                )

            if knowledge_fallback_used:
                parts.append(
                    "A cautious operational interpretation may still be possible, but it should not be treated as a formal rule-based decision."
                )

            if caveats:
                parts.append("\n### Notes")
                for c in caveats:
                    parts.append(f"- {c}")

            return "\n".join(parts)

        if answer_mode == "diagnostic":
            parts = ["### Key Insight"]

            if sql_summary and graph_summary:
                parts.append(
                    "The available evidence supports a plausible link between the observed operational factors."
                )
            else:
                parts.append(
                    "The system found limited but potentially relevant evidence for a diagnostic explanation."
                )

            parts.append("\n### Evidence-Based Analysis")
            if sql_summary and sql_summary.get("derived_insights"):
                for insight in sql_summary["derived_insights"][:3]:
                    parts.append(f"- {insight}")
            if graph_summary and graph_summary.get("graph_interpretation"):
                for insight in graph_summary["graph_interpretation"][:3]:
                    parts.append(f"- {insight}")

            parts.append("\n### Mechanism / Interpretation")
            if graph_summary and graph_summary.get("top_paths"):
                first_path = graph_summary["top_paths"][0]
                parts.append(
                    f"A plausible pathway is: {first_path.get('path_text') or 'graph pathway identified'}."
                )
            else:
                parts.append(
                    "A direct graph pathway was not available, so the interpretation remains weaker."
                )

            if knowledge_fallback_used:
                parts.append(
                    "General operational knowledge may help explain how disruptions propagate, but it should be treated as supplementary interpretation rather than direct retrieved evidence."
                )

            if caveats:
                parts.append("\n### Notes")
                for c in caveats:
                    parts.append(f"- {c}")

            return "\n".join(parts)

        # default / fact lookup / descriptive
        parts = ["### Answer"]

        if sql_summary and sql_summary.get("preview_rows"):
            parts.append("Grounded structured data is available for this query.")
        elif doc_summary and doc_summary.get("best_snippet"):
            parts.append("Relevant document evidence was retrieved for this query.")
        elif rule_summary:
            parts.append("Relevant rule evidence was retrieved for this query.")
        else:
            parts.append("No strong grounded evidence was available for this query.")

        parts.append("\n### Supporting Evidence")
        if sql_summary and sql_summary.get("derived_insights"):
            for insight in sql_summary["derived_insights"][:3]:
                parts.append(f"- {insight}")
        elif doc_summary and doc_summary.get("best_source_file"):
            parts.append(
                f"- Top document source: {doc_summary.get('best_source_file')} (page {doc_summary.get('best_page')})."
            )
        elif rule_summary and rule_summary.get("top_rule"):
            parts.append(
                f"- Top rule: {rule_summary['top_rule'].get('rule_text') or 'rule available'}."
            )
        else:
            parts.append("- No supporting evidence summary available.")

        if caveats:
            parts.append("\n### Notes")
            for c in caveats:
                parts.append(f"- {c}")

        return "\n".join(parts)

    # =========================================================
    # Derived insights
    # =========================================================

    def _derive_sql_insights(
        self,
        preview_rows: List[Dict[str, Any]],
        aggregation: Optional[str],
    ) -> List[str]:
        insights: List[str] = []
        if not preview_rows:
            return insights

        first = preview_rows[0]

        # aggregate-style rows
        aggregate_keys = [
            "avg_berth_delay",
            "avg_berth_productivity",
            "avg_crane_productivity",
            "avg_breakdown_minutes",
            "total_breakdown_minutes",
            "total_calls",
            "min_berth_delay",
            "max_berth_delay",
            "min_crane_productivity",
            "max_crane_productivity",
        ]
        detected_aggregate_keys = [k for k in aggregate_keys if k in first]
        if detected_aggregate_keys:
            if "total_calls" in first:
                insights.append(f"The SQL result summarizes {first['total_calls']} operations.")
            if "avg_berth_delay" in first:
                insights.append(f"Average berth delay is approximately {self._fmt(first['avg_berth_delay'])} hours.")
            if "avg_crane_productivity" in first:
                insights.append(
                    f"Average crane productivity is approximately {self._fmt(first['avg_crane_productivity'])} moves/hour."
                )
            if "avg_breakdown_minutes" in first:
                insights.append(
                    f"Average breakdown time is approximately {self._fmt(first['avg_breakdown_minutes'])} minutes per call."
                )
            if "min_crane_productivity" in first and "max_crane_productivity" in first:
                insights.append(
                    "Crane productivity varies materially across operations, "
                    f"from about {self._fmt(first['min_crane_productivity'])} to {self._fmt(first['max_crane_productivity'])} moves/hour."
                )
            return insights

        # record-style rows
        metric_mentions = []
        for key in ["wave_height_m", "wind_speed_ms", "wind_gust_ms", "berth_productivity_mph", "berth_delay_hours", "arrival_delay_hours"]:
            values = [r.get(key) for r in preview_rows if isinstance(r.get(key), (int, float))]
            if values:
                metric_mentions.append((key, min(values), max(values)))

        for key, lo, hi in metric_mentions[:5]:
            label = key.replace("_", " ")
            if abs(lo - hi) < 1e-9:
                insights.append(f"{label} is approximately {self._fmt(lo)} in the sampled records.")
            else:
                insights.append(
                    f"{label} ranges from approximately {self._fmt(lo)} to {self._fmt(hi)} in the sampled records."
                )

        if aggregation:
            insights.append(f"The SQL plan used {aggregation} aggregation.")
        return insights

    def _derive_graph_insights(self, top_paths: List[Dict[str, Any]]) -> List[str]:
        insights: List[str] = []
        for path in top_paths[:3]:
            path_text = path.get("path_text")
            if path_text:
                insights.append(f"Graph reasoning identified a plausible pathway: {path_text}.")
            explanation = path.get("explanation")
            if explanation:
                insights.append(explanation)
        return insights[:4]

    # =========================================================
    # Helper methods
    # =========================================================

    @staticmethod
    def _required_sections(answer_mode: str) -> List[str]:
        if answer_mode == "diagnostic":
            return [
                "Key Insight",
                "Evidence-Based Analysis",
                "Mechanism / Interpretation",
                "Notes",
            ]
        if answer_mode == "decision_support":
            return [
                "Current Assessment",
                "Evidence-Based Analysis",
                "Recommendation",
                "Notes",
            ]
        return [
            "Answer",
            "Supporting Evidence",
            "Notes",
        ]

    @staticmethod
    def _collect_sources_used(
        docs: List[Dict[str, Any]],
        sql_results: List[Dict[str, Any]],
        rule_results: Dict[str, Any],
        graph_results: Dict[str, Any],
    ) -> List[str]:
        sources = []

        if docs:
            sources.append("documents")

        if sql_results and sql_results[0].get("execution_ok"):
            sources.append("structured_operational_data")

        if rule_results and rule_results.get("matched_rules"):
            sources.append("rules")

        if graph_results and graph_results.get("reasoning_paths"):
            sources.append("graph")

        return sources

    @staticmethod
    def _infer_grounding_status(
        has_docs: bool,
        has_sql: bool,
        has_rules: bool,
        has_graph: bool,
        knowledge_fallback_used: bool,
    ) -> str:
        if knowledge_fallback_used:
            return "fallback_augmented"

        grounded_count = sum([has_docs, has_sql, has_rules, has_graph])
        if grounded_count >= 2:
            return "fully_grounded"
        if grounded_count == 1:
            return "partially_grounded"
        return "weakly_grounded"

    @staticmethod
    def _estimate_confidence(
        answer_mode: str,
        has_docs: bool,
        has_sql: bool,
        has_rules: bool,
        has_graph: bool,
        knowledge_fallback_used: bool,
    ) -> float:
        score = 0.30
        score += 0.15 if has_sql else 0.0
        score += 0.10 if has_docs else 0.0
        score += 0.18 if has_rules else 0.0
        score += 0.17 if has_graph else 0.0

        if answer_mode == "decision_support" and not has_rules:
            score -= 0.10

        if knowledge_fallback_used:
            score -= 0.05

        score = max(0.25, min(score, 0.93))
        return round(score, 2)

    @staticmethod
    def _fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)