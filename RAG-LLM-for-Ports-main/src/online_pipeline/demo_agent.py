# src/online_pipeline/demo_agent.py
"""
Demo script for the Plan-and-Execute Agent pipeline.

Demonstrates:
1. Single-turn agent queries
2. Multi-turn conversation with follow-ups
3. Memory system (short-term entity tracking + long-term persistence)
4. ReAct-style tool observations

Run:
    python -m src.online_pipeline.demo_agent
"""

from __future__ import annotations

import json
from pathlib import Path


def print_section(title: str) -> None:
    print("\n" + "=" * 100)
    print(f"  {title}")
    print("=" * 100)


def print_plan(plan: list) -> None:
    for step in plan:
        status = step.get("status", "?")
        icon = {
            "completed": "+", "failed": "x", "pending": ".",
            "running": "~", "skipped": "-",
        }.get(status, "?")
        print(f"  [{icon}] Step {step.get('step_id', '?')}: "
              f"{step.get('tool_name', '?')} — {step.get('purpose', '')}")
        if step.get("result_summary"):
            print(f"      => {step['result_summary']}")


def print_observations(observations: list) -> None:
    if not observations:
        print("    (no observations — ReAct disabled or single-tool plan)")
        return
    for obs in observations:
        action = obs.get("action", "?")
        icon = {"continue": "->", "modify_next": "~>", "abort_replan": "XX"}.get(action, "??")
        print(f"    [{icon}] {obs.get('tool_name', '?')}: {obs.get('observation', '')[:80]}")
        if action == "modify_next":
            print(f"         Modified next query: {obs.get('modified_query', '')[:60]}")
        elif action == "abort_replan":
            print(f"         ABORT reason: {obs.get('reasoning', '')[:80]}")


def run_single_query(agent, query: str, state_extras: dict = None) -> dict:
    """Run a single agent query and return state."""
    base_state = {
        "user_query": query,
        "reasoning_trace": [],
        "warnings": [],
        "tool_results": [],
        "observations": [],
    }
    if state_extras:
        base_state.update(state_extras)

    return agent.invoke(base_state)


def print_result(state: dict, query: str) -> None:
    """Print formatted agent result."""
    # 1. Execution Plan
    print("\n[1] EXECUTION PLAN")
    print_plan(state.get("plan", []))

    # 2. Iterations + Evidence
    print(f"\n[2] ITERATIONS: {state.get('iteration', 0)}")
    print(f"    Evidence sufficient: {state.get('evidence_sufficient', '?')}")
    gaps = state.get("evidence_gaps", [])
    if gaps:
        print(f"    Evidence gaps: {gaps}")

    # 3. ReAct Observations
    print("\n[3] REACT OBSERVATIONS")
    print_observations(state.get("observations", []))

    # 4. Tool Results Summary
    print("\n[4] TOOL RESULTS")
    for tr in state.get("tool_results", []):
        status = "OK" if tr.get("success") else "FAIL"
        print(f"  [{status}] {tr.get('tool_name', '?')} "
              f"({tr.get('execution_time_s', 0):.2f}s)")

    # 5. Final Answer
    print("\n[5] FINAL ANSWER")
    final = state.get("final_answer", {})
    if isinstance(final, dict):
        answer_text = final.get("answer", "(no answer)")
        print(f"    {answer_text[:500]}")
        print(f"\n    Confidence: {final.get('confidence')}")
        print(f"    Sources: {final.get('sources_used', [])}")
        print(f"    Grounding: {final.get('grounding_status', '?')}")
    else:
        print(f"    {final}")

    # 6. Reasoning Trace
    print("\n[6] REASONING TRACE")
    for step in state.get("reasoning_trace", []):
        print(f"    - {step}")


def demo_single_turn(agent) -> None:
    """Demo 1: Single-turn queries."""
    print_section("DEMO 1: Single-Turn Queries")

    queries = [
        "Why might berth delays be related to weather conditions and crane slowdown?",
        "Under what wind conditions should vessel entry be restricted?",
    ]

    for idx, query in enumerate(queries, start=1):
        print_section(f"Query {idx}: {query}")
        state = run_single_query(agent, query)
        print_result(state, query)


def demo_multi_turn(agent, project_root: Path) -> None:
    """Demo 2: Multi-turn conversation with memory."""
    print_section("DEMO 2: Multi-Turn Conversation")

    from .agent_memory import MemoryManager
    from .session_manager import SessionManager

    memory_mgr = MemoryManager(project_root)
    session_mgr = SessionManager(memory_mgr)

    # Create a session
    session_id, _ = session_mgr.get_or_create()
    print(f"  Session created: {session_id}")

    # Turn 1: Initial question
    conversation = [
        "What is the average berth productivity and are there any wind speed rules?",
        "What about crane operations?",          # Follow-up: should reference berth context
        "Are there any safety thresholds for that?",  # Follow-up: should reference crane ops
    ]

    for turn_idx, raw_query in enumerate(conversation, start=1):
        print_section(f"Turn {turn_idx}: \"{raw_query}\"")

        # Resolve follow-up
        resolved = session_mgr.resolve_query(session_id, raw_query)
        if resolved != raw_query:
            print(f"  [Resolved] → \"{resolved}\"")

        # Build memory context
        state_extras = session_mgr.build_agent_state_extras(session_id, resolved)

        # Show injected context
        if state_extras.get("memory_context"):
            print(f"\n  [Memory context injected]: {state_extras['memory_context'][:200]}...")
        if state_extras.get("active_entities"):
            print(f"  [Active entities]: {list(state_extras['active_entities'].keys())}")

        # Run agent
        state = run_single_query(agent, resolved, state_extras)
        print_result(state, resolved)

        # Record turns
        final = state.get("final_answer", {})
        answer_text = final.get("answer", "") if isinstance(final, dict) else str(final)
        tool_summaries = [
            f"{tr.get('tool_name', '?')}" for tr in state.get("tool_results", [])
            if tr.get("success")
        ]
        session_mgr.record_turn(session_id, "user", raw_query)
        session_mgr.record_turn(session_id, "assistant", answer_text[:200], tool_summaries)

    # End session
    print_section("Session Summary")
    info = session_mgr.get_session_info(session_id)
    print(f"  Total turns: {info.get('total_turns', 0)}")
    print(f"  Active entities: {info.get('active_entities', [])}")
    session_mgr.end_session(session_id)
    print("  Session ended and saved to long-term memory.")

    memory_mgr.close()


def main() -> None:
    from .pipeline_logger import setup_pipeline_logging
    setup_pipeline_logging(level="INFO")

    project_root = Path(__file__).resolve().parents[2]

    # Build Agent graph with ReAct enabled
    from .agent_graph import build_agent_graph
    agent = build_agent_graph(
        project_root=project_root,
        chroma_collection_name=None,
        use_llm_sql_planner=True,
        enable_react_observations=True,
    )

    print_section("Plan-and-Execute Agent Demo (Multi-turn + Memory + ReAct)")
    print(f"Project root: {project_root}")

    # Demo 1: Single-turn
    demo_single_turn(agent)

    # Demo 2: Multi-turn conversation
    demo_multi_turn(agent, project_root)

    print_section("Demo Complete")


if __name__ == "__main__":
    main()
