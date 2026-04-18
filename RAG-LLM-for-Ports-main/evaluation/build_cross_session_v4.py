"""
Cross-session evaluation dataset v4 — written FROM SCRATCH.

Design principles (learned from v3's failures):

1. Session 1 queries use `from_sample_id` to inherit golden fields (so
   the DAG actually retrieves real data and end_session writes a
   meaningful summary). But session-2 queries are **free-form natural
   language** — NOT derived from any base sample.

2. Session 2 queries are deliberately DIVERSE in style:
   - Precise recall ("上次查到的 tide 多少来着")
   - Semantic paraphrase ("sea level observations")
   - Applied decision ("should we halt crane ops given last week's wind?")
   - Vague / open-ended ("最近港口有啥异常没")
   - Comparative ("今年跟去年比呢")
   - Multi-entity ("B3 crane 5 vs B4 crane 7")
   - Chinese / mixed language (realistic for Chinese port managers)

3. No two session-2 queries share the same sentence template. This
   prevents BGE from clustering all recall queries into one region of
   the vector space (the root cause of v3's 15% hit_rate).

4. Coverage: 5 answer_modes × 4 data_sources, each represented at
   least twice among the 25 positive conversations.

5. 10 negative conversations use truly unrelated topics (not just
   different data sources on the same port domain).

Run:
    python evaluation/build_cross_session_v4.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# -------------------------------------------------------------------
# Conversation templates
# -------------------------------------------------------------------
# Session 1: `from_sample_id` → inherits golden fields → DAG runs real
#            retrieval → end_session writes factual summary to LT.
# Session 2: free-form query, no from_sample_id. Evaluated ONLY on
#            whether LT retrieve returns session-1's session_id.

def _s1_turn(sample_id: str) -> Dict[str, Any]:
    return {"from_sample_id": sample_id,
            "evaluation_focus": "session1_establish"}


def _s2_free(query: str, key_fact: str,
             answer_mode: str = "lookup",
             data_source_tag: str = "sql",
             from_order: int = 1) -> Dict[str, Any]:
    """A free-form session-2 turn (no from_sample_id)."""
    return {
        "raw_query": query,
        "evaluation_focus": "cross_session_free_recall",
        "expected_cross_session_hit": True,
        "expected_from_session_order": from_order,
        "answer_mode": answer_mode,
        "data_source_tag": data_source_tag,
        "expected_memory_recall": {
            "from_session": from_order,
            "key_fact": key_fact,
        },
    }


def _neg_s2(query: str) -> Dict[str, Any]:
    return {
        "raw_query": query,
        "evaluation_focus": "cross_session_negative",
        "expected_cross_session_hit": False,
        "expected_from_session_order": None,
        "answer_mode": "lookup",
    }


CONVERSATIONS: List[Dict[str, Any]] = [

    # =================================================================
    #  POSITIVE — SQL source (7 conversations)
    # =================================================================

    # 1. Precise recall, lookup
    {
        "conversation_id": "V4_SQL_01",
        "pattern": "precise_recall",
        "answer_mode": "lookup",
        "data_source": "sql",
        "description": "S1: 2016 avg tide. S2: precise recall in different phrasing.",
        "sessions": [
            {"session_order": 1, "turns": [_s1_turn("V3_SQL_001")]},
            {"session_order": 2, "turns": [
                _s2_free("上次帮我查的潮位平均值是多少来着？",
                         "tide", "lookup", "sql"),
            ]},
        ],
    },

    # 2. Applied decision, decision_support
    {
        "conversation_id": "V4_SQL_02",
        "pattern": "applied_decision",
        "answer_mode": "decision_support",
        "data_source": "sql",
        "description": "S1: wind speed 2015. S2: decision based on that data.",
        "sessions": [
            {"session_order": 1, "turns": [_s1_turn("V3_SQL_005")]},
            {"session_order": 2, "turns": [
                _s2_free("根据之前查到的风速数据，现在这种天气条件下要不要暂停作业？",
                         "wind", "decision_support", "sql"),
            ]},
        ],
    },

    # 3. Vague open-ended, diagnostic
    {
        "conversation_id": "V4_SQL_03",
        "pattern": "vague_openended",
        "answer_mode": "diagnostic",
        "data_source": "sql",
        "description": "S1: crane breakdown hours. S2: vague 'any equipment issues?'",
        "sessions": [
            {"session_order": 1, "turns": [_s1_turn("V3_SQL_015")]},
            {"session_order": 2, "turns": [
                _s2_free("最近有没有设备方面的异常情况？之前好像查过相关数据",
                         "crane", "diagnostic", "sql"),
            ]},
        ],
    },

    # 4. Comparative, comparison
    {
        "conversation_id": "V4_SQL_04",
        "pattern": "comparative",
        "answer_mode": "comparison",
        "data_source": "sql",
        "description": "S1: gate transactions. S2: compare with previous period.",
        "sessions": [
            {"session_order": 1, "turns": [_s1_turn("V3_SQL_010")]},
            {"session_order": 2, "turns": [
                _s2_free("Gate throughput compared to what we discussed before — is it trending up or down?",
                         "gate", "comparison", "sql"),
            ]},
        ],
    },

    # 5. Semantic paraphrase, lookup
    {
        "conversation_id": "V4_SQL_05",
        "pattern": "semantic_paraphrase",
        "answer_mode": "lookup",
        "data_source": "sql",
        "description": "S1: berth delays. S2: totally different words.",
        "sessions": [
            {"session_order": 1, "turns": [_s1_turn("V3_SQL_006")]},
            {"session_order": 2, "turns": [
                _s2_free("Wharf congestion incidents — do we have numbers on ships waiting for docking?",
                         "delay", "lookup", "sql"),
            ]},
        ],
    },

    # 6. Multi-entity, comparison
    {
        "conversation_id": "V4_SQL_06",
        "pattern": "multi_entity",
        "answer_mode": "comparison",
        "data_source": "sql",
        "description": "S1: yard occupancy. S2: multi-entity comparison.",
        "sessions": [
            {"session_order": 1, "turns": [_s1_turn("V3_SQL_009")]},
            {"session_order": 2, "turns": [
                _s2_free("Container yard utilisation vs the vessel arrival volume — any correlation we saw before?",
                         "yard", "comparison", "sql"),
            ]},
        ],
    },

    # 7. Mixed language, descriptive
    {
        "conversation_id": "V4_SQL_07",
        "pattern": "mixed_language",
        "answer_mode": "descriptive",
        "data_source": "sql",
        "description": "S1: vessel calls. S2: Chinese + English mixed.",
        "sessions": [
            {"session_order": 1, "turns": [_s1_turn("V3_SQL_007")]},
            {"session_order": 2, "turns": [
                _s2_free("之前查的 vessel call 数据里，arrival delay 最长的是哪条船？",
                         "vessel", "descriptive", "sql"),
            ]},
        ],
    },

    # =================================================================
    #  POSITIVE — Vector source (6 conversations)
    # =================================================================

    # 8. Document recall, descriptive
    {
        "conversation_id": "V4_VEC_01",
        "pattern": "document_recall",
        "answer_mode": "descriptive",
        "data_source": "vector",
        "description": "S1: HOT lane feasibility. S2: recall specific finding.",
        "sessions": [
            {"session_order": 1, "turns": [_s1_turn("V3_VEC_001")]},
            {"session_order": 2, "turns": [
                _s2_free("那个隧道收费可行性研究的结论是什么？交通分流的估算方法是什么来着？",
                         "HOT lane", "descriptive", "vector"),
            ]},
        ],
    },

    # 9. Policy recall, decision_support
    {
        "conversation_id": "V4_VEC_02",
        "pattern": "policy_recall",
        "answer_mode": "decision_support",
        "data_source": "vector",
        "description": "S1: noise award. S2: decision based on noise policy.",
        "sessions": [
            {"session_order": 1, "turns": [_s1_turn("V3_VEC_004")]},
            {"session_order": 2, "turns": [
                _s2_free("Based on the airline noise program we looked at, which carriers should we prioritise for the next award cycle?",
                         "noise", "decision_support", "vector"),
            ]},
        ],
    },

    # 10. Casual Chinese, lookup
    {
        "conversation_id": "V4_VEC_03",
        "pattern": "casual_chinese",
        "answer_mode": "lookup",
        "data_source": "vector",
        "description": "S1: annual report. S2: casual Chinese recall.",
        "sessions": [
            {"session_order": 1, "turns": [_s1_turn("V3_VEC_008")]},
            {"session_order": 2, "turns": [
                _s2_free("年报那块儿之前看到啥了？主要结论帮我回忆一下",
                         "annual report", "lookup", "vector"),
            ]},
        ],
    },

    # 11. Contradiction check, diagnostic
    {
        "conversation_id": "V4_VEC_04",
        "pattern": "contradiction_check",
        "answer_mode": "diagnostic",
        "data_source": "vector",
        "description": "S1: sustainability report. S2: challenge previous finding.",
        "sessions": [
            {"session_order": 1, "turns": [_s1_turn("V3_VEC_020")]},
            {"session_order": 2, "turns": [
                _s2_free("I recall we found some sustainability data before. Doesn't that contradict the emissions targets in the newer report?",
                         "sustainability", "diagnostic", "vector"),
            ]},
        ],
    },

    # 12. Instruction-style, lookup
    {
        "conversation_id": "V4_VEC_05",
        "pattern": "instruction_style",
        "answer_mode": "lookup",
        "data_source": "vector",
        "description": "S1: terminal doc. S2: instruction-style request.",
        "sessions": [
            {"session_order": 1, "turns": [_s1_turn("V3_VEC_010")]},
            {"session_order": 2, "turns": [
                _s2_free("Pull up what we found in that terminal development document — specifically the JFK redevelopment sponsors.",
                         "terminal", "lookup", "vector"),
            ]},
        ],
    },

    # 13. Reflective, descriptive
    {
        "conversation_id": "V4_VEC_06",
        "pattern": "reflective",
        "answer_mode": "descriptive",
        "data_source": "vector",
        "description": "S1: competition plan. S2: reflective summary request.",
        "sessions": [
            {"session_order": 1, "turns": [_s1_turn("V3_VEC_015")]},
            {"session_order": 2, "turns": [
                _s2_free("Thinking back to the competition plan document, what were the main strategic recommendations we extracted?",
                         "competition plan", "descriptive", "vector"),
            ]},
        ],
    },

    # =================================================================
    #  POSITIVE — Rules source (6 conversations)
    # =================================================================

    # 14. Threshold recall, lookup
    {
        "conversation_id": "V4_RUL_01",
        "pattern": "threshold_recall",
        "answer_mode": "lookup",
        "data_source": "rules",
        "description": "S1: wind restriction. S2: recall the exact number.",
        "sessions": [
            {"session_order": 1, "turns": [_s1_turn("V3_RUL_002")]},
            {"session_order": 2, "turns": [
                _s2_free("那个风速限制的具体阈值是多少？超过之后具体要执行什么操作？",
                         "wind threshold", "lookup", "rules"),
            ]},
        ],
    },

    # 15. Scenario application, decision_support
    {
        "conversation_id": "V4_RUL_02",
        "pattern": "scenario_application",
        "answer_mode": "decision_support",
        "data_source": "rules",
        "description": "S1: vessel entry rules. S2: apply to current scenario.",
        "sessions": [
            {"session_order": 1, "turns": [_s1_turn("V3_RUL_004")]},
            {"session_order": 2, "turns": [
                _s2_free("A 280m LOA container ship is requesting entry now. Based on the rules we reviewed, can it proceed?",
                         "vessel entry", "decision_support", "rules"),
            ]},
        ],
    },

    # 16. What-if, diagnostic
    {
        "conversation_id": "V4_RUL_03",
        "pattern": "whatif",
        "answer_mode": "diagnostic",
        "data_source": "rules",
        "description": "S1: crane safety. S2: what-if scenario.",
        "sessions": [
            {"session_order": 1, "turns": [_s1_turn("V3_RUL_006")]},
            {"session_order": 2, "turns": [
                _s2_free("If crane wind sensors report 22 m/s right now, does that hit the safety cutoff we discussed?",
                         "crane safety", "diagnostic", "rules"),
            ]},
        ],
    },

    # 17. Colloquial, lookup
    {
        "conversation_id": "V4_RUL_04",
        "pattern": "colloquial",
        "answer_mode": "lookup",
        "data_source": "rules",
        "description": "S1: berth allocation. S2: very colloquial.",
        "sessions": [
            {"session_order": 1, "turns": [_s1_turn("V3_RUL_007")]},
            {"session_order": 2, "turns": [
                _s2_free("泊位分配那个规则怎么说的来着？大船优先还是先到先得？",
                         "berth allocation", "lookup", "rules"),
            ]},
        ],
    },

    # 18. Cross-referencing, comparison
    {
        "conversation_id": "V4_RUL_05",
        "pattern": "cross_reference",
        "answer_mode": "comparison",
        "data_source": "rules",
        "description": "S1: temperature threshold. S2: compare with another rule.",
        "sessions": [
            {"session_order": 1, "turns": [_s1_turn("V3_RUL_003")]},
            {"session_order": 2, "turns": [
                _s2_free("How does the temperature threshold from our earlier review compare to the wind speed limit? Which kicks in first operationally?",
                         "temperature threshold", "comparison", "rules"),
            ]},
        ],
    },

    # 19. Urgency-phrased, decision_support
    {
        "conversation_id": "V4_RUL_06",
        "pattern": "urgent_phrasing",
        "answer_mode": "decision_support",
        "data_source": "rules",
        "description": "S1: storm protocol. S2: urgent operational question.",
        "sessions": [
            {"session_order": 1, "turns": [_s1_turn("V3_RUL_005")]},
            {"session_order": 2, "turns": [
                _s2_free("台风预警刚发布，之前查到的暴风操作规程能不能直接启用？需要额外审批吗？",
                         "storm protocol", "decision_support", "rules"),
            ]},
        ],
    },

    # =================================================================
    #  POSITIVE — Graph source (4 conversations)
    # =================================================================

    # 20. Causal recall, diagnostic
    {
        "conversation_id": "V4_GRA_01",
        "pattern": "causal_recall",
        "answer_mode": "diagnostic",
        "data_source": "graph",
        "description": "S1: graph causal for weather→ops. S2: recall causal chain.",
        "sessions": [
            {"session_order": 1, "turns": [_s1_turn("V3_GRA_001")]},
            {"session_order": 2, "turns": [
                _s2_free("上次用知识图谱分析的那条因果链是什么？从天气到作业中断的影响路径帮我理一下",
                         "causal chain", "diagnostic", "graph"),
            ]},
        ],
    },

    # 21. Relationship query, descriptive
    {
        "conversation_id": "V4_GRA_02",
        "pattern": "relationship_query",
        "answer_mode": "descriptive",
        "data_source": "graph",
        "description": "S1: graph entity relationships. S2: ask about connections.",
        "sessions": [
            {"session_order": 1, "turns": [_s1_turn("V3_GRA_005")]},
            {"session_order": 2, "turns": [
                _s2_free("Which operational areas are connected to the berth metrics we mapped out in the graph before?",
                         "graph relationships", "descriptive", "graph"),
            ]},
        ],
    },

    # 22. Impact assessment, decision_support
    {
        "conversation_id": "V4_GRA_03",
        "pattern": "impact_assessment",
        "answer_mode": "decision_support",
        "data_source": "graph",
        "description": "S1: graph for tidal impact. S2: operational decision.",
        "sessions": [
            {"session_order": 1, "turns": [_s1_turn("V3_GRA_002")]},
            {"session_order": 2, "turns": [
                _s2_free("Given the tidal impact paths we found in the knowledge graph, should deep-draft vessels be rescheduled today?",
                         "tidal impact", "decision_support", "graph"),
            ]},
        ],
    },

    # 23. Multi-hop, diagnostic
    {
        "conversation_id": "V4_GRA_04",
        "pattern": "multi_hop_graph",
        "answer_mode": "diagnostic",
        "data_source": "graph",
        "description": "S1: graph multi-hop. S2: trace the path again.",
        "sessions": [
            {"session_order": 1, "turns": [_s1_turn("V3_GRA_003")]},
            {"session_order": 2, "turns": [
                _s2_free("帮我回忆一下之前在图谱里找到的多跳推理路径，是从哪个节点到哪个节点的？",
                         "multi-hop path", "diagnostic", "graph"),
            ]},
        ],
    },

    # =================================================================
    #  POSITIVE — Multi-source (2 conversations, 3 sessions each)
    # =================================================================

    # 24. Cross-source synthesis
    {
        "conversation_id": "V4_MULTI_01",
        "pattern": "cross_source_synthesis",
        "answer_mode": "decision_support",
        "data_source": "multi",
        "description": "S1: SQL data. S2: rules. S3: synthesise both.",
        "sessions": [
            {"session_order": 1, "turns": [_s1_turn("V3_SQL_001")]},
            {"session_order": 2, "turns": [_s1_turn("V3_RUL_002")]},
            {"session_order": 3, "turns": [
                _s2_free("综合之前查到的潮位数据和风速限制规则，今天的作业条件是否满足安全标准？",
                         "tide and wind rules", "decision_support", "multi", 1),
            ]},
        ],
    },

    # 25. Sequential investigation
    {
        "conversation_id": "V4_MULTI_02",
        "pattern": "sequential_investigation",
        "answer_mode": "diagnostic",
        "data_source": "multi",
        "description": "S1: crane SQL. S2: graph causal. S3: tie them together.",
        "sessions": [
            {"session_order": 1, "turns": [_s1_turn("V3_SQL_011")]},
            {"session_order": 2, "turns": [_s1_turn("V3_GRA_001")]},
            {"session_order": 3, "turns": [
                _s2_free("Put together the crane productivity numbers and the causal analysis from the knowledge graph — what's the root cause of the slowdown?",
                         "crane and causal", "diagnostic", "multi", 1),
            ]},
        ],
    },

    # =================================================================
    #  NEGATIVE — Topic drift (10 conversations)
    # =================================================================
    # Session 1 and 2 are on COMPLETELY DIFFERENT topics.
    # Session 2 queries should NOT hit session 1.

    {"conversation_id": "V4_NEG_01", "pattern": "topic_drift",
     "description": "S1: tide data. S2: airport noise program.",
     "sessions": [
         {"session_order": 1, "turns": [_s1_turn("V3_SQL_001")]},
         {"session_order": 2, "turns": [_neg_s2("What airlines received noise reduction awards at Seattle-Tacoma airport?")]},
     ]},

    {"conversation_id": "V4_NEG_02", "pattern": "topic_drift",
     "description": "S1: crane breakdown. S2: HOT lane feasibility.",
     "sessions": [
         {"session_order": 1, "turns": [_s1_turn("V3_SQL_015")]},
         {"session_order": 2, "turns": [_neg_s2("那个收费快速车道的可行性报告结论是啥？")]},
     ]},

    {"conversation_id": "V4_NEG_03", "pattern": "topic_drift",
     "description": "S1: wind speed. S2: JFK terminal sponsors.",
     "sessions": [
         {"session_order": 1, "turns": [_s1_turn("V3_SQL_005")]},
         {"session_order": 2, "turns": [_neg_s2("Who are the main sponsors backing the JFK Terminal One redevelopment project?")]},
     ]},

    {"conversation_id": "V4_NEG_04", "pattern": "topic_drift",
     "description": "S1: berth delays. S2: GHG inventory methodology.",
     "sessions": [
         {"session_order": 1, "turns": [_s1_turn("V3_SQL_006")]},
         {"session_order": 2, "turns": [_neg_s2("PANYNJ 的温室气体盘查用的什么方法论？Scope 1 2 3 分别怎么算的？")]},
     ]},

    {"conversation_id": "V4_NEG_05", "pattern": "topic_drift",
     "description": "S1: yard occupancy. S2: LaGuardia traffic report.",
     "sessions": [
         {"session_order": 1, "turns": [_s1_turn("V3_SQL_009")]},
         {"session_order": 2, "turns": [_neg_s2("How many total passengers went through LaGuardia in the 2010 traffic report?")]},
     ]},

    {"conversation_id": "V4_NEG_06", "pattern": "topic_drift",
     "description": "S1: gate transactions. S2: vessel scheduling rules.",
     "sessions": [
         {"session_order": 1, "turns": [_s1_turn("V3_SQL_010")]},
         {"session_order": 2, "turns": [_neg_s2("What's the minimum notice period for a vessel schedule change under the port authority rules?")]},
     ]},

    {"conversation_id": "V4_NEG_07", "pattern": "topic_drift",
     "description": "S1: wind rule. S2: annual financial report.",
     "sessions": [
         {"session_order": 1, "turns": [_s1_turn("V3_RUL_002")]},
         {"session_order": 2, "turns": [_neg_s2("年度财务报告里的 basis of accounting 是 accrual 还是 cash？")]},
     ]},

    {"conversation_id": "V4_NEG_08", "pattern": "topic_drift",
     "description": "S1: graph causal. S2: tariff liability provisions.",
     "sessions": [
         {"session_order": 1, "turns": [_s1_turn("V3_GRA_001")]},
         {"session_order": 2, "turns": [_neg_s2("Under the 2019 Port Authority tariff, what are the liability provisions for cargo damage?")]},
     ]},

    {"conversation_id": "V4_NEG_09", "pattern": "topic_drift",
     "description": "S1: crane safety rules. S2: competition plan EWR.",
     "sessions": [
         {"session_order": 1, "turns": [_s1_turn("V3_RUL_006")]},
         {"session_order": 2, "turns": [_neg_s2("EWR competition plan 里面 Requesting Airline 的 accommodation 流程是怎样的？")]},
     ]},

    {"conversation_id": "V4_NEG_10", "pattern": "topic_drift",
     "description": "S1: berth allocation. S2: environmental monitoring methodology.",
     "sessions": [
         {"session_order": 1, "turns": [_s1_turn("V3_RUL_007")]},
         {"session_order": 2, "turns": [_neg_s2("What continuous air quality monitoring stations does the Port Authority operate, and what pollutants do they measure?")]},
     ]},
]


# -------------------------------------------------------------------
# Builder
# -------------------------------------------------------------------

def _index_base(base_path: Path) -> Dict[str, Dict[str, Any]]:
    with open(base_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return {s["id"]: s for s in d["samples"]}


def _materialise(conv: Dict[str, Any], base: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    sessions_out = []
    for sess in conv["sessions"]:
        turns_out = []
        for i, t in enumerate(sess["turns"]):
            turn_out = dict(t)
            turn_out["turn_id"] = i + 1
            turn_out["session_order"] = sess["session_order"]
            # Inherit golden fields from base sample if from_sample_id present
            sid = t.get("from_sample_id")
            if sid:
                if sid not in base:
                    raise KeyError(f"unknown base id '{sid}'")
                bs = base[sid]
                turn_out["raw_query"] = t.get("raw_query") or bs["query"]
                for k in ("expected_sources", "needs_vector", "needs_sql",
                          "needs_rules", "needs_graph", "answer_mode",
                          "expected_evidence_keywords", "reference_answer",
                          "difficulty"):
                    if k in bs:
                        turn_out.setdefault(k, bs[k])
            turns_out.append(turn_out)
        sessions_out.append({
            "session_order": sess["session_order"],
            "turns": turns_out,
        })
    return {
        "conversation_id": conv["conversation_id"],
        "pattern": conv["pattern"],
        "description": conv.get("description", ""),
        "sessions": sessions_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path,
                        default=Path("evaluation/golden_dataset_v3_rag.json"))
    parser.add_argument("--out", type=Path,
                        default=Path("evaluation/golden_dataset_v4_cross_session.json"))
    args = parser.parse_args()

    base = _index_base(args.base)
    out_convs = []
    for conv in CONVERSATIONS:
        out_convs.append(_materialise(conv, base))

    pos = sum(1 for c in CONVERSATIONS
              if any(t.get("expected_cross_session_hit") for s in c["sessions"] for t in s["turns"]))
    neg = sum(1 for c in CONVERSATIONS
              if any(t.get("expected_cross_session_hit") is False for s in c["sessions"] for t in s["turns"]))

    output = {
        "description": (
            "Cross-session memory evaluation v4. Session-2 queries are "
            "free-form natural language (Chinese / English / mixed, "
            "varying formality and specificity). NOT derived from "
            "golden_dataset_v3_rag — only session-1 establishment turns "
            "use from_sample_id."
        ),
        "version": "v4.0",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_conversations": len(out_convs),
        "positive_conversations": pos,
        "negative_conversations": neg,
        "total_sessions": sum(len(c["sessions"]) for c in out_convs),
        "total_turns": sum(
            sum(len(s["turns"]) for s in c["sessions"]) for c in out_convs
        ),
        "conversations": out_convs,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.out}")
    print(f"  conversations:  {len(out_convs)}  ({pos} positive + {neg} negative)")
    print(f"  sessions:       {output['total_sessions']}")
    print(f"  turns:          {output['total_turns']}")


if __name__ == "__main__":
    main()
