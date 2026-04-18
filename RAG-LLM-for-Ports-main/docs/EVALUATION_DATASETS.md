# 评测数据集 — 构建方法

> 覆盖三层评测：单轮、多轮（单 session）、跨 session。
> 205 样本 golden set 的完整构建方法论（863 行）见 `evaluation/GOLDEN_DATASET_V3_METHODOLOGY_CN.md`。

---

## 总览

| 层级 | 文件 | 规模 | 测什么 | 怎么建 |
|---|---|---|---|---|
| **单轮** | `golden_dataset_v3_rag.json` | 205 样本 | 路由 / 检索 / 答案 / 护栏 | Claude Opus 从采样 chunk 反向生成 |
| **多轮** | `golden_dataset_v3_multi_turn.json` | 10 段对话 / 31 轮 | 短期记忆（指代消解、实体追踪、主题切换） | 从 205 条基础样本派生的人工模板 |
| **跨 session** | `golden_dataset_v4_cross_session.json` | 35 段对话 / 70 轮 | 长期记忆（跨会话召回、判别力） | Claude 子代理从真实检索数据生成 session-2 查询 |

三层逐级递进——每一层测的是上一层测不了的能力：
- 单轮：流水线能不能正确回答一个问题？
- 多轮：能不能在一段对话中保持上下文？
- 跨 session：能不能在几天后的新会话中回忆起旧会话的知识？

---

## 第 1 层：单轮 Golden 数据集（205 样本）

### 核心设计原则：反向生成（防止数据泄漏）

常规方法（有泄漏风险）：
```
query → 检索器找 chunk → LLM 判断相关性 → ground truth
  ↑                                          ↓
  └────────── 用同一个检索器做评测 ──────────┘  ⚠️ 循环依赖
```

我们的方法（无偏）：
```
v2 chunk（分层采样）→ Claude Opus 生成一个该 chunk 能回答的问题
                      → 该 chunk 本身就是 ground truth（不涉及任何检索器）
```

Ground truth **独立于任何检索器的输出**，因此可以公平评估任何系统变体（v1 RAG / agent v1 / agent v2 / 最终 DAG）。

此外，**出题模型（Claude Opus）与答题模型（Qwen 3.5 Flash）不同**——消除了模型自偏差。

### 覆盖维度（6 个正交轴）

| 维度 | 取值 | 保证方式 |
|---|---|---|
| **数据源组合** | 全部 2^4 = 16 种（含 0 源护栏） | 每种组合设定目标数量；稀有组合故意过采样 |
| **回答模式** | 5 种：lookup / descriptive / comparison / decision_support / diagnostic | 每种 ≥20 样本；Opus prompt 指定模式 |
| **护栏类型** | 9 种：OOD、空证据、不可能查询、证据冲突（3 种）、模糊查询、错误前提、应拒绝 | 共 25 个护栏样本，每种 ≥2 |
| **文档类型** | handbook / policy / sustainability report / annual report / master plan | 从 v2 chunks 按 `doc_type` 分层采样 |
| **时间窗口** | 旧 (<2015) / 中 (2015–2019) / 近 (≥2020) / 无日期 | 按 `publish_year` 分桶分层 |
| **Chunk 属性** | 短/中/长词数；表格 vs 正文 | 按 `is_table` 和 `word_count` 分层 |

### 样本量论证

使用二项分布置信区间公式 `CI = z × sqrt(p(1-p)/n)`，z=1.96（95% 置信）：

| 分层 | 目标 CI | 所需 n | 实际 n |
|---|---|---|---|
| 小（graph-only, rule-only） | ±25% | ≥15 | 15–20 |
| 中（vector, sql） | ±15% | ≥40 | 30–66 |
| 整体（Micro-F1） | ±8% | ≥150 | 205 |

205 是同时满足所有分层约束的最小值。

### 各数据源的采样策略

| 数据源 | 采样方式 | Ground truth |
|---|---|---|
| **Vector** | 从 16,124 个 v2 子 chunk 按 (doc_type × year_bucket × is_table) 分层 → 18 个桶 → 按比例分配 | 采样的 chunk 本身 |
| **SQL** | 15 个 (表, 列, 聚合) 模板在 DuckDB 上实际执行 → 真实查询结果作为 golden 数字 | 真实 SQL 输出（如 `AVG(wind_speed_ms)=5.14`） |
| **Rules** | 从 grounded_rules.json 按完整度排名取前 20 条（有变量 + 操作符 + 阈值 + sql_variable 的优先） | 规则文本、变量、阈值 |
| **Graph** | 从 Neo4j v2 图中随机抽取 15 条带 rule_text 标注的 TRIGGERS 边 | 边路径 + 规则引用 |
| **多源组合** | 11 种组合类型 × N 个，随机选锚 chunk，Opus 生成需要所有列出源才能回答的问题 | 各源 golden 字段的并集 |

### 生成流水线

```
1. 分层采样 → scaffold 文件（按数据源类型）
2. Claude Opus 子代理（并行，每种源一个）：
   - 输入：采样的 chunk/规则/边/SQL 模板 + 目标 answer_mode + 目标难度
   - 输出：query, expected_evidence_keywords, reference_answer, 各源 golden 字段
3. 合并 + 去重 + 验证（所有 expected_sources 已覆盖，无简单问题）
4. 输出：golden_dataset_v3_rag.json（205 样本）
```

**脚本**：`evaluation/build_golden_v3_rag.py`

---

## 第 2 层：多轮单 Session 数据集（10 段对话 / 31 轮）

### 目的

测试单轮评测无法衡量的**短期记忆**能力：
- 指代消解（"那个规则是什么？"）
- 跨轮实体追踪（T1 提到 berth B3，T3 用"它"指代）
- 主题切换检测（T4 转到无关话题——记忆不能带上旧上下文）
- 长程摘要（6+ 轮触发自动压缩）
- 护栏持续性（对话中段插入 OOD，下一轮恢复）

### 构建方法：从 205 条基础样本派生

每一轮都携带 `derived_from_sample_id` 链接到特定基础样本，这意味着：
- 该轮继承所有 golden 字段（`expected_sources`、`needs_*`、`answer_mode`、`expected_evidence_keywords`、`reference_answer`、`golden_vector/sql/rules/graph`）
- **不需要创建新的 ground truth** —— 评测直接复用单轮打分模块
- 数据源/模式/护栏覆盖继承自 205 样本的设计

### 对话设计：6 种 pattern

每种 pattern 针对一个特定的记忆能力：

| Pattern | 数量 | 轮数 | 测什么 |
|---|---|---|---|
| `entity_anchored` | 2 | 3 | 同一实体跨轮；代词消解 |
| `mode_progression` | 2 | 3 | lookup → comparison → decision_support 模式升级 |
| `cross_source_verification` | 2 | 2–3 | SQL 事实 → 规则验证 → 文档解释 |
| `topic_switch` | 2 | 2–3 | **反向测试**：记忆不能携带旧上下文 |
| `long_summarisation` | 1 | 6 | 触发 `_summarise_oldest_half` + key_facts 提取 |
| `guardrail_in_conversation` | 1 | 3 | 对话中段 OOD → 下一轮必须恢复 |

### 跟进轮的标注

T1 之后的每一轮都携带额外的记忆评测字段：
- `rephrase_as`：跟进改述（如 "那个潮位相关的规则是什么？"）
- `expected_resolved_query_contains`：指代消解器应注入的关键词（如 `["tide", "rule"]`）
- `expected_resolved_query_should_not_contain`：主题切换后不应出现的关键词
- `expected_memory_recall`：`{from_turn: 1, key_fact: "tide"}` —— 记忆应该记住什么
- `evaluation_focus`：标识测试的具体能力

### 对话内跨数据源

单段对话故意在不同轮中混合不同数据源：

```
示例（MT3_001 entity_anchored）：
  T1: SQL / lookup     — "2016 年的平均潮位是多少？"
  T2: rules / decision — "那个潮位相关的政策规则是什么？"
  T3: graph / diagnostic — "如果超过那个阈值，哪些作业会受影响？"
```

这反映了真实港口管理者的调查方式：先查数据，再查政策，最后追踪影响。

### 脚本与生成

模板在 `evaluation/build_multi_turn_v3.py` 中以 `CONVERSATIONS` 列表形式手工编写。每个模板指定 `from_sample_id` 引用 + 可选的 `rephrase_as` 覆盖。运行脚本将模板物化为 `golden_dataset_v3_multi_turn.json`。

**为什么手写而不用 LLM 生成**：多轮数据集测试的是*结构化*记忆能力（指代消解、主题切换、摘要触发），需要精确控制轮次顺序和 evaluation_focus 标签。LLM 生成会增加噪声而不增加价值。

---

## 第 3 层：跨 Session 数据集（35 段对话 / 70 轮）

### 目的

测试**长期记忆** —— 在新会话（不同 `session_id`）中回忆旧会话的事实。这是第 1 层和第 2 层都无法评测的能力。

### 构建方法：Claude 子代理从真实数据生成

与第 2 层（手写模板）不同，第 3 层的 session-2 查询由 **Claude 子代理读取真实样本数据后生成**。原因：跨会话召回需要**多样化的自然语言查询**，不能共享句式模板——否则 BGE embedding 聚簇，实验无法区分关键词检索和向量检索。

**流程：**
1. Claude 子代理读取 `golden_dataset_v3_rag.json`（205 样本）
2. 选取 25 个覆盖全部 4 种数据源 × 5 种回答模式的多样样本
3. 对每个选中样本，读取 `query` + `reference_answer` 以了解 session 1 会建立什么知识
4. 生成 **自由形式自然语言** 的 session-2 查询——风格在整个数据集中变化：
   - 精确回忆："上次查到的潮位读数是多少？"
   - 休闲口语："之前看的设备问题有哪些？"
   - 应用决策："根据上周的风速数据，要不要停止吊机作业？"
   - 对比追问："和之前发现的相比怎么样？"
   - 指令式："把我们看过的闸口吞吐量数字调出来"
5. 另外生成 10 段**反向**对话（session 1 关于话题 A，session 2 关于完全无关的话题 B）
6. 每个 session-2 查询都包含 `expected_cross_session_hit: true/false` 和 `expected_memory_recall.key_fact`

**约束**：没有两个 session-2 查询共享相同的句式结构。这防止 embedding 聚簇。

### Session 结构

```
对每段对话：
  Session 1（session_order=1）：
    - from_sample_id = "V3_SQL_001" ← 借用基础样本的查询
    - DAG 真实运行检索 → 真实回答
    - end_session() 把摘要 + 原子 key_facts 写入长期记忆 DuckDB

  Session 2（session_order=2，全新 session_id）：
    - raw_query = "上次查到的海平面数据是多少？" ← Claude 生成
    - build_context() → long_term.retrieve() → 应该找到 session 1 的内容
    - 评分：检索到的 session_id 里是否包含 session 1 的 sid？
```

Session 1 使用 `from_sample_id` 仅用于继承查询文本和 golden 字段。**实际记忆内容在运行时创建** —— DAG 执行、产生回答、`end_session` 写入 DuckDB 长期存储。

### 覆盖范围

| 维度 | 取值 | 数量 |
|---|---|---|
| 数据源 | sql / vector / rules / graph / multi | 7 / 6 / 5 / 4 / 3 |
| 回答模式 | lookup / comparison / decision_support / diagnostic / descriptive | 5 / 5 / 6 / 5 / 4 |
| 查询风格 | 精确 / 休闲 / 应用 / 对比 / 指令 / 同义改写 | 分布在 25 段中 |
| 反向测试 | topic_drift（完全无关话题） | 10 |
| **合计** | | **35 段 / 70 轮** |

### 评测指标（跨 session 专用）

| 指标 | 定义 |
|---|---|
| `cross_session_hit_rate` | 正向 turn 中 `long_term.retrieve()` top-5 包含正确前序 session_id 的比例 |
| `correct_session_recall_rate` | 在命中的 turn 中，命中的是否为 gold 指定的特定 session |
| `cross_session_leak_rate` | 反向 turn 中错误返回了前序 session 内容的比例 |
| `score_gap (pos − neg)` | 正向 turn 的平均 top-score 减去反向 turn 的平均 top-score（判别力） |

### 迭代历史：v3 → v4

| 版本 | 对话数 | hit_rate (Phase B) | 问题 |
|---|---|---|---|
| v3（5 段） | 5 | 75% | 小样本；过于乐观 |
| v3（31 段） | 31 | **15%** | 模板化查询 → embedding 聚簇；叙事性摘要难以区分 |
| **v4（35 段）** | 35 | **76%** | Claude 生成多样化查询 + 结论导向摘要 + 原子 key_facts |

15% → 76% 的提升同时修复了**评测数据和系统本身**：
- **数据端**：多样化查询风格防止 embedding 聚簇
- **系统端**：原子 key_facts 作为独立长期条目 + 结论导向的摘要 prompt

### 脚本

- **生成器**：`evaluation/build_cross_session_v4.py`（Claude 子代理输出）
- **评测驱动**：`evaluation/run_cross_session_evaluation.py`
- **输出**：`evaluation/golden_dataset_v4_cross_session.json`

---

## 延迟评测与埋点

### 流水线级计时（内置于 DAG）

每个 DAG 节点都由 `NodeFactory._timed()` 包装（`langgraph_nodes.py:61–76`）：

```python
def _timed(self, node_name, func, state):
    t0 = time.time()
    result = func(state)
    elapsed = round(time.time() - t0, 4)
    result["_node_timings"][node_name] = elapsed   # 注入 LangGraph state
    return result
```

全部 9 个节点已埋点：`route_query`、`planner`（含子阶段 `planner__query_rewrite`、`planner__plan_total`、`planner__sub_queries__llm_call`）、`retrieve_documents`、`rerank_documents`、`retrieve_rules`、`run_sql`、`run_graph_reasoner`、`merge_evidence`、`synthesize_answer`。

计时字典通过 LangGraph state 流转，由评测驱动在 `per_sample_results[i]["stage_timings"]` 中提取。

### 延迟评测模块（`eval_latency.py`）

计算各阶段的 **p50 / p95 / p99 / mean / max**：

```
阶段                     mean    p50     p95     p99     max   (秒)
-----------------------------------------------------------------------
route_query            21.46   21.49   28.08   29.62   30.01
planner                27.89   28.70   42.81   44.57   45.01
retrieve_documents      0.55    0.38    1.27    1.54    1.61
rerank_documents        0.48    0.51    0.59    0.61    0.61
run_sql                 6.88    6.88    6.88    6.88    6.88
merge_evidence          0.00    0.00    0.00    0.00    0.00
synthesize_answer      40.96   40.05   53.43   55.61   56.15
end_to_end             91.72   92.11  121.29  126.74  128.10
```

（数据来自当前 DAG 的 10 样本 smoke。完整 205 样本 v3 报告：e2e p50 = 72s。）

还跟踪：
- **迭代分布**：`{1: N}`（DAG 始终 1 次迭代，无重规划）
- **重规划触发率**：DAG 为 0%（vs 已退役 ReAct agent 的 66%）
- **ReAct 观察统计**：DAG 为 0（验证无意外的 agent 行为）

### Memory 专有延迟

在 `run_multi_turn_evaluation.py` 和 `run_cross_session_evaluation.py` 中每轮测两个墙钟时间：

| 操作 | 测什么 | 代码位置 | 典型值 |
|---|---|---|---|
| `resolve_followup_ms` | LLM 调用将跟进查询改写为独立形式 | `memory_manager.resolve_followup()` | ~1.7–3.3s（LLM 受限） |
| `build_context_ms` | 短期格式化 + 长期检索（BGE embed_query + DuckDB vss） | `memory_manager.build_context()` | ~4ms（无 embedder）/ ~140ms（有 BGE） |

由 `eval_memory.py` 聚合为 `avg_resolve_followup_ms` 和 `avg_build_context_ms`。

### 延迟对比：v1 Agent → v2 Agent → v3 DAG

来源：`AGENT_FINAL_COMPARISON.md` 和 `rag_v3_n205.json`：

| 阶段 | v1 Agent | v2 Agent | **v3 DAG（当前）** |
|---|---|---|---|
| 端到端 p50 | 117.8s | 61.8s | **72s** |
| 端到端 p95 | 253.1s | 91.9s | **121s** |
| 重规划率 | 66% | 10% | **0%** |

v3 DAG 比 v2 Agent 的 p50 略慢（72 vs 62s），因为 DAG 始终并行运行所有路由源（无提前终止），但 **p95 更可预测**（无重规划拖尾）。

### Memory 的延迟开销

在 v4 跨 session 数据集（35 段对话 / 70 轮）上测量：

| 配置 | resolve_followup | build_context | 每轮总开销 |
|---|---|---|---|
| Phase A（无 embedder） | 1.9s | 11ms | **~2s** |
| Phase B（BGE + vss） | 1.8s | 143ms | **~2s** |

`resolve_followup` 的 LLM 调用占主导（~95% 开销）。`build_context` 即使有 BGE embedding 也可忽略。**Memory 总开销每轮增加 ~2s**，叠加在 72s 的 DAG 延迟上 —— 2.8% 增幅。

### 延迟数据位置

| 文件 | 内容 |
|---|---|
| `evaluation/agent/eval_latency.py` | 百分位计算 + 阶段分解 |
| `evaluation/agent/eval_memory.py` | Memory 专有延迟聚合 |
| `evaluation/agent/reports/rag_v3_n205.json` → `single_turn.latency` | 完整 205 样本各阶段统计 |
| `evaluation/agent/reports/rag_v3_n10_post_cleanup_smoke.json` | Memory 接入后 10 样本回归验证 |
| `evaluation/agent/reports/rag_v4_cross_session_phase_*.json` | 跨 session 每轮 Memory 延迟 |
| `src/online_pipeline/langgraph_nodes.py:55–76` | 埋点源代码 |

---

## 端到端答案质量评测

### 指标体系（7 个客观 + 3 个 LLM judge）

| 指标 | 类型 | 公式/定义 | 需要 golden |
|---|---|---|---|
| **keyword_coverage** | 客观 | `|{kw ∈ expected_kw : kw ∈ answer}| / |expected_kw|` | ✅ `expected_evidence_keywords` |
| **citation_validity** | 客观 | answer 里的 `[sql]/[doc]/[rule]/[graph]` 标签是否对应 evidence_bundle 中真实有数据的源 | ❌ 不需要 |
| **numerical_accuracy** | 客观 | answer 里的数字与 reference_answer 里数字的匹配率 | ✅ `reference_answer` |
| **grounding_flag** | 客观 | `fully_grounded` / `llm_fallback` / `partial` | ❌ 不需要 |
| **embedding_similarity** | 客观 | BGE cosine(answer, reference_answer) | ✅ `reference_answer` |
| **rouge_l_f1** | 客观 | ROUGE-L F1(answer, reference_answer) | ✅ `reference_answer` |
| **faithfulness** (LLM judge) | 主观 | 答案是否基于证据（1=幻觉，5=完全有据） | ✅ `reference_answer` |
| **relevance** (LLM judge) | 主观 | 答案是否切题（1=跑题，5=完全切题） | ✅ `reference_answer` |
| **completeness** (LLM judge) | 主观 | 答案是否覆盖参考答案要点（1=全缺，5=全覆盖） | ✅ `reference_answer` |

### 多轮/跨 session 的差异

| Turn 类型 | 有 golden？ | 可用指标 |
|---|---|---|
| **Session 1 turn**（通过 `from_sample_id` 继承） | ✅ 有 reference + keywords | 全部 |
| **Session 2 自由 turn**（Claude 生成，无 `from_sample_id`） | ❌ 无 reference | 仅 citation_validity + grounding_flag |

### 端到端结果

**多轮（10 段对话 / 31 轮）：**

| 指标 | 值 |
|---|---|
| Keyword coverage | 25.00% |
| **Citation validity** | **100%** |
| Numerical accuracy | 39.58% |
| Embedding similarity | 69.39% |
| ROUGE-L F1 | 9.91% |
| Grounding | fully_grounded=25, llm_fallback=6 |

Keyword coverage 低（25%）因为跟进轮的答案自然偏离原始样本的 golden 关键词——经过指代消解改写后这是预期的。

**跨 session（35 段对话 / 70 轮）：**

| 指标 | 值 |
|---|---|
| Keyword coverage | 49.43% |
| **Citation validity** | **100%** |
| Numerical accuracy | 43.34% |
| Embedding similarity | 76.22% |
| ROUGE-L F1 | 11.67% |
| Grounding | fully_grounded=51, llm_fallback=18 |

**两个场景下 Citation validity 都是 100%** —— synthesizer 的证据锚定机制在多轮和跨 session 上下文中没有退化。这是最强的端到端信号。

---

## 总结：三层评测的设计哲学

| 原则 | 第 1 层（单轮） | 第 2 层（多轮） | 第 3 层（跨 session） |
|---|---|---|---|
| **Ground truth 来源** | 从采样 chunk 反向生成 | 通过 `derived_from_sample_id` 继承自第 1 层 | Session 1 跑真实 DAG；session 2 检查召回 |
| **问题生成者** | Claude Opus（不同于答题模型） | 人工模板 | Claude 子代理（多样化自然语言） |
| **为什么用这种方法** | 避免数据泄漏；无偏 | 结构化记忆测试需要精确控制轮次顺序 | 跨会话召回需要多样化词汇避免 embedding 聚簇 |
| **覆盖保证** | 2^4 源组合 × 5 模式 × 9 护栏 | 6 种记忆压力 pattern | 4 源 × 5 模式 × 6 查询风格 + 10 反向 |
| **避免的反模式** | 循环依赖：检索器生成自己的 gold | 单调模板 → embedding 聚簇 | 同第 2 层（v3 失败，v4 修复） |
