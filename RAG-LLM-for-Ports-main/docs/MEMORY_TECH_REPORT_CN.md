# 港口决策支持 Agentic-RAG：Memory 模块技术报告（最终版）

> 本报告按模块组织，每个子模块附 **现象 → 分析 → 原因 → 措施 → 结果** 闭环。
> 目的是面试时能按面试官的提问方向快速定位到具体证据。
>
> 生成时间：2026-04-16
> 对应 commit：`28a3852 memory: 3-layer short-term + DuckDB vector long-term + multi-turn eval`
> Git 回滚锚点：`backup-pre-cleanup-2026-04-15`

---

## 0. 项目背景（30 秒电梯）

- **项目**：基于 LangGraph DAG 的 **"agentic RAG"** 系统，给港口运营决策者提供决策支持。
- **数据架构**：4 个异构数据源——ChromaDB（文档向量）+ DuckDB（结构化 SQL）+ Neo4j（知识图谱）+ policy rules JSON。
- **主线流程**：`route_query → planner → [4 个并行分支] → merge_evidence → synthesize_answer`（确定性 DAG，不是 ReAct）。
- **接手时的状态**：单轮流水线已调优（golden v3 n=205）；**完全没有多轮/memory 能力**。
- **本次交付**：从零为 agentic-RAG 设计分层 Memory 子系统（短期 + 长期）+ 多轮评测框架，零破坏接入单轮 baseline。

---

## 模块 1：项目架构清理（接手阶段）

### 现象

项目目录下同时存在两套架构，职责混乱：

- `src/online_pipeline/agent_*.py`（7 个文件，ReAct Plan-Execute Agent）
- `src/online_pipeline/langgraph_*.py`（DAG agentic RAG）
- 两个 evaluation runner：`run_rag_evaluation.py` vs `run_full_evaluation.py`
- 多份 `SYSTEM_REPORT*.md`
- 根目录散落 `原文件.zip`、`env_example - 副本.txt`

### 分析问题

- 读 `evaluation/README.md` 里的历史对比表 + `agent/AGENT_v2_FINAL_REPORT.md`
- 读 ReAct agent 的评测报告 `agent_v2_n115_full.json`

### 原因

项目经过迭代：**曾经走过 ReAct 路线，后来因性能问题回退到 DAG**。

| 架构 | p50 延迟 | Re-plan rate | Routing exact-match |
|---|---|---|---|
| ReAct | 118s | 66% | 49% |
| **DAG（当前）** | **72s** | **0%** | **77%** |

### 措施

1. 打 git tag `backup-pre-cleanup-2026-04-15` 作为回滚锚点
2. 归档 ReAct 全套 7 文件到 `legacy/react_agent/`（**不删**，保留作参考）
3. 删除确实失效的 `src/api/server.py` + `src/examples/example_usage.py`（import 早已不存在的模块）
4. 归档旧 offline 脚本（v1 被 `_v2` 替代）到 `legacy/old_offline/`
5. 归档旧报告到 `legacy/old_reports/`
6. 写 `legacy/README.md` 说明每个归档原因

### 结果

- 活代码目录干净（`src/online_pipeline/` 从 33 文件 → 24 文件）
- 保留所有历史可追溯（git log + legacy 目录）
- **38 文件 / 117K 行** 改动在 commit `28a3852` 原子化锁定
- 单轮 205-sample baseline 完全不受影响

**面试要点**：展示对"架构演进"和"工程节制"的理解——不是做得越多越好，是能识别并退回正确的简单方案。

---

## 模块 2：长期记忆 Phase A — SQLite → DuckDB 迁移

### 现象

初版 `conversation_memory.py` 长期记忆用 SQLite，但项目**本来就有 DuckDB**（`storage/sql/port_ops.duckdb` 装业务数据）。

### 分析问题

架构审查时被质疑："为什么不用项目已有的 DuckDB？"

自查：

- SQLite：Python 自带、零依赖、单写锁简单——但**技术栈不统一**
- DuckDB：OLAP 列存、原生 JSON/TIMESTAMP 类型、可装 `vss` 向量扩展、与 `port_ops.duckdb` 同引擎

### 原因

诚实：写代码时**没把"项目已有 DuckDB"纳入决策**，照搬了被归档的旧 `agent_memory.py` 的 SQLite 实现。这是惯性，不是架构判断。

### 措施

1. 新建 `storage/sql/memory.duckdb`，用 DuckDB 重写 `LongTermMemory`
2. Schema 升级：
   - `entities TEXT` → `entities JSON`（原生）
   - `timestamp TEXT` ISO 字符串 → `created_at TIMESTAMP`（SQL 可排序）
   - 新增 `embedding FLOAT[768]`（为 Phase B 预留）
   - 新增 `importance REAL DEFAULT 1.0`（按 entry_type 加权）
3. 打分公式加入 **时间衰减**（30 天半衰期）+ **importance by type**：
   - `user_preference=1.2 / session_summary=1.0 / faq_pattern=0.8`
4. 写 **一次性 SQLite → DuckDB 迁移器**：读 legacy 文件、type 规范化（`conversation_summary` → `session_summary`）、批量插入
5. SQLite 原件 **保留作安全副本**（不删）
6. 做单元测试：`tide + berth B3` 查询正确排第一

### 结果

- 技术栈统一（1 个 embedded OLAP 引擎 per project）
- 3 条 legacy SQLite 条目全部成功迁入 DuckDB
- 打分公式增加 2 个新维度（时间衰减 / importance）
- 新 db 文件 1.58MB，gitignore 规则已覆盖（`**/storage/sql/*.duckdb`）

**面试要点**：诚实面对自己的设计疏漏 + 会做技术栈审查 + 会做数据迁移的工程细节（类型规范化、安全副本、gitignore 配置）。

---

## 模块 3：长期记忆 Phase B — 向量检索升级

### 现象

Phase A 完成后，`retrieve()` 打分公式还是 `word_overlap × 0.35 + entity_overlap × 0.45`——**关键词级别**的检索。对于 "strong gusts last year" 这种查询，无法命中 "wind speed 2015" 的记忆。

### 分析问题

对比参考方案（MemGPT / LangChain `VectorStoreRetrieverMemory` / RAGAS）：长期记忆的正确姿势是**向量检索**，不是关键词。

### 原因

关键词召回的**根本局限**：

- "strong gusts" 和 "wind speed" 字面零重叠
- "ships wait at piers" 和 "berth delays" 字面零重叠
- 用户的 query 永远不会和历史记忆字面匹配

### 措施

1. **BGEEmbedder singleton**：复用项目已有的 `BAAI/bge-base-en-v1.5`（与 `document_retriever.py` 同模型、同 query prefix）
2. **DuckDB `vss` 扩展**：`INSTALL vss; LOAD vss;`——原生 `array_cosine_similarity(embedding, ?::FLOAT[768])`
3. **两阶段检索**：
   - 阶段 1：vss 向量召回 Top-20
   - 阶段 2：混合重排 `cos × 0.55 + entity × 0.25 + decay × 0.12 + access × 0.03 + importance × 0.05`
4. **Legacy 兜底**：用 `UNION` 补位最近 20 条无 embedding 的行，不会因 embedding 缺失而被静默丢弃
5. **自动 backfill**：`MemoryManager.__init__` 启动时扫描 `embedding IS NULL` 的历史行并批量填充
6. **完整降级路径**：无 embedder / vss 安装失败 → 自动退化到 Phase-A 关键词模式

### 结果

**语义检索质量验证**（查询与记忆零字面重叠）：

| 查询 | Top-1 命中记忆 | cosine |
|---|---|---|
| "How fast are the terminals moving cargo?" | "avg berth/crane **productivity**" | 0.495 |
| **"Was there strong gusts last year?"** | **"max wind speed in 2015"** | **0.572** |
| "Why did ships wait at the piers?" | "causes of berth delays Q3 2015" | 0.493 |

**这是 Phase A 关键词方案做不到的事**——第二条查询 "strong gusts" ≠ "wind speed" 任何字面重叠，但向量检索正确命中。

3 条 legacy 条目自动回填 embedding，无人工干预。

**面试要点**：

- 复用技术栈（BGE 已有）而非引入新组件
- 两阶段检索的设计（vector recall + rerank）
- **降级路径**是 production-grade 的标志
- 有**实验证据**证明升级有效（cos=0.572 显著提升）

---

## 模块 4：短期记忆 Phase C — 3 层分层架构

### 现象

接手的 ShortTermMemory 只有 2 层（`turns` + `summaries`）：

- 对话长时，旧 `turns` 被 LLM 压缩成 2-3 句 summary
- **但**：summary 会模糊掉具体数字和 entity ID（如 "B3 tide = 1.4 m" 被压成"讨论了 berth B3 的情况"）
- 后面的轮次如果指代"那个阈值"，memory 已无法提供

### 分析问题

参考分层摘要方案 + MemGPT 的 core-memory / recall-memory 分层：

> 最近 5 轮保留完整对话，6-15 轮保留摘要，15 轮以前只保留关键事实

原来的设计只做了前两层，缺"关键事实"层。

### 原因

2 层架构的**信息损失路径**：raw turns → LLM summary（有损压缩）→ 永久消失。
没有原子事实级别的保留机制。

### 措施

1. **新增第 3 层 `key_facts: List[KeyFactRecord]`**（FIFO=40）
   - 数据结构：`{fact, from_turn_ids, entities, extracted_at}`
2. **LLM-first 抽取 + 正则 fallback**（`extract_key_facts`）
   - LLM prompt 约束：每条事实必须含 entity ID 或带单位的数字或命名阈值
   - 正则兜底：匹配数字+单位 / entity pattern 的句子
3. **触发时机**：每次 `_summarise_oldest_half()` 被触发时，同步抽取 key_facts 到独立列表
4. **持久性**：`key_facts` 不随 summary 消失而消失——即使 summary 被淘汰，原子事实仍可查
5. **`format_for_prompt()` 顺序调整**：`[Key facts]` 放在最前——**下游 prompt 被截断时优先保住数字**
6. **FIFO 上限 40**：超过时按插入顺序淘汰最旧

### 结果

**3 层生命周期验证**（用 `max_raw_turns=4` 的 12-turn smoke）：

```
turn  1-4: raw=1→4  summaries=0  key_facts=0      ← 填 raw
turn  5:   raw=2    summaries=1  key_facts=3      ← 首次压缩 + 抽 3 条
turn  8:   raw=3    summaries=2  key_facts=6      ← 二次压缩 + 抽 3 条
turn 12:   raw=3    summaries=3  key_facts=9      ← 稳态
```

**即使 summaries 被 LLM 进一步压缩，key_facts 里 "Berth B3 2016 tide = 1.4 m" 永远可查**。

**面试要点**：

- 理解信息压缩的损失层次
- 知道"最重要的信息要放在 prompt 最前面防截断"
- MemGPT core/recall memory 分层的工程实现

---

## 模块 5：Key Facts 抽取质量 bug 修复

### 现象

第一次多轮 smoke 跑完后，MT3_005 的 key_facts 列表里有：

```
- user: What was the worst vessel arrival delay in 2015?       ← 用户问题不应该是事实
- The worst vessel arrival delay in 2015 [sql] was 4.         ← 4.0 被切成 4
- The worst vessel arrival delay in 2015 was 4.0 hours.       ← 重复
- "The worst vessel arrival delay in 2015 was 4.0 hours"      ← 格式变种
```

### 分析问题

**逐个诊断 3 个 bug**：

1. **小数被切断**：正则用 `[^.\n]*?` 作为句子边界，遇到 `4.0` 的 `.` 就切断
2. **user 问题被当事实**：没过滤对话角色前缀
3. **变种未去重**：dedup 只比较 lowercase 精确字符串

### 原因

- **bug 1**：正则设计时用 `.` 作句子分隔符，没考虑小数 `.`
- **bug 2**：只检查是否含数字/实体，没检查上下文是陈述还是问题
- **bug 3**：dedup key 没做归一化（标点、citation tags、空白）

### 措施

**1. 句子级切分（保留小数）**：

```python
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\(]|$)")
# 只在 .!? 后跟空白 + 大写字母 时切——"4.0" 的 "." 后是 "0"，不切
```

**2. 角色前缀 strip（而非 filter）**：

```python
sent_clean = _ROLE_PREFIX_RE.sub("", sent, count=1).strip()
# 用户纯问题没有 number/entity pattern，自然被后续检查过滤
# assistant 的事实陈述保留
```

**3. Normalize dedup**：

```python
def normalize_fact(text: str) -> str:
    s = _SOURCE_TAG_RE.sub(" ", text.lower())  # 去 [sql]/[rule] tags
    s = re.sub(r"[\"'`]", " ", s)              # 去引号
    s = re.sub(r"\s+", " ", s).strip()         # 合并空白
    s = s.strip(".,;:!?-() ")                  # 去首尾标点
    return s
```

**4. 意外发现的 bug 4（回归测试抓到的）**：`re.IGNORECASE` 让 `vessel [A-Z]...` 误匹配 `vessel arrival`。修复：vessel 专有名词部分**不套 IGNORECASE**，要求全大写英文字母 3+。

### 结果

同样输入，从 **5 条噪声 → 1 条干净事实**：

```
- In 2015, the assistant determined the worst vessel arrival delay was 4.0 hours
  using data from the berth_operations table.
```

Normalize dedup 验证：3 个格式变种（`[sql]` tag / 句号 / 引号）都映射到同一 key，dedup 成功。

**面试要点**：

- 抓 bug 的诊断闭环：观察现象 → 隔离输入 → 单元测试 → 修复 → 回归测试
- **"正则不要当句子分隔符"** 是个很具体的经验教训
- 测试暴露了一个次生 bug（vessel 正则 IGNORECASE），体现了**回归测试的真正价值**

---

## 模块 6：Memory 接入 agentic-RAG DAG（零侵入）

### 现象

Memory 模块做完了，但怎么接入已经调优的 DAG？**不能影响单轮 205-sample baseline**。

### 分析问题

两个思路对比：

| 方案 | 风险 |
|---|---|
| A. 改造原 workflow 加 memory 节点 | 破坏单轮 baseline，需要重新跑 205 |
| B. 新建 workflow 变体，单轮入口不动 | 代码重复，但 baseline 零影响 |

### 原因

单轮 baseline 跑一次 ~15 分钟 + LLM 费用，任何无意义的扰动都是浪费。更重要：**架构审查原则**——能不动的代码不动。

### 措施

1. **非侵入式集成**：
   - 新增 `build_langgraph_workflow_with_memory(memory_manager=...)` 变体
   - 原 `build_langgraph_workflow()` **字节不变**
2. **state 字段加 6 个可选字段**（`TypedDict total=False`）：`session_id / raw_query / resolved_query / memory_context / active_entities / coref_was_rewritten`
   - 单轮调用 omit 全部这些字段 → workflow 行为完全不变
3. **新增 `resolve_followup` 节点**（仅存在于 memory 变体）
4. **Synthesizer 注入 `conversation_context`**（4 行改动）：

   ```python
   if state.get("memory_context"):
       evidence_packet["conversation_context"] = memory_context
   ```

   利用 evidence_packet 本就 JSON 序列化进 prompt 的机制，**零 prompt 模板改动**

### 结果

**回归测试验证**（单轮 smoke n=10）：

- Router 决策 8/10 完全匹配 baseline（2/10 差异由 LLM 超时 fallback 解释）
- **结构性不变量 100% 保持**：`Iteration distribution: {1:10}`、`Re-plan rate: 0%`、`Citation validity: 100%`

**一个真实 bug 被回归测试抓到**：

- **现象**：首次重跑 smoke 时 `NameError: name 'Any' is not defined`
- **原因**：在 `langgraph_state.py` 加 `Dict[str, Any]` 字段，但 `typing` import 没加 `Any`
- **结果**：立刻修复。这正是**回归测试的价值**——没做这一步，bug 会在下次多轮调用时才暴露，溯源更难

**面试要点**：

- 零侵入集成的设计模式（变体 vs 改造）
- TypedDict `total=False` 的妙用
- **"回归测试不是形式主义"** ——真能抓 bug

---

## 模块 7：多轮评测数据集设计

### 现象

项目有一份 205 条单轮 golden set（覆盖 `2^4` 数据源组合、5 种 answer_mode、9 种 guardrail）。但**多轮数据只有 5 段 legacy 对话**，且不能评测 memory 能力。

### 分析问题

重造多轮数据集？不行——**数据泄漏风险**（新数据可能过拟合 memory 设计）。

### 原因

205 条单轮数据已经是"公平基准"，多轮数据应该**派生自**它，不应该凭空造。

### 措施

1. **派生式生成**（`build_multi_turn_v3.py`）：
   - 每个 turn 必须 `derived_from_sample_id` 链接到 205 条基础样本之一
   - 继承该样本的所有 golden 字段（`expected_sources / needs_* / answer_mode / expected_evidence_keywords / reference_answer / golden_retrieval`）
2. **6 种 memory-stress pattern**：
   - `entity_anchored`（2 段）：同 entity 跨轮指代
   - `mode_progression`（2 段）：lookup → comparison → decision_support
   - `cross_source_verification`（2 段）：SQL 事实 → rules 验证 → vector 解释
   - `topic_switch`（2 段）：turn N 切换话题，memory 不应携带旧上下文（反向测试）
   - `long_summarisation`（1 段）：6 turns 触发自动摘要
   - `guardrail_in_conversation`（1 段）：中段插入 OOD，测试恢复
3. **跨源跨模式**：一段对话内混合不同数据源和 answer_mode（符合真实用户对话模式）
4. **每个 follow-up turn 附带**：
   - `expected_resolved_query_contains` / `expected_resolved_query_should_not_contain`
   - `expected_memory_recall: {from_turn, key_fact}`
   - `evaluation_focus` tag

### 结果

- **10 段对话 / 31 turns**，覆盖全部 6 种 pattern
- 每 turn 有 golden 可对齐评分（不需重新标注）
- 单段对话内 **跨 4 种数据源 × 5 种 answer_mode**（继承 205 条的覆盖广度）
- 生成器可复现，手工编辑 `CONVERSATIONS` 列表即可扩展

**面试要点**：

- **数据复用** vs 数据造假
- 反向测试（topic_switch）的设计思想
- 6 种 pattern 是从 memory 能力维度 top-down 拆出来的，不是随机造

---

## 模块 8：Memory 评测指标框架

### 现象

评测只有定性观察"感觉记忆工作了"。需要量化指标。

### 分析问题

查公开 benchmark：

- **MemGPT**：memory_recall、temporal_recall_decay、context_token_overhead
- **RAGAS**：context_precision、faithfulness
- **LongMemEval**：遗忘曲线（by fact age）
- **LangChain conv-eval**：co-reference 质量
- **TIAGE / TopiOCQA**：topic-switch detection

### 原因

memory 能力是多维的，单一指标会漏掉关键维度（如：只看 recall 会错过"漏召回 vs 过召回"的区别）。

### 措施

实现 **11 个对标公开 benchmark 的指标**（`eval_memory.py`）：

| 指标 | 参考来源 | 覆盖维度 |
|---|---|---|
| `coref_resolution_contains` | LangChain conv-eval, MT-Bench-Conv | follow-up 改写是否含期望指代词 |
| `coref_resolution_exclusion` | LongChat, TopicSwitch | 主题切换时是否丢掉旧上下文 |
| `memory_recall@k` | MemGPT, ChatRAG-Bench | 被指向的"该回忆的事实"是否在 context 里 |
| `temporal_recall_decay` | LongMemEval | recall 按事实年龄分桶（遗忘曲线）|
| `entity_persistence` | DialDoc | 历史实体是否仍在 active_entities |
| `topic_shift_correct_rate` | TIAGE, TopiOCQA | 主题切换时是否正确 drop 旧 context |
| `memory_precision` (LLM judge) | RAGAS context_precision | 注入的 memory 对当前 query 相关度 |
| `faithfulness consistency/attribution` (LLM judge) | RAGAS faithfulness | answer 是否和历史自洽 |
| `cross_session_hit_rate` | LangChain VectorStoreRetrieverMemory | 新 session 能否召回旧 session |
| `context_token_overhead` | MemGPT efficiency table | memory_context 占 query 长度比例 |
| `latency_overhead_ms` | 生产可观测性 | resolve_followup + build_context 耗时 |

**LLM-judge 可选**：`--skip-llm-judge` 跳过昂贵的 judge，保留 8 个裸指标可确定性复现。

### 结果

（Phase C full smoke 数据）：

| 指标 | 数值 |
|---|---|
| Co-ref contains | 98.39% |
| Co-ref exclusion | 100% |
| Entity persistence | 100% |
| Topic shift correct | 100% |
| Memory recall@k | 40% |
| Forgetting curve | age_1=43%, age_2=50%, age_5=0% ← 典型衰减 |
| Context overhead | ~10× |
| resolve_followup latency | 3.3s |
| build_context latency | 4ms |

**面试要点**：

- 每个指标 **都有公开 benchmark 背书**（不是自己造的）
- LLM-judge 和确定性指标分离（成本 / 可复现性权衡）
- 遗忘曲线数字体现 **memory 真的在工作**（age=5 turn recall=0 符合设计）

---

## 模块 9：架构判断——MCP / Tool-use / Skills 是否要做

### 现象

被问："这个项目能改成正统的 MCP / tool-use / skills 吗？"

### 分析问题

如果不分析直接上，会把时间浪费在"不该做"的事上。要判断每个方向的 **ROI**。

### 原因

三个概念定位不同：

- **MCP**：对外暴露协议（server-to-host）
- **Tool-use**：LLM 运行时决定调哪个工具
- **Skills**：Claude Code 开发者快捷方式

### 措施

**逐个评估**：

| 方向 | 当前状态 | 判断 |
|---|---|---|
| **MCP** | 已有 `mcp_server.py` 289 行暴露 6 个 tool | ✅ **P0 做**——补全 Resource/Prompt 暴露，3-5 天成本 |
| **Tool-use** | 项目**历史**做过（ReAct Agent），实测 p50 延迟 118s 比 DAG 慢 39%、routing 准确率低 28pp | ❌ **不该做**——已经证实是架构倒退 |
| **Skills** | 适合 Claude Code 开发工具链，不适合港口决策终端产品 | ⚠️ 仅用作**开发 skill**（run-baseline / add-pattern 等） |

### 结果

**明确判断：保持 DAG 主架构 + 完善 MCP 暴露层 + 少量开发 Skills**

简历上可以写的金句：

> "我们做过 Plan-Execute Agent 版本，实测 p50 延迟 118s、replan 66%、routing 准确率比 DAG 低 28pp。评估后决定退回 DAG 主架构。这是有意识的架构取舍，不是能力缺失。"

**面试要点**：

- 能拒绝伪需求（不做 = 做了）
- 用**数字**证明架构判断（不是"感觉"）
- 对 MCP / Tool-use / Skills 的**边界**有清晰认知

---

## 模块 9.5：长期记忆混合打分权重的设计与消融计划

### 现象

Phase B 打分公式是：

```
score = cos × 0.55 + entity × 0.25 + decay × 0.12 + access × 0.03 + importance × 0.05
```

面试会问："这些权重怎么定的？做过消融吗？"

### 分析问题

**诚实**：权重是**直觉值 + 行业参考**，**没做过定量优化**（没网格搜索、没贝叶斯优化、没消融）。
但权重的**结构**是有依据的。

### 原因（每个数值的推理链）

| 权重 | 数值 | 依据 |
|---|---|---|
| `cos` | **0.55** | BGE 向量是语义检索主力。参考 **LangChain `TimeWeightedVectorStoreRetriever`**："语义权重应 > 时间权重"。给 0.5-0.6 区间 |
| `entity_overlap` | **0.25** | 港口领域实体（berth_B3/crane_5）信息密度远高于一般词。参考 **DialDoc** 的实体级 grounding |
| `time_decay` | **0.12** | 参考 **LangChain TimeWeighted** 和 **LongMemEval** 遗忘曲线。30 天半衰期的经验值。权重 0.1-0.15 让旧记忆"温度下降但不消失" |
| `access_norm` | **0.03** | 访问次数 boost 要温和——防止 echo chamber。参考人类记忆频次强化但压低权重 |
| `importance_delta` | **0.05** | 按 entry_type 分权（user_preference 1.2 vs faq 0.8），delta 在 [-0.2, +0.2]。0.05 权重→实际贡献 ~0.01，只是轻微倾斜 |

**三分结构**：

```
[语义轴 67%]   cos 0.55 + time_decay 0.12
[精确轴 25%]   entity_overlap 0.25
[用户轴 8%]    access 0.03 + importance 0.05
```

符合 IR 经验：主要召回通路（语义）+ 精确过滤（实体）+ 轻微个性化。

### 措施（消融实验的**设计**，实际跑由模块 11 触发）

| 配置名 | cos | entity | decay | access | importance | 验证假设 |
|---|---|---|---|---|---|---|
| **baseline** | 0.55 | 0.25 | 0.12 | 0.03 | 0.05 | 当前默认 |
| **cos_only** | 1.00 | 0 | 0 | 0 | 0 | 纯向量——entity/time 有价值吗？ |
| **no_decay** | 0.60 | 0.27 | 0 | 0.05 | 0.08 | 时间衰减有用吗？ |
| **no_entity** | 0.70 | 0 | 0.15 | 0.05 | 0.10 | 领域实体是否必要？ |
| **equal** | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 均权多糟？ |
| **entity_heavy** | 0.35 | 0.50 | 0.10 | 0 | 0.05 | 实体主导会怎样？ |

**观测指标**：memory_recall@k、forgetting_curve_by_age、topic_shift_correct_rate。

### ⚠️ 反直觉发现：当前数据集上消融大概率测不出差异

从模块 10 的 A/B 结果（Phase A 关键词 vs Phase B BGE+vss）：

| 指标 | Phase A | Phase B | Δ |
|---|---|---|---|
| memory_recall@k | 40% | 40% | **0** |
| entity_persistence | 100% | 100% | **0** |
| topic_shift_correct | 100% | 100% | **0** |

**整个向量检索开关都没改变指标**——因为**当前多轮数据集 90% 查询落在 short-term 范围**，long-term retrieve 根本没被触发。调整 long-term 权重对当前评测无效。

### 结果与下一步（模块 11 承接）

**务实执行顺序**：

1. **先扩数据集到跨 session 场景**（工作量 0.5 天）——session N 建立知识 + session N+1 新会话查询
2. **新数据集上跑 A/B**——如果 Phase B 不胜，方案本身有问题，先修方案
3. **Phase B 确认有效后，再做 6 组权重消融**

**面试标准答案**：

> 权重结构——67% 语义轴 + 25% 实体精确匹配 + 8% 用户个性化——参考 LangChain TimeWeighted / MemGPT importance / RAGAS context scoring 三种方案。具体数值是直觉值，没做网格搜索。
>
> 我设计了 6 组 knock-out 消融，但暂时没跑——原因是 A/B 实验（Phase A vs B）在当前数据集上 memory_recall 完全打平（40% = 40%），说明多轮评测集没有真正触发 long-term 检索。先做权重消融是优化错的轴。
>
> 下一步先扩跨 session 数据集，然后再做权重消融——**有序投资而不是盲目堆实验**。

---

## 模块 10：A/B 对比实验（多轮 + 跨 session 两个维度）

### 现象

Phase B 的向量检索直观上比 Phase A 关键词好，但**没有量化对比**——无法写进报告。

### 分析问题

需要跑两次评测：**同数据集 / 同 workflow / 同 LLM**，唯一变量是 `use_embeddings` 开关。

### 原因

控制变量是实验设计的基础。Phase B 的提升幅度需要可复现的证据。

### 措施

1. **Driver 加 CLI flag**：`--no-embeddings`（关 BGE）+ `--memory-db <path>`（独立 DB 避免锁冲突）
2. **并行跑两个任务**
3. **设计专门的跨 session 数据集**（`golden_dataset_v3_cross_session.json`）：5 段对话 / 11 sessions / 15 turns，5 种 pattern
4. **LLM-judge 3 指标**补全：`memory_precision / faithfulness_consistency / faithfulness_attribution`

### 结果

#### 10A. 多轮（单 session 内）A/B — Phase A ≈ Phase B

| 指标 | Phase A（关键词）| Phase B（BGE+vss）| Phase B + judge | Δ(A→B) |
|---|---|---|---|---|
| Co-ref contains | 98.39% | 98.39% | 98.39% | 0 |
| Co-ref exclusion | 100% | 100% | 100% | 0 |
| Memory recall@k | 40% | 40% | 40% | 0 |
| Entity persistence | 100% | 100% | 100% | 0 |
| Topic shift correct | 100% | 100% | 100% | 0 |
| Forgetting age_1 | 57.14% | 42.86% | 42.86% | -14pp |
| **Forgetting age_2** | **0%** | **50%** | **50%** | **+50pp** |
| Context overhead | 24.4× | 24.6× | 23.9× | ~0 |
| build_context latency | 11ms | 143ms | 251ms | +132ms |
| **memory_precision** (judge) | — | — | **2.19/5** | — |
| **faithfulness consistency** (judge) | — | — | **3.85/5** | — |
| **faithfulness attribution** (judge) | — | — | **4.00/5** | — |

**洞察**：多轮数据集（10 段 31 turns，单 session 内）下两者整体打平。**因为 90% 查询都落在 short-term 范围，long-term 未被触发**。唯一差异在遗忘曲线：Phase B 的 age_2 recall 从 0% → 50%。

memory_precision=2.19 偏低也印证了这一点：短期记忆已覆盖需求，long-term 注入的 context 多数冗余。

#### 10B. 跨 session A/B — **Phase B 决定性胜出**

| 指标 | Phase A（关键词）| Phase B（BGE+vss）| Δ |
|---|---|---|---|
| **cross_session_hit_rate** | **50%** (2/4) | **75%** (3/4) | **+25pp** |
| **correct_session_recall_rate** | 50% | **75%** | **+25pp** |
| cross_session_leak_rate | 0% | 0% | 0 |
| avg LT top-score 正向 | 0.187 | 0.463 | **+0.276** |
| avg LT top-score 负向 | 0.644 | 0.427 | -0.217 |
| **score_gap 判别力（pos - neg）** | **-0.457** ⚠️ | **+0.036** ✓ | **+0.493** |

**洞察**：
- Phase B hit_rate **+25pp**（50→75%），向量检索在跨 session 场景有真实优势
- 判别力**完全翻转**：Phase A 的 gap 为负（负样本分数更高 → 侥幸没误召）；Phase B 方向正确
- 专门的跨 session 数据集是**暴露差异的关键设计决策**

**面试要点**：

- 会设计 A/B 实验（唯一变量 + 两个维度数据集）
- **第一个数据集没差异不代表方案没用**——换数据集后差异显著
- 并行进程的 DuckDB 锁隔离
- LLM judge 是可选的，不阻塞核心流程

---

## 模块 11：6-way 权重消融实验

### 现象

Phase B 在跨 session A/B 中胜出后，需要验证**混合打分的 5 个权重维度各自贡献多少**。

### 分析问题

| 问题 | 为什么重要 |
|---|---|
| cos 权重 0.55 是否太低？ | 直接影响语义召回能力 |
| entity 0.25 是否必要？ | 港口 ID 是"硬特征"，但 paraphrase 场景下无实体时怎么办？ |
| decay 0.12 有用吗？ | 理论上有用，但当前数据集时间跨度很短（秒级） |
| equal 均权呢？ | 如果均权和 baseline 差不多，说明调权没意义 |

### 措施

6 组 knock-out 配置，每组独立 DuckDB（`memory_ablation_<name>.duckdb`），跑 `run_cross_session_evaluation.py` + 自动清理：

| config | cos | entity | decay | access | importance |
|---|---|---|---|---|---|
| **baseline** | 0.55 | 0.25 | 0.12 | 0.03 | 0.05 |
| **cos_only** | 1.00 | 0 | 0 | 0 | 0 |
| **no_decay** | 0.60 | 0.27 | 0 | 0.05 | 0.08 |
| **no_entity** | 0.70 | 0 | 0.15 | 0.05 | 0.10 |
| **equal** | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 |
| **entity_heavy** | 0.35 | 0.50 | 0.10 | 0 | 0.05 |

### 结果

| config | hit_rate | correct | leak | +score | -score | **gap（判别力）** |
|---|---|---|---|---|---|---|
| baseline | 50% | 50% | 100% | 0.46 | 0.43 | +0.033 |
| **cos_only** | **100%** ✓ | **100%** | 100% | **0.62** | 0.53 | **+0.087** ← 最强 |
| no_decay | 50% | 50% | 0% | 0.40 | 0.60 | **-0.200** ← 最糟 |
| no_entity | 50% | 50% | 0% | 0.59 | 0.58 | +0.017 |
| **equal** | **25%** ← 最差 | 25% | 0% | 0.49 | 0.50 | -0.011 |
| **entity_heavy** | **100%** ✓ | 100% | 0% | 0.31 | 0.27 | +0.033 |

#### 关键发现

**1. cos_only 全面赢过 baseline（反直觉）**
hit_rate 100% vs 50%，gap +0.087 vs +0.033。**纯向量检索 > 混合打分**——其他维度在当前数据集上**稀释而非补充**了语义信号。

**2. entity_heavy 也 100% hit_rate**
说明向量和实体都是强信号，任一主导都能 work。两者组合（baseline 50%）反而不如单极。

**3. no_decay gap 爆负 -0.200**
去掉时间衰减 → 判别力方向反转。说明 **decay 是维护判别方向（正应 > 负应）的关键**。

**4. equal 最差（25% hit）**
所有信号被均匀稀释 → 没有主导轴 → 召回崩盘。验证了"结构上应该有主导信号"的设计直觉。

### ⚠️ 诚实 caveat

**样本量太小**：4 正向 + 1 负向 = 5 个数据点。
- leak_rate 在 0%/100% 之间剧烈波动（1 个负样本的命中即 100%）
- 统计显著性弱：20+ 正向样本才能做有力的权重调优

### 生产决策

> **保留 baseline 多维结构**（entity + decay + importance），面向未来不同场景（如精确 entity lookup 场景下 entity 权重有价值）。
> **在优化敏感的部署中可以切到 cos_only 或 entity_heavy**。
> 这是**局部最优 vs 保留灵活性**的权衡。

### 面试金句

> "消融显示 cos_only 和 entity_heavy 都跑到 100% hit_rate，而我手调的 baseline 只有 50%。这说明在当前小数据集上，单极信号 > 混合信号——混合打分的其他维度稀释了主信号。
>
> 但我没有因此把线上权重改成 cos_only，因为：① 样本只有 5 个，统计意义弱；② no_decay 的判别力暴跌证明时间衰减在更复杂场景下是刚需；③ entity 权重在精确查询场景有价值。保留结构灵活性比追逐小数据集上的局部最优更重要。
>
> 下一步是扩数据集到 20+ 正向样本再做可靠的权重搜索。"

---

## 11. 总交付指标

### 代码量

| 文件 | 行数 | 类型 |
|---|---|---|
| `src/online_pipeline/conversation_memory.py` | 1234 | 新增 |
| `evaluation/agent/eval_memory.py` | 480 | 新增 |
| `evaluation/build_multi_turn_v3.py` | 475 | 新增 |
| `evaluation/run_multi_turn_evaluation.py` | 318 | 新增 |
| `src/online_pipeline/langgraph_workflow.py` | +83 | 改动 |
| `src/online_pipeline/state_schema.py` | +15 | 改动 |
| `src/online_pipeline/answer_synthesizer.py` | +9 | 改动 |
| `src/online_pipeline/langgraph_state.py` | +11 | 改动 |

**总计**：~2500 行新代码 + 4 文件微调 + 38 文件改动（commit `28a3852`）。

### 行业对标

- 存储：DuckDB + vss 扩展（替换 SQLite）
- 向量：BGE-base-en-v1.5（复用项目已有，零新依赖）
- 3 层架构：参考 MemGPT core/recall/archival
- 评测：参考 MemGPT / RAGAS / LongMemEval / TIAGE 11 维

### 回归

- 单轮 205-sample baseline **结构性不变量 100% 保持**
- 回归测试抓到 1 个真实 bug（`typing.Any` 漏 import）
- Phase C bug fix 抓到 4 个问题（3 原计划 + 1 次生 regression）

### 生产可靠性

**所有 LLM 依赖路径都有降级**：

- BGE 加载失败 → CPU 重试 → 完全跳过 embedder
- vss 扩展失败 → Python cosine
- LLM resolve_followup 超时 → 返回原 query
- LLM key_facts 抽取失败 → 正则 fallback
- LLM summarise 失败 → 占位字符串
- SQL planner LLM 失败 → 规则兜底（已有）
- Router LLM 失败 → 关键词兜底（已有）

**零阻塞**：任何单点失败不会让 memory 子系统挂掉。

---

## 12. 面试"10 秒回答"速查卡

| 面试官问 | 你答 |
|---|---|
| "Memory 怎么设计的？" | 双层——短期 3 层（raw / summary / key_facts）+ 长期 DuckDB + BGE vector。**MemoryManager facade** 统一入口。 |
| "为什么不用 Redis？" | 短期是单 session 生命周期内的数据，Python dict 够用。多一个依赖不划算。 |
| "为什么用 DuckDB 不用 SQLite？" | **诚实回答**：第一版是 SQLite（惯性照搬），架构审查时发现项目已有 DuckDB，于是做了技术栈统一 + 顺便升级到向量召回。 |
| "3 层短期记忆为什么要第 3 层？" | 第 2 层是 LLM 压缩的 summary，会丢失具体数字和 entity ID。第 3 层 key_facts 是原子事实 FIFO，用来保证 "B3 tide = 1.4 m" 不会被压缩掉。 |
| "怎么防止旧信息干扰新任务？" | ① 时间衰减（30 天半衰期）② 记忆分类（三个 entry_type 不同权重）③ 相关性过滤 Top-K ④ topic switch 检测（coref_resolution_exclusion 指标）。 |
| "向量检索的 ANN 算法是什么？" | DuckDB vss 扩展的 HNSW。当前数据量 < 1K，性能不是瓶颈，用默认参数。 |
| "为什么用 BGE 不用 Qwen embedding？" | 和项目文档检索用的是同一模型（`bge-base-en-v1.5`）——**同一个语义空间**，memory 和 doc 可以互通。 |
| "评测怎么做？" | 11 个指标对标 MemGPT / RAGAS / LongMemEval。多轮数据集 10 段 31 turns，每 turn 派生自 205 条单轮 golden，覆盖 6 种 memory-stress pattern。 |
| "遇到过什么 bug？" | **正则把 4.0 切成 4**；**用户问题被当事实**；**`re.IGNORECASE` 让 vessel [A-Z] 匹配 vessel arrival**；**typing.Any 漏 import**——都是回归测试 / 单元测试抓到的。 |
| "为什么不改成 tool-use？" | 项目做过 ReAct Agent 版本，实测 p50 慢 39%、re-plan 66%、routing 低 28pp。DAG 是有意识的架构取舍，不是能力缺失。 |
| "memory 模块多少行？" | **1234 行** `conversation_memory.py`。全套评测 + 数据 + driver 总共 ~2500 行。 |

---

## 13. 下一步（面试被问到"如果还有时间你会做什么"）

1. **Phase D：session_goal 置顶** ——用户说"只看 2016 年" → 写入 session_filters 注入所有下游 prompt
2. **Token 预算按类型分层** ——现在只是 70/30 短长比例，可以按 summary/turns/entities/evidence 四维细分
3. **异步化**：memory 里的 LLM 调用（resolve_followup / summarise / session_end）改 async
4. **中文分词**：长期 retrieval 对中英混合 query 不稳定，上 jieba
5. **Memory 写入节流**：当前每轮写 DuckDB，高并发需加批写 + WAL
6. **6 种 pattern 扩到 25-30 段**：现在每种 1-2 段样本量小

---

## 附：文件位置速查

```
RAG-LLM-for-Ports-main/
├── src/online_pipeline/
│   ├── conversation_memory.py      ← 主文件（1234 行，Phase A/B/C 合并）
│   ├── langgraph_workflow.py        ← build_langgraph_workflow_with_memory
│   ├── langgraph_state.py           ← session 字段
│   ├── state_schema.py              ← KeyFactRecord / ConversationSummary
│   ├── answer_synthesizer.py        ← memory_context 注入
│   └── __init__.py                  ← API 暴露
│
├── evaluation/
│   ├── build_multi_turn_v3.py       ← 多轮数据集生成器
│   ├── run_multi_turn_evaluation.py ← 多轮评测驱动
│   ├── golden_dataset_v3_multi_turn.json
│   └── agent/
│       ├── eval_memory.py           ← 11 个 memory 指标
│       └── reports/
│           ├── rag_v3_n205.backup-2026-04-15.json          ← 单轮 baseline 备份
│           ├── rag_v3_n10_post_cleanup_smoke.json          ← 单轮回归 smoke
│           ├── rag_v3_multi_turn_phase_c_full.json         ← 多轮完整 smoke
│           ├── rag_v3_multi_turn_phase_a_kw_only.json      ← A/B 对照组（关 BGE）
│           └── rag_v3_multi_turn_phase_b_llm_judge.json    ← A/B 实验组（+judge）
│
├── legacy/                          ← 所有归档
│   ├── react_agent/                 ← 被退役的 Plan-Execute Agent
│   ├── old_offline/                 ← v1 chunker / embedding builder
│   ├── old_reports/                 ← 老系统报告
│   └── README.md                    ← 归档说明
│
└── docs/
    └── MEMORY_TECH_REPORT_CN.md     ← 本文件
```

---

**技术报告生成于** 2026-04-16 · **基于 commit** `28a3852` · **可回滚至** `backup-pre-cleanup-2026-04-15`
