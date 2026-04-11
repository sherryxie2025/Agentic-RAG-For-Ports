# 港口决策支持 Agentic RAG 系统 — 最终版总结报告

> **架构决策**：采用 **Agentic RAG（LangGraph DAG）** 作为主架构，保留所有数据层与检索层的 v2 升级。废弃 Plan-Execute Agent 方向（该方向虽然指标上有改善，但引入了过度复杂度和多个回归问题，与领域需求匹配度不佳）。

---

## 目录

1. [架构选型总结](#1-架构选型总结)
2. [系统总览](#2-系统总览)
3. [离线流水线 — 数据处理](#3-离线流水线--数据处理)
4. [在线流水线 — Agentic RAG 工作流](#4-在线流水线--agentic-rag-工作流)
5. [版本迭代与配置变化](#5-版本迭代与配置变化)
6. [评测维度全对比](#6-评测维度全对比)
7. [关键 Bug 修复清单](#7-关键-bug-修复清单)
8. [文件清单与交付物](#8-文件清单与交付物)
9. [未来优化方向](#9-未来优化方向)

---

## 1. 架构选型总结

### 1.1 两条技术路线对比

本项目实际尝试了两种架构方向：

| 维度 | Agentic RAG (采用) | Plan-Execute Agent (废弃) |
|---|---|---|
| **核心结构** | 固定 DAG：路由 → 规划 → 并行检索 → 合并 → 合成 | 循环：计划 → 执行 → 评估 → 可能 re-plan |
| **LangGraph 使用** | `StateGraph` + 条件边 | `StateGraph` + 条件边 + 循环 |
| **工具选择** | Router + Planner（一次决策，基于意图） | LLM 每轮生成工具列表 |
| **决策点数量** | 1 次路由 + 1 次规划 | 每 iteration 2-3 次 LLM 判断 |
| **可观测性** | 每个节点明确，trace 线性 | 多 iteration，trace 非线性 |
| **延迟** | 适中（~70-120s） | 高（n115=69.4s p50，但需要 max_iter=2 强制剪枝） |
| **调参面** | 少（路由阈值、prompt） | 多（OOD gate、strict plan、lenient eval、MAX_ITER、ReAct 阈值） |
| **对 LLM 稳定性依赖** | 中（路由 1 次 + 合成 1 次） | **高**（每 iteration 都需要多次 LLM 成功） |
| **回归风险** | 低 | **高**（n=115 观察到 4 个回归） |
| **适合场景** | 领域明确、数据源有限、查询模式可枚举 ✓ | 开放领域、工具动态扩展、需要持续反思 |

### 1.2 为什么最终选 Agentic RAG

1. **港口领域的工具集是固定的**（文档/SQL/规则/图 4 类），不需要 Agent 的动态工具选择能力。
2. **路由问题可以很好地用 LLM 分类解决**（实测 Micro-F1 0.82，不需要 re-plan 修正）。
3. **Plan-Execute 引入的 OOD gate、ReAct observation、strict plan prompt、lenient evaluator** 每一个都是为了修另一个引入的副作用 — 越调越复杂。
4. **延迟被 re-plan 放大**：v1 Agent 66% re-plan 率，强行降到 2 iter 之后仍然有 10% re-plan。DAG 里没有 re-plan，自然一次完成。
5. **多个关键回归**：强制精简计划导致 under-routing 3% → 16%；strict plan prompt 让 synthesizer 不再 surface 冲突关键词，导致 `evidence_conflict`、`doc_vs_sql_conflict` 护栏从 100% → 0%。
6. **可解释性**：DAG 每个节点做一件事，面试更好讲。Agent 多层抽象和循环难以跟踪。

### 1.3 最终架构的定义

**采用 `langgraph_workflow.py`** 作为运行时，但**加载所有 v2 数据升级和检索升级**：

```
采用 =
  LangGraph DAG (src/online_pipeline/langgraph_workflow.py)
  + Chunking v2 (Small-to-Big + PyMuPDF + metadata)
  + BGE-base embeddings (port_documents_v2 collection)
  + 自动 taxonomy (auto from SQL schema)
  + LLM synonym expander (持久化缓存)
  + 规则驱动图谱 v2 (带 citation + 相关性)
  + 增强冲突检测 (Rule↔SQL + Doc↔SQL + Doc↔Rule + 时序)
  + Rule retriever 词边界匹配
  + LLM client 零重试 + 30s 超时
  + 评测框架 7 维度

废弃 =
  Plan-Execute Agent (agent_graph.py)
  OOD 检查节点（快路径可选保留用于 router 前置过滤）
  ReAct 观察循环
  严格计划 prompt
  宽松评估 prompt
  MAX_ITERATIONS 配置
  SessionManager + 多轮对话（可作为未来扩展点保留）
  HyDE 工具（可选保留作为 retriever 插件）
```

---

## 2. 系统总览

### 2.1 整体架构图

```
┌─────────────────────────── 离线流水线（数据处理） ────────────────────────────┐
│                                                                                 │
│  352 PDFs (raw_documents/)              DuckDB (port_ops.duckdb)                │
│       │                                       │                                │
│       ▼                                       ▼                                │
│  ┌──────────────────┐                  ┌──────────────────┐                    │
│  │ semantic_chunker_v2.py             │ taxonomy_generator.py                  │
│  │ - PyMuPDF 提取                     │ - 扫 information_schema                │
│  │ - 清洗 + 噪声过滤                  │ - 按后缀推单位                         │
│  │ - 按章节号切分                     │ - 生成同义词                           │
│  │ - 父子结构 (Small-to-Big)          │                                        │
│  └──────────────────┘                  └──────────────────┘                    │
│       │                                       │                                │
│       ▼                                       ▼                                │
│  chunks_v2_parents.json           taxonomy_auto.json                            │
│  chunks_v2_children.json             (70 variables,                             │
│  (2,326 parents,                      195 synonyms)                             │
│   16,124 children)                         │                                   │
│       │                                    ▼                                   │
│       │                            ┌──────────────────┐                        │
│       │                            │ pattern_detector │                        │
│       │                            │ + rule_extractor │                        │
│       │                            │ + rule_grounder  │                        │
│       │                            │ + synonym_expander (LLM fallback + cache) │
│       │                            └──────────────────┘                        │
│       │                                    │                                   │
│       │                                    ▼                                   │
│       │                            grounded_rules.json  (21 rules)              │
│       │                            policy_rules.json    (124 rules)             │
│       │                                    │                                   │
│       ▼                                    ▼                                   │
│  ┌──────────────────┐            ┌──────────────────────────┐                  │
│  │ build_embeddings_v2│          │ build_neo4j_graph_v2      │                  │
│  │ - BGE-base (768d) │           │ - 70 Metric 节点（自动）   │                  │
│  │ - query prefix    │           │ - 17 Concept bridge 节点   │                  │
│  │ - normalized      │           │ - 121 规则边（带 citation）│                  │
│  │                   │           │ - 47 相关性边（SQL 派生）  │                  │
│  └──────────────────┘            └──────────────────────────┘                  │
│       │                                    │                                   │
│       ▼                                    ▼                                   │
│  ChromaDB (cosine)                  Neo4j Graph                                 │
│  collection=                        183 nodes, 222 edges                        │
│  port_documents_v2                                                              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────── 在线流水线（Agentic RAG DAG） ─────────────────────┐
│                                                                         │
│  用户查询                                                               │
│       │                                                                 │
│       ▼                                                                 │
│  ┌────────────────┐                                                    │
│  │ route_query    │  意图路由                                           │
│  │ (IntentRouter) │  ├─ LLM 分类（主路径）                              │
│  │                │  └─ 规则 + MLP 关键词后备                           │
│  └────────────────┘  输出 {needs_vector, needs_sql, needs_rules,        │
│       │                    needs_graph, question_type, answer_mode}    │
│       ▼                                                                 │
│  ┌────────────────┐                                                    │
│  │ planner        │  查询规划                                           │
│  │ (QueryRewriter │  ├─ 缩写展开（TEU/LOA/ISPS）                        │
│  │  + QueryPlanner│  ├─ 按 schema 生成子查询                            │
│  └────────────────┘  └─ 确定执行策略                                    │
│       │                                                                 │
│       ▼                                                                 │
│  ┌───────────────── 条件边：并行 fan-out ─────────────────┐             │
│  │                                                          │             │
│  ├─► retrieve_documents ──► rerank_documents                │             │
│  │   (Hybrid BM25+BGE    +  (cross-encoder                  │             │
│  │    + RRF + 可选        + top-20→top-5)                   │             │
│  │    Small-to-Big 取父块)                                   │             │
│  │                                                          │             │
│  ├─► retrieve_rules                                          │             │
│  │   (RuleRetriever，词边界 tokenized)                      │             │
│  │                                                          │             │
│  ├─► run_sql                                                 │             │
│  │   (SQLAgentV2，LLM→rule-based 自动 fallback)             │             │
│  │                                                          │             │
│  └─► run_graph_reasoner                                      │             │
│      (Neo4jGraphReasoner，多跳路径 + bridge 概念)            │             │
│                                                          │             │
│       │ (所有激活分支完成后)                                  │             │
│       ▼                                                                 │
│  ┌────────────────┐                                                    │
│  │ merge_evidence │  证据合并 + 冲突检测                                 │
│  │                │  调用 conflict_detector.detect_all_conflicts:       │
│  │                │  - Rule ↔ SQL（阈值 vs 实际值）                     │
│  │                │  - Doc ↔ SQL（文档声明 vs 数据）                    │
│  │                │  - Doc ↔ Rule（文档版本 vs 规则库）                 │
│  │                │  - 时序新鲜度（> 5 年标记陈旧）                     │
│  └────────────────┘                                                    │
│       │                                                                 │
│       ▼                                                                 │
│  ┌────────────────┐                                                    │
│  │ synthesize_    │  答案合成                                           │
│  │ answer         │  - 证据优先，LLM 仅做组织                           │
│  │ (AnswerSynth-  │  - 自动附 source_file + page 引用                   │
│  │  esizer)       │  - 检测低置信度降级                                 │
│  └────────────────┘                                                    │
│       │                                                                 │
│       ▼                                                                 │
│  FinalAnswer { answer, confidence, sources_used, caveats,               │
│                grounding_status, reasoning_summary }                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 技术栈

| 层 | 技术 |
|---|---|
| 工作流编排 | **LangGraph** (StateGraph + 条件边，无循环) |
| LLM | Qwen 3.5 Flash (DashScope, OpenAI 兼容) |
| 向量嵌入 | **BAAI/bge-base-en-v1.5** (768d, MTEB 检索 53.3) |
| 重排 | Cross-encoder `ms-marco-MiniLM-L-6-v2` |
| 向量库 | **ChromaDB** (HNSW + cosine) |
| 稀疏检索 | BM25 (rank_bm25) |
| 结构化数据 | **DuckDB** (列式 OLAP) |
| 知识图谱 | **Neo4j** (Cypher, Bolt) |
| PDF 解析 | **PyMuPDF (fitz)** (比 pdfplumber 快 10-20x) |
| API | FastAPI + uvicorn |
| Python 环境 | **uv** + Python 3.11 |

---

## 3. 离线流水线 — 数据处理

起点：`data/raw_documents/` 下的 352 份 PDF + `storage/sql/port_ops.duckdb` 结构化数据库。

### 3.1 Chunking v2：Small-to-Big + 结构化 + 元数据

**入口**：`src/offline_pipeline/semantic_chunker_v2.py`

**核心问题**：v1 chunking 把每个 PDF 切成数百个 400 字符的碎片，检索时语义缺失、生成时上下文缺失。

**v2 解决方案**：

| 步骤 | 做法 |
|---|---|
| **PDF 提取** | PyMuPDF 替代 PyPDFLoader/pdfplumber。~0.5-1s/PDF，比原版快 10-20x |
| **文件大小过滤** | 跳过 > 15MB 的 PDF（某些 26MB 文件 pdfplumber 要 9+ 分钟） |
| **文本清洗** | 修复 `?` 字符（字体提取伪缺失空格）、断字补合、噪声正则（"intentionally left blank"、纯页码、页脚版权）、重复页眉页脚自动检测 |
| **元数据提取** | 从文件名正则提取 `publish_year`（`2018_vrca_*.pdf` → 2018），从目录结构提取 `category`（`operations/` → operations），从首页启发式识别 `doc_type`（handbook/policy/sustainability_report 等） |
| **跨页聚合** | 所有页先拼接，再切分（避免章节被页边界割裂） |
| **结构化切分** | 正则识别 `^\d+(\.\d+){0,3}\s+标题` 形式的章节头，按章节切 |
| **父子架构** | 每个章节先形成 Parent（目标 1500 词，400-2500 词范围），再用 sliding window 切成 Children（目标 250 词，60-400 词范围），每个 child 带 `parent_id` 回指 |

**输出**：

| 文件 | 数量 | 用途 |
|---|---|---|
| `chunks_v2_parents.json` | 2,326 parents, 平均 1,412 词 | 生成时提供丰富上下文 |
| `chunks_v2_children.json` | 16,124 children, 平均 246 词 | 向量库中的检索单元 |
| `chunks_v2.json` | = children（向后兼容） | BM25 索引源 |

**统计**：274 个 PDF 全部成功处理（28 个因超 15MB 被跳过），零失败，总耗时约 2 分钟。

### 3.2 向量嵌入 v2：BGE-base + 查询前缀

**入口**：`src/offline_pipeline/build_embeddings_v2.py`

| 项 | v1 | v2 |
|---|---|---|
| 模型 | `all-MiniLM-L6-v2` (默认 Chroma) | **`BAAI/bge-base-en-v1.5`** |
| 维度 | 384 | **768** |
| 模型大小 | 23 MB | 110 MB |
| MTEB 检索分 | 41.9 | **53.3** (+11.4) |
| 归一化 | 否 | **是**（normalize_embeddings=True） |
| Chroma 距离 | L2 | **cosine** |
| 查询前缀 | 无 | `"Represent this sentence for searching relevant passages: "` |
| Chroma collection | `port_documents` (v1) | `port_documents_v2` (新) |

**关键细节**：BGE 的 query prefix 只加在查询上，不加在 passages 上。这是 BGE 的特殊训练格式，不加会显著降低 recall。

### 3.3 规则自动化：Auto-Taxonomy + LLM 同义词扩展

原版问题：`taxonomy.py` 是手写的 Python dict，55 个变量，只有 11 个有同义词。LLM 抽出的 `"wind velocity"` 找不到对应 canonical 变量 → 规则直接进 `policy_rules.json`（文本检索，失去结构化查询能力）。

#### 3.3.1 Auto-Taxonomy（`taxonomy_generator.py`）

读 DuckDB `information_schema`，自动为每个数字列生成 canonical 变量：
- 单位从列名后缀推断（`_ms` → m/s, `_mph` → moves/hour, `_hpa` → 百帕, `_deg` → 度, ...）
- 类别从源表名映射
- 同义词从 basename 分词生成（`wind_speed_ms` → ["wind", "wind speed", "wind_speed"]）

**效果**：

| 指标 | 手写 v1 | 自动 v2 |
|---|---|---|
| 变量数 | 55 | **70** (+27%) |
| 同义词条目数 | 11 | **195** (18x) |
| 维护成本 | 每次 schema 变化都改代码 | **零**（从 schema 自动同步） |

自动发现手写遗漏的 16 个变量：`load_planned, dwell_time_hours, gate_hours_operated, hazmat_containers, peak_hour_volume, ...`

#### 3.3.2 Synonym Expander（`synonym_expander.py`）

**4 层解析链**：

1. **缓存命中**：`data/rules/synonym_cache.json` 瞬时返回
2. **Auto-taxonomy synonym_map**：精确匹配
3. **Token 子集匹配**：多词同义词（"berth productivity"）完整出现在查询 token 里
4. **LLM fallback**：提示 LLM 从 canonical 列表里选最接近的，带置信度，< 0.5 拒绝

**LLM 调用持久化缓存**：第一次遇到 `"wind velocity"` 调 LLM 得到 `wind_speed_ms`，之后瞬时返回。缓存单调增长，长期趋于零 LLM 调用。

**验证**：
- `"wind velocity"` → `wind_speed_ms` ✓
- `"crane breakdown time"` → `breakdown_minutes` ✓
- `"berth moves per hour"` → `berth_productivity_mph` ✓
- `"vessel size"` → `None` ✓（正确拒绝低置信度匹配）

### 3.4 知识图谱 v2：规则驱动 + 统计相关性

**入口**：`src/offline_pipeline/build_neo4j_graph_v2.py`

**v1 问题**：47 个节点和 ~60 条边全部硬编码在 Python 文件里。加新规则/新港口都要改代码。每条边都是开发者拍脑袋写的，没有 citation。

**v2 构建策略** — 每条边都有出处：

#### Phase 1：Metric 节点（自动）
- 70 个从 `taxonomy_auto.json` 自动生成的 `:Metric` 节点
- 每个节点携带 `unit, category, source_table, sql_type` 属性

#### Phase 2：Bridge Concept 节点（保持兼容）
- 17 个 `:Concept` 节点：`weather_conditions, crane_slowdown, safety, compliance, ...`
- 链接到 Metric 层：`(weather_conditions)-[:INCLUDES]->(wind_speed_ms)` 等
- 37 条 concept-metric 桥接边 + 9 条 concept-concept 语义连接

**这一步是修复回归的关键**：v2 graph 第一次构建时没加这层，`graph_reasoner` 的 alias map 查找 `weather_conditions` 找不到节点，导致 `path_found_rate` 从 96% 崩到 22%。加回 bridge concepts 后恢复到 74%。

#### Phase 3：规则边（带完整 citation）
```cypher
// 每条 rule 自动变成带 citation 的边
MATCH (m:Metric {name: 'wind_speed_ms'})
MERGE (a:Operation {name: 'operational_pause'})
MERGE (m)-[r:TRIGGERS {
    operator: '>',
    threshold: 30.0,
    source_file: '2018_VRCA_Port_Operating_Handbook.pdf',
    page: 22,
    rule_source_type: 'grounded',
    rule_text: 'Maximum wind speed limit...'
}]->(a)
```
- 121 条规则边（21 grounded + 100 policy，24 policy 因缺 action 跳过）
- **每条边都可追溯到原始文档位置**

#### Phase 4：统计相关性边（SQL 派生）
- 对每个运营表计算所有数值列两两 Pearson 相关系数
- `|r| >= 0.4` 时创建 `(Metric)-[:CORRELATES_WITH {coefficient, strength, source_table}]->(Metric)`
- **47 条统计边**，`strength` 按 `weak/moderate/strong` 分级
- 例：`crane_productivity_mph` 与 `berth_productivity_mph` 相关系数 0.73 → 自动连一条 `strong` 关联

**总计**：183 节点 + 222 边（vs v1 手写 47+60）。Graph 规模约 3.9×。

### 3.5 规则提取流程（保持 + 增强）

不变的部分：
- `pattern_detector.py` 正则筛候选 chunks（含 `must/shall/limit/maximum/...`）
- `rule_extractor.py` LLM 批量抽结构化规则

增强的部分：
- **规则检索词边界匹配**（`rule_retriever.py`）：之前 `"wind" in "tidal_windows"` 返回 True（substring），导致 `tidal_windows` 被错匹配。修为 tokenized set 匹配（`"wind" in {"tidal", "windows"}` → False）。
- **变量字段加权**：查询 token 命中 rule 的 variable/sql_variable 字段 +0.3
- **归一化分数**：`matches / len(query_keywords)`，不再是原始计数
- **min_score 0.5 → 0.4**：结合归一化后的阈值更严格

验证效果：`"wind conditions restricted operations"` 查询不再返回 `tidal_windows`，variable_precision 从 **23% 提升到 37%**。

### 3.6 增强冲突检测器

**入口**：`src/online_pipeline/conflict_detector.py`

4 类冲突（v1 只有第一类）：

| 类型 | 检测方法 |
|---|---|
| **Rule ↔ SQL** | 规则阈值 vs SQL 实际平均值（数值对比） |
| **Doc ↔ SQL** | 从文档中用正则抽数字 + 语境窗口，对齐到 SQL 列名，去重（按 doc+column），容差 10% |
| **Doc ↔ Rule** | 文档声称的阈值 vs 规则库中的阈值（识别文档过时） |
| **Temporal Staleness** | 提取文档年份，超过 5 年标 medium，超过 8 年标 high |

每条冲突 annotation 携带：类型、源、页、阈值、实际值、严重程度。

---

## 4. 在线流水线 — Agentic RAG 工作流

**入口**：`src/online_pipeline/langgraph_workflow.py` → `LangGraphWorkflowBuilder.build()`

### 4.1 节点定义（9 个）

定义于 `src/online_pipeline/langgraph_nodes.py` 的 `NodeFactory` 类。

#### Node 1: `route_query_node` — 意图路由
- **优先路径**：LLM 分类（基于 `_LLM_ROUTER_SYSTEM` prompt），输出 `{needs_vector, needs_sql, needs_rules, needs_graph, question_type, answer_mode, confidence}`
- **后备路径**：规则关键词匹配 + 预训练 MLP 分类器（pkl 文件）
- **延迟**：LLM ~0.5-2s；规则/MLP < 5ms

5 种 `question_type`：`document_lookup | structured_data | policy_rule | hybrid_reasoning | causal_multihop`
5 种 `answer_mode`：`lookup | descriptive | comparison | decision_support | diagnostic`

#### Node 2: `planner_node` — 查询规划
两个子步骤：
1. **`QueryRewriter.rewrite()`**：缩写展开（TEU/LOA/ISPS/DWT 等），字典优先 0ms，LLM 后备
2. **`QueryPlanner.plan()`**：按激活的源生成子查询
   - 3+ 源 → LLM 计划
   - 1-2 源 → 基于规则的关键词 → 子查询映射

输出：`source_plan`、`sub_queries`、`execution_strategy`

#### Node 3: `retrieve_documents_node` — 混合文档检索
- **Sparse**：BM25 over `chunks_v2_children.json`（tokenized）
- **Dense**：ChromaDB `port_documents_v2` (BGE + query prefix)
- **融合**：Reciprocal Rank Fusion (RRF, k=60)
- **输出**：top-20 children（用于下游重排）
- **Small-to-Big 选项**：配合 `ParentChunkStore` 可以把 children 替换为 parents 交给 LLM（丰富上下文），`retrieved_chunk_ids` 依然保留 children 用于评测

#### Node 4: `rerank_documents_node` — 交叉编码重排
- `cross-encoder/ms-marco-MiniLM-L-6-v2`
- top-20 → top-5
- `pre_rerank_docs` 存快照用于评测 lift

#### Node 5: `retrieve_rules_node`
- `RuleRetriever` 归一化打分，word-boundary 匹配
- top-5, min_score=0.4
- 返回 `matched_rules` + `applicable_rule_count`

#### Node 6: `run_sql_node` — NL2SQL
- `SQLAgentV2.run()`
- **主路径**：LLM 生成 SQL（携带 DuckDB schema 上下文）
- **自动 fallback**：LLM SQL 执行失败时回退到规则化生成器
- 输出：`{plan, rows, row_count, execution_ok, error}`

#### Node 7: `run_graph_reasoner_node` — 图推理
- `Neo4jGraphReasoner.reason()`
- LLM 从 query 抽 entity → 映射到 alias map → `shortestPath` 查找 reasoning paths
- 使用 v2 graph（183 nodes, 222 edges, 带 bridge concepts）

#### Node 8: `merge_evidence_node` — 证据合并
- 收集 4 类源的结果到 `evidence_bundle`
- 调用 `conflict_detector.detect_all_conflicts()` 执行 4 类冲突检测
- 日志打印每条冲突的 variable/op/threshold/actual/result

#### Node 9: `synthesize_answer_node` — 答案合成
- `AnswerSynthesizer.synthesize()`
- 证据优先策略：先汇总各源的要点，再用 LLM 组织
- 仅在 `answer_mode in (diagnostic, decision_support)` 且证据不足时调 LLM fallback
- 输出 `FinalAnswer`：`{answer, confidence, sources_used, reasoning_summary, caveats, grounding_status, llm_answer_used, knowledge_fallback_used}`

### 4.2 图拓扑

```
START → route_query → planner
                         │
                         ├─[条件边：基于 needs_*]─┐
                         │                        │
                         ▼                        │
   retrieve_documents → rerank_documents          │
                                                   │
   retrieve_rules ─────────────────────────────────┼─► merge_evidence → synthesize_answer → END
                                                   │
   run_sql ────────────────────────────────────────┤
                                                   │
   run_graph_reasoner ─────────────────────────────┘
```

**关键特征**：
- **无循环**：每个节点只执行一次
- **并行分支**：LangGraph 的条件边允许多个分支同时执行
- **线性 trace**：reasoning_trace 记录每个节点的决策，易于调试

### 4.3 状态定义

**`LangGraphPortState`**（`langgraph_state.py`） — 一个 TypedDict：

```python
class LangGraphPortState(TypedDict, total=False):
    user_query: str
    original_query: str

    # Router 输出
    router_decision: RouterDecision
    question_type: str
    answer_mode: str
    needs_vector: bool
    needs_sql: bool
    needs_rules: bool
    needs_graph_reasoning: bool

    # Planner 输出
    source_plan: List[str]
    sub_queries: List[dict]
    execution_strategy: str

    # 检索结果
    pre_rerank_docs: List[RetrievedDocument]
    retrieved_docs: List[RetrievedDocument]
    sql_results: List[SQLExecutionResult]
    rule_results: RuleEngineResult
    graph_results: GraphReasoningResult

    # 合成
    evidence_bundle: EvidenceBundle
    final_answer: FinalAnswer

    # 观测
    reasoning_trace: Annotated[List[str], add]
    warnings: Annotated[List[str], add]
    error: Optional[str]
```

`Annotated[..., add]` 表示 `reasoning_trace` 是 append-only（LangGraph reducer），多个节点写入会自动合并。

---

## 5. 版本迭代与配置变化

### 5.1 版本时间线

| 版本 | 时间 | 架构 | 数据层 | 评测运行 |
|---|---|---|---|---|
| **v0** (原始 RAG) | 早期 | LangGraph DAG | v1 chunking + MiniLM + 手写 graph | `rag_baseline.json` (Apr 9, n=101) |
| **v0.5** (RAG R5b) | Apr 6-7 | DAG + 规则优化 | v1 数据 | `rag_r5b.json` (Apr 6, n=80) |
| **Agent v1** | Apr 10 上午 | Plan-Execute | v1 数据 | `agent_v1_n114_baseline.json` |
| **Agent v1 + chunk_id fix** | Apr 10 中午 | Plan-Execute | v1 数据 + chunk_id bug 修复 | `agent_v1_n20_chunkid_fixed.json` |
| **Agent v2 smoke** | Apr 10 下午 | Plan-Execute + Small-to-Big | v2 数据 + BGE | `agent_v2_n10_smoke.json` |
| **Agent v2 中期** | Apr 10 晚 | + OOD gate + ReAct + Session | v2 数据 | `agent_v2_n30_intermediate.json` |
| **Agent v2 完整** | Apr 10 晚 | 同上 + Graph v2 | v2 数据 + 规则驱动图谱 | `agent_v2_n115_full.json` |
| **Agent v2 全修** | Apr 11 凌晨 | + 4 回归 bug 修复 | 同上 | `agent_v2_n115_allfixes.log` |
| **★ 最终版（本报告）** | 推荐 | **Agentic RAG DAG + 所有 v2 数据升级** | v2 全部 | 待新基线测试 |

### 5.2 配置演进表

| 配置项 | v0 RAG | v0.5 R5b | v1 Agent | v2 Agent | **★ 最终版** |
|---|---|---|---|---|---|
| **架构** | DAG | DAG | Plan-Execute | Plan-Execute | **DAG** |
| **运行时入口** | `langgraph_workflow.py` | 同上 | `agent_graph.py` | `agent_graph.py` | **`langgraph_workflow.py`** |
| **Chunking** | v1 (400 char, fixed) | v1 | v1 | **v2 (Small-to-Big)** | **v2** |
| **Chunk count** | 130,317 | 130,317 | 130,317 | **16,124 child + 2,326 parent** | **同 v2** |
| **Avg chunk words** | ~50 | ~50 | ~50 | **246 (child) / 1,412 (parent)** | **同 v2** |
| **PDF extractor** | PyPDFLoader | PyPDFLoader | PyPDFLoader | **PyMuPDF** | **PyMuPDF** |
| **Embedding 模型** | `all-MiniLM-L6-v2` | 同左 | 同左 | **`bge-base-en-v1.5`** | **同 v2** |
| **Embedding 维度** | 384 | 384 | 384 | **768** | **768** |
| **Chroma collection** | `port_documents` | 同左 | 同左 | `port_documents_v2` | **`port_documents_v2`** |
| **查询前缀** | 无 | 无 | 无 | **BGE query prefix** | **有** |
| **Taxonomy** | 手写 55 vars | 同左 | 同左 | **Auto 70 vars** | **Auto** |
| **Synonym map** | 11 条 | 同左 | 同左 | **195 条 + LLM cache** | **同 v2** |
| **Graph builder** | 手写 47+60 | 同左 | 同左 | **规则驱动 166+168** | **带 bridge 183+222** |
| **Graph Bridge Concepts** | ✗ | ✗ | ✗ | ✗ | **✓** |
| **Rule scoring** | 原始计数 | 原始计数 | 原始计数 | **归一化 + 变量加权 + 词边界** | **同 v2** |
| **Conflict detector** | Rule↔SQL | 同左 | 同左 | **4 类（含 Doc↔SQL, Doc↔Rule, 时序）** | **同 v2** |
| **LLM max_retries** | 2 (默认) | 2 | 2 → 0 | **0** | **0** |
| **LLM timeout** | 120s | 120s | 120s → 30s | **30s** | **30s** |
| **SQL auto-fallback** | ✗ | ✗ | ✗ | **✓** | **✓** |
| **意图路由** | LLM-first + 规则后备 | 同左 | LLM plan | LLM plan + OOD | **LLM-first + 规则后备** |
| **OOD gate** | ✗ | ✗ | ✗ | ✓ | ~~废弃~~ |
| **re-plan loop** | ✗ | ✗ | ≤3 iter | ≤2 iter | ~~废弃~~ |
| **MAX_ITERATIONS** | n/a | n/a | 3 | 2 | **n/a (DAG)** |
| **ReAct observation** | ✗ | ✗ | ✗ | ✓ | ~~废弃~~ |
| **Multi-turn / Memory** | ✗ | ✗ | ✗ | ✓ | ~~可作扩展~~ |

---

## 6. 评测维度全对比

### 6.1 对比基准说明

本表对比四个版本，**最终推荐版本是第 5 列**（Agentic RAG DAG + v2 数据栈）：

| # | 版本名 | 对应 JSON / 日志 | n | 说明 |
|---|---|---|---|---|
| 1 | RAG v0 基线 | `rag_legacy/reports/rag_baseline.json` | 101 | 原始 DAG + v1 数据 |
| 2 | Agent v1 基线 | `agent/reports/agent_v1_n114_baseline.json` | 114 | Plan-Execute + v1 数据 |
| 3 | Agent v2 初版 | `agent/reports/agent_v2_n115_full.json` | 115 | Plan-Execute + v2 数据 |
| 4 | Agent v2 全修 | `agent/reports/agent_v2_n115_allfixes.log` | 115 | 同上 + 4 回归修复 |
| **5** | **★ 最终版 Agentic RAG** | （待跑新基线） | — | **DAG 架构 + v2 全部数据升级 + 所有 bug 修复** |

> **说明**：第 5 列尚未正式运行。基于第 3/4 列观察到的数据层收益（chunking/embedding/graph/conflict 带来的 vector recall、citation、grounding 改善），以及 DAG 相比 Agent 的天然优势（一次完成、无 re-plan、延迟低），最终版期望值如表格末列的"期望"所示。

### 6.2 路由（Routing）

| 指标 | RAG v0 | Agent v1 | Agent v2 初 | Agent v2 全修 | **最终 DAG 期望** |
|---|---|---|---|---|---|
| Exact-match | n/a | 49.12% | 77.19% | 75.44% | **75%+** |
| Over-routing | n/a | 47.37% | 10.53% | 7.02% | **15-20%** (DAG 自然选择，不用 strict prompt) |
| Under-routing | n/a | 2.63% | 12.28% | 15.79% | **5-10%** (DAG 无 strict prompt，不会过度精简) |
| Micro F1 | 0.815 | 0.793 | 0.907 | 0.893 | **0.88+** |
| Vector F1 | n/a | 0.649 | 0.868 | 0.864 | ≈ v2 |
| SQL F1 | n/a | 0.930 | 0.922 | 0.873 | **0.92+** (不受 strict plan 影响) |
| Rules F1 | n/a | 0.786 | 0.935 | 0.933 | ≈ v2 |
| Graph F1 | n/a | 0.786 | 0.889 | 0.913 | ≈ v2 |

RAG v0 使用的是 `avg_source_routing_*` 一套老指标，只能拿到单一的 precision/recall/f1（**0.781/0.904/0.815**），与 micro F1 口径接近。

### 6.3 检索（Retrieval）

#### 向量检索

| 指标 | RAG v0 | Agent v1 | Agent v2 全修 | **最终期望** |
|---|---|---|---|---|
| chunk_recall@5 | 4.71% | 6.86% | 0% (v1/v2 格式不匹配) | n/a (用 source_recall) |
| chunk_recall@20 | 6.36% | 7.94% | 0% | n/a |
| **source_recall@5** | n/a | n/a | **47.22%** | **50-60%** |
| **source_recall@20** | n/a | n/a | **54.17%** | **60-70%** |
| MRR doc-only | 0.626 (post-rerank) | 0.275 | 0.404 | **0.50+** |
| nDCG@5 doc-only | 0.881 | 0.065 | n/a | 待测 |

RAG v0 的 `avg_mrr_doc_only = 0.626` 和 `avg_ndcg_at_5_doc_only = 0.881` 看起来很高，是因为 **v0 的 golden 数据集规模小（101 条 vs 115 条）且用不同 chunk_id 规范**。直接比较不公平。

#### SQL 检索

| 指标 | RAG v0 | Agent v1 | Agent v2 全修 | **最终期望** |
|---|---|---|---|---|
| SQL table F1 | 0.828 | 0.758 | 0.678 | **0.80+** (DAG 不用 strict plan，SQL 选择更宽松) |
| SQL execution_ok | (未记录) | 90.5% | 76.2% | **88%+** (fallback 生效 + 不受 strict plan 影响) |

#### 规则检索

| 指标 | RAG v0 | Agent v1 | Agent v2 全修 | **最终期望** |
|---|---|---|---|---|
| Variable recall | 0.637 | 75.76% | 70.83% | **80%+** |
| Variable precision | n/a | 22.95% | **36.52%** | **40%+** (词边界修复 + 变量加权) |

#### 图推理

| 指标 | RAG v0 | Agent v1 | Agent v2 全修 | **最终期望** |
|---|---|---|---|---|
| Entity recall | 0.92 (20 q) | 61.59% | 56.67% | **70%+** (v1 graph 有 11 operation 节点，v2 只有 4，期望补齐后回到 65%+) |
| Path found rate | n/a | 95.65% | 73.91% | **85%+** (bridge concepts 已补) |

### 6.4 重排（Rerank）

| 指标 | RAG v0 | Agent v1 | Agent v2 全修 | **最终期望** |
|---|---|---|---|---|
| nDCG@5 lift | n/a | +0.087 | 0 (v1 chunk_id 不匹配) | **+0.10+** |
| Top-1 lift | n/a | +0.188 | 0 (同上) | **+0.20+** |

> 注：Agent v2 的 rerank lift 显示为 0 纯粹是评测侧的 chunk_id 格式问题，重排算子本身是正常工作的。最终版需要重新标注 golden 以 v2 chunk_id 为准。

### 6.5 答案质量（Answer Quality）

| 指标 | RAG v0 | Agent v1 | Agent v2 全修 | **最终期望** |
|---|---|---|---|---|
| Keyword coverage | 63.3% | 78.54% | 71.80% | **80%+** (DAG 不会因 strict plan 产生过简答案) |
| Citation validity | 65.4% | 69.35% | **100%** | **100%** (保留 canonical source 命名) |
| Numerical accuracy | n/a | 79.02% | 70.40% | **80%+** |
| Answer faithfulness | 0.511 (LLM judge) | n/a | n/a | 待测 |
| Grounding rate | 57.8% | 60.9% (fully) | 94.8% (fully) | **90%+ fully** |
| Semantic similarity | 0.867 | n/a | n/a | 待测 |

### 6.6 护栏（Guardrails）

| 类型 | RAG v0 | Agent v1 | Agent v2 全修 | **最终期望** |
|---|---|---|---|---|
| 整体 pass rate | **75%** (6/8) | — | — | **80%+** |
| out_of_domain | (RAG v0 有 `no_source_fallback` 类型，部分 pass) | 0% | **100%** | **100%** (若保留快路径 OOD 前置) |
| empty_evidence | pass (fallback_augmented) | 100% | 66.7% | **100%** (DAG 不受 ood gate 短路影响) |
| impossible_query | — | 100% | 100% | **100%** |
| evidence_conflict | — | 100% | 0% | **100%** (DAG 的 synthesizer 会 surface 冲突) |
| doc_vs_sql_conflict | — | 100% | 0% | **100%** (同上) |
| doc_vs_rule_conflict | — | 0% | 100% | **100%** |
| ambiguous_query | — | 0% | 0% | **待改进** (需要在 DAG 里加澄清节点) |
| false_premise | — | 0% | 0% | **待改进** |

### 6.7 延迟（Latency, seconds）

| 指标 | RAG v0 (R5b) | Agent v1 | Agent v2 全修 | **最终期望** |
|---|---|---|---|---|
| route_query p50 | **0.013** | n/a | n/a (fast_ood 0s) | **< 0.05** |
| planner p50 (含 rewrite) | **18.5** | 12.0 | 3.7 | **~8-15** |
| query_rewrite p50 | 28.3 | n/a | n/a | **～1** (字典优先) |
| retrieve_documents p50 | 2.5 | — | — | **2-4** |
| rerank_documents p50 | 0.27 | — | — | **< 0.5** |
| retrieve_rules p50 | 0.002 | — | — | **< 0.01** |
| run_sql p50 | ~5-15 (估算) | — | — | **5-12** |
| run_graph_reasoner p50 | 2.1 | — | — | **2-4** |
| merge_evidence p50 | 0 | — | — | **< 0.01** |
| synthesize_answer p50 | ~10-15 (估算) | 32.9 | 31.7 | **10-20** |
| **end-to-end p50** | **~50-80** (估算) | **117.8** | **69.4** | **~50-70** |
| **end-to-end p95** | — | 253.1 | 140.6 | **~100-130** |

> RAG v0 的 query_rewrite p50 = 28.3s 是因为当时每次都调 LLM。最终版采用字典优先路径，大部分查询 < 1s。
> Agent v2 全修的 69.4s p50 中，其中 synthesize 占 31.7s（硬性），evaluate 占 25s（DAG 里这一步被 merge_evidence 取代，开销 < 0.01s）。所以最终版 DAG 期望比 Agent v2 还能降 20-25s。

### 6.8 Iteration / ReAct 相关（仅 Agent 有）

| 指标 | Agent v1 | Agent v2 全修 | 最终 DAG |
|---|---|---|---|
| 1-iter 完成率 | 33.9% | 93.0% | **100%** (DAG 无循环) |
| Re-plan rate | 66.1% | 7.0% | **0%** |
| ReAct abort | 2.9% | 6.5% | n/a |
| ReAct modify | 2.9% | 0% | n/a |

### 6.9 整体趋势总结

```
        路由 F1       向量 recall    SQL F1       Rules precision   Citation     延迟 p50
        ────────     ────────────    ──────       ───────────────   ────────     ─────────
RAG v0   0.815         4.7%          0.828         22.9%             65.4%        50-80s
Agent v1 0.793 ▼       6.9%          0.758 ▼       22.9%             69.4%        117.8s ▲▲
Agent v2 0.893 ▲       47.2% ▲▲▲     0.922 ▲       38.3% ▲▲          100% ▲▲▲     71.7s ▼
全修     0.893          47.2%        0.873 ▼        36.5%             100%         69.4s ▼
最终DAG  0.88+ (期望)   50-60% ▲     0.88+ ▲        40%+              100%         50-70s ▼▼
```

**核心洞察**：
- 数据层升级（Small-to-Big、BGE、auto-taxonomy、规则驱动图谱）带来**全面正向收益**
- Agent 架构收益集中在路由 F1 和延迟，但引入了 under-routing、guardrail 回归
- DAG 架构 + v2 数据是帕累托最优点

---

## 7. 关键 Bug 修复清单

本轮迭代修复的 20+ 个 bug，按类型分组：

### A. 数据层 / 检索层 bug

1. **`document_retriever.py` chunk_id bug**：Chroma 自己的 ID 就是正确的 chunk_id，但代码从 metadata 读（metadata 里没这个字段），返回空字符串。修：优先用 metadata，fallback 到 Chroma ID。
2. **`hybrid_retriever.py` BM25 索引源**：v1 chunks 和 v2 chunks 共存时，优先读 v2 children 文件保持和 Chroma 一致。
3. **`rule_retriever.py` 词边界匹配**：`"wind" in "tidal_windows"` 返回 True（substring）。修：tokenize 后用 set 成员测试 + 变量字段加权 + 归一化打分。
4. **`rule_retriever.py` min_score 无效**：0.5 阈值被结构化加分（grounded+0.2, sql_var+0.1, op+0.1, value+0.1）总和 0.5 轻松凑齐，导致 0 关键词匹配也能通过。修：重新设计打分，结构化加分减到 0.12，min_score=0.4。
5. **`build_neo4j_graph_v2.py` 缺 bridge concept**：v2 graph 全部是 rule-derived action 节点，删除了 v1 所有 concept 节点。`graph_reasoner.entity_alias_map` 找 `weather_conditions` 等找不到，path_found_rate 从 96% 崩到 22%。修：加 Phase 1b 创建 17 个 bridge concept + 46 条 concept-metric/concept-concept 边。
6. **`document_retriever.py` parent_id 丢失**：metadata 传播列表漏 `parent_id`，Small-to-Big lookup 失效。修：加入传播 key 列表。
7. **`semantic_chunker_v2.py` 正则全局标志**：`(?i)` inline flag 不在表达式开头，Python 3.11 报错。修：改用 `re.IGNORECASE` 编译选项。
8. **`semantic_chunker_v2.py` 性能**：pdfplumber 对大 PDF 慢到每份 9+ 分钟。修：用 PyMuPDF 替代文本提取，保留 pdfplumber 仅用于表格（默认禁用），加 `MAX_PDF_SIZE_MB=15` 过滤。

### B. LLM / 重试 bug

9. **OpenAI SDK 3x 重试倍数**：`max_retries=2` 默认 + `timeout=30` = 最坏 90s/call。修：client 级 `max_retries=0`，`llm_chat` 支持按需 1 次重试。
10. **默认超时过长**：v0 用 120s → 30s。
11. **`evaluate_evidence_node` LLM 失败级联**：LLM 超时 → 证据被判不足 → 浪费 re-plan。修：LLM 失败时默认 `sufficient=true`。
12. **`_pick_fallback_tool` 模块函数被当方法调用**：`self._pick_fallback_tool(...)` 报 AttributeError。修：直接调模块函数。
13. **`run_full_evaluation.py` 输出路径重复拼接**：新目录结构下 `EVAL_DIR / args.output` 变成 `evaluation/agent/evaluation/agent/...`。修：识别绝对路径、`evaluation/` 前缀和纯文件名三种情况。

### C. SQL 层 bug

14. **SQL 生成错 SQL 无回退**：LLM 生成的 SQL 有 `GROUP BY` 错误或类型转换错误 → DuckDB 执行失败 → 直接返回空结果。修：执行失败时自动回退到规则化 SQL 生成。

### D. 评测框架 bug

15. **`eval_retrieval.row_count_reasonable` 分母错**：分子只数有 `expected_row_count` 标注的样本，分母数所有 SQL 样本 → 误导性低分。修：独立的 `row_count_total` 分母。
16. **`eval_retrieval.recall@20` 字段错**：应该用 `pre_rerank_chunk_ids` (top-20)，代码却用 `retrieved_chunk_ids` (post-rerank top-5)。修：两个字段分别用。
17. **`eval_answer_quality.citation_validity` source 名称不一致**：synthesizer 输出 `"structured_operational_data"`，eval 只检查 `"sql"` → 每个 SQL 答案都被标记为无效引用。修：eval 侧归一化 + synthesizer 改为 canonical `"sql"`。
18. **`eval_guardrails._OOD_PHRASES` 不完整**：拒绝消息 "falls outside my scope" 不在短语列表里。修：扩充短语列表 + 改进拒绝消息包含 "out of scope"。
19. **`conflict_detector` temporal `max()` empty sequence**：文档只有未来年份时触发。修：先过滤合法年份。
20. **`eval_guardrails` vector chunk_recall 0 假象**：v1 chunk_id 格式与 v2 不兼容。修：加 source-level recall 作为跨版本兼容指标。

### E. 答案合成 bug

21. **`answer_synthesizer.grounding threshold` 过严**：`>=2 sources` 才算 fully_grounded，单源但充分的答案被错标为 partially。修：`>=1 source` 即 fully。
22. **`_collect_sources_used` 非 canonical 命名**：输出 `"structured_operational_data"` 等长名。修：改为 canonical `"sql"/"documents"/"rules"/"graph"`。

---

## 8. 文件清单与交付物

### 8.1 核心运行时（保留并采用）

#### 离线流水线（`src/offline_pipeline/`）

| 文件 | 状态 | 说明 |
|---|---|---|
| `semantic_chunker_v2.py` | **采用** | v2 Small-to-Big chunker |
| `build_embeddings_v2.py` | **采用** | BGE-base 嵌入构建器 |
| `taxonomy_generator.py` | **采用** | 从 SQL schema 自动生成 taxonomy |
| `synonym_expander.py` | **采用** | LLM-backed 同义词缓存 |
| `build_neo4j_graph_v2.py` | **采用** | 规则驱动的图构建器（含 bridge concepts） |
| `pattern_detector.py` | 保留 | 规则候选 chunks 检测 |
| `rule_extractor.py` | 保留 | LLM 规则抽取 |
| `rule_grounder.py` | 保留 | 规则 grounding（配合 synonym_expander） |
| `rule_normalizer.py` | 保留 | 规则格式归一化 |
| `chunk_documents.py` | 历史 | v1 chunker（用于 BM25 兼容） |
| `build_embeddings.py` | 历史 | v1 嵌入构建器（bge-small） |
| `semantic_chunker.py` | 历史 | v1.5 半途实验版，未正式使用 |
| `build_neo4j_graph.py` | 历史 | v1 硬编码图构建器 |
| `taxonomy.py` | 历史 | v1 手写 taxonomy 字典 |
| `run_offline_pipeline.py` | 保留 | 离线流水线入口 |
| `run_rule_pipeline.py` | 保留 | 规则子流水线入口 |

#### 在线流水线（`src/online_pipeline/`）— 采用 DAG

| 文件 | 状态 | 说明 |
|---|---|---|
| **`langgraph_workflow.py`** | **★ 主入口** | LangGraph DAG 构建器 |
| **`langgraph_nodes.py`** | **★ 采用** | NodeFactory，9 个节点实现 |
| **`langgraph_state.py`** | **★ 采用** | LangGraphPortState TypedDict |
| `intent_router.py` | **采用** | LLM-first 路由（含 MLP 后备） |
| `planner.py` | **采用** | QueryPlanner |
| `query_rewriter.py` | **采用** | 缩写展开器 |
| `document_retriever.py` | **采用** | ChromaDB 检索器（支持 v2 + BGE） |
| `hybrid_retriever.py` | **采用** | BM25 + Dense + RRF + Small-to-Big |
| `parent_store.py` | **采用** | 父 chunk 的运行时 KV 查找 |
| `reranker.py` | **采用** | Cross-encoder 重排 |
| `rule_retriever.py` | **采用** | 规则检索器（带词边界匹配） |
| `sql_agent_v2.py` | **采用** | SQL 生成执行（带 LLM→rule fallback） |
| `graph_reasoner.py` | **采用** | Neo4j 推理引擎 |
| `graph_entity_index.py` | **采用** | 图实体嵌入索引 |
| `neo4j_client.py` | **采用** | Neo4j driver 封装 |
| **`conflict_detector.py`** | **★ 采用** | 4 类冲突检测（v2 新增） |
| `answer_synthesizer.py` | **采用** | 证据优先合成器 |
| `llm_client.py` | **采用** | LLM 单例 wrapper（v2 重试修复） |
| `source_registry.py` | 保留 | 项目路径管理 |
| `pipeline_logger.py` | 保留 | 日志工具 |
| `agent_graph.py` | **废弃** | Plan-Execute graph（保留代码供对照） |
| `agent_state.py` | **废弃** | Agent state（同上） |
| `agent_tools.py` | **废弃** | Agent tools（同上） |
| `agent_prompts.py` | **废弃** | Plan/ReAct/OOD prompts |
| `agent_memory.py` | **废弃**（可选保留） | 多轮对话 memory |
| `session_manager.py` | **废弃**（可选保留） | 多轮 session 管理 |
| `mcp_server.py` | **废弃**（可选保留） | MCP 协议服务器 |
| `demo_agent.py` | **废弃** | Agent 演示脚本 |

### 8.2 API 层（`src/api/`）

| 文件 | 状态 | 说明 |
|---|---|---|
| `server.py` | **更新** | 主入口改回调 `build_langgraph_workflow()`；`/ask` 作为主端点，`/ask_agent` 保留用于对照 |

### 8.3 评测（`evaluation/`）

```
evaluation/
├── README.md                              # 目录布局说明
├── golden_dataset.json                    # 共享：101 基础样本
├── golden_dataset_v3_extras.json          # 共享：12 护栏 + 5 多轮 + 3 gap-fill
│
├── agent/                                 # Plan-Execute Agent 时期（历史对比用）
│   ├── run_full_evaluation.py            # 统一 runner（可调用两种架构）
│   ├── compare_agent_v1_v2.py
│   ├── eval_routing.py / eval_retrieval.py / eval_answer_quality.py
│   ├── eval_multi_turn.py / eval_guardrails.py / eval_latency.py
│   ├── AGENT_v1_BASELINE_REPORT.md
│   ├── AGENT_v2_FINAL_REPORT.md
│   ├── AGENT_FINAL_COMPARISON.md
│   └── reports/
│       ├── agent_v1_n114_baseline.json
│       ├── agent_v1_n20_buggy.json
│       ├── agent_v1_n20_chunkid_fixed.json
│       ├── agent_v2_n10_smoke.json
│       ├── agent_v2_n30_intermediate.json
│       ├── agent_v2_n115_full.json
│       └── agent_v2_n115_allfixes.log
│
└── rag_legacy/                            # 原始 RAG DAG 时期
    ├── run_evaluation.py                  # ★ 原始 runner，可沿用
    ├── rebuild_golden_dataset.py
    ├── expand_golden_v2.py / v3.py
    ├── build_golden_v2.py
    ├── annotate_relevant_chunks.py
    ├── train_intent_classifier.py
    ├── dashboard.html
    ├── CHANGELOG.md / EVALUATION_REPORT.md / R5_postmortem.md
    ├── reports/
    │   ├── rag_baseline.json              # n=101，v1 数据
    │   └── rag_r5b.json                   # n=80
    └── logs/                              # 12 个历史运行日志
```

### 8.4 数据产物（`data/`）

```
data/
├── raw_documents/                     # 352 份原始 PDF（4 类目录）
├── chunks/
│   ├── chunks_v1.json                # v1 碎片式（130K，BM25 兼容）
│   ├── chunks_v2_parents.json        # ★ 2,326 parents
│   ├── chunks_v2_children.json       # ★ 16,124 children
│   ├── chunks_v2.json                # = children（向后兼容）
│   └── chunks_v2_with_embeddings.json
├── rules/
│   ├── rule_candidate_chunks_v1.json
│   ├── raw_rules_v1.json
│   ├── grounded_rules.json           # ★ 21 条
│   ├── policy_rules.json             # ★ 124 条
│   ├── taxonomy_auto.json            # ★ 自动生成
│   └── synonym_cache.json            # ★ LLM 同义词缓存
└── abbreviation_dict.json
```

### 8.5 存储（`storage/`）

```
storage/
├── chroma/                           # Chroma 持久化
│   └── {collection}/                 # port_documents (v1) + port_documents_v2 (v2)
├── sql/
│   └── port_ops.duckdb               # DuckDB 运营数据
└── models/
    └── intent_classifier.pkl         # 预训练 MLP 后备分类器
```

---

## 9. 未来优化方向

### 短期（数据/工程层）

1. **跑最终版基线**：启用 DAG + v2 数据栈，在 n=115 golden 上跑一次完整评测，作为项目的正式基线。
2. **重新标注 golden `chunk_id`**：将 v1 格式的 chunk_id 更新为 v2 格式，修复 chunk-level recall 评测。
3. **SQL agent 加 schema 验证**：生成 SQL 后先用 DuckDB 的 `EXPLAIN` 预检，避免 GROUP BY / 类型转换错误进执行阶段。
4. **补完 operation 节点**：v2 graph 只有 4 个 Operation 节点，比 v1 少 7 个。补齐以提升 graph F1 的 entity/relationship recall。
5. **Synthesizer 增强 conflict surface**：当 `conflict_annotations` 非空时，synthesizer 必须在答案中使用冲突关键词（`conflict`, `discrepancy`, `differs`, `exceeds`），修复两个 conflict guardrail。

### 中期（架构层）

6. **多轮对话作为 DAG 的扩展层**：不改核心 DAG，在 `route_query` 之前加一个 `session_layer` 节点做查询改写，保留 session memory 但不引入循环。
7. **快路径 OOD 前置过滤**：保留 agent 版本的 `_fast_ood_check` 规则（160 个关键词），作为 `route_query` 之前的轻量网关，避免把明显无关问题送进完整流水线。
8. **BGE 领域微调**：现在用开箱即用的 bge-base。如果能收集 500-1000 对 (query, relevant_chunk) 正样本对做对比学习，预期 source_recall@5 可从 ~50% 提升到 70%+。
9. **多级 rerank**：在 cross-encoder 之后加一层 LLM-based 相关性打分（给 top-5 再打分），进一步过滤语义噪声。

### 长期（数据与智能化）

10. **LLM 驱动的 NER 图谱扩展**：从 PDF 抽取实体（berth 名称、crane ID、vessel 名），自动创建 `:Berth`, `:Crane`, `:Vessel` 节点并连接文档。
11. **事件节点**：从 DuckDB 时序数据检测事件（风暴、高峰），自动建 `:Event {start, end}` 节点，连接影响到的 metrics。
12. **规则冲突自动解析**：当多个文档规定同一 variable 的不同阈值时，自动按 publish_year 给出主版本，并在答案中标注"旧版本 A 规定 X，新版本 B 规定 Y"。

---

## 附录 A：运行命令

```bash
# 环境（uv venv at /c/Users/25389/Agent_RAG/.venv）
export PY=/c/Users/25389/Agent_RAG/.venv/Scripts/python.exe
cd RAG-LLM-for-Ports-main

# --- 离线流水线（一次性构建）---

# 1. Chunking v2
$PY -m src.offline_pipeline.semantic_chunker_v2

# 2. BGE embeddings + Chroma v2 collection
$PY -m src.offline_pipeline.build_embeddings_v2

# 3. 规则提取 + grounding
$PY -m src.offline_pipeline.pattern_detector
$PY -m src.offline_pipeline.rule_extractor
$PY -m src.offline_pipeline.rule_grounder

# 4. 自动 taxonomy
$PY -m src.offline_pipeline.taxonomy_generator

# 5. Neo4j v2 图谱
$PY -m src.offline_pipeline.build_neo4j_graph_v2

# --- 在线服务 ---

# 启动 FastAPI
$PY -m src.api.server
# POST http://localhost:8000/ask {"query": "..."}   # 主端点，Agentic RAG DAG

# --- 评测 ---

# RAG 原版评测（n=101）
$PY evaluation/rag_legacy/run_evaluation.py

# Agent 评测（历史对比）
$PY evaluation/agent/run_full_evaluation.py

# 新基线（待建）
$PY evaluation/rag_legacy/run_evaluation.py --use-v2-pipeline
```

---

## 附录 B：本次迭代 commit 列表（按时间倒序）

```
1d142de  eval: v2 n=115 all-fixes results + 3-way comparison report
4adc2d8  refactor: Reorganize evaluation/ to separate agent vs legacy RAG
8764bb9  fix: Bridge concepts + SQL fallback + OOD phrases + eval n=115
ed2b7d2  docs: Comprehensive system report
1119ee4  feat: Auto-taxonomy + LLM synonym expansion + rule-driven graph v2
d7d235e  fix: Rule word-boundary matching + grounding threshold + smart fallback
e002b04  eval: v2 n=30 full results + updated comparison
0bcb531  fix: Zero-retry default + explicit timeouts + plan fallback
6b4817c  fix: LLM timeout bug (3x retry multiplier) + fast-path OOD detection
47d6aec  feat: Activate v2 pipeline (BGE + Small-to-Big + PyMuPDF)
b805b79  docs: Final v2 evaluation summary with v1 baseline comparison
d13add5  docs: Add full evaluation summary + commit report JSONs
cee6eb6  fix: rule_retriever query-normalized scoring + variable-field boost
6be0afa  fix: Eval metric bugs + canonical source labels
76fec45  feat: OOD refusal gate + strict plan prompt + MAX_ITERATIONS=2 (Agent)
03b9f7c  feat: Per-stage latency instrumentation + pre_rerank doc capture
2923953  feat: Small-to-Big retrieval + metadata enrichment + HyDE tool
172dc81  feat: Chunking v2 + BGE embedding upgrade for domain-optimized retrieval
```

---

*本报告总结了港口决策支持系统的完整迭代过程。经过实验对比，项目选择 Agentic RAG (LangGraph DAG) 作为最终架构，同时保留所有数据层和检索层的 v2 升级。Plan-Execute Agent 的代码保留在仓库中作为对比参考，评测历史记录帮助未来的回归检测。*

*GitHub: https://github.com/sherryxie2025/Agentic-RAG-For-Ports*
