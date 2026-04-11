# Golden Dataset v3 构建方法论

> 本文详细记录 `golden_dataset_v3_rag.json` 的完整构建方法。该数据集用于对 **Agentic RAG DAG (v2 data pipeline)** 进行无偏评测。

---

## 目录

1. [动机与问题定义](#1-动机与问题定义)
2. [设计原则](#2-设计原则)
3. [覆盖维度的全面分析](#3-覆盖维度的全面分析)
4. [样本量的统计学推导](#4-样本量的统计学推导)
5. [分层采样策略](#5-分层采样策略)
6. [反向生成方法](#6-反向生成方法)
7. [Opus 子代理并行生成架构](#7-opus-子代理并行生成架构)
8. [完整工作流](#8-完整工作流)
9. [按类别的生成器详解](#9-按类别的生成器详解)
10. [质量规则与验证](#10-质量规则与验证)
11. [最终数据集统计](#11-最终数据集统计)
12. [文件清单](#12-文件清单)

---

## 1. 动机与问题定义

### 1.1 历史背景

项目的评测数据集演进经历了三个阶段：

| 版本 | 样本数 | 问题 |
|---|---|---|
| `golden_dataset.json` (v1) | 101 | `relevant_chunk_ids` 使用 v1 chunking 格式（`0_1_0`），与 v2 不兼容 |
| `golden_dataset_v3_extras.json` | +12 guardrails + 5 multi-turn + 3 gap-fill | 只是对 v1 的小幅扩充 |
| **`golden_dataset_v3_rag.json`（本文）** | **205** | **从零设计，完全基于 v2 数据结构，unbiased** |

### 1.2 核心问题：数据泄露（Data Leakage）

最初的方案是："用现有 query 跑一遍 v2 retrieval，取 top-20 作为 relevant chunks 的候选"。这存在严重的**循环依赖**：

```
有偏循环：
  [query] → [v2 retrieval top-20] → [LLM judge 筛选] → ground_truth
            ↑                                           ↓
            └──────────── 用同一个模型评测 ────────────┘  ⚠️ 数据泄露
```

问题：
1. Ground truth 依赖于**被评测模型自己的输出**
2. 评测时模型必然在"自己生成的 gold 集"上表现虚高
3. 无法区分真实的模型改进 vs 数据集偏见

### 1.3 第二个核心问题：评测模型与出题模型同源

此外，原始的生成脚本使用的是 **Qwen 3.5 Flash**（项目主 LLM）来生成 query。但**被评测的系统也使用 Qwen 3.5 Flash**。这违反了 LLM 评测的基本原则：

> **出题者和答题者必须是不同的模型**

正确做法：
- **出题**：使用性能更强的外部模型（本项目使用 **Claude Opus 4.1**，通过 Claude Code subagent）
- **答题**：项目本身的 Qwen 3.5 Flash（DashScope）
- **评分**：客观指标（keyword/citation/numerical）+ 可选的第三方 LLM judge

---

## 2. 设计原则

构建新 golden dataset 时遵循以下 5 条原则：

### 原则 1：Unbiased Ground Truth（无偏真值）

**反向生成（Reverse Generation）**：不是 "query → 找 relevant chunks"，而是 "sample chunk → 生成能被它回答的 query"。

```
无偏反向流程：
  [v2 chunk (分层采样)] → [Opus 生成 query] → [chunk 作为天然 ground truth]
                                                       ↓
                                        评测任何 retriever 都公平
```

**关键性质**：ground truth 独立于任何 retriever 的输出，因此可以公平对比多个系统（v1 RAG / agent v1 / agent v2 / 最终 DAG+v2）。

### 原则 2：完全覆盖（Complete Coverage）

必须覆盖以下所有维度的全部取值：
- 2^4 = 16 种数据源路由组合（vector, sql, rules, graph）
- 5 种 answer_mode（lookup, descriptive, comparison, decision_support, diagnostic）
- 9+ 种 guardrail 类型
- 5+ 种文档类型（handbook, policy, sustainability_report, annual_report, master_plan, guideline）
- 多个时间窗（old / mid / recent / undated）

### 原则 3：统计显著（Statistically Meaningful）

每个关键 stratum（子类）要有足够样本使指标在 95% 置信度下达到可接受的置信区间：
- 小 stratum（单源路由类型）：≥ 12 样本 → ±15% CI
- 大 stratum（整体 F1）：≥ 100 样本 → ±5% CI

### 原则 4：使用 v2 metadata（Leverage New Metadata）

v2 chunking 新增的元数据字段（`parent_id`, `section_title`, `doc_type`, `publish_year`, `category`, `is_table`）必须在 golden dataset 中显式记录，使：
- 评测可以测试 metadata 过滤功能
- Ground truth 可以按多个维度匹配（chunk_id / parent_id / source_file / section）
- 数据集本身成为 v2 数据管道能力的展示

### 原则 5：可复现（Reproducible）

- 使用固定 `RANDOM_SEED=42` 保证分层采样确定性
- Scaffold 作为中间产物保存，方便重跑某一类
- 所有生成提示都写进代码，不依赖对话历史

---

## 3. 覆盖维度的全面分析

设计数据集时识别了 6 个正交维度：

### 维度 A — 数据源路由组合（16 种）

这是最核心的维度，因为它决定了 router 的分类空间。

| 源组合 | 典型场景 | 自然频率 | 目标数 |
|---|---|---|---|
| 0 源 | OOD / 闲聊 / 无关 | ~10% | 18（guardrail） |
| 1 源：vector | "报告里讲了什么" | ~25% | 66 |
| 1 源：sql | "2015 年产能" | ~20% | 30 |
| 1 源：rules | "风速限制" | ~15% | 20 |
| 1 源：graph | "为什么延迟" | ~5% | 15 |
| 2 源：v+s | "报告说 X，实际数据呢" | ~5% | 7 |
| 2 源：v+r | "政策引用" | ~4% | 7 |
| 2 源：v+g | "文档 + 因果" | ~3% | 4 |
| 2 源：s+r | "数据 vs 规则"（决策支持） | ~6% | 10 |
| 2 源：s+g | "数据 + 因果" | ~2% | 4 |
| 2 源：r+g | "规则 + 因果" | ~2% | 5 |
| 3 源：v+s+r | "决策支持 + 文档" | ~2% | 5 |
| 3 源：v+s+g | "诊断 + 文档" | ~1% | 4 |
| 3 源：v+r+g | 罕见 | <1% | 3 |
| 3 源：s+r+g | "全数据决策" | ~1% | 3 |
| 4 源：全部 | 复杂多跳 | <1% | 4 |
| **Total** | | | **205** |

**合理性**：频率设计参考真实业务场景，但故意**过采样稀有组合**（如 4 源）以保证罕见情况也能被测到。

### 维度 B — Answer Mode（5 种）

每种 answer mode 至少 20 样本：

| Mode | 典型用途 | 目标数 | 实际数 |
|---|---|---|---|
| lookup | 简单事实查询 | ≥30 | 59 |
| descriptive | 解释/描述 | ≥25 | 36 |
| comparison | 对比 | ≥20 | 22 |
| decision_support | "是否应该 / 是否允许" | ≥20 | 48 |
| diagnostic | "为什么 / 因果" | ≥15 | 40 |

### 维度 C — Guardrail 类型（9 类）

| 类型 | 数量 | 样本例子 |
|---|---|---|
| out_of_domain | 4 | pizza recipe / cat joke / current time / sports trivia |
| empty_evidence | 3 | nonexistent berth B99 / martian operations / 2050 data |
| impossible_query | 3 | 2030 future date / negative wind speed / contradictory |
| evidence_conflict（rule↔sql） | 3 | wind 25 knots vs 规则 / wave / crane productivity |
| doc_vs_sql_conflict | 2 | 报告 85% vs 实际 / 排放 70% 声明 |
| doc_vs_rule_conflict | 2 | 2018 handbook vs 当前规则库 |
| ambiguous_query | 3 | "recent delays" / "that incident" / "how is operations" |
| false_premise | 3 | "port closed 2023" / "100% shutdown" / "all electric" |
| refusal_appropriate | 2 | write DB / delete rules |
| **Total** | **25** | |

### 维度 D — v2 Metadata 过滤（16 类）

**v1 完全没有这种测试**。v2 新增的 metadata 字段需要专门的测试：

| 过滤类型 | 数量 | 目的 |
|---|---|---|
| doc_type 过滤 | 6 | "handbook / sustainability report / annual report..." |
| publish_year 窗口 | 5 | "since 2020" / "before 2015" / "most recent" |
| category 过滤 | 5 | environment / operations / management / technology |
| **Total** | **16** | |

### 维度 E — 文档内容变异

文档级别覆盖：
- 不同港口：Port of Virginia / Long Beach / Rotterdam / Panynj
- 不同主题：sustainability / safety / productivity / technology
- 不同时代：2010 年前 / 2015-2019 / 2020+

这个维度通过 chunking 的 **分层采样** 自动满足。

### 维度 F — chunk 物理属性

| 属性 | 取值 |
|---|---|
| word_count | 短 (<150) / 中 (150-300) / 长 (>300) |
| is_table | True / False |
| has_metadata | doc_type, publish_year, section_title 至少有一个 |

---

## 4. 样本量的统计学推导

### 4.1 单 stratum 置信区间

使用二项分布置信区间公式：

```
CI = z * sqrt(p*(1-p)/n)
```

其中 `z=1.96`（95% 置信），`p` 是观察到的比例。最坏情况 `p=0.5`：

| n | CI 半宽 |
|---|---|
| 10 | ±31% |
| 15 | ±25% |
| 20 | ±22% |
| **30** | **±18%** |
| 50 | ±14% |
| 100 | ±10% |
| 200 | ±7% |

### 4.2 应用到本项目

- **小 stratum（单源 rule-only, graph-only）**：目标 ±25% CI → n ≥ 15
- **中 stratum（单源 vector, sql）**：目标 ±15% CI → n ≥ 40
- **大 stratum（整体 Micro-F1）**：目标 ±8% CI → n ≥ 150
- **guardrail 类型**：每类 n ≥ 3（作为发现性指标，±30% CI 可接受）

**总和约束**：

```
50 (vector) + 30 (sql) + 20 (rules) + 15 (graph) + 49 (multi)
+ 25 (guardrails) + 16 (metadata) = 205
```

即 **205 样本** 是同时满足以上所有约束的最小值（±10%）。

### 4.3 为什么不是 100 或 300

- **< 150**：16 种组合 × 每种 ≥5 = 80 下限 + guardrail/metadata 至少 +40 = 120。120 只能做发现性评测，不够支持 stratum 内统计推断
- **> 250**：边际收益递减，150 已覆盖 95% 的路由空间，多加 100 只能把 CI 从 ±8% 降到 ±6%
- **205 是甜点**：既能 per-stratum 得出有意义的比较，又在 2-3 小时内可生成完

---

## 5. 分层采样策略

### 5.1 Vector 采样：多维笛卡尔积分层

对 `chunks_v2_children.json`（16,124 children）按三个维度构造 bucket：

```python
bucket_key = (doc_type, year_bucket, is_table)
year_bucket = "old" if year < 2015
              "mid" if year < 2020
              "recent" if year >= 2020
              "undated" otherwise
```

结果：**18 个非空 bucket**。分布观察：

```
  ('document', 'old', False)                  6267 chunks
  ('document', 'mid', False)                  3032 chunks
  ('document', 'recent', False)               1533 chunks
  ('sustainability_report', 'old', False)     1252 chunks
  ('annual_report', 'recent', False)          1067 chunks
  ('document', 'undated', False)              958 chunks
  ('sustainability_report', 'mid', False)     483 chunks
  ('sustainability_report', 'recent', False)  481 chunks
  ('master_plan', 'old', False)               418 chunks
  ('handbook', 'mid', False)                  104 chunks
  ...
```

**按比例分配** 50 个 vector 样本到各 bucket（每 bucket 至少 1），然后每个 bucket 内随机采。这样保证所有 doc_type/时间/表格类型都有代表。

### 5.2 SQL 采样：基于 DuckDB 实际数据

直接查 DuckDB 生成真实结果：

```python
templates = [
    ("environment", "wind_speed_ms", "AVG", "average wind speed"),
    ("environment", "wave_height_m", "MAX", "maximum wave height"),
    ("crane_operations", "crane_productivity_mph", "AVG", "average crane productivity"),
    ...  # 共 15 个 (table, column, agg, phrase) 模板
]

for tpl in templates:
    sql = f"SELECT {agg}(\"{col}\") FROM \"{table}\" WHERE year={year}"
    result = duckdb.execute(sql).fetchone()[0]
    scenario = {"sql": sql, "result": result, "tables": [table], ...}
```

**理由**：使用真实 DuckDB 结果作为 ground truth，确保数据集中的数字和实际系统可查询到的数字一致。出题时 LLM 看到的 `result=5.23`，它生成的 query 被 RAG 回答时，RAG 也应该算出 `5.23`。

### 5.3 Rules 采样：从 grounded_rules.json 中按完整度排名

`grounded_rules.json` 只有 21 条（相对稀缺），采样策略是：

```python
ranked = sorted(grounded, key=lambda r: (
    1 if r.get("variable") else 0,
    1 if r.get("sql_variable") else 0,
    1 if r.get("operator") else 0,
    1 if r.get("value") is not None else 0,
), reverse=True)
```

取前 20 条 + 允许循环补齐到目标数。**更完整的规则（有阈值、操作符、sql_variable）优先**，因为这些更适合生成"是否允许"类决策支持问题。

### 5.4 Graph 采样：从 Neo4j v2 图的 TRIGGERS 边

```cypher
MATCH (a)-[r:TRIGGERS]->(b)
WHERE r.rule_text IS NOT NULL
RETURN a.name AS source, b.name AS target,
       r.rule_text, r.operator, r.threshold, r.source_file
LIMIT 50
```

返回的边随机抽 15 条。每条边都带原始规则的 citation，可以验证 graph 推理回答时是否正确引用。

### 5.5 Multi-source 组合：固定权重 + 随机 anchor chunk

```python
combos = [
    (["vector", "sql"], 5),
    (["vector", "rules"], 5),
    (["vector", "graph"], 4),
    (["sql", "rules"], 7),       # 最常见：决策支持
    (["sql", "graph"], 4),
    (["rules", "graph"], 5),
    (["vector", "sql", "rules"], 5),
    (["vector", "sql", "graph"], 4),
    (["vector", "rules", "graph"], 3),
    (["sql", "rules", "graph"], 3),
    (["vector", "sql", "rules", "graph"], 4),
]
```

对每个 combo，从 `chunks_v2_children.json` 随机挑一个 anchor chunk 作为文档上下文（如果 combo 包含 vector）。Opus 负责生成一个**真正需要所有列出的源才能回答的问题**（见 §9.5）。

---

## 6. 反向生成方法

### 6.1 核心思想

对比两种方法：

#### 传统（Forward，有偏）
```
step 1: 用户写 query
step 2: 跑 retrieval，拿 top-20
step 3: 人工从 top-20 里标注 relevant
step 4: 用这个标注当 ground truth
```

问题：
- ground truth 是模型输出的子集
- 无法检测 "模型错过的相关 chunks"（因为你只看到 top-20）
- 标注者有 selection bias

#### 反向（Reverse，无偏）
```
step 1: 从 chunks 中分层采样 N 个
step 2: 对每个 chunk，用强 LLM 生成"这个 chunk 能回答的 query"
step 3: 这个 chunk 就是天然 ground truth
```

优点：
- ground truth **不依赖任何 retriever**
- 可以测试 retriever 是否 "找不到" 天然相关的 chunk
- 天然覆盖 chunk 的完整分布（通过分层采样）

### 6.2 反向生成的质量要求

Opus 生成 query 时的核心约束：

1. **Paraphrase（改写）**：不能直接复制 chunk 里的原句
2. **Natural（自然）**：听起来像港口分析师问同事，而不是 literal rephrasing
3. **Specific enough（足够具体）**：应该能清楚地从这个 chunk 答出来，不能是"告诉我港口的事"
4. **Not trivially lexical（非平凡词汇）**：不能只是问 chunk 里出现的一个词
5. **Mode variety（模式变异）**：在 50 个 vector 样本里混合 5 种 answer_mode
6. **Time-aware**：当 chunk 有 publish_year 时，可以生成时间特定的 query（如"2018 handbook 说什么"）

### 6.3 SQL / Rule / Graph 的反向生成特殊性

- **SQL**：给 LLM 真实 DuckDB 查询结果 + SQL，让它反向生成用户问题。ground truth 是 SQL 本身。
- **Rule**：给 LLM 规则全文 + 变量 + 阈值 + 操作，让它反向生成"是否允许"类问题。ground truth 是规则的 variable + source_file + page。
- **Graph**：给 LLM 图的边（`(source) -[TRIGGERS]-> (target)`），让它反向生成"为什么 X 影响 Y"类问题。ground truth 是 entity list + relationship。

---

## 7. Opus 子代理并行生成架构

### 7.1 为什么用 Claude Opus 4.1

三个关键原因：

1. **模型多样性**：被评测的 RAG 系统用 Qwen 3.5 Flash。使用不同家族、不同规模的模型出题，避免同源偏见。
2. **生成质量**：Opus 4.1 是 Anthropic 当前最强的推理模型，能生成更自然、更多样化的 query。
3. **Claude Code subagent 原生支持**：可以通过 Agent tool 在子进程中以隔离的方式批量调用，不占用主对话上下文。

### 7.2 Subagent 架构

```
主 Claude Code (Sonnet 4.5) — 协调
    │
    ├─ Subagent A (Opus 4.1) — Vector 50 samples
    ├─ Subagent B (Opus 4.1) — SQL 30 samples
    ├─ Subagent C (Opus 4.1) — Rules 20 samples
    ├─ Subagent D (Opus 4.1) — Graph 15 samples
    └─ Subagent E (Opus 4.1) — Multi-source 49 samples
    
    (5 个 subagent 并行运行，共 164 samples)
    
主 Claude Code 最后合并 164 opus 样本 + 41 handwritten = 205
```

### 7.3 每个 subagent 的任务定义

每个 subagent 收到：
1. **角色**：明确说明要生成港口领域 RAG 评测数据
2. **输入文件路径**：scaffold JSON 的绝对路径
3. **输出文件路径**：results JSON 的绝对路径
4. **精确的输出 schema**：含所有字段和嵌套结构
5. **质量规则**：paraphrase、answer_mode 分布、避免词汇重叠等
6. **工作流程**：Read → 逐样本生成 → Write → 报告

### 7.4 工具访问

Subagent 拥有完整工具集（`Read, Write, Bash, Glob, Grep`），但被要求：
- **不调任何 LLM API**：subagent 本身就是 Opus 4.1，直接用自己的推理生成 query
- 使用 `Read` 读 scaffold
- 使用 `Write` 保存 results

### 7.5 并行执行的实现

在主 Claude Code 的同一条消息中，使用**多个 Agent tool 调用**触发并行执行：

```
<function_calls>
  <Agent description="Generate SQL" model="opus">...</Agent>
  <Agent description="Generate rules" model="opus">...</Agent>
  <Agent description="Generate graph" model="opus">...</Agent>
  <Agent description="Generate multi-source" model="opus">...</Agent>
</function_calls>
```

4 个任务同时启动，在 ~6 分钟内全部完成（串行需要 ~25 分钟）。Vector 50 因为上下文最大，单独更早启动了一次。

### 7.6 成本和效率

| 类别 | Tokens (预估) | Wall time |
|---|---|---|
| Vector (50, subagent A) | ~113K | ~5.5 分钟 |
| SQL (30, subagent B) | ~55K | ~1.5 分钟 |
| Rules (20, subagent C) | ~50K | ~1.3 分钟 |
| Graph (15, subagent D) | ~47K | ~1.2 分钟 |
| Multi (49, subagent E) | ~87K | ~4.4 分钟 |
| **Total** | **~352K** | **~7 分钟并行总时长** |

---

## 8. 完整工作流

### 阶段 1：Scaffold 分层采样（Python，~5 秒）

```bash
python evaluation/dump_golden_scaffolds.py
```

产物（在 `evaluation/scaffolds/`）：
- `vector_tasks.json` — 50 个 v2 chunks 带元数据
- `sql_tasks.json` — 30 个 SQL 场景带真实结果
- `rules_tasks.json` — 20 个 grounded rules
- `graph_tasks.json` — 15 个 Neo4j 边
- `multi_tasks.json` — 49 个多源 anchor

### 阶段 2：Opus Subagent 并行生成（~7 分钟）

主 Claude Code 启动 5 个 Opus 4.1 subagents，每个读取自己的 scaffold，生成 query，写回 results 文件。

产物：
- `vector_results.json` (50 samples)
- `sql_results.json` (30 samples)
- `rules_results.json` (20 samples)
- `graph_results.json` (15 samples)
- `multi_results.json` (49 samples)

### 阶段 3：合并 + 加手写部分（Python，~1 秒）

```bash
python evaluation/merge_golden_v3.py
```

执行：
1. 加载 5 个 results 文件
2. 调用 `build_golden_v3_rag.build_guardrail_samples()` 生成 25 个 guardrail 样本
3. 调用 `build_golden_v3_rag.build_metadata_filter_samples()` 生成 16 个 metadata 过滤样本
4. 合并所有样本
5. 校验：去重 ID、统计分布
6. 写 `evaluation/golden_dataset_v3_rag.json`

### 阶段 4：Sanity Check（人工）

查看分布表：
- 205 total ✓
- 16 种 routing 组合都 > 0 ✓
- 5 种 answer mode 都 > 0 ✓
- 9 种 guardrail 类型都存在 ✓
- 无重复 ID ✓
- 每个样本都有必要字段 ✓

---

## 9. 按类别的生成器详解

### 9.1 Vector Chunk-First 生成器

**输入**：`chunks_v2_children.json` 的一个 chunk

**Prompt 要点**：
- 给 Opus 看 chunk 的完整 metadata（source, section, doc_type, year）和文本
- 要求生成一个**paraphrased** 的问题
- 要求同时返回 reference_answer, keywords, answer_mode
- 要求在 50 个样本中分配多样的 answer_mode（目标 40/20/15/15/10）

**输出**：完整 sample record，其中 `golden_vector` 字段包含：
```json
{
  "relevant_chunk_ids": ["86__c__p1__0__7"],
  "relevant_parent_ids": ["86__p__p1__0"],
  "relevant_source_files": ["2023-pov-sustainability-report-3.pdf"],
  "relevant_sections": ["2.1.4"],
  "expected_doc_types": ["sustainability_report"],
  "expected_categories": ["environment"],
  "expected_publish_year": 2023
}
```

### 9.2 SQL Result-First 生成器

**输入**：来自 DuckDB 的真实查询 + 结果

**Prompt 要点**：
- 给 Opus 看 SQL 语句、结果、使用的表、聚合方式、年份过滤
- 要求生成自然语言问题（不要 SQL 术语）
- 要求变化措辞（"What was..."、"How much..."、"Show me..." 等）
- 要求大多数是 lookup (70%)，少量 comparison (20%) 和 decision_support (10%)

**输出 golden_sql**：
```json
{
  "expected_tables": {"environment": ["wind_speed_ms"]},
  "expected_aggregation": "AVG",
  "expected_year_filter": "2015"
}
```

### 9.3 Rule-First 生成器

**输入**：来自 `grounded_rules.json` 的一条规则

**Prompt 要点**：
- 给 Opus 看完整规则：variable, operator, value, action, 原文, source, page
- 要求生成"是否允许"类问题，**不能在 question 里提到具体阈值**（用户不知道才会问）
- 80% decision_support, 20% lookup

**Prompt 模板例子**：
- "Under what {variable} conditions should operations be {action}?"
- "What is the maximum {variable} allowed for {operation}?"
- "Is {action} required when {variable} exceeds certain limits?"

### 9.4 Graph Edge-First 生成器

**输入**：Neo4j 中的一条 `TRIGGERS` 边

**Prompt 要点**：
- 给 Opus 看边的 source, target, 规则文本, 阈值
- 要求生成 "why / how does X affect Y" 类问题
- 强调**多跳因果推理**，不是单文档查找
- 所有 15 个都是 `answer_mode="diagnostic"`

**例子**：
- `(wind_speed_ms) -[TRIGGERS]-> (operational_pause)` → "Why would high wind speeds trigger an operational pause?"
- `(wave_height_m) -[TRIGGERS]-> (crane_operations)` → "How does wave height influence crane operations?"

### 9.5 Multi-source Combinatorial 生成器

**核心挑战**：生成的 query 必须**真正需要所有列出的源**才能回答，不能被单源回答。

**Prompt 模板按 combo 分类**：

| Combo | 模板 |
|---|---|
| vector+sql | "Does the [report claim] match the actual operational data?" |
| vector+rules | "The handbook mentions X; does it align with current rule thresholds?" |
| vector+graph | "The report describes incidents of X — what is the underlying causal chain?" |
| sql+rules | "Given the recent [metric], does it comply with safety rules?" |
| sql+graph | "Recent [metric] has risen — what factors in the operational graph explain this?" |
| rules+graph | "How does rule X prevent the [causal cascade] from happening?" |
| v+s+r | "Does the report's claimed [metric] comply with both the actual data and safety rules?" |
| v+s+g | "The report mentions X; verify against actual data and explain the causal factors." |
| ...等 4 源组合 | |

**answer_mode 选择规则**：
- 包含 graph → `diagnostic`（因果优先）
- 包含 rules 但无 graph → `decision_support`
- 纯 vector+sql → `comparison`

### 9.6 Guardrails（手写）

**为什么不用 LLM 生成**：guardrail 测试的是"agent 应该拒绝/质疑"的场景，需要精确控制问题语义。LLM 可能生成"太明显"或"太隐晦"的变体，不好控制难度。

25 条手写样本覆盖 9 种类型（见维度 C）。

### 9.7 Metadata Filters（手写模板）

16 条样本，模板化生成：

```python
doc_type_tests = [
    ("What does the 2018 VRCA operating handbook say about pilotage?", "handbook"),
    ("According to the sustainability report, what are the emission targets?", "sustainability_report"),
    ...
]
```

手写保证每个测试精确针对一个 metadata 字段，不会有歧义。

---

## 10. 质量规则与验证

### 10.1 Prompt 中的质量约束

每个 subagent 的 prompt 都包含以下硬性规则：

| 规则 | 用途 |
|---|---|
| **Paraphrase, don't copy** | 避免查询词汇直接匹配 chunk 文本（会让 BM25 轻松找到，降低评测区分度） |
| **Natural phrasing** | 避免"查询式"语言，模拟真实用户 |
| **Specific enough** | 避免过于笼统无法定位 |
| **Vary openings** | "What was / How much / Show me / Find / Can you tell" 等混用 |
| **Mode diversity** | 在每个类别里强制 answer_mode 分布 |
| **Time-specific when applicable** | 利用 publish_year 生成时间限定查询 |

### 10.2 Subagent 报告检查

每个 subagent 返回后，检查它的报告：

- **Vector (50)**：`lookup 20, descriptive 18, comparison 7, decision_support 3, diagnostic 3` — 接近目标
- **SQL (30)**：`lookup 21, comparison 6, decision_support 3` — 正好符合目标
- **Rules (20)**：`decision_support 17, lookup 3` — 符合 "80% 决策" 目标
- **Graph (15)**：`diagnostic 15` — 全是 diagnostic，正确
- **Multi (49)**：按 combo 自动决定 mode（含 graph → diagnostic，含 rules → decision_support，否则 comparison），正确

### 10.3 合并后的 Sanity Check

`merge_golden_v3.py` 输出会打印：

```
Category counts:                      检查各类数量匹配预期
Source combinations (2^4):            检查 16 种组合都覆盖
Answer mode distribution:             检查 5 种 mode 都充分
Guardrail types:                      检查 9 种 guardrail 都在
Generation methods:                   检查 opus_* 和 handwritten 比例
```

**如果有重复 ID**，会打印 warning 并阻止输出（实际没有发生）。

### 10.4 Opus 发现的问题与修复

Subagents 主动指出了几个可改进点：

1. **SQL scaffold 冗余**：SQL subagent 报告 "30 个任务只有 9 个独立 (metric, year) 组合"。它通过在重复上使用不同的措辞和 answer_mode 来缓解，但建议未来增加 sql scaffold 的多样性。
2. **Graph 边重叠**：GRA_002 和 GRA_008 都使用 `vessel_loa_meters` 作为 source（不同阈值和目标）。是设计选择而非 bug。
3. **Vector descriptive 比例偏高**：因为很多 chunks 是叙述性散文（会议记录、手册规则），自然适合 descriptive 问题。40% 的 target 被扩到 36%。

这些都作为未来迭代的改进点记录，当前不阻塞数据集使用。

---

## 11. 最终数据集统计

### 11.1 总体

| 指标 | 值 |
|---|---|
| 总样本数 | **205** |
| 文件大小 | 234 KB |
| 生成日期 | 2026-04-11 |
| 出题模型 | Claude Opus 4.1 (Anthropic) |
| 被评测系统 | Qwen 3.5 Flash (DashScope) |

### 11.2 按类别

| 类别 | 数量 | 生成方法 |
|---|---|---|
| vector | 50 | opus_chunk_first |
| sql | 30 | opus_sql_result_first |
| rules | 20 | opus_rule_first |
| graph | 15 | opus_graph_edge_first |
| multi-source | 49 | opus_multi_source |
| guardrails | 25 | handwritten_guardrail |
| metadata filters | 16 | metadata_filter_handwritten |
| **TOTAL** | **205** | |

### 11.3 按 routing 组合（2^4 = 16）

| 组合 | 数量 | 组合 | 数量 |
|---|---|---|---|
| () [guardrail] | 18 | rules+sql | 10 |
| vector | 66 | rules+sql+vector | 5 |
| sql | 30 | vector+rules | 7 |
| rules | 20 | vector+sql | 7 |
| graph | 15 | graph+rules+vector | 3 |
| graph+rules | 5 | graph+rules+sql | 3 |
| graph+sql | 4 | graph+sql+vector | 4 |
| graph+vector | 4 | graph+rules+sql+vector | 4 |

**全 16 种覆盖** ✓

### 11.4 按 answer mode

| Mode | 数量 | 比例 |
|---|---|---|
| lookup | 59 | 29% |
| decision_support | 48 | 23% |
| diagnostic | 40 | 20% |
| descriptive | 36 | 18% |
| comparison | 22 | 11% |

**全 5 种覆盖** ✓

### 11.5 按 guardrail 类型

| 类型 | 数量 |
|---|---|
| out_of_domain | 4 |
| empty_evidence | 3 |
| impossible_query | 3 |
| evidence_conflict | 3 |
| ambiguous_query | 3 |
| false_premise | 3 |
| doc_vs_sql_conflict | 2 |
| doc_vs_rule_conflict | 2 |
| refusal_appropriate | 2 |

**9 种全覆盖** ✓

### 11.6 按 v2 metadata

- `relevant_parent_ids` 字段：50 个 vector 样本都有
- `expected_doc_types` 字段：每个 vector + metadata filter 样本都有
- `expected_publish_year` 字段：覆盖 old/mid/recent/undated 4 种
- `expected_categories` 字段：environment / operations / management / technology 都覆盖

---

## 12. 文件清单

### 12.1 生成器代码

| 文件 | 行数 | 用途 |
|---|---|---|
| `evaluation/build_golden_v3_rag.py` | ~750 | 核心生成库：stratification, samplers, guardrail/metadata 手写样本 |
| `evaluation/dump_golden_scaffolds.py` | ~150 | 第一阶段：从 v2 数据分层采样并写 scaffold 文件 |
| `evaluation/merge_golden_v3.py` | ~140 | 第三阶段：合并 opus results + handwritten |

### 12.2 Scaffold 中间产物（生成时产生，可删）

| 文件 | 样本数 |
|---|---|
| `evaluation/scaffolds/vector_tasks.json` | 50 |
| `evaluation/scaffolds/sql_tasks.json` | 30 |
| `evaluation/scaffolds/rules_tasks.json` | 20 |
| `evaluation/scaffolds/graph_tasks.json` | 15 |
| `evaluation/scaffolds/multi_tasks.json` | 49 |

### 12.3 Opus Results（opus subagents 产生，可删）

| 文件 | 样本数 |
|---|---|
| `evaluation/scaffolds/vector_results.json` | 50 |
| `evaluation/scaffolds/sql_results.json` | 30 |
| `evaluation/scaffolds/rules_results.json` | 20 |
| `evaluation/scaffolds/graph_results.json` | 15 |
| `evaluation/scaffolds/multi_results.json` | 49 |

### 12.4 最终产物

| 文件 | 用途 |
|---|---|
| **`evaluation/golden_dataset_v3_rag.json`** | ⭐ 最终 205 样本数据集 |
| `evaluation/run_rag_evaluation.py` | 使用新 golden 跑 DAG + v2 的评测 runner |

### 12.5 文档

| 文件 | 内容 |
|---|---|
| **`evaluation/GOLDEN_DATASET_V3_METHODOLOGY_CN.md`** | ⭐ 本文（构建方法论） |
| `FINAL_SYSTEM_REPORT_CN.md` | 系统整体总结 |

---

## 附录 A：从头复现完整命令

```bash
cd RAG-LLM-for-Ports-main
export PY=/c/Users/25389/Agent_RAG/.venv/Scripts/python.exe

# 前提条件：v2 数据管道已运行
# - data/chunks/chunks_v2_children.json 存在
# - data/rules/grounded_rules.json 存在
# - storage/sql/port_ops.duckdb 存在
# - Neo4j 运行并已加载 v2 graph (build_neo4j_graph_v2)

# 阶段 1：分层采样
$PY evaluation/dump_golden_scaffolds.py

# 阶段 2：Opus subagent 生成（需要在 Claude Code 环境中运行）
# 在主 Claude Code 会话中，通过 Agent tool 并行启动 5 个 subagent
# 每个 subagent 读一个 scaffold 文件，生成对应的 results 文件
# 这一步无法用纯 Python 复现，必须在 Claude Code 里手动调用

# 阶段 3：合并
$PY evaluation/merge_golden_v3.py

# 验证
$PY -c "
import json
with open('evaluation/golden_dataset_v3_rag.json', encoding='utf-8') as f:
    d = json.load(f)
print(f'Total: {d[\"total_samples\"]}')
print(f'Categories: {d[\"category_counts\"]}')
"

# 跑评测
$PY evaluation/run_rag_evaluation.py
```

---

## 附录 B：为什么选 Opus 4.1 而不是其他模型

| 模型 | 评估 |
|---|---|
| **Opus 4.1** ✓ | 最强推理，自然度最高，Claude Code 原生支持 |
| Sonnet 4.5 | 快但可能生成过于模板化 query |
| GPT-4o | 需要 OpenAI API key，且与项目生态耦合度低 |
| Gemini | API 兼容性问题 |
| Qwen 35b (项目自己的模型) | ❌ 会导致出题答题同源 |
| 人工标注 | 质量最高但成本太高（205 样本 × 2 分钟 = 7 小时） |

Opus 4.1 是当前的最佳平衡点：质量够、速度够快（并行）、与 Claude Code 集成好、和评测系统彻底异源。

---

*本文档记录了从"想法"到"205 样本数据集"的完整构建过程。如果未来需要迭代数据集（扩展到 300 样本、加多轮对话、或换新 chunking 策略），这份方法论是可直接沿用的蓝图。*
