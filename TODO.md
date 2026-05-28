# MyDataStorm - TODO & Known Issues

## Pending Features

### 1. Recommendation 生成能力

**现状**: DataSTORM 只生成数据分析 insights，不生成 actionable recommendations。
InsightBench 的 GT 中有 3/8 条（G6, G7 部分）是建议性内容（如 "Regular Updates and Maintenance", "Proactive Monitoring"）。

**需要做的**:
- 在 Report Generation 阶段（或 Insight Condensation 阶段）增加一个 recommendation 生成步骤
- 基于已有 insights 生成 2-3 条 actionable recommendations
- 可以作为 `_extract_insights()` 的一部分，或者单独一个 prompt

**优先级**: Medium — 对 InsightBench 评分有直接影响（~3/8 GT 条目需要 recommendations）

---

### 2. Thesis-driven 过滤过于激进

**现状**: `INSIGHT_BANK_FILTER` 在有 thesis 时会 deprioritize "off-topic" findings，导致与 thesis 方向不同但客观重要的发现被丢弃（例：Hardware 增长趋势被 "Priority Neglect" thesis 淘汰）。

**可能方案**:
- 在 InsightBank 中保留一个 "non-thesis" 槽位（如 max 50 中保留 10 给 thesis-unrelated findings）
- 或者在 insight condensation 时不用 thesis 过滤，而是从原始 executor responses 直接提取
- 或者让 thesis refinement 更积极地 Pivot

**优先级**: High — 这是当前 flag-4 低分的主要原因之一

---

### 3. 文本分析能力

**现状**: Executor 只做 SQL 查询，不做 NLP/文本分析。InsightBench GT 中有 "Specific hardware issues are predominantly mentioned in incident descriptions"（G8）这类需要分析 short_description 文本内容的 insight。

**可能方案**:
- 在 Planner prompt 中提示：如果有文本列（如 description），应该生成关键词分析/主题提取的问题
- 在 execute_python_from_sql 中支持简单的文本分析（已有 nltk, wordcloud 等包）
- 或者在 warm-start 阶段自动做一轮文本列的词频统计

**优先级**: Medium

---

### 4. 基础切片强制探索

**现状**: Layer 1 的 Planner 完全自由决定探索方向。如果它一开始就锁定了某个 thesis 方向，后续所有层都会在同一方向打转。

**可能方案**:
- Layer 1 强制生成 "按各维度切片" 的基础问题：按 category / priority / agent / time 各跑一个 GROUP BY COUNT
- 这些基础统计作为 Layer 2+ 的前置 context，让 Planner 能看到全局分布再做决策
- 类似于 EDA (Exploratory Data Analysis) 的自动化第一步

**优先级**: High — 能直接解决 "漏掉 Hardware 增长" 这类问题

---

## Known Issues

- Planner 生成问题数已做硬截断（代码层面 `[:max_q]`），但 prompt 里的 "up to N" 对 deepseek-v4-pro 约束力不足
- `execute_python_from_sql` 的 sandbox 不允许 `import` 语句（安全限制），agent 需要使用预注入的变量名（np, pd, stats 等）
- Warm-start 阶段因为 Serper API 禁用，只基于 dataset description 臆想 insights（对 InsightBench 场景是纯噪声）
