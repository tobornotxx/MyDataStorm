"""DataSTORM 所有 Prompt 模板。

严格 1:1 对应论文附录 Table 5 - Table 23 中的所有 Prompt。
每个常量名标注了对应的论文表格编号和 Prompt 编号。

Prompt 编号与论文表格对应关系:
- Prompt 5  → Table 5:  探索问题生成 (后续层)
- Prompt 6  → Table 6:  探索直接 SQL 生成
- Prompt 7  → Table 7:  查询一致性模块
- Prompt 8  → Table 8:  洞察库过滤
- Prompt 9  → Table 9:  论点生成
- Prompt 10 → Table 10: 论点精炼
- Prompt 11 → Table 11: 初始问题生成 (第一层)
- Prompt 12 → Table 12: 大纲生成
- Prompt 13 → Table 13: 章节草稿
- Prompt 14 → Table 14: 引用验证
- Prompt 15 → Table 15: 章节修订
- Prompt 16 → Table 16: 最终润色
- Prompt 17 → Table 17: 参考诱导评估标准生成
- Prompt 18 → Table 18: 参考诱导评估标准评分
- Prompt 19 → Table 19: 原子分解
- Prompt 20 → Table 20: 洞察归因
- Prompt 21 → Table 21: InsightBench 评估
- Prompt 22 → Table 22: Executor 主 Prompt
- Prompt 23 → Table 23: InsightBench 摘要
"""

# =============================================================================
# Prompt 5 (Table 5): 探索问题生成 Prompt — 后续层使用
# Planner 基于全局洞察库和当前论点生成新的探索问题
# =============================================================================
EXPLORATION_QUESTIONS_GENERATION = """\
# instruction
You are an analytical reasoning engine that explores a relational database. Your goal is to discover
surprising or meaningful insights. Your task
now is to ask new questions based on the table returned. A list of global insights is provided to you. You
should NOT ask questions that are
already covered by the global insights.

Generate EXACTLY {{ max_questions }} questions to further explore relevant topics.
- You MUST generate exactly {{ max_questions }} questions — no more, no less.
- Make each question self-contained and clearly scoped.
- Each question should investigate one specific aspect.
- Cover diverse dimensions (time trends, category breakdown, agent performance, correlations, anomalies).

For each question, also specify a "destination" to indicate where the question should be routed:
- "database": The question can be answered by querying the database (e.g., aggregations, distributions,
  trends, filters, correlations, rankings,
  or any computation over the data).
- "internet": The question requires external context NOT available in the database.

Most questions should be "database" - only use "internet" when the answer genuinely cannot come from the
database.

Output a JSON object with EXACTLY this structure:
{
  "chain_of_thought": "your reasoning about what aspects to explore",
  "questions": [
    {"question": "...", "destination": "database"},
    ... (EXACTLY {{ max_questions }} items)
  ]
}

# input
Description of database content: {{ db_description }}

Global insights: {{ global_insights }}

Conversation history: {{ dialogue_turns }}

Topic/Question you are writing: {{ topic }}

{% if thesis %}
You are building evidence for the following thesis: "{{ thesis }}"

Research strategy: {{ research_strategy }}

Prioritize questions that help build, test, or refine this argument. You may also ask questions that
challenge or qualify the thesis - strong analysis addresses counter-arguments.
{% endif %}
"""

# =============================================================================
# Prompt 5b: 双模式问题生成 — 基于问题树的跟进 + 探索
# 每一步基于已有问题提出至少 m 个跟进问题 + 至少 n 个全新探索方向,
# agent 自主决定每层最终提几个问题 (m/n 为下限, 每类上限5个)。
# =============================================================================
TREE_BASED_QUESTION_GENERATION = """\
# instruction
You are an analytical reasoning engine exploring a relational database.
Your goal is to discover surprising, meaningful, and previously unknown insights.

You have access to:
1. A QUESTION TREE showing all previously asked questions, organized by layer,
   with their answers and results.
2. A list of GLOBAL INSIGHTS that have been extracted from those answers.
3. A research THESIS (if one has been formed).

Your task is to generate TWO groups of questions:

## Group A: FOLLOW-UP questions (AT LEAST {{ m }}, AT MOST 5 questions)
- These should deepen EXISTING lines of investigation found in the question tree.
- Look at questions that revealed interesting partial results or anomalies,
  and drill deeper into those findings.
- Each follow-up question MUST reference which existing question(s) it extends,
  using the "parent_ids" field with question node IDs from the tree.
- Follow-up questions should be more specific and targeted than their parents.
- Generate more than {{ m }} if the data suggests many lines of investigation
  are worth pursuing deeper.

## Group B: EXPLORATORY questions (AT LEAST {{ n }}, AT MOST 5 questions)
- These should open ENTIRELY NEW investigation directions NOT covered by existing questions.
- Cover different dimensions: time trends, category breakdowns, correlations,
  anomalies, cross-variable interactions, distributions, outliers.
- Ensure these questions are self-contained and independent.
- Exploratory questions should have an empty "parent_ids" list.
- Generate more than {{ n }} if there are many unexplored dimensions in the data.

## Routing
For each question, specify a "destination":
- "database": answerable by querying the database
- "internet": requires external context (use sparingly)

## Rules
- You MUST generate AT LEAST {{ m }} follow-up questions AND AT LEAST {{ n }} exploratory questions.
- Each type MUST NOT exceed 5 questions. If there aren't enough good questions, fewer is fine — but never exceed 5.
- Do NOT ask questions already covered by existing questions or global insights.
- Ensure diversity across all generated questions.
- Make each question clear, specific, and scoped to one aspect.
- Be decisive: generate the number of questions that is appropriate for the current state
  of exploration. Do NOT pad with weak questions just to hit a number.

{% if thesis %}
## Current Thesis
"{{ thesis }}"
Research strategy: {{ research_strategy }}
Prioritize questions that help build, test, or refine this argument.
Also ask questions that challenge or qualify the thesis - strong analysis addresses counter-arguments.
{% endif %}

Output a JSON object with this structure:
{
  "chain_of_thought": "your reasoning about the question tree, gaps, and new directions",
  "follow_up_questions": [
    {
      "question": "...",
      "destination": "database",
      "parent_ids": ["<node_id_from_tree>", ...]
    }
  ],
  "exploratory_questions": [
    {
      "question": "...",
      "destination": "database",
      "parent_ids": []
    }
  ]
}

# input
Description of database content: {{ db_description }}

Topic: {{ topic }}

QUESTION TREE (all previously asked questions with their answers):
{{ question_tree }}

GLOBAL INSIGHTS extracted so far:
{{ global_insights }}
"""

# =============================================================================
# Prompt 6 (Table 6): 探索直接 SQL 生成 Prompt
# 用于生成带 filter/groupby 的 SELECT * SQL, 供汇总统计使用
# =============================================================================
EXPLORATION_DIRECT_SQL = """\
# instruction

You are an analytical reasoning engine that explores a relational database. Your goal is to discover
surprising or meaningful insights. Your task now is to isolate a view of the tables (with filters
and/or groupbys) on which I will compute summary statistics for all columns to derive interesting
insights. Generate a total of 1-{{ max_questions }} number of questions. You don't have to generate the
maximum number of questions.

In each question:
- You should return a SQL of form: `SELECT * ...` based on filters and groupbys identified in the past round.
- For instance, if in the previous turn you have identified that categroy A has the most amount of
  population, then in this turn you might want to investiate `SELECT * FROM category = A.` Note here you
  should not select from multiple cateogories because summary statistics will be computed based on the
  result.
- First generate your reasoning and then generate the actual SQL.
- The database is PostgreSQL, so make sure to respect the syntax, such as wrapping tables in double quotes if
  the table name contains upper case letters.

# input
Description of database content: {{ db_description }}

Conversation history: {{ dialogue_turns }}

Topic/Question you are writing: {{ topic }}
"""

# =============================================================================
# Prompt 7 (Table 7): 查询一致性模块 Prompt
# 检测并修正并行 Executor 运行产生的 SQL 谓词不一致
# =============================================================================
QUERY_CONSISTENCY = """\
# instruction
You are given a list of SQL responses related to the same topic. Your task is to:
1. identify any inconsistencies in the SQL predicates used and standarize any inconsistencies. For the nodes you would like to correct, issue a follow-up question with the desired SQL predicates. You can directly instruct what to modify in the SQLs. DO NOT instruct new variables not seen in the current SQL. DO NOT instruct it correct any variables.
2. Some noes will be given to you as examples. These examples will be marked with "example_node": True, and you do not need to issue a follow-up question for them.
3. make sure the SQLs reflect the conversation context presented in previous_queries. If any SQL appears to have forgotten the conversational context, issue a follow-up question to resolve it.
4. If no follow-up question is needed, set "follow_up_question": None.

Output a JSON following examples.

# input
{
  "example_node_0": {
    "query": "Show me the top 20 countries by the number of missile or artillery attacks that they have targetted by?",
    "SQL": "SELECT country, COUNT(*) AS attack_count FROM events WHERE sub_event_type IN ('Shelling/artillery/missile attack') GROUP BY country ORDER BY attack_count DESC LIMIT 20;",
    "example_node": True,
    "note": "no need to generate follow_up_question"
  },
  "query0": {
    "previous_queries": None,
    "query": "What specific regions or countries in the Middle East have seen the most significant increase in ISIS-related activities in 2025?",
    "SQL": "SELECT region, country, COUNT(*) AS event_count FROM events WHERE year = 2025 AND region = 'Middle East' AND (actor1 LIKE '%ISIS%' OR actor2 LIKE '%ISIS%' OR assoc_actor_2 LIKE '%ISIS%') GROUP BY region, country ORDER BY event_count DESC;"
  },
  "query1": {
    "previous_queries": None,
    "query": "How have shifts in geopolitical alliances, such as Israel's potential normalization with Saudi Arabia, influenced ISIS activities in the Middle East during 2025?",
    "SQL": "SELECT * FROM events WHERE year = 2025 AND region = 'Middle East' AND (actor1 LIKE '%ISIS%' OR actor2 LIKE '%ISIS%' OR assoc_actor_2 LIKE '%ISIS%');"
  }
}

# output
{
  "query0": {
    "query": "What specific regions or countries in the Middle East have seen the most significant increase in ISIS-related activities in 2025?",
    "follow_up_question": None
  },
  "query1": {
    "query": "How have shifts in geopolitical alliances, such as Israel's potential normalization with Saudi Arabia, influenced ISIS activities in the Middle East during 2025?",
    "follow_up_question": "Please include actor1 LIKE '%ISIS%' OR assoc_actor_1 LIKE '%ISIS%' OR actor2 LIKE '%ISIS%' OR assoc_actor_2 LIKE '%ISIS%' to be consistent with query0's actor filtering pattern."
  }
}

# input
{{ input }}
"""

# =============================================================================
# Prompt 8 (Table 8): 洞察库过滤 Prompt
# 从候选洞察中选择最相关的, 更新全局洞察库
# =============================================================================
INSIGHT_BANK_FILTER = """\
# instruction
You are given a list of candidate insights derived from database exploration on a topic.
Your task is to select the most valuable insights, capped at {{ max_num_insights }}.

## Selection criteria (in priority order):
1. **Anomalies and surprises**: findings that reveal unexpected patterns, outliers, or counter-intuitive results (e.g., one category growing while others shrink, a metric behaving opposite to expectation)
2. **Significant trends**: statistically significant changes over time, strong correlations, or clear distributional skews
3. **Actionable findings**: observations that point to root causes or have clear operational implications
4. **Breadth of coverage**: prefer a diverse set of insights covering different dimensions (time, category, agent, priority) over many insights about the same dimension

## De-prioritize:
- Redundant findings that repeat what another selected insight already says
- Trivial or expected observations (e.g., "data has 500 rows", "5 categories exist")
- Findings where the analysis failed or produced no result

The topic is: {{ topic }}

The database context: {{ db_description }}

{% if thesis %}
## Current working thesis (for context, NOT as a filter):
"{{ thesis }}"
Note: Do NOT filter out findings just because they don't support the thesis.
Surprising findings that contradict or are orthogonal to the thesis are often the most valuable.
{% endif %}

Output a JSON dict, where each key is a node_id and the value is the insight for that node_id.

# input
{{ input }}
"""

# =============================================================================
# Prompt 9 (Table 9): 论点生成 Prompt
# 在探索 p 层后, 从洞察库中生成候选论点
# =============================================================================
THESIS_GENERATION = """\
# instruction

You are a senior analyst at a world-class publication (think The Economist, Foreign Affairs, or
FiveThirtyEight). You have been given a general
topic and a batch of findings produced by a preliminary data exploration agent.

Your job is NOT to describe what the data shows. Your job is to REASON about what the findings mean - to
identify non-obvious patterns, causal
claims, counter-narratives, strategic implications, or surprising tensions - and to distill them into
compelling, defensible thesis statements.

Each thesis should be the kind of bold, original argument that could anchor a top-tier analytical article
written for a general audience.

Generate at most 3 thesis candidates.

Rules:
- Each thesis is a CONCISE TITLE - maximum 10 words. Think magazine cover line or op-ed headline, NOT a full
  sentence or a data summary.
- A good thesis takes a POSITION. It argues something. It should be possible to disagree with it. Avoid bland
  descriptive titles like "Trends in
  X" or "Overview of Y."
- Do NOT embed statistics, numbers, or data citations in the thesis title.
- The thesis should capture a non-obvious, thought-provoking argument that would make an informed reader want
  to read the full article.
- For each thesis, provide a research_strategy: a concrete plan for how a writer should develop this argument
  into a full analytical article.
  Specify what evidence to marshal, what comparisons to draw, what counter-arguments to address, what narrative
  structure to follow, and what
  conclusions to build toward. This will be handed to a downstream research agent that will write the article.
- If the findings in this batch don't support 3 strong theses, output fewer. Quality over quantity.

# input
Description of database content: {{ db_description }}

Topic: {{ topic }}

Below are findings from a preliminary data exploration on this topic. Reason about what these findings reveal
- the patterns, tensions, and
implications - and propose up to 3 thesis statements that could each serve as the central argument of a
top-tier analytical article.

{{ context }}
"""

# =============================================================================
# Prompt 10 (Table 10): 论点精炼 Prompt
# 根据新证据精炼 (Sharpen / Pivot / Confirm) 当前论点
# =============================================================================
THESIS_REFINEMENT = """\
# instruction

You are a senior analyst at a world-class publication (think The Economist, Foreign Affairs, or
FiveThirtyEight).

You previously proposed a working thesis to guide research on a topic. Since then, a research agent has
gathered
additional findings from the database. Your task is to re-examine that thesis in light of the new evidence and
decide whether to:

1. Sharpen - narrow or deepen the original argument using new supporting evidence
2. Pivot - shift to a better-supported or more compelling argument uncovered by the new findings
3. Confirm - keep the thesis essentially unchanged if the evidence continues to support it strongly

Output exactly one refined thesis and the updated research strategy.

# input
Description of database content: {{ db_description }}

Topic: {{ topic }}

Current Thesis: {{ current_thesis }}

Current Research strategy: {{ current_research_strategy }}

Current findings:
{{ context }}
"""

# =============================================================================
# Prompt 11 (Table 11): 初始问题生成 Prompt — 第一层使用
# 从预热报告 r₀ 生成初始探索问题
# =============================================================================
INITIAL_QUESTIONS_GENERATION = """\
You are conducting research on a goal/topic: "{{ topic }}". The goal here is to extract previously unknown
insights by exploring and observing the information in the database with the following description: {{ db_description }}.

Generate EXACTLY {{ num_questions }} questions that an investigator will be interested in.
- You MUST generate exactly {{ num_questions }} questions — no more, no less.
- The questions will be used to generate search queries in the database to help answer them.
- The questions should be self-contained (include any specific years, months, locations, etc.)
  and related to the goal/topic: "{{ topic }}".
- Each question should investigate one specific aspect. Do not include too many subquestions inside a single question.
- The questions should be independent of each other.

CRITICAL: Layer 1 questions MUST prioritize fundamental EDA (Exploratory Data Analysis). Include these types:
1. Distribution/breakdown by key categorical columns (e.g., "What is the distribution of X across categories?")
2. Time trends (e.g., "How does the volume/metric change over time? Is there an increasing or decreasing trend?")
3. Correlations or lack thereof between key numeric variables (e.g., "Is there a correlation between volume and resolution time?")
4. Per-group comparisons (e.g., "Is performance/metric uniform across all agents/groups, or does one stand out?")

Do NOT jump to advanced analysis (statistical tests, Gini coefficients, outlier detection) in Layer 1.
Start with basic counts, averages, and trend lines. Advanced analysis belongs in later layers.

Output a JSON object with EXACTLY this structure:
{
  "questions": [
    {"question": "...", "destination": "database"},
    ... (EXACTLY {{ num_questions }} items)
  ]
}

{% if article %}
Here is more background information on the goal/topic based on the internet: "{{ article }}".
{% endif %}
"""

# =============================================================================
# Prompt 12 (Table 12): 大纲生成 Prompt — 报告 Stage A
# =============================================================================
OUTLINE_GENERATION = """\
# instruction

You are planning a publication-ready analytical narrative report.
Design a single flowing report that proves the central claim through evidence.
TREE evidence is the core spine; external context should enrich but not replace it.
This is for readers, not a thesis committee.

Rules:
- The thesis is an internal organizing claim, not a standalone section heading.
- Do not plan separate "Thesis" or "Key Findings" sections.
- Use a narrative arc: opening hook -> escalation/mechanism -> geography/human stakes -> implications.
- Each section should advance the story and hand off naturally to the next.
- Every section must still materially support, test, or sharpen the thesis.
- Prefer 4 substantive sections, each capable of roughly 450-700 words.
- For each section, provide at most 3 web queries in web_queries. Queries must be specific and aimed at
  authoritative sources (major NGOs, official documents, major outlets), not generic search phrases.

Return JSON only with this schema:
{
  "lede_strategy": "...",
  "key_findings": ["...", "..."],
  "sections": [
    {
      "section_id": "S1",
      "heading": "...",
      "purpose": "...",
      "must_include_evidence_ids": [1,2],
      "key_points": ["..."],
      "storytelling_moves": ["..."],
      "web_queries": ["query 1", "query 2"]
    }
  ],
  "closing_strategy": "..."
}

# input
TOPIC:
{{ topic }}

THESIS:
{{ thesis }}

TITLE PACKAGE:
title: {{ title }}
subtitle: {{ subtitle }}
editorial_angle: {{ editorial_angle }}

CORE EVIDENCE NOTES:
{{ note_digest }}

WARMSTART CONTEXT (optional background hints):
{{ warmstart_text }}

VALID EVIDENCE IDS (must_include_evidence_ids must use only these):
{{ valid_ids }}

Create the report plan now.
"""

# =============================================================================
# Prompt 13 (Table 13): 章节草稿 Prompt — 报告 Stage B
# =============================================================================
SECTION_DRAFT = """\
# instruction
You are writing one section of a publication-ready analytical narrative report. Think of medias such as New
York Times or the Economist.

Rules:
- TREE evidence is the core spine: prioritize it and explicitly use it.
- Supplemental web context can provide background, reactions, and scene-setting.
- Every factual claim must have inline citation(s) in [N] format.
- Use only citation numbers from ALLOWED_CITATIONS.
- Do not invent citations or facts.
- Do not repeat the section heading inside section_markdown.
- Do not open with phrases like "This section" or "This evidence". Lead with the most consequential finding.
- Move from data -> mechanism -> human stakes -> repercussions.
- Quant style: write numbers like a reporter with evidence, not a methods appendix.
- Avoid in-line statistical jargon (e.g., "mean", "std", "p-value", "significant", "contemporaneous",
  "lagging", "correlation
  coefficient", "r=") unless the coefficient itself is the only faithful representation of the evidence.
- Prefer plain-language comparatives first ("about twice as high", "tracked closely", "rose sharply"), then
  give the exact
  numbers in a second clause or sentence.
- If you include r/lag/etc., translate immediately in plain language and avoid stacking multiple
  coefficients in one sentence.
- Prefer one numeric claim per sentence; avoid dense parenthetical math.
- Avoid meta signposting like "as later sections will detail"; use a natural bridge sentence instead.
- For each evidence citation provided, include at least one substantive use tied to that citation.
- If the web packet contains clearly relevant authoritative sources, use at least 1-2 web citations for
  context or external
  validation. Do not force weak web sources.
- Prefer smooth prose over bullets or mini-subheadings; use internal subheadings only if truly necessary.
- End with a forward-driving sentence that naturally sets up the next section.
- Do not add any Sources/References/Citations section.

Return JSON only:
{
  "section_id": "...",
  "heading": "...",
  "section_markdown": "...",
  "used_citations": [1,2,3]
}

# input
TOPIC:
{{ topic }}

THESIS:
{{ thesis }}

REPORT TITLE:
{{ report_title }}

SECTION SPEC:
- section_id: {{ section_id }}
- heading: {{ heading }}
- purpose: {{ purpose }}
- key_points: {{ key_points }}
- storytelling_moves: {{ storytelling_moves }}

ALLOWED_CITATIONS:
{{ allowed }}

CORE TREE EVIDENCE (mandatory):
{{ core_packet }}

SUPPLEMENTAL WEB CONTEXT (optional):
{{ web_packet }}

TARGET SECTION LENGTH:
{{ target_words }} words (hard ceiling - stay under this)

Draft this section now. Keep it highly analytical, readable, and citation-grounded.
"""

# =============================================================================
# Prompt 14 (Table 14): 引用验证 Prompt — 报告 Stage C
# =============================================================================
CITATION_GROUNDING = """\
You are a precise fact-checker for data journalism reports.

You will be given a SENTENCE taken from a report and the SOURCE text that the sentence cites. Your job is to
determine whether every factual claim in the
SENTENCE is supported (entailed) by the SOURCE. You will be given the context leading up to the citation.

Rules:
- Set is_entailed to true only if every factual claim in the SENTENCE can be directly verified from the
  SOURCE. Minor rephrasing or summarisation is fine
  as long as nothing contains factual errors. For instance, fatality vs. incident count would be a factual
  difference.
- Set is_entailed to false if the SENTENCE adds, omits, or distorts any fact relative to the SOURCE.
- If is_entailed is false, concisely identify the issue in your output (one sentence).

DO NOT flag:
- Reasonable interpretations or paraphrases of the evidence
- Stylistic differences or summarization

# input
SENTENCE:
{{ sentence }}

SOURCE:
{{ sources }}
"""

# =============================================================================
# Prompt 15 (Table 15): 章节修订 Prompt — 报告 Stage D
# =============================================================================
SECTION_REVISION = """\
# instruction
You are revising one section of a publication-ready analytical narrative report.

You will be given the previous draft and a list of criticisms. Each criticism identifies a specific sentence
that
makes a claim not supported by the cited evidence.

Rules:
- Fix ONLY the criticized sentences. Do not rewrite or restructure anything else.
- For each criticism, either:
  a) Rewrite the sentence to remove or qualify the unsupported claim, keeping any supported parts intact, or
  b) Remove the sentence entirely if no part of it is supportable.
- Keep all citations that remain accurate. Do not add new citations outside ALLOWED_CITATIONS.
- Do not invent new facts.
- Preserve the section's structure, flow, and all uncriticized content verbatim.
- Do not add a Sources/References/Citations section.

Return JSON only:
{
  "section_id": "...",
  "heading": "...",
  "section_markdown": "...",
  "used_citations": [1,2,3]
}

# input
TOPIC:
{{ topic }}

THESIS:
{{ thesis }}

REPORT TITLE:
{{ report_title }}

SECTION SPEC:
- section_id: {{ section_id }}
- heading: {{ heading }}
- purpose: {{ purpose }}
- key_points: {{ key_points }}
- storytelling_moves: {{ storytelling_moves }}

ALLOWED_CITATIONS:
{{ allowed }}

CORE TREE EVIDENCE (mandatory):
{{ core_packet }}

SUPPLEMENTAL WEB CONTEXT (optional):
{{ web_packet }}

PREVIOUS DRAFT:
{{ previous_draft }}

CRITICISMS:
{{ criticisms }}

Revise the section now. Change only what the criticisms require.
"""

# =============================================================================
# Prompt 16 (Table 16): 最终润色 Prompt — 报告 Stage E
# =============================================================================
FINAL_POLISH = """\
# instruction
You are a senior editor polishing a near-final publication-ready report draft. Think of medias such as New
York Times or the Economist.

Rules:
- Preserve and improve analytical flow, and add smooth transitions between sections.
- Preserve or expand substance; do not compress the draft into a summary.
- Keep all existing valid citations; do not invent new citation numbers.
- Use only citation numbers listed in ALLOWED_CITATIONS.
- Keep markdown headings and publication-ready prose.
- Do NOT add Sources/References/Citations section (it will be appended programmatically).
- The final polished report body must not exceed {{ target_total_words }} words (excluding the sources
  appendix). Cut ruthlessly for concision while preserving every cited claim.
- You should include a conclusion section at the end.
- Do not explictly include a "thesis" block

Return JSON only:
{
  "report_markdown": "..."
}

# input
TOPIC:
{{ topic }}

THESIS:
{{ thesis }}

TITLE:
{{ title }}
{{ subtitle }}

PLAN (for structural intent):
{{ plan_json }}

ALLOWED_CITATIONS:
{{ allowed_citations }}

DRAFT REPORT:
{{ draft_markdown }}
"""

# =============================================================================
# Prompt 17 (Table 17): 参考诱导评估标准生成 Prompt
# =============================================================================
REFERENCE_CRITERIA_GENERATION = """\
# instruction

You are an expert analyst. Given a reference research article, extract a list of evaluation criteria
describing
what analytical points a good report on this topic should cover. Focus on general trends and patterns - do not
reference specific numbers, dates, or proper nouns that would make the criteria too narrow.

Read the reference article carefully. Identify the key analytical points it makes - the insights, trends, and
conclusions that a thorough report on this topic should include.

For each criterion:
1. Give it a short **name** (3-6 words)
2. Write a **description** of the general trend or pattern to look for (1-2 sentences, no specific numbers or
   dates needed but include e.g. the general trend)

Return as a JSON object with a "criteria" array, each item having "name" and "description" fields.

# input

## Research Task
{{ task_prompt }}

## Reference Article
{{ reference_article }}
"""

# =============================================================================
# Prompt 18 (Table 18): 参考诱导评估标准评分 Prompt
# =============================================================================
REFERENCE_CRITERIA_GRADING = """\
# instruction

You are an expert evaluator of analytical research articles. Given a set of evaluation criteria and a
generated
article, grade how well the article addresses each criterion.

For each criterion, assess how well the generated article addresses it on a 0.0-1.0 scale:
- **1.0** - Fully addresses: the article clearly covers this analytical point
- **0.75** - Mostly addresses: covered but with gaps or insufficient depth
- **0.5** - Partially addresses: touches on it but misses key aspects
- **0.25** - Barely addresses: only a brief or tangential mention
- **0.0** - Not addressed: completely absent from the article

Return as a JSON object with fields: criterion_scores (array of {name, score, explanation}).

# input

{% if score_reminder %}
{{ score_reminder }}
{% endif %}

## Research Task
{{ task_prompt }}

## Evaluation Criteria
{{ criteria }}

## Generated Article
{{ generated_article }}
"""

# =============================================================================
# Prompt 19 (Table 19): 原子分解 Prompt
# =============================================================================
ATOMIC_BREAKDOWN = """\
Given an input article, your task is to break down the insights in the article into itemized points. Each
insight should be self-contained.

Input article: {{ article }}
"""

# =============================================================================
# Prompt 20 (Table 20): 洞察归因 Prompt
# =============================================================================
INSIGHT_ATTRIBUTION = """\
You are an expert analyst evaluating whether a piece of evidence from a generated article is derived from
ACLED (Armed Conflict
Location & Event Data) data.

Article topic: {{ article_topic }}

Evidence to classify: {{ evidence }}

ACLED data includes:
- Conflict event counts, incident reports, and event descriptions
- Violence against civilians statistics
- Battle-related data (battles, explosions/remote violence, riots, protests)
- Fatality counts and casualty figures from conflict events
- Geographic conflict data (locations of events, subnational breakdowns)
- Conflict trend analysis and temporal patterns derived from event data
- Armed group activity and actor-level data
- Conflict index scores (e.g., ACLED Conflict Index)
- Data explicitly attributed to ACLED or its datasets

NOT ACLED data:
- General geopolitical analysis or commentary not tied to specific event data
- Economic indicators (GDP, inflation, trade figures)
- Humanitarian statistics from UN agencies (UNHCR refugee counts, OCHA displacement figures) unless
  explicitly tied to ACLED
- Demographic or census data
- Policy statements, diplomatic actions, or government declarations
- Media reports or journalistic analysis without specific conflict event data
- Academic or think-tank analysis not grounded in ACLED event data
- Data from other conflict databases (e.g., UCDP, GTD, IISS) unless attributed to ACLED
"""

# =============================================================================
# Prompt 21 (Table 21): InsightBench 评估 Prompt
# =============================================================================
INSIGHTBENCH_EVAL = """\
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Provided Answer:
{{ answer }}

Ground Truth Answer:
{{ gt_answer }}

Follow these instructions when writing your response:
* On a scale of 1-10, provide a numerical rating for how close the provided answer is to the ground truth answer, with 10 denoting that the provided answer is the same as ground truth answer.
* Your response should contain only the numerical rating. DONOT include anything else like the provided answer, the ground truth answer, or an explanation of your rating scale in your response.
* Check very carefully before answering.
* Follow the output format as shown in the example below:

Example response:
<rating>7</rating>

### Response:
"""

# =============================================================================
# Prompt 22 (Table 22): Executor 主 Prompt — ReAct 风格数据库代理
# =============================================================================
EXECUTOR_MAIN = """\
# instruction
Your task is to write a **{{ database_type }}** query to answer the given question.

- User question is contextual. If needed, the current date is {{ curr_date }}
- Do NOT repeat the same action, as the results will be the same.
- Output one final SQL at the end that contains all results (not multiple ";" separated queries).
- For stats/visualizations, call execute_python_from_sql AFTER you have determined the final SQL query.

IMPORTANT: BE EFFICIENT. You have a LIMITED number of turns. Once you have a SQL result that
answers the question, call stop() IMMEDIATELY.

{% if db_description %}
# Database Schema & Environment (already provided — go directly to execute_sql)
{{ db_description }}
{% endif %}

The required output format is EXACTLY:
Thought: <your reasoning>
Action: <action_name>(<arguments>)

Do NOT wrap your response in markdown code blocks or add any other formatting.

Possible actions are:
- get_tables(): Retrieves all tables (skip this if schema is provided above).
- retrieve_tables_details([table_names]): Retrieve column details (skip if schema is provided above).
- execute_sql(sql): Runs a SQL query and returns results.
- execute_python_from_sql(sql, python_code): Executes Python on the SQL result DataFrame.
  Available: numpy (np), pandas (pd), scipy, scipy.stats (stats). Variable: sql_results (DataFrame).
  Example: execute_python_from_sql("SELECT * FROM t", "print(sql_results.describe())")
- stop(): Marks the last executed SQL as final answer. CALL THIS when you have the result.

# input
{% if conversation_history %}
Prior turn contexts:
--
{% for turn in conversation_history %}
User Question: {{ turn.question }}
Action history:
{% for action in turn.action_history %}
{{ action }}
{% endfor %}
Agent Response: {{ turn.response }}
--
{% endfor %}
{% endif %}

Current-turn User Question: {{ question }}

{% if action_history %}
Action history:
{% for action in action_history %}
{{ action }}
{% endfor %}
{% endif %}

Output one "Thought" and one "Action":
"""

# =============================================================================
# Prompt 23 (Table 23): InsightBench 摘要 Prompt
# =============================================================================
INSIGHTBENCH_SUMMARY = """\
You are given a list of data insights derived from a dataset analysis.
Write a concise, coherent paragraph (3-5 sentences) that summarizes the key findings.
Focus on the most important patterns and avoid repetition.

Insights:
{{ insights }}
"""
