# DataSTORM: Deep Research on Large-Scale Databases using EDA and Data Storytelling

基于论文 "DataSTORM: Deep Research on Large-Scale Databases using Exploratory Data Analysis and Data Storytelling" 的复现实现。

## 架构概览

```
datastorm/
├── types.py              # 数据类型定义
├── config.py             # 配置管理
├── llm/
│   └── client.py         # LLM 客户端封装
├── prompts/
│   ├── templates.py      # 所有 Prompt 模板 (1:1 对应论文)
│   └── renderer.py       # Jinja2 模板渲染
├── database/
│   └── connector.py      # PostgreSQL 数据库连接与执行
├── internet/
│   └── search.py         # 互联网搜索接口
├── agents/
│   ├── planner.py        # Planner Agent (高层探索问题生成)
│   └── executor.py       # Executor Agent (ReAct 风格 SQL 执行)
├── modules/
│   ├── warmstart.py      # 阶段1: 互联网预热研究
│   ├── exploration.py    # 阶段2: 多智能体探索框架
│   ├── consistency.py    # 查询一致性检测模块
│   ├── insight_bank.py   # 全局洞察库管理
│   ├── statistics.py     # 自底向上归纳统计
│   ├── thesis.py         # 论点生成与精炼
│   └── report.py         # 阶段3: 报告生成流水线
├── evaluation/
│   └── evaluator.py      # 评估模块
└── pipeline.py           # DataSTORM 主流水线

main.py                   # 程序入口
```

## 系统三阶段

1. **Warm-Start (互联网预热)**: 使用互联网研究生成初始报告 r₀ 和洞察库 B₀
2. **Multi-Agent Exploration (多智能体探索)**: m 层迭代探索，含 Planner-Executor 分解、查询一致性检测、归纳统计、论点驱动
3. **Report Generation (报告生成)**: 大纲 → 章节草稿 → 引用验证 → 章节修订 → 最终润色

## 使用

```bash
pip install -r requirements.txt
cp .env.example .env  # 配置 API keys
python main.py --query "Your research query" --db-url "postgresql://..." 
```
