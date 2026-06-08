# 基于 LLM 的产业分析报告生成系统

根据多地区多维考核指标 Excel 数据，自动完成**数据分析 → 报告撰写 → 文本润色**全流程，输出结构化报告。

## 架构概览

```
用户输入地区名 (如 "渝北区")
      │
      ▼
 [1] 数据分析 (data_analysis)
      ├─ 读取考核评估总表 (data/overview_data/)
      ├─ 自动查找 data/detailed_data/ 下匹配地区名的补充材料 Excel
      ├─ describe_dataframes_schema() 生成表结构描述
      ├─ Planning LLM 根据 schema 生成 N 条查询指令
      └─ CodeAgent 逐条执行查询 → analysis_result (str)
      │
      ▼
 [2] 报告撰写 (doc_writing)
      └─ Writing LLM 根据 analysis_result + 原始 DataFrame → draft
      │
      ▼
 [3] 文本改写 (rewriting)
      └─ Rewriting LLM 对 draft 润色 → final_report
      │
      ▼
 保存至 output/{地区名}_报告.md
```

三个阶段使用**独立的 LLM 实例**，可分别配置不同模型和参数。

## 项目结构

```
├── main.py                      # 主入口，串联三阶段流程
├── data_analysis.py             # 数据分析：查找补充材料 + LLM 生成查询 + CodeAgent 执行
├── doc_writing.py               # 报告撰写：DocWriter 类
├── rewriting.py                 # 文本改写：Rewriter 类
├── code_agent.py                # smolagents CodeAgent 封装
│
├── llm/
│   ├── __init__.py
│   └── llm.py                  # LLM 通用基类 (BaseLLM / OpenAILikeLLM / LLMConfig)
│
├── utils/
│   ├── data_inspector.py       # DataFrame schema 描述 + AI 查询 + MCP Tool 包装
│   ├── file_io.py              # read_all_excel / data_save
│   ├── prompt_renderer.py      # Jinja2 模板渲染器
│   ├── prompts.py              # CodeAgent 读取指令生成
│   ├── temp_file.py            # 变量序列化到临时文件
│   ├── logger.py               # 日志（终端 + 文件）
│   └── helper.py               # 预留
│
├── prompts/                     # Jinja2 模板 (.j2)
│   ├── data_analysis_system.j2
│   ├── data_analysis_user.j2
│   ├── doc_writing_system.j2   # 待填充
│   ├── doc_writing_user.j2
│   └── rewriting_system.j2     # 待填充
│
├── data/
│   ├── overview_data/           # 考核评估总表
│   ├── detailed_data/           # 各地区补充材料 (如 渝北区-25-06.xlsx)
│   └── test_data/               # 测试数据
│
├── output/                      # 生成的报告输出目录
├── logs/                        # 按日期滚动的日志文件
└── requirements.txt
```

## 快速开始

### 1. 环境配置

```bash
conda create -n mlenv python=3.13
conda activate mlenv
pip install -r requirements.txt
```

### 2. 配置环境变量

在项目根目录创建 `.env` 文件：

```env
API_BASE_DEFAULT=https://your-api-endpoint/v1
API_KEY_DEFAULT=your-api-key
MODEL_DEFAULT=your-default-model-name
```

### 3. 配置高级模型（可选）

编辑 `main.py` 中的 `_create_writing_llm()` 和 `_create_rewriting_llm()`，将 `PLACEHOLDER_*` 替换为实际的闭源模型配置：

```python
def _create_writing_llm() -> OpenAILikeLLM:
    return OpenAILikeLLM(config=LLMConfig(
        model="实际模型名",
        api_base="https://实际API地址/v1",
        api_key="实际API密钥",
        temperature=0.7,
    ))
```

### 4. 准备数据

- 将考核评估总表 Excel 放入 `data/overview_data/`
- 将各地区补充材料 Excel 放入 `data/detailed_data/`，**文件名必须包含地区名原文**（如 `渝北区-25-06.xlsx`）

### 5. 运行

```bash
# 命令行参数方式
python main.py 渝北区

# 交互输入方式
python main.py
# > 请输入地区名称: 渝北区
```

报告将保存至 `output/渝北区_报告.md`。

## 核心模块说明

### LLM 基类 (`llm/llm.py`)

提供 OpenAI-compatible API 的统一封装，支持所有兼容 OpenAI 格式的服务。

| 类 / 函数 | 说明 |
|-----------|------|
| `LLMConfig` | 连接与生成参数（model / api_base / api_key / temperature / 重试策略等），优先级：显式传参 > 环境变量 > 默认值 |
| `BaseLLM` | 抽象基类，提供 `chat()` / `generate()` / `stream()` / `batch()` + 自动重试 + 对话历史管理 |
| `OpenAILikeLLM` | 基于 openai SDK 的具体实现，支持同步 / 异步 / 流式调用 |
| `create_llm()` | 工厂函数，快速创建实例 |

> **注意**：`code_agent.py` 中的 `MyCodeAgent` 使用 `smolagents` + `LiteLLMModel`（独立于本 LLM 模块）。

### 数据检查 (`utils/data_inspector.py`)

| 函数 / 类 | 说明 |
|-----------|------|
| `describe_dataframes_schema(dfs)` | 将 `{sheet_name: DataFrame}` 字典转为结构化文本描述（表头层级、列名、数据类型、示例值、唯一值），支持 MultiIndex |
| `query_dataframes(dfs, instruction)` | 利用 `MyCodeAgent` 根据自然语言指令生成并执行 Python 代码查询 DataFrame |
| `inspect_and_query(file_path, instruction)` | 一站式接口：读取 Excel → 生成 schema → AI 查询 |
| `DataInspectorMCPTool` | MCP Tool 包装器，统一暴露 `describe` / `query` / `inspect` 三个 action |

### Prompt 模板系统 (`utils/prompt_renderer.py`)

使用 **Jinja2** 模板引擎管理所有 prompt，模板文件存放在 `prompts/` 目录下。

**容错机制**：
- 模板引用了但调用方未传入的变量 → **替换为空字符串 + 警告日志**（不报错）
- 调用方传入了但模板中未使用的多余变量 → **警告日志**（不报错）

```python
from utils.prompt_renderer import render_prompt

text = render_prompt(
    "data_analysis_user.j2",
    region_name="渝北区",
    assessment_schema="...",
    max_queries=5,
)
```

模板支持完整 Jinja2 语法（`{{ 变量 }}`、`{% if %}`、`{% for %}`、过滤器等）。

### 文件读写 (`utils/file_io.py`)

| 函数 | 说明 |
|------|------|
| `read_all_excel(file_path, sheet_name, header)` | 灵活读取 Excel 多 sheet，支持 MultiIndex 表头、自动前向填充、Unnamed 列名清理 |
| `data_save(data, file_path, file_type)` | 保存 DataFrame / str / dict / list 到文件，支持 xlsx / csv / json / txt / md / html，文件名冲突自动加后缀 |

### 临时文件序列化 (`utils/temp_file.py`)

为 `CodeAgent` 传递变量设计。根据变量类型自动选择最优序列化格式：

| 变量类型 | 文件格式 | 备注 |
|----------|---------|------|
| `DataFrame`（普通列） | `.parquet` | 高效列式存储 |
| `DataFrame`（MultiIndex 列） | `.pkl` | parquet 不支持 MultiIndex，自动回退 pickle |
| `ndarray` | `.npy` | numpy 原生格式 |
| `list` / `dict` | `.json` | JSON 序列化 |
| `str` | `.txt` | 纯文本 |

## 环境变量

| 变量名 | 用途 | 默认值 |
|--------|------|--------|
| `API_BASE_DEFAULT` | LLM API 地址 | — |
| `API_KEY_DEFAULT` | LLM API 密钥 | — |
| `MODEL_DEFAULT` | 默认模型名称 | `siliconflow/Qwen/Qwen3-8B`（CodeAgent 回退值） |

## 核心依赖

| 用途 | 包 |
|------|---|
| LLM 调用 | `openai`, `litellm` |
| Agent 框架 | `smolagents` |
| 数据处理 | `pandas`, `numpy`, `openpyxl`, `pyarrow`, `fastparquet` |
| 模板引擎 | `Jinja2` |
| 环境变量 | `python-dotenv` |

最后两个模块都应该可以在单次LLM调用下完成，主要任务是Prompt Engineering。暂时先不考虑。主要要做的是，顺利引导Orchestrator观察-思考给定的城区在各个指标上的表现情况，然后给出一系列可计算的统计学问题。SQL和DA两个模块如何使用CodeAgent完成，需要仔细考虑，其中细节还需要修改。