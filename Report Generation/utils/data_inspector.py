"""
Data Inspector 模块
提供两个核心功能：
1. describe_dataframes_schema: 结构化描述 Excel/DataFrame 的表头和数据类型
2. query_dataframes: 利用 AI Agent 根据自然语言指令查询 DataFrame 数据
"""

import os
from typing import Union, Dict, Any, List, Optional
from pathlib import Path
import pandas as pd

from utils import logger


# ============================================================
# 1. Schema 描述
# ============================================================

def describe_dataframes_schema(
    dfs: Dict[str, pd.DataFrame],
    max_sample_rows: int = 0,
    max_unique_values: int = 0,
) -> str:
    """
    将 read_all_excel 返回的 {sheet_name: DataFrame} 字典，转化为一段结构化的
    文本描述，仅包含每个 sheet 的列名和数据类型。

    Args:
        dfs: read_all_excel 返回的字典，key=sheet名，value=DataFrame
        max_sample_rows: 每列展示的示例值行数，默认 0（不展示）
        max_unique_values: 展示 unique 值的最大数量，默认 0（不展示）

    Returns:
        str: 结构化描述字符串
    """
    lines: List[str] = []
    lines.append("=" * 50)
    lines.append("Excel 文件数据结构概览")
    lines.append(f"共包含 {len(dfs)} 个 Sheet\n")

    for sheet_idx, (sheet_name, df) in enumerate(dfs.items(), 1):
        lines.append(f"【Sheet {sheet_idx}】 \"{sheet_name}\"  行数: {len(df)}  列数: {len(df.columns)}")

        if isinstance(df.columns, pd.MultiIndex):
            n_levels = df.columns.nlevels
            lines.append(f"  表头: MultiIndex({n_levels}层)")
            for col_idx, col in enumerate(df.columns):
                # 转义列名中的换行符，让模型能看到\n并在代码中正确使用
                col_path = " > ".join(str(c).replace('\n', '\\n') for c in col) if isinstance(col, tuple) else str(col).replace('\n', '\\n')
                dtype = str(df.iloc[:, col_idx].dtype)
                col_line = f"    [{col_idx}] {col_path}  ({dtype})"
                if max_sample_rows > 0:
                    sample_vals = _get_sample_values(df.iloc[:, col_idx], max_sample_rows)
                    col_line += f"  示例: {sample_vals}"
                if max_unique_values > 0:
                    nunique = df.iloc[:, col_idx].nunique()
                    if 0 < nunique <= max_unique_values:
                        uniques = df.iloc[:, col_idx].dropna().unique().tolist()
                        uniques = [_truncate_str(v) for v in uniques]
                        col_line += f"  唯一值: {uniques}"
                lines.append(col_line)
        else:
            lines.append("  表头: 单层")
            for col_idx, col_name in enumerate(df.columns):
                dtype = str(df.iloc[:, col_idx].dtype)
                col_name_escaped = str(col_name).replace('\n', '\\n')
                col_line = f"    [{col_idx}] \"{col_name_escaped}\"  ({dtype})"
                if max_sample_rows > 0:
                    sample_vals = _get_sample_values(df.iloc[:, col_idx], max_sample_rows)
                    col_line += f"  示例: {sample_vals}"
                if max_unique_values > 0:
                    nunique = df.iloc[:, col_idx].nunique()
                    if 0 < nunique <= max_unique_values:
                        uniques = df.iloc[:, col_idx].dropna().unique().tolist()
                        uniques = [_truncate_str(v) for v in uniques]
                        col_line += f"  唯一值: {uniques}"
                lines.append(col_line)

        lines.append("")

    # 生成 DataFrame 变量引用指南
    lines.append("数据引用指南:")
    for sheet_name, df in dfs.items():
        lines.append(f'  dfs["{sheet_name}"]  → shape={df.shape}')
    lines.append("")

    return "\n".join(lines)


def _truncate_str(v, max_len: int = 30):
    """如果是字符串且超长则截断，保留原始类型"""
    if isinstance(v, str) and len(v) > max_len:
        return v[:max_len] + '...'
    return v


def _get_sample_values(series: pd.Series, n: int = 3, max_str_len: int = 30) -> str:
    """获取一列的前 n 个非空示例值，格式化为字符串，长文本截断"""
    non_null = series.dropna()
    if len(non_null) == 0:
        return "[全部为空]"
    samples = non_null.head(n).tolist()
    formatted = [repr(_truncate_str(v, max_str_len)) for v in samples]
    null_count = series.isna().sum()
    suffix = ""
    if null_count > 0:
        suffix = f"  (空值数: {null_count})"
    return f"[{', '.join(formatted)}]{suffix}"


# ============================================================
# 2. AI 查询
# ============================================================

def query_dataframes(
    dfs: Dict[str, pd.DataFrame],
    instruction: str,
    schema_str: Optional[str] = None,
    model: str = None,
    api_base: str = None,
    api_key: str = None,
    max_steps: int = 3,
    **agent_kwargs,
) -> str:
    """
    根据自然语言指令，利用 AI Agent 生成并执行代码来查询 DataFrame 数据。

    流程:
        1. 如果未提供 schema_str，自动调用 describe_dataframes_schema 生成
        2. 将 schema_str + 自然语言指令组合成 prompt
        3. 调用 CodeAgent 生成代码操作 DataFrame
        4. 执行代码并返回结果字符串

    Args:
        dfs: read_all_excel 返回的字典，key=sheet名，value=DataFrame
        instruction: 自然语言指令，描述需要查询/分析的内容
        schema_str: 表结构描述字符串（可选，不传则自动生成）
        model: AI 模型标识符，默认从环境变量读取
        api_base: API base URL，默认从环境变量读取
        api_key: API key，默认从环境变量读取
        max_steps: Agent 最大执行步数
        **agent_kwargs: 传递给 CodeAgent 的额外参数（如 temperature, top_p 等）

    Returns:
        str: AI 查询结果的字符串
    """
    from code_agent import create_code_agent

    # 1. 生成或使用已有的 schema 描述
    if schema_str is None:
        schema_str = describe_dataframes_schema(
            dfs, max_sample_rows=3, max_unique_values=8
        )
    
    # 2. 构建完整的 prompt（不包含文件读取指令，读取指令由 Agent 自动生成）
    prompt = _build_query_prompt(schema_str, instruction, dfs)
    logger.info(f"Query prompt constructed, instruction: {instruction}")

    # 3. 初始化 Agent
    # CodeAgent 使用单独的环境变量控制模型，避免与普通 LLM 混用
    code_agent_model_env = os.getenv("CODE_AGENT_MODEL_NAME")
    model_id = model or code_agent_model_env or os.getenv(
        "MODEL_DEFAULT", "siliconflow/Qwen/Qwen3-8B"
    )
    base_url = api_base or os.getenv("API_BASE_DEFAULT")
    key = api_key or os.getenv("API_KEY_DEFAULT")

    agent = create_code_agent(
        model=model_id,
        api_base=base_url,
        api_key=key,
        additional_authorized_imports=["pandas", "numpy", "re", "math", "collections"],
        **agent_kwargs,
    )

    # 4. 将所有 DataFrame 作为 additional_args 传入
    #    key 格式: sheet_<idx> 以避免特殊字符问题
    #    注意: CodeAgent.run() 内部会自动根据 DataFrame 列类型
    #    选择正确的序列化方式（parquet 或 pickle），并生成对应的读取指令
    additional_args = {}
    for idx, (sheet_name, df) in enumerate(dfs.items()):
        var_name = f"sheet_{idx}"
        additional_args[var_name] = df

    # 5. 执行查询
    result = agent.run(
        input=prompt,
        max_steps=max_steps,
        additional_args=additional_args,
    )

    if result is None:
        logger.warning("AI Agent 未返回有效结果")
        return "查询未返回有效结果，请检查指令或数据。"

    return str(result)


def _build_query_prompt(
    schema_str: str,
    instruction: str,
    dfs: Dict[str, pd.DataFrame],
) -> str:
    """
    构建给 AI Agent 的查询 prompt。
    
    注意: 此 prompt 只包含数据结构描述 + 变量映射 + 用户指令 + 编码要求。
    文件读取指令由 CodeAgent.run() 内部通过 get_simple_agent_var_instruction() 自动生成，
    会根据 DataFrame 列类型（普通列用 parquet，MultiIndex 列用 pickle）
    生成正确的读取代码示例，不在此处重复指定，以避免指令冲突。
    """
    # 构建变量映射说明
    var_mapping_lines = []
    for idx, (sheet_name, df) in enumerate(dfs.items()):
        var_name = f"sheet_{idx}"
        is_multi = isinstance(df.columns, pd.MultiIndex)
        col_info = f"MultiIndex({df.columns.nlevels}层)" if is_multi else "单层表头"
        var_mapping_lines.append(
            f'  变量 `{var_name}` → Sheet "{sheet_name}", '
            f"shape={df.shape}, {col_info}"
        )
    var_mapping_str = "\n".join(var_mapping_lines)

    prompt = f"""你是数据分析助手。你必须在一个代码块内完成所有分析，直接调用 final_answer() 返回结果。禁止使用 print()，禁止分步探索。

<数据结构信息>
{schema_str}
</数据结构信息>

<变量映射>
{var_mapping_str}
</变量映射>

<用户指令>
{instruction}
</用户指令>

<注意事项>
- 如果指令中提到的列名不存在，先检查实际列名（可能存在拼写差异），用模糊匹配找到最接近的列
- 如果需要的指标不是现成列（如解决时间），主动从现有列推导（如 closed_at - opened_at）
- 对日期字段先用 pd.to_datetime() 转换
- 对可能为空的列先 .dropna()
</注意事项>

请在一个代码块中：读取数据 → 筛选查询 → 计算统计量 → 用 final_answer(结果字符串) 返回。不要 print，不要分步。
"""
    return prompt


# ============================================================
# 3. 便捷函数：从文件路径直接完成 "读取 → 描述 → 查询" 全流程
# ============================================================

def inspect_and_query(
    file_path: Union[str, Path],
    instruction: str,
    sheet_name=None,
    header=0,
    model: str = None,
    **kwargs,
) -> str:
    """
    一站式接口：读取 Excel → 生成结构描述 → AI 查询

    Args:
        file_path: Excel 文件路径
        instruction: 自然语言查询指令
        sheet_name: 要读取的 sheet（同 read_all_excel）
        header: 表头配置（同 read_all_excel）
        model: AI 模型标识符
        **kwargs: 传递给 query_dataframes 的额外参数

    Returns:
        str: 查询结果
    """
    from utils.file_io import read_all_excel

    # 读取 Excel
    dfs = read_all_excel(file_path, sheet_name=sheet_name, header=header)
    logger.info(f"已读取 {len(dfs)} 个 Sheet")

    # 生成结构描述
    schema_str = describe_dataframes_schema(dfs)
    logger.info(f"Schema 描述已生成:\n{schema_str}")

    # AI 查询
    result = query_dataframes(
        dfs=dfs,
        instruction=instruction,
        schema_str=schema_str,
        model=model,
        **kwargs,
    )

    return result


# ============================================================
# 4. MCP Tool 包装器
# ============================================================

class DataInspectorMCPTool:
    """
    MCP Tool 包装器：支持 describe_dataframes_schema、query_dataframes、inspect_and_query 三大功能。
    
    用法：
        tool = DataInspectorMCPTool()
        result = tool.run({"action": "describe", "dfs": {...}})
    
    返回值统一为 dict：
        成功: {"result": <str>}
        失败: {"error": <str>}
    """

    def run(self, params: Dict[str, Any]) -> Dict[str, str]:
        """
        params: dict, 必须包含 'action' 字段，可选值：'describe', 'query', 'inspect'
        其余参数按原函数要求传递。
        
        Returns:
            dict: {"result": str} 或 {"error": str}
        """
        action = params.get("action")
        if action not in ("describe", "query", "inspect"):
            return {"error": f"Unknown action '{action}'. Supported: describe, query, inspect."}

        try:
            if action == "describe":
                return self._handle_describe(params)
            elif action == "query":
                return self._handle_query(params)
            else:  # inspect
                return self._handle_inspect(params)
        except Exception as e:
            logger.error(f"DataInspectorMCPTool error (action={action}): {e}")
            return {"error": f"{type(e).__name__}: {e}"}

    def _handle_describe(self, params: Dict[str, Any]) -> Dict[str, str]:
        dfs = params.get("dfs")
        if dfs is None:
            return {"error": "Missing required parameter 'dfs' for action 'describe'."}
        result = describe_dataframes_schema(
            dfs,
            max_sample_rows=params.get("max_sample_rows", 3),
            max_unique_values=params.get("max_unique_values", 8),
        )
        return {"result": result}

    def _handle_query(self, params: Dict[str, Any]) -> Dict[str, str]:
        dfs = params.get("dfs")
        instruction = params.get("instruction")
        if dfs is None or instruction is None:
            return {"error": "Missing required parameter 'dfs' or 'instruction' for action 'query'."}
        result = query_dataframes(
            dfs=dfs,
            instruction=instruction,
            schema_str=params.get("schema_str"),
            model=params.get("model"),
            api_base=params.get("api_base"),
            api_key=params.get("api_key"),
            max_steps=params.get("max_steps", 3),
            **params.get("agent_kwargs", {}),
        )
        return {"result": result}

    def _handle_inspect(self, params: Dict[str, Any]) -> Dict[str, str]:
        file_path = params.get("file_path")
        instruction = params.get("instruction")
        if file_path is None or instruction is None:
            return {"error": "Missing required parameter 'file_path' or 'instruction' for action 'inspect'."}
        result = inspect_and_query(
            file_path=file_path,
            instruction=instruction,
            sheet_name=params.get("sheet_name"),
            header=params.get("header", 0),
            model=params.get("model"),
            **params.get("kwargs", {}),
        )
        return {"result": result}

    # 兼容不同 MCP 框架的调用接口
    def dispatch(self, params: Dict[str, Any]) -> Dict[str, str]:
        return self.run(params)

    def handle(self, params: Dict[str, Any]) -> Dict[str, str]:
        return self.run(params)


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    from utils.file_io import read_all_excel

    # 示例: 读取 Excel 并展示结构
    test_file = "data/test_data/test_load.xlsx"
    if Path(test_file).exists():
        dfs = read_all_excel(test_file, header=[[1], [0], [0, 1, 2, 3]])
        schema = describe_dataframes_schema(dfs)
        print(schema)

        # 示例: AI 查询（需要配置好模型环境变量）
        # result = query_dataframes(
        #     dfs=dfs,
        #     instruction="请列出每个 Sheet 中所有列的汇总统计信息",
        # )
        # print(result)
    else:
        # 使用内存数据演示
        demo_dfs = {
            "销售数据": pd.DataFrame({
                "日期": ["2024-01", "2024-02", "2024-03"],
                "产品": ["A", "B", "C"],
                "销量": [100, 200, 150],
                "金额": [1000.5, 2000.0, 1500.75],
            }),
            "库存数据": pd.DataFrame({
                "仓库": ["北京", "上海", "广州"],
                "产品": ["A", "B", "C"],
                "当前库存": [500, 300, 400],
            }),
        }
        schema = describe_dataframes_schema(demo_dfs)
        print(schema)
