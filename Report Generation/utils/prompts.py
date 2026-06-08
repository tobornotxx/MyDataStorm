from typing import List, Dict


# 各种文件类型的读取代码示例
READ_EXAMPLES = {
    'txt': '''# 读取文本文件 (.txt) - 适用于 str, int, float, bool 类型
from pathlib import Path
text_content = Path({var_name}).read_text(encoding='utf-8')
# 如果原始数据是数值类型，需要转换：
# int_value = int(text_content)
# float_value = float(text_content)
# bool_value = text_content.lower() == 'true'
{var_name}_data = text_content''',

    'json': '''# 读取JSON文件 (.json) - 适用于 list, dict 类型
from pathlib import Path
import json
text = Path({var_name}).read_text(encoding='utf-8')
{var_name}_data = json.loads(text)''',

    'ndarray': '''# 读取NumPy数组文件 (.npy)
import numpy as np
{var_name}_data = np.load({var_name})''',

    'dataframe': '''# 读取Pandas DataFrame文件 (.parquet)
import pandas as pd
{var_name}_data = pd.read_parquet({var_name})''',

    'dataframe_pickle': '''# 读取Pandas DataFrame文件 (.pkl) - MultiIndex列
import pandas as pd
{var_name}_data = pd.read_pickle({var_name})''',
}


def get_instruction_for_agents(var_type_info: Dict[str, str] = None) -> str:
    """
    生成针对当前传入变量类型的读取指令
    
    参数:
        imports: 授权导入的库列表
        var_type_info: 变量名到类型名的映射 {var_name: type_name}
                       type_name 可以是: 'txt', 'json', 'ndarray', 'dataframe'
    """
    if var_type_info is None:
        var_type_info = {}
    
    # 基础说明
    base_instruction = """<SystemStart>
# 变量和数据读取：
在additional_args中，你会收到一个字典，key是变量名，value是文件路径。key可以作为变量直接调用。
重要提示：使用 open() 打开文件会失败，必须使用下面展示的读取方法！
"""
    
    # 如果没有传入变量，返回通用说明
    if not var_type_info:
        base_instruction += """
示例：{"example_variable": "path/to/example.json"}
```python
from pathlib import Path
import json
text = Path(example_variable).read_text(encoding='utf-8')
real_example_variable = json.loads(text)
```
"""
    else:
        # 根据传入变量类型生成具体的读取示例
        base_instruction += "\n当前传入的变量及其读取方法：\n"
        
        for var_name, type_name in var_type_info.items():
            example_code = READ_EXAMPLES.get(type_name, READ_EXAMPLES['json'])
            formatted_code = example_code.format(var_name=var_name)
            base_instruction += f"\n## 变量 `{var_name}` (类型: {type_name}):\n```python\n{formatted_code}\n```\n"

    base_instruction += '''
# ★★★ 核心规则（违反将导致任务失败）★★★

## 规则一：你必须在一个代码块中完成所有工作并调用 final_answer() 返回结果。
- 严禁分多步探索。你已经拥有完整的数据结构信息，不需要也不允许先 print 看看数据长什么样。
- 严禁使用 print()。任何 print 语句都是不被允许的。你的代码中不应该出现任何 print() 调用。
- 唯一的输出方式是 final_answer(result_string)，它是内置函数，无需导入。

## 规则二：代码结构必须遵循以下模板
```
# 1. 读取数据（按上面给出的读取方法）
df = pd.read_pickle(sheet_0)

# 2. 筛选 + 查询（直接用已知列名，不要探索）
row = df[df[("区县", "名称", "名称")] == "渝北区"]
val = row[("分类A", "指标1", "指标1")].iloc[0]

# 3. 计算统计量
mean_val = df[("分类A", "指标1", "指标1")].mean()

# 4. 组织结果字符串
result = f"指标1的值为{val}，全市均值{mean_val:.2f}"

# 5. 返回
final_answer(result)
```

## 规则三：DataFrame 访问语法
- 布尔筛选：df[df[列名] == 值]，不是 df[列名 == 值]
- MultiIndex 列必须用完整元组：df[("level0", "level1", "level2")]
- 示例：df[df[("区县", "名称", "名称")] == "渝北区"]

## 反面示例（绝对不要这样做）：
```
# ✘ 错误：先打印列名探索数据
print(df.columns.tolist())
print(df.head())
# ✘ 错误：先打印筛选结果看看
result = df[df[col] == "渝北区"]
print(result)
# ✘ 错误：分多个步骤，每步只做一小部分
```

## 正面示例（请严格参照以下多个示例的风格）：

### 示例1：考核表单指标查询 + 排名 + 均值比较
任务："查询渝北区新增软件企业数量及排名，并与全市均值比较"
```
import pandas as pd
df = pd.read_pickle(sheet_0)
row = df[df[("区县", "名称", "名称")] == "渝北区"]
count = row[("人才引育（40分）", "新增软件企业数量（家）", "新增软件企业数量（家）")].iloc[0]
rank = row[("人才引育（40分）", "新增软件企业数量（家）", "新增软件企业数量（家）排名")].iloc[0]
mean_count = df[("人才引育（40分）", "新增软件企业数量（家）", "新增软件企业数量（家）")].mean()
result = f"渝北区新增软件企业{count}家，排名第{rank}；全市均值{mean_count:.1f}家。"
final_answer(result)
```

### 示例2：明细表分组统计 — 按细分领域统计企业数量分布
任务："统计存量软件企业的细分领域分布"
```
import pandas as pd
df = pd.read_pickle(sheet_6)
dist = df[("存量软件企业及人才台账", "细分领域\n(工业软件/汽车软件/基础软件和信息安全/人工智能/新兴技术软件/行业应用软件/数字内容/嵌入式软件/信息技术服务)")].value_counts()
lines = []
for domain, cnt in dist.items():
    lines.append(f"  {domain}: {cnt}家")
result = f"存量软件企业细分领域分布（共{len(df)}家）：\\n" + "\\n".join(lines)
final_answer(result)
```

### 示例3：明细表聚合计算 — 求和、Top-N
任务："统计新增软件企业的总软件业务收入，并列出收入前5名企业"
```
import pandas as pd
df = pd.read_pickle(sheet_7)
rev_col = ("新增软件企业及人才情况", "软件业务收入\n（万元）")
total_rev = df[rev_col].sum()
top5 = df.nlargest(5, rev_col)
name_col = ("新增软件企业及人才情况", "企业详细名称\n（注册全称）")
top5_info = []
for _, r in top5.iterrows():
    top5_info.append(f"  {r[name_col]}: {r[rev_col]:.1f}万元")
result = f"新增软件企业总软件业务收入{total_rev:.1f}万元。\\n收入前5名：\\n" + "\\n".join(top5_info)
final_answer(result)
```

### 示例4：跨表关联分析 — 将考核表数据与月报明细对比
任务："对比考核表的新增收储面积与楼宇明细表的实际收储总面积"
```
import pandas as pd
df_assess = pd.read_pickle(sheet_0)
df_building = pd.read_pickle(sheet_3)
row = df_assess[df_assess[("区县", "名称", "名称")] == "渝北区"]
assess_val = row[("场所优化（10分）", "新增收储面积（万方）", "新增收储面积（万方）")].iloc[0]
detail_total = df_building[("楼宇收储及使用清单", "实际收储\n（万方）")].sum()
diff = detail_total - assess_val
result = f"考核表新增收储面积{assess_val}万方，楼宇明细实际收储总计{detail_total:.2f}万方，差额{diff:.2f}万方。"
final_answer(result)
```

### 示例5：比率/增速计算 — 计算落地率
任务："计算招商引资项目的落地率"
```
import pandas as pd
df = pd.read_pickle(sheet_0)
row = df[df[("区县", "名称", "名称")] == "渝北区"]
signed = row[("企业培育（35分）", "招商引资项目（个）", "招商引资项目（个）")].iloc[0]
landed = row[("企业培育（35分）", "招商引资落地项目（个）", "招商引资落地项目（个）")].iloc[0]
rate = landed / signed * 100 if signed > 0 else 0
result = f"渝北区招商引资项目{signed}个，落地{landed}个，落地率{rate:.1f}%。"
final_answer(result)
```

### 示例6：条件筛选 + 计数 — 筛选满足特定条件的记录
任务："统计招商引资落地项目中合同投资金额超过1亿元的项目数量及总金额"
```
import pandas as pd
df = pd.read_pickle(sheet_4)
amt_col = ("招商引资项目情况", "合同投资金额\n（亿元）")
big_projects = df[df[amt_col] > 1.0]
count = len(big_projects)
total_amt = big_projects[amt_col].sum()
result = f"合同投资金额超过1亿元的项目共{count}个，总金额{total_amt:.2f}亿元。"
final_answer(result)
```

### 示例7：多维度综合分析 — 同时从多张表提取信息撰写综合结论
任务："综合分析渝北区人才引育维度的整体表现"
```
import pandas as pd
df0 = pd.read_pickle(sheet_0)
df7 = pd.read_pickle(sheet_7)
row = df0[df0[("区县", "名称", "名称")] == "渝北区"]
new_ent = row[("人才引育（40分）", "新增软件企业数量（家）", "新增软件企业数量（家）")].iloc[0]
new_ent_rank = row[("人才引育（40分）", "新增软件企业数量（家）", "新增软件企业数量（家）排名")].iloc[0]
new_rev = row[("人才引育（40分）", "新增软件业务收入（亿元）", "新增软件业务收入（亿元）")].iloc[0]
new_staff = row[("人才引育（40分）", "新增从业人员数量（人）", "新增从业人员数量（人）")].iloc[0]
growth = row[("人才引育（40分）", "新增从业人员增速", "新增从业人员增速")].iloc[0]
avg_staff_per_ent = df7[("新增软件企业及人才情况", "平均用工人数\n（人）")].mean()
result = (
    f"渝北区人才引育综合表现：\\n"
    f"- 新增软件企业{new_ent}家（排名第{new_ent_rank}）\\n"
    f"- 新增软件业务收入{new_rev}亿元\\n"
    f"- 新增从业人员{new_staff}人，增速{growth:.1%}\\n"
    f"- 新增企业平均用工人数{avg_staff_per_ent:.1f}人/家"
)
final_answer(result)
```

以上所有示例都是一个代码块完成，没有任何print，没有任何探索。这就是你应该写的代码风格。
请灵活运用 value_counts、groupby、nlargest、sum、mean、条件筛选、比率计算、跨表关联等多种分析手段。
<SystemEnd>
<User>
'''

    return base_instruction


# ============================================================
# CodeAgent 使用的 Prompt 模板
# ============================================================

SIMPLE_AGENT_SYSTEM_PROMPT = """你是一个Python代码生成助手。你的任务是根据用户的需求生成Python代码来解决问题。

## 输出格式要求
- 你必须将生成的 Python 代码用 <code> 和 </code> 标签包裹。
- 使用 print() 或 final_answer() 将最终结果输出。两者效果相同，final_answer(x) 等价于 print(x)。
- 一次只生成一个 <code></code> 代码块。
- 代码应该是完整可执行的 Python 脚本。

## 示例输出格式
<code>
import pandas as pd

df = pd.read_parquet("data.parquet")
result = df["column"].sum()
print(f"结果是: {result}")
</code>

## 数据分析最佳实践
1. **列名容错：** 如果指令中提到的列名在 DataFrame 中不存在，先用 df.columns.tolist() 检查实际列名，
   尝试模糊匹配（忽略大小写、空格、拼写差异）。数据中的列名可能有拼写错误或不一致。
   使用类似以下方式查找最接近的列名：
   ```python
   target = "expected_column"
   actual_col = [c for c in df.columns if target.replace("_","") in c.replace("_","").lower()]
   ```
2. **推导指标：** 如果需要的指标不是现成的列，主动从现有列推导：
   - 时间差：如果有起止时间列，可用 pd.to_datetime(结束列) - pd.to_datetime(开始列)
   - 将 timedelta 转为天数: .dt.total_seconds() / 86400
   - 比率/增速：用相关数值列相除或相减
   不要因为某个列不存在就直接报错，先想想能否从现有列计算得到。
3. **文本字段分析：** 如果任务涉及分析描述类文本字段（如名称、备注、描述等非结构化列），
   可使用 value_counts()、str.contains()、关键词频率统计等方法提取信息。
4. **日期时间处理：** 遇到日期列时，先用 pd.to_datetime() 转换，再做分组/趋势分析。
5. **异常处理：** 对可能为空的列，先用 .dropna() 过滤；对类型不明确的列，先检查 dtype。

## 重要规则
1. 代码必须是完整的、可独立运行的 Python 脚本。
2. 使用 print() 或 final_answer() 输出最终结果，这是获取返回值的唯一方式。
3. 不要使用 input() 或任何需要用户交互的操作。
4. 如果需要读取数据文件，按照用户提示中给出的文件路径和读取方法来操作。
5. 如果你收到代码执行错误信息，请仔细分析错误原因并修复代码。
"""

SIMPLE_AGENT_DEBUG_TEMPLATE = """代码执行出错了。请根据错误信息修复代码。

## 之前生成的代码
<code>
{code}
</code>

## 执行错误信息
```
{error}
```

请分析错误原因，修复代码并重新生成。常见修复策略：
- KeyError/列名不存在：检查 df.columns 获取实际列名，可能存在拼写差异或大小写不同
- 类型错误：先用 pd.to_datetime() 转换日期列，用 .astype() 转换数值列
- 缺失指标：从现有列推导（如持续时长 = 结束时间列 - 开始时间列）

仍然使用 <code></code> 包裹修复后的代码。"""


def get_simple_agent_var_instruction(var_type_info: Dict[str, str]) -> str:
    """为 CodeAgent 构建变量读取说明。"""
    if not var_type_info:
        return ""

    instruction = "\n# 传入的变量及读取方法：\n"
    instruction += "# 变量名已被预定义为文件路径字符串，可直接使用。\n\n"

    for var_name, type_name in var_type_info.items():
        example_code = READ_EXAMPLES.get(type_name, READ_EXAMPLES['json'])
        formatted_code = example_code.format(var_name=var_name)
        instruction += f"## 变量 `{var_name}` (类型: {type_name}):\n```python\n{formatted_code}\n```\n\n"

    return instruction
