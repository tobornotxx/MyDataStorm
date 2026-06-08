"""
招商金额数据查询脚本

数据源: 江北区-25-06.xlsx → 第5个Sheet（招商相关），表头在 Excel 第3行（header=2）

查询逻辑（自然语言）:
    从招商数据表中，筛选出年份为 2025 或年份为空的记录（空行通常是上方数据的延续），
    提取投资主体、合同投资金额、细分领域、员工人数 4 个字段，
    按「细分领域」分组，组内按「合同投资金额」降序排列。

SQL-like:
    SELECT
        E  AS 投资主体,
        I  AS 合同投资金额（亿元）,
        P  AS 细分领域,
        X  AS 员工人数（人）
    FROM Sheet5
    WHERE C = 2025
       OR C IS NULL
       OR TRIM(C) = ''
    ORDER BY 细分领域 ASC, 合同投资金额（亿元） DESC;

返回结果: 招商金额.csv
    列: 投资主体, 合同投资金额（亿元）, 细分领域, 员工人数（人）
    排序: 按细分领域分组、组内按合同投资金额降序
"""

import pandas as pd
import os
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.file_io import read_all_excel

# ==========================================
# 1. 用户控制变量区 (请在此处修改你的配置)
# ==========================================
INPUT_EXCEL_NAME = "data/detailed_data/江北区-25-06.xlsx"  # 输入的Excel文件名
HEADER_ROW = [0,1,2]                             # 表头所在的行 (0表示第一行，如果没有表头填 None)
OUTPUT_CSV_NAME = "招商金额.csv"      # 输出的CSV文件名
# 设置输出的列名 (必须是4个，分别对应提取的 E, I, P, X 列)
OUTPUT_COLUMNS = ['投资主体', '合同投资金额（亿元）', '细分领域', '员工人数（人）'] 

# ==========================================
# 2. 路径处理
# ==========================================
OUTPUT_DIR = "human_validation/validation_reference/"
# 确保输出目录存在，如果不存在则自动创建
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, OUTPUT_CSV_NAME)

# ==========================================
# 3. 数据处理区
# ==========================================
# 读取第5个sheet (pandas中索引从0开始，所以第5个sheet是4)
# df = pd.read_excel(INPUT_EXCEL_NAME, sheet_name=4, header=HEADER_ROW)
df = read_all_excel(INPUT_EXCEL_NAME, sheet_name=4, header=HEADER_ROW)['招商引资落地项目台账(D)']

# 获取 Column C 的数据 (索引为2)
col_c = df.iloc[:, 2]

# 构建筛选条件：Column C = 2025 或者 空白 (NaN, None, 空字符串, 全空格的字符串)
# 注意：这里兼容了数字类型的2025和字符串类型的'2025'，全程避免使用 dropna
condition_2025 = (col_c == 2025) | (col_c == "2025") | (col_c == 2025.0)
condition_blank = col_c.isna() | (col_c.astype(str).str.strip() == "") | (col_c.astype(str).str.lower() == "nan")

# 应用筛选条件
df_filtered = df[condition_2025 | condition_blank].copy()

# 提取指定的列：E(4), I(8), P(15), X(23)
# 注意：使用 iloc 纯按位置索引提取
df_extracted = df_filtered.iloc[:, [4, 8, 15, 23]].copy()

# 排序："按照P划分之后按照I列分别降序排列"
# 在提取出的 df_extracted 中：
# - P列现在的相对索引是 2
# - I列现在的相对索引是 1
# 我们按照 P 列升序(划分聚集)，然后按照 I 列降序排列
col_p_current_name = df_extracted.columns[2]
col_i_current_name = df_extracted.columns[1]

df_sorted = df_extracted.sort_values(
    by=[col_p_current_name, col_i_current_name], 
    ascending=[True, False] # P列升序(分组效果)，I列降序
)

# 赋予用户自定义的输出列名
df_sorted.columns = OUTPUT_COLUMNS

# ==========================================
# 4. 结果输出
# ==========================================
# 导出为CSV文件
df_sorted.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"处理完成！文件已保存至: {output_path}")