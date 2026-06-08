"""
楼宇数据查询脚本

数据源: 江北区-25-06.xlsx → 第4个Sheet（楼宇相关），表头在 Excel 第2行

查询逻辑（自然语言）:
    从楼宇数据表中，筛选出 2025 年 1-6 月的记录，
    按「园区名称」(F列) 和「楼宇名称」(G列) 分组，对「新增软件产业使用（万方）」(K列) 求和，
    按求和结果降序排列，并在第一行额外展示所有分组的总计值。

SQL-like:
    -- 分组明细
    SELECT F AS 园区名称, G AS 楼宇名称 SUM(K) AS 新增软件产业使用面积合计
    FROM Sheet4
    WHERE C = 2025
      AND D IN ('1月', '2月', '3月', '4月', '5月', '6月')
    GROUP BY F, G
    ORDER BY 新增软件产业使用面积合计 DESC;

    -- 全局汇总（附在结果第一行的额外列中）
    SELECT SUM(K) AS K列筛选后求和
    FROM Sheet4
    WHERE C = 2025
      AND D IN ('1月', '2月', '3月', '4月', '5月', '6月');

返回结果: 楼宇.csv
    列: 园区名称, 楼宇名称, 新增软件产业使用面积合计, K列筛选后求和（仅第一行显示总计值，其他行为空）
    排序: 按 新增软件产业使用面积合计 降序
"""

import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.file_io import read_all_excel
# --- 1. 用户需要配置的参数 ---

# 【请修改】输入您的Excel文件名
excel_file_path = 'data/detailed_data/江北区-25-06.xlsx'  # 示例文件名，请替换为您自己的文件名

# 【请修改】指定表头的索引（start from 0）
header_excel_row = [0,1,2]

# 输出目录和文件名 (无需修改)
output_dir = 'human_validation/validation_reference/'
output_csv_path = os.path.join(output_dir, '楼宇.csv')


# --- 2. 代码执行部分 ---

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 将Excel行号转换为pandas使用的0-based索引
# pandas从0开始计数，所以第5行对应的header索引是4


# 步骤一：读取Excel的第4个sheet，并使用指定的行作为表头
print(f"正在读取文件 '{excel_file_path}' 的第4个sheet...")
print(f"将使用Excel中的第 {header_excel_row} 行作为表头...")

# sheet_name=3 表示第四个sheet (0-indexed)
df = read_all_excel(excel_file_path, sheet_name=3, header=header_excel_row)
(df, ) = df.values()

print("文件读取成功！原始数据的前5行：")
print(df.head())
print("-" * 50)

# 步骤二：按照列的位置进行筛选
# Column C -> 索引为 2
# Column D -> 索引为 3
# .columns[index] 会获取该位置列的实际名称
col_c_name = df.columns[2]
col_d_name = df.columns[3]

print(f"将根据以下条件筛选数据:")
print(f"  - 列 C (名称: '{col_c_name}') = 2025")
print(f"  - 列 D (名称: '{col_d_name}') 在 ['1月', '2月', ..., '6月'] 中")

months_to_filter = ['1月', '2月', '3月', '4月', '5月', '6月']

filtered_df = df[
    (df[col_c_name] == 2025) &
    (df[col_d_name].isin(months_to_filter))
].copy()

# 步骤三：按位置选取 F, G, K 列
# Column F -> 索引为 5
# Column G -> 索引为 6
# Column K -> 索引为 10
col_f_name = df.columns[5]
col_g_name = df.columns[6]
col_k_name = df.columns[10]

print(f"选取列 F ('{col_f_name}'), G ('{col_g_name}'), K ('{col_k_name}')...")

# 检查所需列是否存在
required_cols = [col_f_name, col_g_name, col_k_name]
if not all(col in filtered_df.columns for col in required_cols):
    raise ValueError("筛选后的数据中，找不到所有指定的 F, G, K 列，请检查源文件。")

result_df = filtered_df[required_cols].copy()
# 步骤四：将K列转换为数值类型，清洗空值，为后续运算做准备
print(f"将列 K ('{col_k_name}') 转换为数值类型...")
result_df[col_k_name] = pd.to_numeric(result_df[col_k_name], errors='coerce')
result_df.dropna(subset=[col_k_name], inplace=True)

# 步骤五：按列 F 和列 G 进行 Group By 分组，并对列 K 求和
print(f"按列 F ('{col_f_name}') 和列 G ('{col_g_name}') 分组求和...")
# as_index=False 确保分组的键(F,G)依然作为普通的列保留，而不是变成索引
# MultiIndex 列名是 tuple；GroupBy[] 里必须写成 [[col_k_name]]，否则会把 tuple 当成「多个列名」
grouped_df = result_df.groupby([col_f_name, col_g_name], as_index=False, dropna=False)[[col_k_name]].sum()

# 步骤六：按分组求和后的列 K 降序排列
print(f"按求和后的列 K ('{col_k_name}') 降序排列...")
grouped_df.sort_values(by=[col_k_name], ascending=False, inplace=True)

# 重置索引，让排序后的 DataFrame 索引从 0 重新开始，这对于下一步精准给第一行赋值很重要
grouped_df.reset_index(drop=True, inplace=True)

# 步骤七：计算全局总和，并在新列的【第一行】显示
print("计算全局总和并添加到新列（仅第一行显示）...")
total_sum = grouped_df[col_k_name].sum()

# 先将新列全部初始化为空值 (None 会在输出CSV时变成空)
grouped_df['K列筛选后求和'] = None

# 判断一下如果数据不为空，则仅在第一行（index=0）填入求和数值
if not grouped_df.empty:
    grouped_df.loc[0, 'K列筛选后求和'] = total_sum

print("数据处理完成！最终结果预览：")
print(grouped_df.head())
print("-" * 50)

# 步骤八：保存到CSV文件
print(f"正在将结果保存到: '{output_csv_path}'")
grouped_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
print("任务完成！CSV文件已成功保存。")
