"""
存量新增收入 & 用工数据查询脚本

数据源: 江北区-25-06.xlsx → 第7个Sheet（存量新增相关），表头在 Excel 第2行（header=1）

查询逻辑（自然语言）:
    从存量新增数据表中，筛选出「2025年」且统计周期为「1-6月」的记录，
    提取企业名称、主营业务、收入、用工、人才、税收、领域等 8 个字段。
    生成两份排序结果:
      1) 按「细分领域」分组，组内按「新增软件业务收入」降序 → 存量新增收入.csv
      2) 按「细分领域」分组，组内按「新增用工人数」降序   → 存量新增用工.csv

SQL-like:
    -- 结果1: 按收入降序
    SELECT
        E  AS 企业详细名称,
        H  AS 主营业务,
        J  AS 新增软件业务收入（万元）,
        L  AS 新增用工人数（人）,
        N  AS 新增中高端软件人才数（人）（20-50万）,
        P  AS 新增中高端软件人才数（人）（50万以上）,
        R  AS 新增个人所得税（万元）,
        V  AS 细分领域
    FROM Sheet7
    WHERE B = '2025年'
      AND D = '1-6月'
    ORDER BY 细分领域 ASC, 新增软件业务收入（万元） DESC;

    -- 结果2: 按用工人数降序
    SELECT ... (同上)
    ORDER BY 细分领域 ASC, 新增用工人数（人） DESC;

返回结果:
    存量新增收入.csv — 按细分领域分组、组内按收入降序
    存量新增用工.csv — 按细分领域分组、组内按用工人数降序
    列: 企业详细名称, 主营业务, 新增软件业务收入（万元）, 新增用工人数（人）,
        新增中高端软件人才数（人）（20-50万）, 新增中高端软件人才数（人）（50万以上）,
        新增个人所得税（万元）, 细分领域
"""

import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.file_io import read_all_excel
# ==========================================
# 1. 用户控制变量区 (请在此处修改你的配置)
# ==========================================
INPUT_EXCEL_NAME = "data/detailed_data/江北区-25-06.xlsx"  # 输入的Excel文件名
HEADER_ROW = 1                             # 表头所在的行 (0表示第一行，如果没有表头填 None)
OUTPUT_CSV_NAME = "存量新增收入.csv"    # 输出的CSV文件名
OUTPUT_CSV_NAME_2 = "存量新增用工.csv"
# 设置输出的列名 (必须是8个，分别对应提取的 E, H, J, L, N, P, R, V 列)
OUTPUT_COLUMNS = [
    "企业详细名称", "主营业务", "新增软件业务收入（万元）", "新增用工人数（人）", 
    "新增中高端软件人才数（人）（20-50万）", "新增中高端软件人才数（人）（50万以上）", "新增个人所得税（万元）", "细分领域"
]

# ==========================================
# 2. 路径处理
# ==========================================
OUTPUT_DIR = "human_validation/validation_reference/"
# 确保输出目录存在，如果不存在则自动创建
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, OUTPUT_CSV_NAME)
output_path_2 = os.path.join(OUTPUT_DIR, OUTPUT_CSV_NAME_2)

# ==========================================
# 3. 数据处理区
# ==========================================
# 读取第7个sheet (pandas中索引从0开始，所以第7个sheet是6)
# df = pd.read_excel(INPUT_EXCEL_NAME, sheet_name=6, header=HEADER_ROW)
df = read_all_excel(INPUT_EXCEL_NAME, sheet_name=6, header=HEADER_ROW)
(df, ) = df.values()

# 获取 Column B(索引1) 和 Column D(索引3) 的数据
col_b = df.iloc[:, 1]
col_d = df.iloc[:, 3]

# 构建筛选条件：Column B = '2025年' 且 Column D = '1-6月'
# 为了防止Excel单元格里存在首尾空格导致漏筛，这里将其转为字符串并去除空格后再比对
condition_b = col_b.astype(str).str.strip() == "2025年"
condition_d = col_d.astype(str).str.strip() == "1-6月"

# 应用筛选条件 (按位与 &)，全程避免使用 dropna
df_filtered = df[condition_b & condition_d].copy()

# 提取指定的列：E(4), H(7), J(9), L(11), N(13), P(15), R(17), V(21)
# 纯按位置索引提取，不受原表列名的影响
target_indices = [4, 7, 9, 11, 13, 15, 17, 21]
df_extracted = df_filtered.iloc[:, target_indices].copy()

# 排序："按照V分组，每个组内用column J降序重新排列"
# 在提取出的 df_extracted 中，我们刚刚提取了8列：
# - J列 是第 3 个提取的，现在的相对索引是 2
# - V列 是第 8 个提取的，现在的相对索引是 7
col_v_current_name = df_extracted.columns[7]  # 对应原 V 列
col_j_current_name = df_extracted.columns[2]  # 对应原 J 列
col_l_current_name = df_extracted.columns[3]  # 对应原 L 列

# 利用多重排序来实现"分组内排序"：先按V列升序把相同的聚集在一起，再按J列降序
df_sorted = df_extracted.sort_values(
    by=[col_v_current_name, col_j_current_name], 
    ascending=[True, False] 
)
df_sorted_2 = df_extracted.sort_values(
    by=[col_v_current_name, col_l_current_name],
    ascending=[True, False]
)

# 赋予用户自定义的输出列名
df_sorted.columns = OUTPUT_COLUMNS
df_sorted_2.columns = OUTPUT_COLUMNS

# ==========================================
# 4. 结果输出
# ==========================================
# 导出为CSV文件，使用 utf-8-sig 防止中文乱码
df_sorted.to_csv(output_path, index=False, encoding='utf-8-sig')
df_sorted_2.to_csv(output_path_2, index=False, encoding='utf-8-sig')

print(f"处理完成！文件已保存至: {output_path}与{output_path_2}")