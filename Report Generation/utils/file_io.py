import json
import pandas as pd
from pathlib import Path
from typing import Union, Dict, Any, List
from utils import logger
def read_all_excel(
    file_path: Union[str, Path],
    sheet_name: Union[str, int, list, None] = None,
    header: Union[int, List[int], List[List[int]], Dict[str, Union[int, List[int]]]] = 0
) -> Dict[str, pd.DataFrame]:
    """
    读取Excel文件的指定或所有sheet。
    
    Args:
        file_path: Excel文件路径
        sheet_name: 要读取的sheet，可以是：
                    - None: 读取所有sheet
                    - str: 单个sheet名称
                    - int: 单个sheet索引
                    - list: 多个sheet名称或索引的列表
        header: 表头行配置（0-based索引），可以是：
                - int: 所有sheet使用同一行作为表头，如 header=0 表示第1行为表头
                - List[int]: 所有sheet使用相同的多行表头，如 header=[0,1] 表示前两行为表头（MultiIndex）
                - List[List[int]]: 每个sheet分别配置表头，按sheet顺序对应
                  如 header=[[0], [0,1], [1]] 表示第1个sheet用第1行，第2个sheet用前两行，第3个sheet用第2行
                - Dict[str, int|List[int]]: 按sheet名称映射表头配置
                  如 header={"Sheet1": 0, "Sheet2": [0,1]}
                   
    Returns:
        Dict[str, pd.DataFrame]: 以sheet名为key，DataFrame为value的字典
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 先获取所有sheet名称
    xlsx = pd.ExcelFile(file_path)
    all_sheet_names = xlsx.sheet_names
    logger.info(f"Reading excel file: {file_path}")
    # 确定要读取的sheet列表
    if sheet_name is None:
        sheets_to_read = all_sheet_names
    elif isinstance(sheet_name, (str, int)):
        sheets_to_read = [sheet_name] if isinstance(sheet_name, str) else [all_sheet_names[sheet_name]]
    else:
        # list 类型
        sheets_to_read = [all_sheet_names[s] if isinstance(s, int) else s for s in sheet_name]
    
    # 判断 header 是否为 list of list（每个 sheet 分别配置）
    def is_list_of_lists(obj) -> bool:
        if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], list):
            return True
        return False
    
    # 构建每个 sheet 的 header 配置
    def get_header(sheet_idx: int, sheet_nm: str) -> Union[int, List[int]]:
        if isinstance(header, int):
            # 单个 int，所有 sheet 用同一行
            return header
        elif isinstance(header, dict):
            # dict 映射，找不到则默认 0
            return header.get(sheet_nm, 0)
        elif isinstance(header, list):
            if is_list_of_lists(header):
                # list of list，按顺序对应每个 sheet
                return header[sheet_idx] if sheet_idx < len(header) else 0
            else:
                # 普通 list（如 [0, 1]），所有 sheet 用相同的多行表头
                return header
        return 0
    
    # 逐个读取 sheet
    dfs = {}
    for idx, sn in enumerate(sheets_to_read):
        logger.info(f"Reading sheet: {sn}")
        hdr = get_header(idx, sn)
        df = pd.read_excel(
            xlsx,
            sheet_name=sn,
            header=hdr
        )
        # 对每一列做前向填充
        for col in df.columns:
            df[col] = df[col].ffill()
        try:
            if isinstance(df.columns, pd.MultiIndex):
                # Multirow header case.
                cols = df.columns.to_frame(index=False)
                cols = cols.replace(r'^Unnamed:.*', None, regex=True)
                cols = cols.ffill(axis=1)
                df.columns = pd.MultiIndex.from_frame(cols)
            else:
                # Single row of header, means type=Index
                header_df = df.columns.to_series()
                header_df[header_df.str.contains('Unnamed', na=False)] = None
                header_df = header_df.ffill()
                df.columns = header_df
        except Exception as e:
            logger.info(f"Error normalizing header for sheet '{sn}': {e}")

        dfs[sn] = df
    
    xlsx.close()
    return dfs


def data_save(
    data: Any,
    file_path: Union[str, Path],
    file_type: str = "xlsx"
) -> Path:
    """
    保存数据到文件。如果文件已存在，自动添加数字后缀。
    
    Args:
        data: 要保存的数据，支持 DataFrame, str, dict, list 等类型
        file_path: 目标文件路径（可以带或不带扩展名）
        file_type: 文件类型，支持 "xlsx", "csv", "json", "txt", "md", "html" 等
        
    Returns:
        Path: 实际保存的文件路径
    """
    file_path = Path(file_path)
    
    # 确保父目录存在
    if file_path.parent != Path("."):
        file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 分离文件名和扩展名
    # 如果路径已经有扩展名，使用路径中的扩展名
    if file_path.suffix:
        stem = file_path.stem
        suffix = file_path.suffix
        parent = file_path.parent
    else:
        stem = file_path.name
        suffix = f".{file_type}"
        parent = file_path.parent
    
    # 构建初始完整路径
    target_path = parent / f"{stem}{suffix}"
    
    # 如果文件已存在，添加数字后缀
    counter = 1
    while target_path.exists():
        target_path = parent / f"{stem}_{counter}{suffix}"
        counter += 1
    
    # 根据文件类型保存
    suffix_lower = suffix.lower()
    
    # DataFrame 专用格式
    if suffix_lower in [".xlsx", ".xls"]:
        if isinstance(data, pd.DataFrame):
            data.to_excel(target_path, index=False)
        else:
            pd.DataFrame(data).to_excel(target_path, index=False)
    
    elif suffix_lower == ".csv":
        if isinstance(data, pd.DataFrame):
            data.to_csv(target_path, index=False, encoding="utf-8-sig")
        elif isinstance(data, str):
            target_path.write_text(data, encoding="utf-8-sig")
        else:
            pd.DataFrame(data).to_csv(target_path, index=False, encoding="utf-8-sig")
    
    elif suffix_lower == ".parquet":
        if isinstance(data, pd.DataFrame):
            data.to_parquet(target_path, index=False)
        else:
            pd.DataFrame(data).to_parquet(target_path, index=False)
    
    elif suffix_lower in [".pkl", ".pickle"]:
        if isinstance(data, pd.DataFrame):
            data.to_pickle(target_path)
        else:
            import pickle
            with open(target_path, "wb") as f:
                pickle.dump(data, f)
    
    # 文本格式
    elif suffix_lower in [".txt", ".md", ".markdown", ".html", ".htm", ".log", ".py", ".js", ".css"]:
        if isinstance(data, str):
            target_path.write_text(data, encoding="utf-8")
        elif isinstance(data, (list, tuple)):
            # 列表/元组按行写入
            target_path.write_text("\n".join(str(item) for item in data), encoding="utf-8")
        elif isinstance(data, pd.DataFrame):
            target_path.write_text(data.to_string(index=False), encoding="utf-8")
        else:
            target_path.write_text(str(data), encoding="utf-8")
    
    # JSON 格式
    elif suffix_lower == ".json":
        if isinstance(data, pd.DataFrame):
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(data.to_json(orient="records", force_ascii=False, indent=2))
        elif isinstance(data, str):
            target_path.write_text(data, encoding="utf-8")
        else:
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    
    # YAML 格式
    elif suffix_lower in [".yaml", ".yml"]:
        try:
            import yaml
            with open(target_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
        except ImportError:
            raise ImportError("需要安装 pyyaml: pip install pyyaml")
    
    # XML 格式
    elif suffix_lower == ".xml":
        if isinstance(data, pd.DataFrame):
            data.to_xml(target_path, index=False, encoding="utf-8")
        elif isinstance(data, str):
            target_path.write_text(data, encoding="utf-8")
        else:
            raise ValueError("XML格式仅支持 DataFrame 或字符串")
    
    # 二进制格式
    elif suffix_lower in [".bin", ".dat"]:
        if isinstance(data, bytes):
            target_path.write_bytes(data)
        elif isinstance(data, str):
            target_path.write_bytes(data.encode("utf-8"))
        else:
            import pickle
            with open(target_path, "wb") as f:
                pickle.dump(data, f)
    
    else:
        # 默认当作文本处理
        if isinstance(data, str):
            target_path.write_text(data, encoding="utf-8")
        elif isinstance(data, bytes):
            target_path.write_bytes(data)
        elif isinstance(data, pd.DataFrame):
            data.to_csv(target_path, index=False, encoding="utf-8-sig")
        else:
            target_path.write_text(str(data), encoding="utf-8")
    
    return target_path

if __name__ == "__main__":
    data = read_all_excel("data/test_data/test_load.xlsx", header=[0,1,2])
    for sheet_name, sheet_data in data.items():
        print(sheet_name)
        print(sheet_data)
    data_save(data['Sheet1'], 'data/test_data/test_save', 'csv')

    logger.info(f"Sheet 1 header schema: {data["Sheet1"].columns}")
    logger.info(f"Sheet 1 column 0:{data['Sheet1'][('时间','时间','时间')]}")
    