from typing import Any, Tuple
import pandas as pd
import numpy as np

def get_var_storage_info(value: Any) -> Tuple[str, str]:

    """
    根据变量类型确定存储格式和文件后缀
    
    返回: (file_suffix, type_name)
    - str/数值/bool → .txt
    - list/dict → .json
    - np.ndarray → .npy
    - pd.DataFrame (MultiIndex columns) → .pkl (pickle)
    - pd.DataFrame (普通列) → .parquet
    """
    # 检查 pandas DataFrame
    if isinstance(value, pd.DataFrame):
        # MultiIndex 列无法被 parquet 正确序列化，改用 pickle
        if isinstance(value.columns, pd.MultiIndex):
            return '.pkl', 'dataframe_pickle'
        return '.parquet', 'dataframe'
    
    # 检查 numpy array
    if isinstance(value, np.ndarray):
        return '.npy', 'ndarray'
    
    # 检查 list 和 dict
    if isinstance(value, (list, dict)):
        return '.json', 'json'
    
    # 检查基本类型：str, 数值, bool
    if isinstance(value, (str, int, float, bool, complex)):
        return '.txt', 'txt'
    
    # 默认使用 json（尝试序列化）
    return '.json', 'json'


def save_variable_to_temp(key: str, value: Any, suffix: str, type_name: str) -> str:
    """
    根据类型将变量保存到临时文件，返回文件路径
    """
    import tempfile
    import os
    import json
    temp_fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=f'{key}_', text=(type_name == 'txt'))
    
    if type_name == 'txt':
        # 文本类型：str, 数值, bool
        with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
            f.write(str(value))
    
    elif type_name == 'json':
        # JSON 类型：list, dict
        with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
    
    elif type_name == 'ndarray':
        # Numpy array
        os.close(temp_fd)  # numpy.save 需要自己管理文件
        np.save(temp_path, value)
    
    elif type_name == 'dataframe':
        # Pandas DataFrame (普通列)
        os.close(temp_fd)  # pandas 需要自己管理文件
        value.to_parquet(temp_path, index=False)
    
    elif type_name == 'dataframe_pickle':
        # Pandas DataFrame (MultiIndex 列) - parquet 无法保留 MultiIndex
        os.close(temp_fd)
        value.to_pickle(temp_path)
    
    else:
        # 默认尝试 JSON
        with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
    
    return temp_path   