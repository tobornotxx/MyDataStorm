"""通用辅助函数"""

import re
from typing import Dict, Optional


def extract_code_from_response(response_text: str) -> Optional[str]:
    """从 LLM 响应中提取 <code></code> 包裹的代码块。

    如果存在多个 <code></code> 块，则拼接所有代码块。
    """
    pattern = r'<code>\s*\n?(.*?)\s*</code>'
    matches = re.findall(pattern, response_text, re.DOTALL)
    if not matches:
        return None
    return "\n\n".join(match.strip() for match in matches)


def build_variable_preamble(var_paths: Dict[str, str]) -> str:
    """生成变量路径赋值的前置代码行，注入到脚本顶部。"""
    lines = []
    for var_name, path in var_paths.items():
        # 使用 repr 安全地转义路径中的反斜杠等特殊字符
        lines.append(f'{var_name} = {repr(path)}')
    return "\n".join(lines)
