"""Prompt 模板渲染器。

使用 Jinja2 渲染论文中的所有 Prompt 模板。
"""

from __future__ import annotations

from typing import Any

from jinja2 import BaseLoader, Environment

# 创建 Jinja2 环境，使用字符串模板
_env = Environment(loader=BaseLoader(), keep_trailing_newline=True)


def render(template_str: str, **kwargs: Any) -> str:
    """渲染 Jinja2 模板字符串。

    Args:
        template_str: 模板字符串
        **kwargs: 模板变量

    Returns:
        渲染后的字符串
    """
    tmpl = _env.from_string(template_str)
    return tmpl.render(**kwargs)
