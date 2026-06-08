"""
Prompt 模板渲染器
使用 Jinja2 模板引擎加载和填充 .j2 格式的 prompt 模板。

特性:
  - 模板中引用但调用方未传入的变量 → 警告 + 替换为空字符串（不报错）
  - 调用方传入但模板中未使用的多余变量 → 警告（不报错）
  - 支持 Jinja2 完整语法（条件判断 {% if %}、循环 {% for %}、过滤器等）

用法:
    from utils.prompt_renderer import render_prompt, PromptRenderer

    # 便捷函数（使用默认 prompts/ 目录）
    text = render_prompt("data_analysis_user.j2", region_name="渝北区", max_queries=5)

    # 或手动创建渲染器
    renderer = PromptRenderer(template_dir="prompts")
    text = renderer.render("doc_writing_user.j2", analysis_result="...", df_text="...")
"""

from pathlib import Path
from typing import Any, Optional, Union

from jinja2 import Environment, FileSystemLoader, Undefined, meta

from utils import logger


# 默认模板目录
_DEFAULT_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "prompts"


# ============================================================
# 自定义 Undefined：遇到缺失变量不报错，返回空字符串并警告
# ============================================================

class _SilentUndefined(Undefined):
    """
    渲染时遇到未提供的模板变量，不抛出异常，
    而是返回空字符串并记录一条警告日志。
    """

    def __str__(self) -> str:
        logger.warning(
            f"[PromptRenderer] 模板变量未提供: '{self._undefined_name}'，已替换为空字符串"
        )
        return ""

    # 保证 {% if undefined_var %} 判定为 False
    def __bool__(self) -> bool:
        return False

    def __iter__(self):
        return iter([])

    def __len__(self) -> int:
        return 0


# ============================================================
# 渲染器
# ============================================================

class PromptRenderer:
    """
    Prompt 模板渲染器。

    Args:
        template_dir: 模板文件所在目录，默认为项目根目录下的 prompts/
    """

    def __init__(self, template_dir: Union[str, Path] = _DEFAULT_TEMPLATE_DIR):
        self.template_dir = Path(template_dir)
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            undefined=_SilentUndefined,
            keep_trailing_newline=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        logger.info(f"PromptRenderer initialized: template_dir={self.template_dir}")

    def render(
        self,
        template_name: str,
        **kwargs: Any,
    ) -> str:
        """
        渲染指定模板文件。

        Args:
            template_name: 模板文件名（如 "data_analysis_user.j2"）
            **kwargs: 填充变量

        Returns:
            str: 渲染后的文本
        """
        # 加载模板
        template = self.env.get_template(template_name)

        # AST 静态分析：找出模板中引用的变量名
        source = self.env.loader.get_source(self.env, template_name)[0]
        ast = self.env.parse(source)
        template_vars = meta.find_undeclared_variables(ast)

        # 检查多余变量（调用方提供了但模板中未引用）
        provided_vars = set(kwargs.keys())
        extra_vars = provided_vars - template_vars
        if extra_vars:
            logger.warning(
                f"[PromptRenderer] 模板 '{template_name}' 未使用以下变量: "
                f"{extra_vars}，已忽略"
            )

        # 渲染（缺失变量由 _SilentUndefined 在运行时处理并警告）
        rendered = template.render(**kwargs)
        return rendered.strip()

    def render_string(
        self,
        template_str: str,
        **kwargs: Any,
    ) -> str:
        """
        从字符串模板渲染（不依赖文件）。

        Args:
            template_str: Jinja2 模板字符串
            **kwargs: 填充变量

        Returns:
            str: 渲染后的文本
        """
        template = self.env.from_string(template_str)
        return template.render(**kwargs).strip()

    def list_templates(self) -> list:
        """列出模板目录下所有可用模板文件"""
        return self.env.list_templates()


# ============================================================
# 模块级单例 & 便捷函数
# ============================================================

_default_renderer: Optional[PromptRenderer] = None


def get_renderer(
    template_dir: Union[str, Path] = _DEFAULT_TEMPLATE_DIR,
) -> PromptRenderer:
    """获取（或创建）默认渲染器单例"""
    global _default_renderer
    if (
        _default_renderer is None
        or _default_renderer.template_dir != Path(template_dir)
    ):
        _default_renderer = PromptRenderer(template_dir)
    return _default_renderer


def render_prompt(template_name: str, **kwargs: Any) -> str:
    """
    便捷函数：使用默认渲染器渲染指定模板。

    Args:
        template_name: 模板文件名
        **kwargs: 填充变量

    Returns:
        str: 渲染后的文本
    """
    return get_renderer().render(template_name, **kwargs)
