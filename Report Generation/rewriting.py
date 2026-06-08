"""
Rewriting 模块
使用 LLM 对文本进行改写/润色。

用法:
    from llm import OpenAILikeLLM, LLMConfig
    from rewriting import Rewriter

    llm = OpenAILikeLLM(config=LLMConfig(
        model="your-model-name",
        api_base="https://your-api-endpoint",
        api_key="your-api-key",
    ))

    rewriter = Rewriter(llm=llm)
    result = rewriter.rewrite("需要改写的文本")
    print(result)
"""

from typing import Optional

from llm import BaseLLM
from utils import logger
from utils.prompt_renderer import render_prompt


class Rewriter:
    """
    文本改写器。
    
    Args:
        llm: BaseLLM 实例（必须传入，不使用默认模型）
        system_prompt: 系统提示词字符串。如果不传，则从 j2 模板渲染。
        system_template: 系统提示词模板文件名，默认为 rewriting_system.j2
    """

    def __init__(
        self,
        llm: BaseLLM,
        system_prompt: Optional[str] = None,
        system_template: str = "rewriting_system.j2",
    ):
        self.llm = llm

        # 加载 system prompt
        if system_prompt is not None:
            self._system_prompt = system_prompt
        else:
            self._system_prompt = render_prompt(system_template)

        logger.info(
            f"Rewriter initialized: model={self.llm.config.model}, "
            f"prompt_len={len(self._system_prompt)}"
        )

    def rewrite(
        self,
        text: str,
        **kwargs,
    ) -> str:
        """
        改写单段文本。
        
        Args:
            text: 需要改写的原始文本
            **kwargs: 覆盖 LLM 生成参数（temperature, max_tokens 等）
            
        Returns:
            str: 改写后的文本
        """
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": text})

        response = self.llm.generate(messages, **kwargs)
        logger.info(
            f"Rewrite done: input_len={len(text)}, output_len={len(response.content)}"
        )
        return response.content

    def rewrite_batch(
        self,
        texts: list[str],
        **kwargs,
    ) -> list[str]:
        """
        批量改写多段文本。
        
        Args:
            texts: 需要改写的文本列表
            **kwargs: 覆盖 LLM 生成参数
            
        Returns:
            list[str]: 改写后的文本列表
        """
        results = []
        for i, text in enumerate(texts):
            logger.info(f"Rewriting [{i+1}/{len(texts)}]...")
            result = self.rewrite(text, **kwargs)
            results.append(result)
        return results
