"""
LLM 模块
提供 OpenAI-compatible API 的统一封装。
"""

from llm.llm import (
    BaseLLM,
    OpenAILikeLLM,
    LLMConfig,
    LLMResponse,
    Message,
    create_llm,
)

__all__ = [
    "BaseLLM",
    "OpenAILikeLLM",
    "LLMConfig",
    "LLMResponse",
    "Message",
    "create_llm",
]
