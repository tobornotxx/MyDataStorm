"""
LLM 通用基类模块
提供 OpenAI-compatible API 的统一封装，支持：
- 同步 / 异步调用
- 流式 / 非流式输出
- 多轮对话管理
- 自动重试与错误处理
- 轻松扩展子类

本模块服务于项目中所有需要调用 LLM 的场景，包括 CodeAgent 和其他模块。
"""

import os
import time
from abc import ABC, abstractmethod
from typing import (
    Any, Dict, List, Optional, Union, Generator, AsyncGenerator, Callable,
)
from dataclasses import dataclass, field

from dotenv import load_dotenv

from utils import logger

load_dotenv()


# ============================================================
# 数据结构
# ============================================================

@dataclass
class Message:
    """单条消息"""
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    name: Optional[str] = None  # 用于 tool 消息

    def to_dict(self) -> Dict[str, str]:
        d = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        return d


@dataclass
class LLMResponse:
    """LLM 调用结果的统一表示"""
    content: str
    model: str = ""
    usage: Optional[Dict[str, int]] = None  # prompt_tokens, completion_tokens, total_tokens
    finish_reason: Optional[str] = None
    raw_response: Optional[Any] = None  # 保留原始响应对象

    def __str__(self) -> str:
        return self.content


@dataclass
class LLMConfig:
    """
    LLM 配置，集中管理所有连接和生成参数。
    
    优先级: 显式传参 > 环境变量 > 默认值
    """
    model: str = ""
    api_base: str = ""
    api_key: str = ""

    # 生成参数
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: Optional[int] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None

    # 重试参数
    max_retries: int = 3
    retry_delay: float = 1.0  # 首次重试等待秒数
    retry_backoff: float = 2.0  # 指数退避倍数
    timeout: Optional[float] = 120.0  # 请求超时秒数

    def __post_init__(self):
        # 从环境变量补全空值
        if not self.model:
            self.model = os.getenv("MODEL_DEFAULT", "")
        if not self.api_base:
            self.api_base = os.getenv("API_BASE_DEFAULT", "")
        if not self.api_key:
            self.api_key = os.getenv("API_KEY_DEFAULT", "")


# ============================================================
# 基类
# ============================================================

class BaseLLM(ABC):
    """
    LLM 调用的抽象基类。
    
    子类只需实现 _call_api() 和可选的 _call_api_stream()，
    即可获得重试、日志、对话管理等通用能力。
    
    用法示例::
    
        class MyLLM(BaseLLM):
            def _call_api(self, messages, **kwargs):
                # 调用具体 API
                ...
        
        llm = MyLLM(config=LLMConfig(model="gpt-4"))
        response = llm.chat("你好")
    """

    def __init__(self, config: Optional[LLMConfig] = None, **kwargs):
        self.config = config or LLMConfig(**kwargs)
        self._history: List[Message] = []
        self._system_prompt: Optional[str] = None

    # ---- 核心抽象方法 ----

    @abstractmethod
    def _call_api(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> LLMResponse:
        """
        子类实现：发送请求到 LLM API 并返回结果。
        
        Args:
            messages: OpenAI 格式的消息列表
            **kwargs: 额外参数（会覆盖 config 中的同名参数）
        
        Returns:
            LLMResponse
        """
        ...

    def _call_api_stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Generator[str, None, None]:
        """
        子类可选实现：流式调用。
        默认实现回退到非流式调用。
        """
        response = self._call_api(messages, **kwargs)
        yield response.content

    # ---- 公开接口 ----

    def set_system_prompt(self, prompt: str) -> "BaseLLM":
        """设置系统提示词，返回 self 以支持链式调用"""
        self._system_prompt = prompt
        return self

    def clear_history(self) -> "BaseLLM":
        """清空对话历史"""
        self._history.clear()
        return self

    @property
    def history(self) -> List[Message]:
        """返回对话历史的副本"""
        return list(self._history)

    def chat(
        self,
        message: str,
        *,
        keep_history: bool = True,
        **kwargs,
    ) -> LLMResponse:
        """
        发送消息并获取回复（自动管理对话历史 + 重试）。
        
        Args:
            message: 用户消息
            keep_history: 是否将本轮对话追加到历史
            **kwargs: 覆盖 config 中的生成参数
            
        Returns:
            LLMResponse
        """
        messages = self._build_messages(message)
        response = self._call_with_retry(messages, **kwargs)

        if keep_history:
            self._history.append(Message(role="user", content=message))
            self._history.append(Message(role="assistant", content=response.content))

        return response

    def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> LLMResponse:
        """
        无状态调用：直接传入完整消息列表（不使用内部历史）。
        
        Args:
            messages: OpenAI 格式消息列表
            **kwargs: 覆盖 config 中的生成参数
            
        Returns:
            LLMResponse
        """
        return self._call_with_retry(messages, **kwargs)

    def stream(
        self,
        message: str,
        *,
        keep_history: bool = True,
        **kwargs,
    ) -> Generator[str, None, None]:
        """
        流式发送消息，逐 chunk 返回文本。
        
        Args:
            message: 用户消息
            keep_history: 是否保存完整回复到历史
            **kwargs: 覆盖 config 中的生成参数
            
        Yields:
            str: 文本片段
        """
        messages = self._build_messages(message)
        full_content = []
        for chunk in self._call_api_stream(
            messages, **self._merge_kwargs(kwargs)
        ):
            full_content.append(chunk)
            yield chunk

        if keep_history:
            complete_text = "".join(full_content)
            self._history.append(Message(role="user", content=message))
            self._history.append(Message(role="assistant", content=complete_text))

    def batch(
        self,
        messages_list: List[str],
        *,
        keep_history: bool = False,
        **kwargs,
    ) -> List[LLMResponse]:
        """
        批量调用（串行），每条消息独立处理。
        
        Args:
            messages_list: 多条用户消息
            keep_history: 是否保存到历史
            **kwargs: 覆盖 config 中的生成参数
            
        Returns:
            List[LLMResponse]
        """
        results = []
        for msg in messages_list:
            resp = self.chat(msg, keep_history=keep_history, **kwargs)
            results.append(resp)
        return results

    # ---- 内部方法 ----

    def _build_messages(self, user_message: str) -> List[Dict[str, str]]:
        """组装完整消息列表：system + history + 当前消息"""
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        for msg in self._history:
            messages.append(msg.to_dict())
        messages.append({"role": "user", "content": user_message})
        return messages

    def _merge_kwargs(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """将 config 参数与调用时覆盖参数合并"""
        base = {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        }
        if self.config.max_tokens is not None:
            base["max_tokens"] = self.config.max_tokens
        if self.config.seed is not None:
            base["seed"] = self.config.seed
        if self.config.stop is not None:
            base["stop"] = self.config.stop
        base.update(overrides)
        return base

    def _call_with_retry(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> LLMResponse:
        """带重试的调用包装"""
        merged = self._merge_kwargs(kwargs)
        last_error = None
        delay = self.config.retry_delay

        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = self._call_api(messages, **merged)
                if attempt > 1:
                    logger.info(f"LLM call succeeded on attempt {attempt}")
                return response
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries:
                    logger.warning(
                        f"LLM call failed (attempt {attempt}/{self.config.max_retries}): "
                        f"{type(e).__name__}: {e}. Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay *= self.config.retry_backoff
                else:
                    logger.error(
                        f"LLM call failed after {self.config.max_retries} attempts: "
                        f"{type(e).__name__}: {e}"
                    )

        raise last_error  # type: ignore[misc]


# ============================================================
# OpenAI-compatible 实现
# ============================================================

class OpenAILikeLLM(BaseLLM):
    """
    基于 openai SDK 的 OpenAI-compatible API 实现。
    
    支持所有兼容 OpenAI API 格式的服务（OpenAI / Azure OpenAI / vLLM /
    Ollama / LM Studio / SiliconFlow / DeepSeek / 通义千问 等）。
    
    用法::
    
        # 使用环境变量配置
        llm = OpenAILikeLLM()
        
        # 显式配置
        llm = OpenAILikeLLM(config=LLMConfig(
            model="deepseek-chat",
            api_base="https://api.deepseek.com/v1",
            api_key="sk-xxx",
            temperature=0.3,
        ))
        
        # 单轮
        resp = llm.chat("你好")
        print(resp.content)
        
        # 多轮
        llm.set_system_prompt("你是一个数据分析助手。")
        llm.chat("分析一下这个表格...")
        llm.chat("再帮我看看趋势")
        
        # 流式
        for chunk in llm.stream("写一首诗"):
            print(chunk, end="", flush=True)
    """

    def __init__(self, config: Optional[LLMConfig] = None, **kwargs):
        super().__init__(config, **kwargs)
        self._client = None
        self._async_client = None

    @property
    def client(self):
        """延迟初始化 OpenAI client"""
        if self._client is None:
            from openai import OpenAI
            client_kwargs: Dict[str, Any] = {
                "api_key": self.config.api_key,
            }
            if self.config.api_base:
                client_kwargs["base_url"] = self.config.api_base
            if self.config.timeout:
                client_kwargs["timeout"] = self.config.timeout
            self._client = OpenAI(**client_kwargs)
            logger.info(
                f"OpenAI client initialized: model={self.config.model}, "
                f"base_url={self.config.api_base or 'default'}"
            )
        return self._client

    @property
    def async_client(self):
        """延迟初始化 AsyncOpenAI client"""
        if self._async_client is None:
            from openai import AsyncOpenAI
            client_kwargs: Dict[str, Any] = {
                "api_key": self.config.api_key,
            }
            if self.config.api_base:
                client_kwargs["base_url"] = self.config.api_base
            if self.config.timeout:
                client_kwargs["timeout"] = self.config.timeout
            self._async_client = AsyncOpenAI(**client_kwargs)
        return self._async_client

    def _call_api(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> LLMResponse:
        """调用 OpenAI-compatible chat completions API"""
        request_params = {
            "model": self.config.model,
            "messages": messages,
            **kwargs,
        }
        # 移除值为 None 的参数
        request_params = {k: v for k, v in request_params.items() if v is not None}

        response = self.client.chat.completions.create(**request_params)

        choice = response.choices[0]
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            content=choice.message.content or "",
            model=response.model or self.config.model,
            usage=usage,
            finish_reason=choice.finish_reason,
            raw_response=response,
        )

    def _call_api_stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Generator[str, None, None]:
        """流式调用 OpenAI-compatible API"""
        request_params = {
            "model": self.config.model,
            "messages": messages,
            "stream": True,
            **kwargs,
        }
        request_params = {k: v for k, v in request_params.items() if v is not None}

        stream = self.client.chat.completions.create(**request_params)
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def achat(
        self,
        message: str,
        *,
        keep_history: bool = True,
        **kwargs,
    ) -> LLMResponse:
        """异步版 chat"""
        messages = self._build_messages(message)
        merged = self._merge_kwargs(kwargs)
        response = await self._acall_api(messages, **merged)

        if keep_history:
            self._history.append(Message(role="user", content=message))
            self._history.append(Message(role="assistant", content=response.content))

        return response

    async def _acall_api(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> LLMResponse:
        """异步调用 OpenAI-compatible API"""
        request_params = {
            "model": self.config.model,
            "messages": messages,
            **kwargs,
        }
        request_params = {k: v for k, v in request_params.items() if v is not None}

        response = await self.async_client.chat.completions.create(**request_params)

        choice = response.choices[0]
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            content=choice.message.content or "",
            model=response.model or self.config.model,
            usage=usage,
            finish_reason=choice.finish_reason,
            raw_response=response,
        )

    async def astream(
        self,
        message: str,
        *,
        keep_history: bool = True,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """异步流式调用"""
        messages = self._build_messages(message)
        merged = self._merge_kwargs(kwargs)

        request_params = {
            "model": self.config.model,
            "messages": messages,
            "stream": True,
            **merged,
        }
        request_params = {k: v for k, v in request_params.items() if v is not None}

        full_content = []
        stream = await self.async_client.chat.completions.create(**request_params)
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                full_content.append(text)
                yield text

        if keep_history:
            complete_text = "".join(full_content)
            self._history.append(Message(role="user", content=message))
            self._history.append(Message(role="assistant", content=complete_text))


# ============================================================
# 便捷工厂函数
# ============================================================

def create_llm(
    model: str = "",
    api_base: str = "",
    api_key: str = "",
    **kwargs,
) -> OpenAILikeLLM:
    """
    快速创建 LLM 实例的工厂函数。
    
    Args:
        model: 模型名称，默认读取 MODEL_DEFAULT 环境变量
        api_base: API 地址，默认读取 API_BASE_DEFAULT 环境变量
        api_key: API 密钥，默认读取 API_KEY_DEFAULT 环境变量
        **kwargs: 其他 LLMConfig 参数（temperature, max_tokens 等）
        
    Returns:
        OpenAILikeLLM 实例
        
    示例::
    
        llm = create_llm(model="deepseek-chat", temperature=0.3)
        print(llm.chat("你好"))
    """
    config = LLMConfig(
        model=model,
        api_base=api_base,
        api_key=api_key,
        **kwargs,
    )
    return OpenAILikeLLM(config=config)
