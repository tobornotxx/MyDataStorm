"""LLM 客户端封装。

提供统一的 LLM 调用接口, 支持:
- 文本生成 (chat completion)
- JSON 模式输出
- 重试逻辑
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from openai import OpenAI

from datastorm.config import LLMConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """OpenAI LLM 客户端封装。"""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        kwargs = {"api_key": config.api_key}
        if config.api_base:
            kwargs["base_url"] = config.api_base
        self._client = OpenAI(**kwargs)

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_completion_tokens: int | None = None,
        json_mode: bool = False,
        max_retries: int = 3,
    ) -> str:
        """生成文本。

        Args:
            prompt: 用户 prompt
            model: 模型名称, 默认使用 exploration_model
            system_prompt: 系统 prompt
            temperature: 温度
            max_completion_tokens: 最大 token 数
            json_mode: 是否要求 JSON 输出
            max_retries: 最大重试次数

        Returns:
            生成的文本
        """
        model = model or self._config.exploration_model
        temperature = temperature if temperature is not None else self._config.temperature
        max_completion_tokens = max_completion_tokens or self._config.max_completion_tokens

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # OpenAI 要求使用 json_object 模式时 messages 中必须包含 "json" 一词
        if json_mode:
            has_json_word = any("json" in m["content"].lower() for m in messages)
            if not has_json_word:
                if messages[0]["role"] == "system":
                    messages[0]["content"] += "\nRespond in JSON format."
                else:
                    messages.insert(0, {"role": "system", "content": "Respond in JSON format."})

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        logger.debug(
            "LLM request: model=%s, temperature=%s, max_completion_tokens=%s, json_mode=%s",
            model, temperature, max_completion_tokens, json_mode,
        )
        logger.debug("LLM prompt (%d chars):\n%s", len(prompt), prompt[:2000])

        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content or ""
                usage = response.usage
                logger.debug(
                    "LLM response (%d chars, tokens: prompt=%s completion=%s):\n%s",
                    len(content),
                    usage.prompt_tokens if usage else "?",
                    usage.completion_tokens if usage else "?",
                    content[:2000],
                )
                return content
            except Exception as e:
                logger.warning("LLM call failed (attempt %d/%d): %s", attempt + 1, max_retries, e)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

        return ""  # unreachable

    def generate_json(
        self,
        prompt: str,
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_completion_tokens: int | None = None,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """生成 JSON 输出。

        Returns:
            解析后的 JSON 字典
        """
        text = self.generate(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            json_mode=True,
            max_retries=max_retries,
        )
        # 尝试解析 JSON, 处理可能的 markdown 包裹
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            parsed = json.loads(text)
            logger.debug("LLM JSON parsed keys: %s", list(parsed.keys()))
            return parsed
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON from LLM response: %s", text[:200])
            return {}
