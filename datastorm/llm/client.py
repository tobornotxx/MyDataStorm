"""LLM 客户端封装。

提供统一的 LLM 调用接口, 支持:
- 文本生成 (chat completion)
- JSON 模式输出
- 重试逻辑
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from openai import OpenAI

from datastorm.config import LLMConfig, ScenarioConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """OpenAI LLM 客户端封装。

    支持按「场景 (scenario)」调用不同模型/端点: 每个场景在 ``llm_config.json``
    的 ``scenarios`` 字段里可单独配置 model_name / api_base / api_key 等。
    内部按 ``(api_base, api_key)`` 缓存多个 OpenAI 客户端, 调用时根据场景选用。
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        # (api_base, api_key) → OpenAI 客户端缓存, 避免重复构造
        self._clients: dict[tuple[str, str], OpenAI] = {}

    def _get_client(self, api_base: str, api_key: str) -> OpenAI:
        """按 (api_base, api_key) 复用 OpenAI 客户端。"""
        cache_key = (api_base or "", api_key or "")
        client = self._clients.get(cache_key)
        if client is None:
            kwargs: dict[str, Any] = {"api_key": api_key}
            if api_base:
                kwargs["base_url"] = api_base
            client = OpenAI(**kwargs)
            self._clients[cache_key] = client
        return client

    def generate(
        self,
        prompt: str,
        scenario: str = "default",
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
            scenario: 场景名, 决定使用的 model/api_base/api_key 及默认 temperature/
                max_completion_tokens。在 ``llm_config.json`` 的 ``scenarios`` 中配置,
                未配置的字段继承 ``default``。
            model: 显式模型名, 覆盖该场景的 model_name
            system_prompt: 系统 prompt
            temperature: 温度, 覆盖该场景默认值
            max_completion_tokens: 最大 token 数, 覆盖该场景默认值
            json_mode: 是否要求 JSON 输出
            max_retries: 最大重试次数

        Returns:
            生成的文本
        """
        sc: ScenarioConfig = self._config.scenario(scenario)
        model = model or sc.model_name
        temperature = temperature if temperature is not None else sc.temperature
        max_completion_tokens = max_completion_tokens or sc.max_completion_tokens
        client = self._get_client(sc.api_base, sc.api_key)

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
        logger.debug("LLM prompt (%d chars):\n%s", len(prompt), prompt[:6000])

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content or ""
                usage = response.usage
                logger.debug(
                    "LLM response (%d chars, tokens: prompt=%s completion=%s):\n%s",
                    len(content),
                    usage.prompt_tokens if usage else "?",
                    usage.completion_tokens if usage else "?",
                    content[:6000],
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
        scenario: str = "default",
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_completion_tokens: int | None = None,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """生成 JSON 输出。

        Args:
            prompt: 用户 prompt
            scenario: 场景名 (见 :meth:`generate`)
            model: 显式模型名, 覆盖该场景的 model_name
            system_prompt: 系统 prompt
            temperature: 温度, 覆盖该场景默认值
            max_completion_tokens: 最大 token 数, 覆盖该场景默认值
            max_retries: 最大重试次数

        Returns:
            解析后的 JSON 字典
        """
        text = self.generate(
            prompt=prompt,
            scenario=scenario,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            json_mode=True,
            max_retries=max_retries,
        )
        # 尝试解析 JSON, 处理可能的 markdown 包裹
        text = text.strip()

        # 方法 1: 去除 markdown 代码块包裹
        # 处理 ```json / ```JSON / ``` 等各种变体
        code_block_match = re.search(
            r"```(?:json|JSON|js)?\s*\n?(.*?)```", text, re.DOTALL
        )
        if code_block_match:
            text = code_block_match.group(1).strip()
        else:
            # 没有代码块包裹，去除常见前缀
            if text.startswith("```"):
                text = re.sub(r"^```\w*\s*\n?", "", text)
            if text.endswith("```"):
                text = text[:-3].strip()

        # 方法 2: 直接 JSON 解析
        try:
            parsed = json.loads(text)
            logger.debug("LLM JSON parsed keys: %s", list(parsed.keys()) if isinstance(parsed, dict) else type(parsed).__name__)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            pass

        # 方法 3: 从文本中提取第一个 JSON 对象 (处理 LLM 在 JSON 前后加了闲话的情况)
        # 匹配最外层的 { ... }
        brace_match = re.search(r"\{", text)
        if brace_match:
            start = brace_match.start()
            # 从第一个 { 开始，找到匹配的 }
            depth = 0
            end = start
            in_string = False
            escape_next = False
            for i in range(start, len(text)):
                ch = text[i]
                if escape_next:
                    escape_next = False
                    continue
                if ch == "\\":
                    escape_next = True
                    continue
                if ch == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break

            if end > start:
                json_candidate = text[start:end]
                try:
                    parsed = json.loads(json_candidate)
                    logger.debug("LLM JSON extracted from text, keys: %s", list(parsed.keys()) if isinstance(parsed, dict) else type(parsed).__name__)
                    return parsed if isinstance(parsed, dict) else {}
                except json.JSONDecodeError:
                    pass

        logger.error("Failed to parse JSON from LLM response: %s", text[:500])
        return {}
