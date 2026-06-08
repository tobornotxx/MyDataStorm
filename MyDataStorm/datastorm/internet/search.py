"""互联网搜索接口。

提供 Web 搜索能力, 用于:
1. Warm-start 阶段的互联网研究 (论文 Section 3.1)
2. Planner 路由到 "internet" 的问题 (论文 Section 3.2.1)
3. 报告生成阶段的补充 web 查询 (论文 Section 3.3)

默认使用 Serper API, 论文中实现了自定义 web 搜索服务器,
支持域名屏蔽和日期限制以防止信息泄漏。
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from datastorm.config import InternetConfig

logger = logging.getLogger(__name__)


class WebSearchResult:
    """单条搜索结果。"""

    def __init__(self, title: str, url: str, snippet: str, content: str = "") -> None:
        self.title = title
        self.url = url
        self.snippet = snippet
        self.content = content

    def to_text(self) -> str:
        parts = [f"Title: {self.title}", f"URL: {self.url}", f"Snippet: {self.snippet}"]
        if self.content:
            parts.append(f"Content: {self.content[:2000]}")
        return "\n".join(parts)


class WebSearcher:
    """Web 搜索器, 基于 Serper API。

    论文 Section 4.2: 实现了自定义 web 搜索服务器 via Serper,
    支持屏蔽 ACLED 域名和限制发布日期。
    """

    SERPER_URL = "https://google.serper.dev/search"

    def __init__(self, config: InternetConfig) -> None:
        self._config = config

    def search(self, query: str, num_results: int | None = None) -> list[WebSearchResult]:
        """执行 Web 搜索。

        Args:
            query: 搜索查询
            num_results: 返回结果数量

        Returns:
            搜索结果列表
        """
        if not self._config.serper_api_key:
            logger.warning("No Serper API key configured, returning empty results")
            return []

        num_results = num_results or self._config.max_results_per_query

        headers = {
            "X-API-KEY": self._config.serper_api_key,
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "q": query,
            "num": num_results,
        }
        # 论文 Section 4.2: 日期限制
        if self._config.date_restrict:
            payload["tbs"] = f"cdr:1,cd_max:{self._config.date_restrict}"

        try:
            resp = requests.post(self.SERPER_URL, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error("Web search failed: %s", e)
            return []

        results = []
        for item in data.get("organic", [])[:num_results]:
            url = item.get("link", "")
            # 论文 Section 4.2: 域名屏蔽
            if any(blocked in url for blocked in self._config.blocked_domains):
                continue
            results.append(
                WebSearchResult(
                    title=item.get("title", ""),
                    url=url,
                    snippet=item.get("snippet", ""),
                )
            )

        return results

    def search_and_format(self, query: str, num_results: int | None = None) -> str:
        """搜索并格式化为文本。"""
        results = self.search(query, num_results)
        if not results:
            return f"No web results found for: {query}"
        parts = [f"Web search results for: {query}\n"]
        for i, r in enumerate(results, 1):
            parts.append(f"[{i}] {r.to_text()}\n")
        return "\n".join(parts)
