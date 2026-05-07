"""查询一致性检测模块。

论文 Section 3.2.1 (Query consistency module):
独立并行的探索分支可能在 SQL 谓词上产生不一致。
例如, 在 ACLED 数据库中, 多个列编码事件中的参与者 (actor1, assoc_actor_1, actor2, assoc_actor_2),
独立的 Executor 运行可能以不同方式处理这些列。

该模块:
1. 比较同一层中所有 Executor 的 SQL 查询
2. 参考全局洞察库中已有的 question-query 对作为上下文
3. 检测并标准化不一致的谓词
4. 生成跟进查询以修正不一致
"""

from __future__ import annotations

import json
import logging
from typing import Any

from datastorm.config import DataSTORMConfig
from datastorm.llm.client import LLMClient
from datastorm.prompts import renderer, templates
from datastorm.types import ConsistencyFollowUp, ExecutorResponse, Insight

logger = logging.getLogger(__name__)


class QueryConsistencyModule:
    """查询一致性检测模块 (论文 Section 3.2.1, Prompt 7)。"""

    def __init__(self, llm: LLMClient, config: DataSTORMConfig) -> None:
        self._llm = llm
        self._config = config

    def detect_inconsistencies(
        self,
        responses: list[ExecutorResponse],
        existing_insights: list[Insight],
    ) -> list[ConsistencyFollowUp]:
        """检测一组 Executor 响应中的 SQL 谓词不一致。

        论文 Section 3.2.1:
        模块接收数据库导向的 question-query 对 (q_{i,j}, s_{i,j}),
        并参考 B_{i-1} 中的 question-query 对作为上下文,
        生成一致性跟进查询 q'_{i,j}。

        Args:
            responses: 当前层所有 Executor 响应
            existing_insights: 全局洞察库中已有的洞察

        Returns:
            每个响应对应的一致性跟进
        """
        if len(responses) <= 1:
            # 只有一个查询, 无需一致性检测
            logger.debug("Consistency check skipped: only %d response(s)", len(responses))
            return [
                ConsistencyFollowUp(
                    original_question=r.question,
                    original_sql=r.sql,
                    follow_up_question=None,
                )
                for r in responses
            ]

        # 构造输入 JSON (遵照 Prompt 7 的格式)
        input_data: dict[str, Any] = {}

        # 添加已有洞察作为 example_nodes (不需要跟进)
        for i, insight in enumerate(existing_insights[:3]):  # 限制数量
            if insight.sql:
                input_data[f"example_node_{i}"] = {
                    "query": insight.question,
                    "SQL": insight.sql,
                    "example_node": True,
                    "note": "no need to generate follow_up_question",
                }

        # 添加当前层的查询
        for i, resp in enumerate(responses):
            if not resp.sql:
                continue
            input_data[f"query{i}"] = {
                "previous_queries": None,
                "query": resp.question,
                "SQL": resp.sql,
            }

        logger.debug(
            "Consistency check: %d responses, %d example nodes",
            len(responses),
            sum(1 for k in input_data if k.startswith("example_node")),
        )

        if not any(k.startswith("query") for k in input_data):
            return [
                ConsistencyFollowUp(
                    original_question=r.question,
                    original_sql=r.sql,
                    follow_up_question=None,
                )
                for r in responses
            ]

        # 调用 LLM (Prompt 7, Table 7)
        prompt = renderer.render(
            templates.QUERY_CONSISTENCY,
            input=json.dumps(input_data, indent=2),
        )

        result = self._llm.generate_json(prompt, temperature=0.0)

        # 解析结果
        follow_ups = []
        for i, resp in enumerate(responses):
            key = f"query{i}"
            if key in result:
                follow_up = result[key].get("follow_up_question")
                follow_ups.append(
                    ConsistencyFollowUp(
                        original_question=resp.question,
                        original_sql=resp.sql,
                        follow_up_question=follow_up if follow_up and follow_up != "None" else None,
                    )
                )
                if follow_up and follow_up != "None":
                    logger.debug(
                        "Consistency: query%d needs follow-up: %r", i, follow_up[:200]
                    )
            else:
                follow_ups.append(
                    ConsistencyFollowUp(
                        original_question=resp.question,
                        original_sql=resp.sql,
                        follow_up_question=None,
                    )
                )

        logger.debug(
            "Consistency check done: %d/%d queries need follow-up",
            sum(1 for f in follow_ups if f.follow_up_question), len(follow_ups),
        )

        return follow_ups
