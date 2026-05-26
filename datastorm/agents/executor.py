"""Executor Agent — ReAct 风格数据库代理。

论文 Section 3.2.1 & Appendix B:
Executor 负责低层查询翻译, 通过动态探索数据库内容来确定最合适的 SQL 响应。
使用 ReAct 循环 (Yao et al., 2023), 基于 Liu et al. (2024a) 的实现。

可用动作 (论文 Appendix B):
1. get_tables()                               → 获取所有表及简要描述
2. retrieve_tables_details([table_names])      → 获取表详细信息
3. execute_sql(sql)                            → 执行 SQL 查询
4. execute_python_from_sql(sql, python_code)   → 基于 SQL 结果执行 Python
5. stop()                                      → 标记最终 SQL 并终止

代理运行直到调用 stop() 或达到 15 轮上限 (论文 Appendix B)。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from datastorm.config import DataSTORMConfig
from datastorm.database.connector import DatabaseConnector
from datastorm.llm.client import LLMClient
from datastorm.prompts import renderer, templates
from datastorm.types import ExecutorResponse

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Executor 对话历史中的一轮。"""

    question: str
    action_history: list[str] = field(default_factory=list)
    response: str = ""


class ExecutorAgent:
    """ReAct 风格 Executor Agent (论文 Appendix B, Prompt 22)。

    在每一轮中, 代理检查当前观察, 选择一个动作及参数,
    系统执行它并将结果追加到上下文中。
    """

    def __init__(
        self,
        llm: LLMClient,
        db: DatabaseConnector,
        config: DataSTORMConfig,
    ) -> None:
        self._llm = llm
        self._db = db
        self._config = config
        self._conversation_history: list[ConversationTurn] = []

    def execute(self, question: str, context: str = "") -> ExecutorResponse:
        """执行一个探索问题。

        实现论文 Appendix B 的 ReAct 循环:
        代理运行直到调用 stop() 或达到 max_turns 上限。

        Args:
            question: 自然语言探索问题 (q_{i,j})
            context: 可选的额外上下文 (如一致性跟进的原始 SQL)

        Returns:
            ExecutorResponse 包含答案、SQL、汇总统计等
        """
        max_turns = self._config.exploration.executor_max_turns
        action_history: list[str] = []
        last_sql = ""
        last_result_df = None
        full_question = f"{question}\n\nAdditional context: {context}" if context else question
        consecutive_parse_errors = 0

        logger.debug("Executor starting: question=%r, max_turns=%d", question, max_turns)

        for turn in range(max_turns):
            # 构造 Prompt (Prompt 22, Table 22)
            prompt = renderer.render(
                templates.EXECUTOR_MAIN,
                database_type=self._config.database.database_type,
                curr_date=datetime.now().strftime("%Y-%m-%d"),
                conversation_history=[
                    {
                        "question": t.question,
                        "action_history": t.action_history,
                        "response": t.response,
                    }
                    for t in self._conversation_history
                ],
                question=full_question,
                action_history=action_history,
            )

            # 调用 LLM 获取 Thought + Action
            response = self._llm.generate(
                prompt,
                model=self._config.llm.exploration_model,
                temperature=0.0,
            )

            # 解析 Thought 和 Action
            thought, action_name, action_args = self._parse_response(response)

            logger.debug(
                "Executor turn %d: Thought=%r, Action=%s(%s)",
                turn, thought[:200], action_name, action_args[:300],
            )

            if not action_name:
                consecutive_parse_errors += 1
                logger.warning(
                    "Turn %d: Could not parse action from response (consecutive errors: %d)",
                    turn, consecutive_parse_errors,
                )
                logger.debug("Turn %d: raw LLM response:\n%s", turn, response[:500])
                action_history.append(f"Turn {turn}: [Parse Error] {response[:200]}")
                # 连续解析失败 2 次则提前终止，避免浪费轮次
                if consecutive_parse_errors >= 2:
                    logger.warning("Executor: %d consecutive parse errors, terminating early", consecutive_parse_errors)
                    break
                continue
            else:
                consecutive_parse_errors = 0

            # 执行动作
            observation, executed_sql, result_df = self._execute_action(action_name, action_args)

            logger.debug(
                "Executor turn %d: Observation (%d chars):\n%s",
                turn, len(observation), observation[:500],
            )

            # 记录动作历史
            action_entry = (
                f"Turn {turn}:\n"
                f"Thought: {thought}\n"
                f"Action: {action_name}({action_args})\n"
                f"Observation: {observation[:2000]}"
            )
            action_history.append(action_entry)

            # 追踪最后的 SQL 和结果
            if executed_sql:
                last_sql = executed_sql
                if result_df is not None:
                    last_result_df = result_df

            # 检查是否终止
            if action_name == "stop":
                break

        # 生成最终响应摘要
        answer = self._generate_answer_summary(question, action_history)

        logger.debug(
            "Executor finished: question=%r, turns=%d, last_sql=%r, answer=%r",
            question, len(action_history), last_sql[:200], answer[:300],
        )

        # 保存到对话历史
        self._conversation_history.append(
            ConversationTurn(
                question=question,
                action_history=action_history,
                response=answer,
            )
        )

        return ExecutorResponse(
            question=question,
            answer=answer,
            sql=last_sql,
            raw_results=last_result_df,
            row_count=len(last_result_df) if last_result_df is not None else 0,
        )

    def _parse_response(self, response: str) -> tuple[str, str, str]:
        """解析 LLM 响应中的 Thought 和 Action。

        Returns:
            (thought, action_name, action_args)
        """
        thought = ""
        action_name = ""
        action_args = ""

        # 清洗 markdown 包裹（``` 代码块、**bold** 等）
        cleaned = response
        # 去掉可能的 ``` 代码块包裹
        for tag in ("```", "```markdown", "```text", "```plaintext"):
            if cleaned.startswith(tag):
                cleaned = cleaned[len(tag):]
            if cleaned.rstrip().endswith("```"):
                cleaned = cleaned.rstrip()[:-3]
        cleaned = cleaned.strip()

        # 去掉可能的 **bold** 包裹 (e.g., **Thought:** / **Action:**)
        cleaned = re.sub(r"\*\*(Thought|Action)\*\*:", r"\1:", cleaned)

        # 提取 Thought (支持 Thought: / thought: / **Thought:** 等变体)
        thought_match = re.search(
            r"[Tt]hought\s*:\s*(.*?)(?=[Aa]ction\s*:|$)", cleaned, re.DOTALL
        )
        if thought_match:
            thought = thought_match.group(1).strip()

        # 提取 Action — 匹配 Action: name(args) 或 Action: name
        action_match = re.search(
            r"[Aa]ction\s*:\s*(\w+)\s*\((.*)\)\s*$", cleaned, re.DOTALL
        )
        if not action_match:
            action_match = re.search(
                r"[Aa]ction\s*:\s*(\w+)\s*\((.*)\)", cleaned, re.DOTALL
            )
        if not action_match:
            action_match = re.search(
                r"[Aa]ction\s*:\s*(\w+)\s*$", cleaned, re.DOTALL
            )

        if action_match:
            action_name = action_match.group(1).strip()
            if action_match.lastindex and action_match.lastindex >= 2:
                action_args = action_match.group(2).strip()

        if not action_name:
            logger.error(
                "Executor: failed to parse action. Raw response:\n%s",
                response[:800],
            )

        return thought, action_name, action_args

    def _execute_action(
        self, action_name: str, action_args: str
    ) -> tuple[str, str, Any]:
        """执行解析出的动作。

        对应论文 Appendix B 中的 5 个可用动作。

        Returns:
            (observation_text, executed_sql, result_dataframe_or_None)
        """
        try:
            if action_name == "get_tables":
                return self._db.get_tables(), "", None

            elif action_name == "retrieve_tables_details":
                table_names = self._parse_list_arg(action_args)
                return self._db.retrieve_tables_details(table_names), "", None

            elif action_name == "execute_sql":
                sql = action_args.strip("'\"")
                df, summary = self._db.execute_sql(sql)
                return summary, sql, df

            elif action_name == "execute_python_from_sql":
                parts = self._parse_tuple_arg(action_args)
                if len(parts) >= 2:
                    result = self._db.execute_python_from_sql(parts[0], parts[1])
                    return result, parts[0], None
                return "Error: expected (sql, python_code) tuple", "", None

            elif action_name == "stop":
                return "Process terminated.", "", None

            else:
                return f"Unknown action: {action_name}", "", None

        except Exception as e:
            logger.error("Action '%s' failed: %s", action_name, e)
            return f"Error: {e}", "", None

    def _parse_list_arg(self, args: str) -> list[str]:
        """解析列表参数, 如 ["table1", "table2"]。"""
        args = args.strip()
        # 移除方括号
        if args.startswith("["):
            args = args[1:]
        if args.endswith("]"):
            args = args[:-1]
        # 分割并清理
        items = []
        for item in args.split(","):
            item = item.strip().strip("'\"")
            if item:
                items.append(item)
        return items

    def _parse_tuple_arg(self, args: str) -> list[str]:
        """解析元组参数, 如 ("sql", "python")。"""
        # 尝试匹配两个引号包裹的字符串
        matches = re.findall(r'"((?:[^"\\]|\\.)*)"', args)
        if len(matches) >= 2:
            return matches
        # 回退: 按逗号分割
        parts = args.split(",", 1)
        return [p.strip().strip("'\"") for p in parts]

    def _generate_answer_summary(self, question: str, action_history: list[str]) -> str:
        """根据执行历史生成自然语言答案摘要。"""
        history_text = "\n\n".join(action_history[-5:])  # 使用最近的历史
        prompt = (
            f"Based on the following database exploration for the question: '{question}'\n\n"
            f"Exploration history:\n{history_text}\n\n"
            f"Provide a concise natural language answer summarizing the key findings. "
            f"Include specific numbers and patterns discovered."
        )
        return self._llm.generate(prompt, temperature=0.3, max_completion_tokens=1024)

    def reset_history(self) -> None:
        """重置对话历史。"""
        self._conversation_history.clear()
