"""问题链和继承关系日志模块。

记录每一层提出的问题、问题的父节点关系、执行结果,
并持久化到本地 JSON 文件, 便于追踪问题的演化路径。
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datastorm.types import QuestionNode

logger = logging.getLogger(__name__)


class QuestionLogger:
    """问题树日志管理器。

    维护所有问题的完整记录，包括:
    - 问题的链和继承关系 (parent_ids)
    - 每个问题的类别 (follow_up / exploratory)
    - 执行状态和结果摘要
    - 时间戳

    日志保存在 question_tree.json 中。
    """

    def __init__(self, output_dir: str | None = None) -> None:
        self._output_dir = output_dir or os.getcwd()
        self._nodes: list[dict[str, Any]] = []
        self._start_time = datetime.now(timezone.utc).isoformat()

    def set_output_dir(self, output_dir: str) -> None:
        """设置输出目录。"""
        self._output_dir = output_dir

    def log_node(self, node: QuestionNode) -> None:
        """记录一个问题节点。"""
        entry = {
            "id": node.id,
            "question": node.question,
            "destination": node.destination.value if hasattr(node.destination, "value") else str(node.destination),
            "layer": node.layer,
            "parent_ids": node.parent_ids[:],
            "category": node.category,
            "has_answer": bool(node.answer),
            "answer_preview": node.answer[:200] if node.answer else "",
            "sql_preview": node.sql[:200] if node.sql else "",
            "row_count": node.row_count,
            "logged_at": datetime.now(timezone.utc).isoformat(),
        }
        self._nodes.append(entry)
        logger.debug("Logged question node [%s] (layer %d, %s)", node.id, node.layer, node.category)

    def log_nodes(self, nodes: list[QuestionNode]) -> None:
        """批量记录问题节点。"""
        for node in nodes:
            self.log_node(node)

    def save(self, filepath: str | None = None) -> str:
        """持久化问题树到 JSON 文件。"""
        if filepath is None:
            filepath = str(Path(self._output_dir) / "question_tree.json")

        output = {
            "session_start": self._start_time,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "total_nodes": len(self._nodes),
            "by_layer": self._build_layer_summary(),
            "by_category": self._build_category_summary(),
            "nodes": self._nodes,
            "lineage": self._build_lineage_graph(),
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info("Question tree saved to %s (%d nodes)", filepath, len(self._nodes))
        return filepath

    def _build_layer_summary(self) -> dict[str, Any]:
        """构建按层的汇总统计。"""
        by_layer: dict[int, list[str]] = {}
        for node in self._nodes:
            layer = node["layer"]
            by_layer.setdefault(layer, []).append(node["id"])
        return {
            str(k): {"node_count": len(v), "node_ids": v}
            for k, v in sorted(by_layer.items())
        }

    def _build_category_summary(self) -> dict[str, int]:
        """构建按类别的汇总统计。"""
        counts: dict[str, int] = {}
        for node in self._nodes:
            cat = node["category"]
            counts[cat] = counts.get(cat, 0) + 1
        return counts

    def _build_lineage_graph(self) -> list[dict[str, Any]]:
        """构建问题继承关系图 (parent → children 边)。"""
        edges = []
        for node in self._nodes:
            for parent_id in node["parent_ids"]:
                edges.append({
                    "from": parent_id,
                    "to": node["id"],
                    "type": node["category"],
                    "layer": node["layer"],
                })
        return edges
