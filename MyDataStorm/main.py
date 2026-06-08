"""DataSTORM 命令行入口。

用法:
    python main.py --query "Research topic" --db-url "postgresql://..." --db-description "..."

环境变量:
    OPENAI_API_KEY     OpenAI API Key
    DATABASE_URL       PostgreSQL 连接字符串
    SERPER_API_KEY     Serper API Key (用于 Web 搜索)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from datastorm.config import DataSTORMConfig
from datastorm.pipeline import DataSTORMPipeline


def setup_logging(verbose: bool = False) -> None:
    """配置日志。"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DataSTORM: Deep Research on Large-Scale Databases using EDA and Data Storytelling"
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Research query/topic to investigate",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="PostgreSQL database URL (overrides DATABASE_URL env var)",
    )
    parser.add_argument(
        "--db-description",
        default="",
        help="Description of database content",
    )
    parser.add_argument(
        "--output",
        default="datastorm_report.md",
        help="Output markdown file path",
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        default=None,
        help="Maximum exploration layers (default: 5 per paper)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def save_report(report, output_path: str) -> None:
    """保存最终报告到 markdown 文件。"""
    path = Path(output_path)

    # 组装完整 markdown
    parts = [
        f"# {report.title}",
        f"*{report.subtitle}*",
        "",
        report.markdown,
        "",
        "## Sources",
        "",
    ]

    for ref in report.references:
        ref_id = ref.get("id", "")
        source = ref.get("source", "")
        question = ref.get("question", "")
        sql = ref.get("sql", "")
        parts.append(f"[{ref_id}] {source}: {question}")
        if sql:
            parts.append(f"```sql\n{sql}\n```")
        parts.append("")

    path.write_text("\n".join(parts), encoding="utf-8")
    print(f"Report saved to: {path}")

    # 同时保存 JSON 元数据
    meta_path = path.with_suffix(".json")
    meta = {
        "title": report.title,
        "subtitle": report.subtitle,
        "thesis": {
            "title": report.thesis.title,
            "research_strategy": report.thesis.research_strategy,
        },
        "references": report.references,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Metadata saved to: {meta_path}")


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    # 构建配置
    config = DataSTORMConfig()
    if args.db_url:
        config.database.url = args.db_url
    if args.db_description:
        config.db_description = args.db_description
    if args.max_layers:
        config.exploration.max_layers = args.max_layers

    # 验证必要配置
    if not config.llm.api_key:
        print("Error: OPENAI_API_KEY is required", file=sys.stderr)
        return 1
    if not config.database.url:
        print("Error: DATABASE_URL or --db-url is required", file=sys.stderr)
        return 1

    pipeline = DataSTORMPipeline(config)
    try:
        output_dir = str(Path(args.output).parent) if args.output else None
        report = pipeline.run(args.query, output_dir=output_dir)
        save_report(report, args.output)
        return 0
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
