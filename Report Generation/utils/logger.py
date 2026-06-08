"""
Logger Module - 提供全局日志功能
支持终端输出和文件日志记录
"""

import logging
import os
from datetime import datetime
from pathlib import Path


def _get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).parent.parent


def _ensure_logs_dir() -> Path:
    """确保 logs 目录存在"""
    logs_dir = _get_project_root() / "logs"
    logs_dir.mkdir(exist_ok=True)
    return logs_dir


def _setup_logger() -> logging.Logger:
    """配置并返回 logger 实例"""
    logger = logging.getLogger("report_generation")
    
    # 避免重复添加 handler
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)
    
    # 日志格式
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )
    file_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 终端 Handler - 彩色输出
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_format)
    
    # 文件 Handler - 按日期生成日志文件
    logs_dir = _ensure_logs_dir()
    log_filename = datetime.now().strftime("%Y-%m-%d") + ".log"
    file_handler = logging.FileHandler(
        logs_dir / log_filename,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_format)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


# 创建全局 logger 实例
_logger = _setup_logger()

# 导出便捷方法
debug = _logger.debug
info = _logger.info
warning = _logger.warning
error = _logger.error
critical = _logger.critical
exception = _logger.exception


def set_level(level: str) -> None:
    """
    设置日志级别
    
    Args:
        level: 日志级别 ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
    """
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    _logger.setLevel(level_map.get(level.upper(), logging.INFO))


def get_logger() -> logging.Logger:
    """获取 logger 实例，用于需要更多控制的场景"""
    return _logger


import uuid as _uuid

def log_to_file(content: str, label: str = "RECORD") -> str:
    """
    将完整内容写入日志文件，不做任何截断。
    终端只打印摘要（标识符 + 长度），完整内容仅存于日志文件。

    Args:
        content: 要记录的完整内容（代码、结果、prompt 等）
        label: 标签，用于标识内容类型（如 "CODE", "RESULT", "PROMPT"）

    Returns:
        str: 本次记录的唯一标识符，方便在日志文件中搜索
    """
    uid = _uuid.uuid4().hex[:8]
    marker = f"[{label}#{uid}]"
    border = "+" + "-" * 68 + "+"

    # 写入日志文件（完整内容，无截断）
    _logger.debug(
        f"\n{border}\n"
        f"| {marker}  len={len(content)}\n"
        f"{border}\n"
        f"{content}\n"
        f"{border}"
    )

    # 终端只打印摘要
    _logger.info(f"{marker} 已写入日志文件 (长度 {len(content)} 字符)")

    return uid

