"""统一打分器 —— 所有 benchmark 入口共用。

使用 G-Eval (LLM-as-Judge) 方法对 insight / summary 进行语义评分。
优先使用 logprobs 加权，API 不支持时自动回退到 Monte Carlo 采样。

配置来源：MyDataStorm/datastorm/llm_config.json
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)

# ============================================================
# 配置加载
# ============================================================

_CFG_PATH = (
    Path(__file__).resolve().parents[2] / "MyDataStorm" / "datastorm" / "llm_config.json"
)


def _load_config() -> dict:
    try:
        if _CFG_PATH.is_file():
            return json.loads(_CFG_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


_CFG = _load_config()

_SCORER_API_KEY = _CFG.get("api_key") or os.getenv("OPENAI_API_KEY", "")
_SCORER_API_BASE = _CFG.get("api_base") or os.getenv("OPENAI_API_BASE", "")
_SCORER_MODEL = _CFG.get("model_name", "deepseek-v4-pro")
_SCORER_TEMPERATURE = float(_CFG.get("temperature", 0.7))
_SCORER_MAX_TOKENS = int(_CFG.get("max_completion_tokens", 4096))

# Monte Carlo 采样次数
_MC_SAMPLES = 5
# logprobs 检测标志：None=未检测, True=支持, False=不支持
_logprobs_supported: bool | None = None


def _create_client() -> OpenAI:
    kwargs: dict[str, str] = {"api_key": _SCORER_API_KEY}
    if _SCORER_API_BASE:
        kwargs["base_url"] = _SCORER_API_BASE
    return OpenAI(**kwargs)


# ============================================================
# G-Eval Prompt
# ============================================================

_G_EVAL_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n"
    "Provided Answer:\n{answer}\n\n"
    "Ground Truth Answer:\n{gt_answer}\n\n"
    "Follow these instructions when writing your response:\n"
    "* On a scale of 1-10, provide a numerical rating for how close the provided answer "
    "is to the ground truth answer, with 10 denoting that the provided answer is the same "
    "as ground truth answer.\n"
    "* Your response should contain only the numerical rating. "
    "DONOT include anything else like the provided answer, the ground truth answer, "
    "or an explanation of your rating scale in your response.\n"
    "* Wrap your numerical rating inside <rating></rating> tags.\n"
    "* Check very carefully before answering.\n"
    "* Follow the output format as shown in the example below:\n"
    "Example response:\n<rating>7</rating>\n\n"
    "### Response:\n"
)

_SYSTEM_MESSAGE = (
    "You are a high school teacher evaluating student responses to a question. "
    "You are tasked with grading the response based on how well it answers the question. "
    "You are to provide a numerical rating for how well the response answers the question "
    "based on the ground truth answer."
)


# ============================================================
# logprobs 支持检测
# ============================================================

def _detect_logprobs(client: OpenAI, model: str) -> bool:
    """发送一次最小化 G-Eval 调用，检测 API 是否支持 logprobs 参数。"""
    prompt = _G_EVAL_TEMPLATE.format(answer="test", gt_answer="test")
    try:
        client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_completion_tokens=20,
            logprobs=True,
            top_logprobs=3,
        )
        logger.info("Scorer: logprobs supported by %s", model)
        return True
    except Exception as e:
        msg = str(e)
        if any(kw in msg.lower() for kw in ("logprobs", "log_probs", "top_logprobs", "unsupported parameter")):
            logger.info("Scorer: logprobs NOT supported by %s, will use Monte Carlo", model)
            return False
        # 其他错误（网络、认证等）不判定为不支持
        logger.warning("Scorer: logprobs detection got unexpected error: %s", msg[:200])
        raise


# ============================================================
# 核心评分
# ============================================================

def _score_pair_logprobs(client: OpenAI, model: str, answer: str, gt_answer: str) -> float:
    """使用 logprobs 加权的 G-Eval 评分。"""
    prompt = _G_EVAL_TEMPLATE.format(answer=answer, gt_answer=gt_answer)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_completion_tokens=50,
        logprobs=True,
        top_logprobs=5,
    )
    raw = response.choices[0].message.content or ""
    rating_match = re.findall(r"<rating>(\d+)</rating>", raw)
    if not rating_match:
        return _extract_fallback_rating(raw)

    rating_str = rating_match[0]
    logprobs_content = response.choices[0].logprobs.content
    if not logprobs_content:
        return float(rating_str) / 10.0

    tokens = [o.token for o in logprobs_content]
    if rating_str not in tokens:
        return float(rating_str) / 10.0

    idx = tokens.index(rating_str)
    top_lps = logprobs_content[idx].top_logprobs
    if not top_lps:
        return float(rating_str) / 10.0

    probs = [np.exp(lp.logprob) for lp in top_lps]
    probs = [p / sum(probs) for p in probs]
    ratings = [float(lp.token) if lp.token.isdigit() else 0 for lp in top_lps]
    score = sum(r * p for r, p in zip(ratings, probs))
    return score / 10.0


def _score_pair_monte_carlo(client: OpenAI, model: str, answer: str, gt_answer: str) -> float:
    """Monte Carlo 采样评分：多次调用取平均，作为 logprobs 的替代。"""
    prompt = _G_EVAL_TEMPLATE.format(answer=answer, gt_answer=gt_answer)
    ratings: list[float] = []
    for _ in range(_MC_SAMPLES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_completion_tokens=50,
            )
            raw = response.choices[0].message.content or ""
            rating_match = re.findall(r"<rating>(\d+)</rating>", raw)
            if rating_match:
                ratings.append(float(rating_match[0]))
        except Exception:
            continue
    if not ratings:
        return 0.0
    return sum(ratings) / len(ratings) / 10.0


def _extract_fallback_rating(raw: str) -> float:
    """从未包裹 <rating> 标签的响应中提取数值。"""
    nums = re.findall(r"\b(\d+)\b", raw)
    if nums:
        return float(nums[0]) / 10.0
    return 0.0


# ============================================================
# 公开 API
# ============================================================

def score_insights(pred_insights: list[str], gt_insights: list[str]) -> float:
    """对一组预测 insight 进行 G-Eval 评分（many-to-many best-match）。

    对每个 GT insight，在所有预测中找最高分的匹配，取平均。
    """
    global _logprobs_supported
    client = _create_client()
    model = _SCORER_MODEL

    if _logprobs_supported is None:
        try:
            _logprobs_supported = _detect_logprobs(client, model)
        except Exception:
            logger.warning("Scorer: logprobs detection failed, using Monte Carlo")
            _logprobs_supported = False

    score_func = _score_pair_logprobs if _logprobs_supported else _score_pair_monte_carlo

    best_scores: list[float] = []
    for gt in gt_insights:
        best = 0.0
        for pred in pred_insights:
            s = score_func(client, model, pred, gt)
            if s > best:
                best = s
        best_scores.append(best)

    return float(np.mean(best_scores)) if best_scores else 0.0


def score_summary(pred_summary: str, gt_summary: str) -> float:
    """对单条 summary 进行 G-Eval 评分。"""
    global _logprobs_supported
    client = _create_client()
    model = _SCORER_MODEL

    if _logprobs_supported is None:
        try:
            _logprobs_supported = _detect_logprobs(client, model)
        except Exception:
            _logprobs_supported = False

    score_func = _score_pair_logprobs if _logprobs_supported else _score_pair_monte_carlo
    return score_func(client, model, pred_summary, gt_summary)


def get_scorer_config() -> dict[str, str]:
    """返回当前打分器配置（用于日志/调试）。"""
    return {
        "api_base": _SCORER_API_BASE or "https://api.openai.com/v1",
        "model": _SCORER_MODEL,
        "logprobs": str(_logprobs_supported),
        "mc_samples": str(_MC_SAMPLES) if not _logprobs_supported else "N/A",
    }
