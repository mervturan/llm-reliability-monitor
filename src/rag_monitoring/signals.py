from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class SignalResult:
    name: str
    score: float
    raw_value: float | None = None
    description: str = ""


def cosine_to_confidence(score: float) -> float:
    return float(np.clip((score + 1.0) / 2.0, 0.0, 1.0))


def retrieval_signal(retrieval_scores: list[float]) -> SignalResult:
    raw = float(retrieval_scores[0])
    return SignalResult(
        name="retrieval",
        score=cosine_to_confidence(raw),
        raw_value=raw,
        description="Top retrieved-document cosine similarity.",
    )


def faithfulness_signal(
    answer: str,
    context: str,
    similarity_function: Callable[[str, str], float],
) -> SignalResult:
    raw = float(similarity_function(answer, context))
    return SignalResult(
        name="faithfulness",
        score=cosine_to_confidence(raw),
        raw_value=raw,
        description="Semantic similarity between answer and retrieved evidence.",
    )


def consistency_signal(
    answers: list[str],
    similarity_function: Callable[[str, str], float],
) -> SignalResult:
    if len(answers) <= 1:
        return SignalResult(
            name="consistency",
            score=float("nan"),
            raw_value=None,
            description="Unavailable because only one answer was generated.",
        )

    pairwise_scores: list[float] = []
    for left in range(len(answers)):
        for right in range(left + 1, len(answers)):
            pairwise_scores.append(
                cosine_to_confidence(
                    similarity_function(answers[left], answers[right])
                )
            )

    score = float(np.mean(pairwise_scores))
    return SignalResult(
        name="consistency",
        score=score,
        raw_value=score,
        description="Mean pairwise similarity across repeated generations.",
    )


def combine_signals(
    signals: list[SignalResult],
    enabled_signals: list[str],
    weights: dict[str, float],
) -> float:
    enabled = set(enabled_signals)
    numerator = 0.0
    denominator = 0.0

    for signal in signals:
        if signal.name not in enabled or np.isnan(signal.score):
            continue
        weight = float(weights.get(signal.name, 0.0))
        if weight <= 0:
            continue
        numerator += weight * signal.score
        denominator += weight

    if denominator <= 0:
        raise ValueError("No valid enabled monitoring signals have positive weights.")

    return float(numerator / denominator)
