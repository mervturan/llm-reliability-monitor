from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class SignalResult:
    """
    Represents a single monitoring signal.

    score:
        Confidence value normalised to [0,1].

    raw_value:
        Original value before normalisation (optional).

    description:
        Human-readable explanation for logging/debugging.
    """

    name: str
    score: float
    raw_value: float | None = None
    description: str = ""


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def cosine_to_confidence(score: float) -> float:
    """
    Convert cosine similarity [-1,1] to confidence [0,1].
    """
    return float(np.clip((score + 1.0) / 2.0, 0.0, 1.0))


# ---------------------------------------------------------
# Retrieval
# ---------------------------------------------------------

def retrieval_signal(
    retrieval_scores: list[float],
) -> SignalResult:

    raw = retrieval_scores[0]

    return SignalResult(
        name="retrieval",
        score=cosine_to_confidence(raw),
        raw_value=raw,
        description="Top retrieved document similarity.",
    )


# ---------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------

def faithfulness_signal(
    answer: str,
    context: str,
    similarity_function: Callable[[str, str], float],
) -> SignalResult:

    raw = similarity_function(answer, context)

    return SignalResult(
        name="faithfulness",
        score=cosine_to_confidence(raw),
        raw_value=raw,
        description="Semantic similarity between answer and retrieved evidence.",
    )


# ---------------------------------------------------------
# Consistency
# ---------------------------------------------------------

def consistency_signal(
    answers: list[str],
    similarity_function: Callable[[str, str], float],
) -> SignalResult:

    if len(answers) <= 1:
        return SignalResult(
            name="consistency",
            score=np.nan,
            raw_value=None,
            description="Consistency unavailable (single generation).",
        )

    similarities = []

    for i in range(len(answers)):
        for j in range(i + 1, len(answers)):
            similarities.append(
                cosine_to_confidence(
                    similarity_function(
                        answers[i],
                        answers[j],
                    )
                )
            )

    score = float(np.mean(similarities))

    return SignalResult(
        name="consistency",
        score=score,
        raw_value=score,
        description="Average pairwise similarity between generations.",
    )


# ---------------------------------------------------------
# Combination
# ---------------------------------------------------------

def combine_signals(
    signals: list[SignalResult],
    weights: dict[str, float],
) -> float:
    """
    Weighted confidence combination.

    Missing signals (NaN) are ignored automatically.
    """

    numerator = 0.0
    denominator = 0.0

    for signal in signals:

        if np.isnan(signal.score):
            continue

        weight = weights.get(signal.name, 0.0)

        numerator += weight * signal.score
        denominator += weight

    if denominator == 0:
        raise ValueError(
            "No valid monitoring signals were supplied."
        )

    return numerator / denominator