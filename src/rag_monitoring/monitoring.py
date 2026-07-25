from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np


@dataclass
class Thresholds:
    accept: float
    flag: float
    target_risk: float
    target_achieved: bool
    accepted_count_at_threshold: int
    empirical_risk_at_threshold: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def minmax_cosine(score: float) -> float:
    return float(np.clip((score + 1.0) / 2.0, 0.0, 1.0))


def consistency_score(answers: list[str], similarity_function) -> float | None:
    if len(answers) <= 1:
        return None

    similarities = []
    for left in range(len(answers)):
        for right in range(left + 1, len(answers)):
            similarities.append(
                minmax_cosine(similarity_function(answers[left], answers[right]))
            )
    return float(np.mean(similarities)) if similarities else None


def combined_confidence(
    retrieval_score: float,
    faithfulness_score: float,
    consistency: float | None,
    weights: dict[str, float],
) -> float:
    components = {
        "retrieval": minmax_cosine(retrieval_score),
        "faithfulness": minmax_cosine(faithfulness_score),
    }
    active_weights = {
        "retrieval": float(weights.get("retrieval", 0.0)),
        "faithfulness": float(weights.get("faithfulness", 0.0)),
    }

    if consistency is not None and float(weights.get("consistency", 0.0)) > 0:
        components["consistency"] = float(np.clip(consistency, 0.0, 1.0))
        active_weights["consistency"] = float(weights["consistency"])

    total_weight = sum(active_weights.values())
    if total_weight <= 0:
        raise ValueError("Active monitoring score weights must sum to a positive value.")

    return sum(active_weights[name] * components[name] for name in active_weights) / total_weight


def calibrate_thresholds(
    rows: list[dict],
    target_risk: float,
    flag_margin: float,
) -> Thresholds:
    if not rows:
        raise ValueError("Calibration requires at least one row.")

    candidates = sorted({float(row["combined_confidence"]) for row in rows})

    for threshold in candidates:
        accepted = [row for row in rows if row["combined_confidence"] >= threshold]
        empirical_risk = 1.0 - float(np.mean([row["exact_match"] for row in accepted]))

        if empirical_risk <= target_risk:
            return Thresholds(
                accept=threshold,
                flag=max(0.0, threshold - flag_margin),
                target_risk=target_risk,
                target_achieved=True,
                accepted_count_at_threshold=len(accepted),
                empirical_risk_at_threshold=empirical_risk,
            )

    maximum_confidence = max(candidates)
    return Thresholds(
        accept=float(np.nextafter(maximum_confidence, np.inf)),
        flag=max(0.0, maximum_confidence - flag_margin),
        target_risk=target_risk,
        target_achieved=False,
        accepted_count_at_threshold=0,
        empirical_risk_at_threshold=None,
    )


def decision(confidence: float, thresholds: Thresholds) -> str:
    if confidence >= thresholds.accept:
        return "accept"
    if confidence >= thresholds.flag:
        return "flag"
    return "escalate"
