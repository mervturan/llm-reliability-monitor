from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from rag_monitoring.signals import (
    SignalResult,
    combine_signals,
    consistency_signal,
    faithfulness_signal,
    retrieval_signal,
)


@dataclass
class MonitoringResult:
    signals: list[SignalResult]
    combined_confidence: float

    def signal_scores(self) -> dict[str, float | None]:
        return {
            signal.name: None if np.isnan(signal.score) else float(signal.score)
            for signal in self.signals
        }

    def raw_signal_values(self) -> dict[str, float | None]:
        return {signal.name: signal.raw_value for signal in self.signals}


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


def monitor_answer(
    prediction: str,
    retrieved_contexts: list[str],
    retrieval_scores: list[float],
    all_predictions: list[str],
    similarity_function,
    enabled_signals: list[str],
    weights: dict[str, float],
) -> MonitoringResult:
    joined_context = "\n".join(retrieved_contexts)
    signals = [
        retrieval_signal(retrieval_scores),
        faithfulness_signal(prediction, joined_context, similarity_function),
        consistency_signal(all_predictions, similarity_function),
    ]
    confidence = combine_signals(signals, enabled_signals, weights)
    return MonitoringResult(signals=signals, combined_confidence=confidence)


def calibrate_thresholds(
    rows: list[dict],
    target_risk: float,
    flag_margin: float,
    correctness_field: str = "exact_match",
) -> Thresholds:
    if not rows:
        raise ValueError("Calibration requires at least one row.")

    candidates = sorted({float(row["combined_confidence"]) for row in rows})

    for threshold in candidates:
        accepted = [
            row for row in rows
            if float(row["combined_confidence"]) >= threshold
        ]
        accuracy = float(np.mean([float(row[correctness_field]) for row in accepted]))
        empirical_risk = 1.0 - accuracy

        if empirical_risk <= target_risk:
            return Thresholds(
                accept=float(threshold),
                flag=max(0.0, float(threshold) - float(flag_margin)),
                target_risk=float(target_risk),
                target_achieved=True,
                accepted_count_at_threshold=len(accepted),
                empirical_risk_at_threshold=float(empirical_risk),
            )

    maximum_confidence = max(candidates)
    return Thresholds(
        accept=float(np.nextafter(maximum_confidence, np.inf)),
        flag=max(0.0, maximum_confidence - float(flag_margin)),
        target_risk=float(target_risk),
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
