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
    """
    Monitoring output for one generated answer.
    """

    signals: list[SignalResult]
    combined_confidence: float

    def signal_scores(self) -> dict[str, float | None]:
        """
        Return signal scores in a CSV/JSON-friendly form.
        """
        output: dict[str, float | None] = {}

        for signal in self.signals:
            if np.isnan(signal.score):
                output[signal.name] = None
            else:
                output[signal.name] = float(signal.score)

        return output

    def raw_signal_values(self) -> dict[str, float | None]:
        """
        Return raw, pre-normalisation values for reproducibility.
        """
        return {
            signal.name: signal.raw_value
            for signal in self.signals
        }


@dataclass
class Thresholds:
    """
    Thresholds learned from the calibration split.
    """

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
    weights: dict[str, float],
) -> MonitoringResult:
    """
    Calculate all enabled monitoring signals for one answer.

    Signal availability is controlled through the supplied weights.
    Signals with a weight of zero do not contribute to the final score.
    """

    joined_context = "\n".join(retrieved_contexts)

    signals = [
        retrieval_signal(
            retrieval_scores=retrieval_scores,
        ),
        faithfulness_signal(
            answer=prediction,
            context=joined_context,
            similarity_function=similarity_function,
        ),
        consistency_signal(
            answers=all_predictions,
            similarity_function=similarity_function,
        ),
    ]

    confidence = combine_signals(
        signals=signals,
        weights=weights,
    )

    return MonitoringResult(
        signals=signals,
        combined_confidence=float(confidence),
    )


def calibrate_thresholds(
    rows: list[dict],
    target_risk: float,
    flag_margin: float,
    correctness_field: str = "exact_match",
) -> Thresholds:
    """
    Select the lowest confidence threshold whose accepted calibration
    examples satisfy the requested empirical risk.

    This is currently an empirical thresholding baseline. It is not yet
    a formal conformal guarantee.
    """

    if not rows:
        raise ValueError(
            "Threshold calibration requires at least one calibration example."
        )

    if not 0.0 <= target_risk <= 1.0:
        raise ValueError(
            "target_risk must be between 0 and 1."
        )

    candidates = sorted(
        {
            float(row["combined_confidence"])
            for row in rows
        }
    )

    for threshold in candidates:
        accepted = [
            row
            for row in rows
            if float(row["combined_confidence"]) >= threshold
        ]

        if not accepted:
            continue

        accepted_accuracy = float(
            np.mean([
                float(row[correctness_field])
                for row in accepted
            ])
        )

        empirical_risk = 1.0 - accepted_accuracy

        if empirical_risk <= target_risk:
            return Thresholds(
                accept=float(threshold),
                flag=max(
                    0.0,
                    float(threshold) - float(flag_margin),
                ),
                target_risk=float(target_risk),
                target_achieved=True,
                accepted_count_at_threshold=len(accepted),
                empirical_risk_at_threshold=float(empirical_risk),
            )

    maximum_confidence = max(candidates)

    # Put the accept threshold immediately above every observed
    # calibration confidence so that no answer is falsely reported
    # as satisfying the requested risk.
    unavailable_accept_threshold = float(
        np.nextafter(maximum_confidence, np.inf)
    )

    return Thresholds(
        accept=unavailable_accept_threshold,
        flag=max(
            0.0,
            float(maximum_confidence) - float(flag_margin),
        ),
        target_risk=float(target_risk),
        target_achieved=False,
        accepted_count_at_threshold=0,
        empirical_risk_at_threshold=None,
    )


def decision(
    confidence: float,
    thresholds: Thresholds,
) -> str:
    """
    Convert a confidence value into a monitoring decision.
    """

    if confidence >= thresholds.accept:
        return "accept"

    if confidence >= thresholds.flag:
        return "flag"

    return "escalate"