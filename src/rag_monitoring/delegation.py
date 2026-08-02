from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from rag_monitoring.monitoring import Thresholds, decision


@dataclass
class DelegationResult:
    decision: str
    should_escalate: bool
    policy_name: str
    delegation_score: float | None = None


@dataclass
class CTDPolicy:
    """
    Learned CTD-inspired delegation-value policy.

    The model estimates whether replacing the small-model answer with
    the large-model answer is likely to improve Exact Match.
    """

    model: LogisticRegression
    threshold: float
    target_delegation_rate: float
    feature_names: list[str]

    def score(self, features: list[float]) -> float:
        feature_array = np.asarray(
            features,
            dtype=float,
        ).reshape(1, -1)

        return float(
            self.model.predict_proba(feature_array)[0, 1]
        )


def threshold_delegation(
    confidence: float,
    thresholds: Thresholds,
) -> DelegationResult:
    monitoring_decision = decision(
        confidence,
        thresholds,
    )

    return DelegationResult(
        decision=monitoring_decision,
        should_escalate=monitoring_decision == "escalate",
        policy_name="threshold_baseline",
    )


def frugalgpt_delegation(
    confidence: float,
    threshold: float,
) -> DelegationResult:
    should_escalate = confidence < threshold

    return DelegationResult(
        decision="escalate" if should_escalate else "accept",
        should_escalate=should_escalate,
        policy_name="frugalgpt_style",
        delegation_score=float(confidence),
    )


def ctd_features(row: dict) -> list[float]:
    """
    Features available before invoking the large model.
    """

    return [
        float(row["retrieval_signal"]),
        float(row["faithfulness_signal"]),
        float(row["combined_confidence"]),
    ]


def fit_ctd_policy(
    calibration_rows: list[dict],
    target_delegation_rate: float,
    seed: int = 42,
    training_fraction: float = 0.70,
) -> CTDPolicy:
    """
    Fit a CTD-inspired delegation-value model.

    Calibration rows must already contain outputs from both models.

    The rows are divided into:
      1. probe-training rows for logistic regression;
      2. threshold-calibration rows for choosing a delegation threshold.

    This controls the delegation rate empirically; it does not provide
    the formal guarantees from the original CTD paper.
    """

    if not 0.0 < target_delegation_rate <= 1.0:
        raise ValueError(
            "target_delegation_rate must be in (0, 1]."
        )

    if not 0.0 < training_fraction < 1.0:
        raise ValueError(
            "training_fraction must be between 0 and 1."
        )

    if len(calibration_rows) < 10:
        raise ValueError(
            "CTD requires at least 10 calibration examples. "
            "Use 50 or more for a meaningful smoke test."
        )

    features: list[list[float]] = []
    benefit_labels: list[int] = []

    for row in calibration_rows:
        if row.get("large_exact_match") is None:
            raise ValueError(
                "Every CTD calibration row must contain a "
                "large-model output and large_exact_match."
            )

        features.append(ctd_features(row))

        benefit = (
            float(row["large_exact_match"])
            > float(row["small_exact_match"])
        )
        benefit_labels.append(int(benefit))

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(calibration_rows))

    split_index = int(
        len(calibration_rows) * training_fraction
    )
    split_index = min(
        max(split_index, 1),
        len(calibration_rows) - 1,
    )

    training_indices = indices[:split_index]
    threshold_indices = indices[split_index:]

    x = np.asarray(features, dtype=float)
    y = np.asarray(benefit_labels, dtype=int)

    x_train = x[training_indices]
    y_train = y[training_indices]

    if len(np.unique(y_train)) < 2:
        raise ValueError(
            "CTD probe-training data contains only one class. "
            "Increase calibration_size or use another seed."
        )

    model = LogisticRegression(
        class_weight="balanced",
        random_state=seed,
        max_iter=1000,
    )
    model.fit(x_train, y_train)

    threshold_scores = model.predict_proba(
        x[threshold_indices]
    )[:, 1]

    # Delegate approximately the requested fraction of examples
    # with the highest predicted delegation benefit.
    quantile = 1.0 - target_delegation_rate
    threshold = float(
        np.quantile(
            threshold_scores,
            quantile,
            method="higher",
        )
    )

    return CTDPolicy(
        model=model,
        threshold=threshold,
        target_delegation_rate=float(
            target_delegation_rate
        ),
        feature_names=[
            "retrieval_signal",
            "faithfulness_signal",
            "combined_confidence",
        ],
    )


def ctd_delegation(
    row: dict,
    policy: CTDPolicy,
) -> DelegationResult:
    score = policy.score(
        ctd_features(row)
    )
    should_escalate = score >= policy.threshold

    return DelegationResult(
        decision=(
            "escalate"
            if should_escalate
            else "accept"
        ),
        should_escalate=should_escalate,
        policy_name="ctd_style",
        delegation_score=score,
    )