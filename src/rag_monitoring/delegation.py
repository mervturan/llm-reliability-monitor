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
    model: LogisticRegression
    threshold: float
    target_delegation_rate: float

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
        delegation_score=None,
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
        delegation_score=confidence,
    )


def fit_ctd_policy(
    calibration_rows: list[dict],
    target_delegation_rate: float,
) -> CTDPolicy:
    """
    Learn whether escalation is likely to improve Exact Match.

    This is a CTD-inspired adaptation, not a full reproduction of
    the paper's latent-space delegation-value probe or formal
    multiple-hypothesis-testing guarantee.
    """

    if not 0.0 < target_delegation_rate <= 1.0:
        raise ValueError(
            "target_delegation_rate must be in (0, 1]."
        )

    features: list[list[float]] = []
    benefit_labels: list[int] = []

    for row in calibration_rows:
        if row.get("large_exact_match") is None:
            raise ValueError(
                "CTD calibration requires large-model outputs "
                "for every calibration example."
            )

        features.append(
            [
                float(row["retrieval_signal"]),
                float(row["faithfulness_signal"]),
                float(row["combined_confidence"]),
            ]
        )

        benefit = (
            row["small_exact_match"] == 0
            and row["large_exact_match"] == 1
        )
        benefit_labels.append(int(benefit))

    if len(set(benefit_labels)) < 2:
        raise ValueError(
            "CTD calibration requires both beneficial and "
            "non-beneficial escalation examples."
        )

    model = LogisticRegression(
        class_weight="balanced",
        random_state=42,
        max_iter=1000,
    )
    model.fit(features, benefit_labels)

    scores = model.predict_proba(
        np.asarray(features, dtype=float)
    )[:, 1]

    # Select the threshold that delegates approximately the desired
    # fraction of examples with the highest predicted benefit.
    quantile = 1.0 - target_delegation_rate
    threshold = float(np.quantile(scores, quantile))

    return CTDPolicy(
        model=model,
        threshold=threshold,
        target_delegation_rate=target_delegation_rate,
    )


def ctd_delegation(
    features: list[float],
    policy: CTDPolicy,
) -> DelegationResult:
    delegation_score = policy.score(features)
    should_escalate = delegation_score >= policy.threshold

    return DelegationResult(
        decision="escalate" if should_escalate else "accept",
        should_escalate=should_escalate,
        policy_name="ctd_style",
        delegation_score=delegation_score,
    )