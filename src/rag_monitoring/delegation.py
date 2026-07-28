from __future__ import annotations

from dataclasses import dataclass

from rag_monitoring.monitoring import Thresholds, decision


@dataclass
class DelegationResult:
    decision: str
    should_escalate: bool
    policy_name: str


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