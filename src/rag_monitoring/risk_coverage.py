from __future__ import annotations

import pandas as pd


def build_risk_coverage_table(
    rows: list[dict],
    correctness_field: str = "exact_match",
) -> pd.DataFrame:
    columns = [
        "threshold",
        "accepted_count",
        "coverage",
        "selective_accuracy",
        "selective_risk",
        "escalation_rate",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    ordered = sorted(
        rows,
        key=lambda row: float(row["combined_confidence"]),
        reverse=True,
    )

    total = len(ordered)
    records: list[dict] = []
    correct_so_far = 0.0

    for index, row in enumerate(ordered, start=1):
        correct_so_far += float(row[correctness_field])
        coverage = index / total
        selective_accuracy = correct_so_far / index
        records.append({
            "threshold": float(row["combined_confidence"]),
            "accepted_count": index,
            "coverage": coverage,
            "selective_accuracy": selective_accuracy,
            "selective_risk": 1.0 - selective_accuracy,
            "escalation_rate": 1.0 - coverage,
        })

    return pd.DataFrame(records, columns=columns)
