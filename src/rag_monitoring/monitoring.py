from dataclasses import dataclass
import numpy as np

@dataclass
class Thresholds:
    accept: float
    flag: float

def scale_cosine(score):
    return float(np.clip((score + 1.0) / 2.0, 0.0, 1.0))

def consistency_score(answers, similarity_function):
    if len(answers) <= 1:
        return 1.0
    scores = [scale_cosine(similarity_function(answers[i], answers[j]))
              for i in range(len(answers)) for j in range(i + 1, len(answers))]
    return float(np.mean(scores))

def combined_confidence(retrieval_score, faithfulness_score, consistency, weights):
    values = {"retrieval": scale_cosine(retrieval_score),
              "faithfulness": scale_cosine(faithfulness_score),
              "consistency": float(np.clip(consistency, 0.0, 1.0))}
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Monitoring score weights must be positive")
    return sum(weights[k] * values[k] for k in values) / total

def calibrate_thresholds(rows, target_risk, flag_margin):
    threshold = 1.0
    for candidate in sorted({float(r["combined_confidence"]) for r in rows}):
        accepted = [r for r in rows if r["combined_confidence"] >= candidate]
        if accepted and 1.0 - np.mean([r["exact_match"] for r in accepted]) <= target_risk:
            threshold = candidate
            break
    return Thresholds(threshold, max(0.0, threshold - flag_margin))

def decision(confidence, thresholds):
    if confidence >= thresholds.accept:
        return "accept"
    if confidence >= thresholds.flag:
        return "flag"
    return "escalate"
