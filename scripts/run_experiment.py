from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_monitoring.config import load_config
from rag_monitoring.data import load_squad_splits
from rag_monitoring.generation import AnswerGenerator
from rag_monitoring.io_utils import (
    create_run_directory,
    save_json,
    save_jsonl,
    save_summary_csv,
    save_yaml,
)
from rag_monitoring.metrics import exact_match, token_f1
from rag_monitoring.monitoring import calibrate_thresholds, decision, monitor_answer
from rag_monitoring.reproducibility import environment_info, set_seed
from rag_monitoring.retrieval import DenseRetriever
from rag_monitoring.risk_coverage import build_risk_coverage_table


def normalize_context(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def find_gold_context_rank(
    gold_context: str,
    retrieved_contexts: list[str],
) -> int | None:
    normalized_gold = normalize_context(gold_context)
    for rank, context in enumerate(retrieved_contexts, start=1):
        if normalize_context(context) == normalized_gold:
            return rank
    return None


def evaluate_examples(
    examples: list[dict],
    retriever: DenseRetriever,
    generator: AnswerGenerator,
    config: dict,
) -> list[dict]:
    rows: list[dict] = []
    retrieval_config = config["retrieval"]
    generation_config = config["generation"]
    monitoring_config = config["monitoring"]

    for example in tqdm(examples, desc="Evaluating"):
        retrieval_result = retriever.retrieve(
            example["question"],
            top_k=retrieval_config["top_k"],
        )

        raw_answers: list[str] = []
        cleaned_answers: list[str] = []

        for sample_index in range(generation_config["num_consistency_samples"]):
            temperature = generation_config["temperature"]
            if sample_index > 0 and temperature == 0:
                temperature = 0.7

            raw_answer, cleaned_answer = generator.generate(
                question=example["question"],
                contexts=retrieval_result.contexts,
                max_new_tokens=generation_config["max_new_tokens"],
                temperature=temperature,
            )
            raw_answers.append(raw_answer)
            cleaned_answers.append(cleaned_answer)

        prediction = cleaned_answers[0]
        gold_context_rank = find_gold_context_rank(
            example["context"], retrieval_result.contexts
        )
        reciprocal_rank = 1.0 / gold_context_rank if gold_context_rank else 0.0

        monitoring_result = monitor_answer(
            prediction=prediction,
            retrieved_contexts=retrieval_result.contexts,
            retrieval_scores=retrieval_result.scores,
            all_predictions=cleaned_answers,
            similarity_function=retriever.semantic_similarity,
            enabled_signals=monitoring_config["enabled_signals"],
            weights=monitoring_config["score_weights"],
        )
        signal_scores = monitoring_result.signal_scores()
        raw_signal_values = monitoring_result.raw_signal_values()
        references = example["answers"]["text"]

        rows.append({
            "id": example["id"],
            "question": example["question"],
            "raw_prediction": raw_answers[0],
            "cleaned_prediction": prediction,
            "all_raw_predictions": raw_answers,
            "all_cleaned_predictions": cleaned_answers,
            "prediction": prediction,
            "references": references,
            "gold_context_retrieved": gold_context_rank is not None,
            "gold_context_rank": gold_context_rank,
            "reciprocal_rank": reciprocal_rank,
            "retrieved_contexts": retrieval_result.contexts,
            "retrieval_scores": retrieval_result.scores,
            "retrieval_top_score": retrieval_result.scores[0],
            "retrieval_signal": signal_scores["retrieval"],
            "faithfulness_signal": signal_scores["faithfulness"],
            "consistency_signal": signal_scores["consistency"],
            "retrieval_signal_raw": raw_signal_values["retrieval"],
            "faithfulness_signal_raw": raw_signal_values["faithfulness"],
            "consistency_signal_raw": raw_signal_values["consistency"],
            "combined_confidence": monitoring_result.combined_confidence,
            "exact_match": exact_match(prediction, references),
            "token_f1": token_f1(prediction, references),
        })

    return rows


def aggregate_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {}

    accepted = [row for row in rows if row.get("decision") == "accept"]
    retrieved = [row for row in rows if row["gold_context_retrieved"]]
    not_retrieved = [row for row in rows if not row["gold_context_retrieved"]]

    return {
        "n_examples": len(rows),
        "exact_match": float(np.mean([row["exact_match"] for row in rows])),
        "token_f1": float(np.mean([row["token_f1"] for row in rows])),
        "mean_confidence": float(np.mean([row["combined_confidence"] for row in rows])),
        "retrieval": {
            "top_k": max(len(row["retrieved_contexts"]) for row in rows),
            "recall_at_k": len(retrieved) / len(rows),
            "mean_reciprocal_rank": float(np.mean([row["reciprocal_rank"] for row in rows])),
            "mean_gold_rank_when_retrieved": (
                float(np.mean([row["gold_context_rank"] for row in retrieved]))
                if retrieved else None
            ),
        },
        "generation_given_retrieval": {
            "exact_match_when_gold_retrieved": (
                float(np.mean([row["exact_match"] for row in retrieved]))
                if retrieved else None
            ),
            "exact_match_when_gold_not_retrieved": (
                float(np.mean([row["exact_match"] for row in not_retrieved]))
                if not_retrieved else None
            ),
        },
        "acceptance_rate": len(accepted) / len(rows),
        "selective_exact_match": (
            float(np.mean([row["exact_match"] for row in accepted]))
            if accepted else None
        ),
        "selective_risk": (
            1.0 - float(np.mean([row["exact_match"] for row in accepted]))
            if accepted else None
        ),
        "decision_counts": {
            label: sum(row.get("decision") == label for row in rows)
            for label in ["accept", "flag", "escalate"]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["run"]["seed"])

    run_dir = create_run_directory(
        config["run"]["output_root"],
        config["run"]["name"],
    )
    save_yaml(run_dir / "config.yaml", config)
    save_json(run_dir / "environment.json", environment_info())

    calibration_examples, test_examples, contexts = load_squad_splits(
        calibration_size=config["dataset"]["calibration_size"],
        test_size=config["dataset"]["test_size"],
        context_pool_size=config["dataset"]["context_pool_size"],
        seed=config["run"]["seed"],
    )

    retriever = DenseRetriever(
        model_name=config["retrieval"]["embedding_model"],
        contexts=contexts,
        batch_size=config["retrieval"]["batch_size"],
    )
    generator = AnswerGenerator(
        model_name=config["generation"]["model_name"],
        device=config["generation"]["device"],
    )

    calibration_rows = evaluate_examples(calibration_examples, retriever, generator, config)
    thresholds = calibrate_thresholds(
        calibration_rows,
        target_risk=config["monitoring"]["target_risk"],
        flag_margin=config["monitoring"]["flag_margin"],
    )
    for row in calibration_rows:
        row["decision"] = decision(row["combined_confidence"], thresholds)

    test_rows = evaluate_examples(test_examples, retriever, generator, config)
    for row in test_rows:
        row["decision"] = decision(row["combined_confidence"], thresholds)

    threshold_payload = thresholds.to_dict()
    metrics = {
        "signal_configuration": {
            "enabled_signals": config["monitoring"]["enabled_signals"],
            "score_weights": config["monitoring"]["score_weights"],
        },
        "calibration": aggregate_metrics(calibration_rows),
        "test": aggregate_metrics(test_rows),
        "calibration_status": threshold_payload,
    }

    save_jsonl(run_dir / "calibration_predictions.jsonl", calibration_rows)
    save_jsonl(run_dir / "test_predictions.jsonl", test_rows)
    save_json(run_dir / "thresholds.json", threshold_payload)
    save_json(run_dir / "metrics.json", metrics)
    save_summary_csv(run_dir / "summary.csv", calibration_rows, test_rows)

    build_risk_coverage_table(calibration_rows).to_csv(
        run_dir / "risk_coverage_calibration.csv", index=False
    )
    build_risk_coverage_table(test_rows).to_csv(
        run_dir / "risk_coverage_test.csv", index=False
    )

    print(f"\nRun completed: {run_dir}")
    print(metrics)


if __name__ == "__main__":
    main()
