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

from rag_monitoring.delegation import (
    CTDPolicy,
    ctd_delegation,
    fit_ctd_policy,
    frugalgpt_delegation,
    threshold_delegation,
    always_large_delegation,
)

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

def best_reference_semantic_similarity(
    prediction: str,
    references: list[str],
    similarity_function,
) -> float:
    """
    Compare a prediction with all acceptable reference answers
    and return the highest semantic similarity.
    """

    if not references:
        return 0.0

    return max(
        similarity_function(prediction, reference)
        for reference in references
    )

def apply_cascade(
    rows: list[dict],
    examples: list[dict],
    large_generator: AnswerGenerator,
    large_generation_config: dict,
    thresholds,
    policy_name: str,
    similarity_function,
    ctd_policy: CTDPolicy | None = None,
) -> list[dict]:
    """
    Apply the selected delegation policy.

    The large model is called only for delegated examples, except that
    CTD calibration rows may already contain large-model outputs generated
    earlier for fitting the delegation-value model.
    """

    if len(rows) != len(examples):
        raise ValueError(
            "Rows and examples must contain the same number of items."
        )

    for row, example in tqdm(
        zip(rows, examples),
        total=len(rows),
        desc="Applying cascade",
    ):
        # ---------------------------------------------------------
        # Select the delegation policy
        # ---------------------------------------------------------
        if policy_name == "frugalgpt":
            delegation_result = frugalgpt_delegation(
                confidence=row["combined_confidence"],
                threshold=thresholds.accept,
            )

        elif policy_name == "threshold_baseline":
            delegation_result = threshold_delegation(
                confidence=row["combined_confidence"],
                thresholds=thresholds,
            )

        elif policy_name == "ctd":
            if ctd_policy is None:
                raise ValueError(
                    "CTD policy has not been fitted."
                )

            delegation_result = ctd_delegation(
                row=row,
                policy=ctd_policy,
            )

        elif policy_name == "always_large":
            delegation_result = always_large_delegation()

        else:
            raise ValueError(
                f"Unknown delegation policy: {policy_name}"
            )

        row["decision"] = delegation_result.decision
        row["delegation_policy"] = delegation_result.policy_name
        row["delegation_score"] = delegation_result.delegation_score

        # ---------------------------------------------------------
        # Record small-model result
        # ---------------------------------------------------------
        small_prediction = row["cleaned_prediction"]
        references = row["references"]

        row["small_prediction"] = small_prediction
        row["small_exact_match"] = exact_match(
            small_prediction,
            references,
        )
        row["small_token_f1"] = token_f1(
            small_prediction,
            references,
        )
        row["small_semantic_similarity"] = row["semantic_similarity"]

        # CTD calibration may already contain a large-model output.
        existing_large_raw = row.get("large_raw_prediction")
        existing_large_prediction = row.get("large_prediction")

        # Default behaviour: keep the small-model answer.
        row["escalated"] = False
        row["final_prediction"] = small_prediction

        if not delegation_result.should_escalate:
            # Keep any existing CTD calibration output for analysis,
            # but do not use it as the final answer.
            if existing_large_prediction is None:
                row["large_raw_prediction"] = None
                row["large_prediction"] = None
                row["large_exact_match"] = None
                row["large_token_f1"] = None
                row["large_semantic_similarity"] = None
            else:
                row["large_semantic_similarity"] = (
                    best_reference_semantic_similarity(
                        prediction=existing_large_prediction,
                        references=references,
                        similarity_function=similarity_function,
                    )
                )

        else:
            # -----------------------------------------------------
            # Obtain the large-model answer
            # -----------------------------------------------------
            if existing_large_prediction is not None:
                large_raw = existing_large_raw
                large_cleaned = existing_large_prediction
            else:
                large_raw, large_cleaned = large_generator.generate(
                    question=example["question"],
                    contexts=row["retrieved_contexts"],
                    max_new_tokens=large_generation_config[
                        "max_new_tokens"
                    ],
                    temperature=large_generation_config[
                        "temperature"
                    ],
                )

            row["escalated"] = True
            row["large_raw_prediction"] = large_raw
            row["large_prediction"] = large_cleaned
            row["large_exact_match"] = exact_match(
                large_cleaned,
                references,
            )
            row["large_token_f1"] = token_f1(
                large_cleaned,
                references,
            )
            row["large_semantic_similarity"] = (
                best_reference_semantic_similarity(
                    prediction=large_cleaned,
                    references=references,
                    similarity_function=similarity_function,
                )
            )
            row["final_prediction"] = large_cleaned

        # ---------------------------------------------------------
        # Evaluate the final cascade answer
        # ---------------------------------------------------------
        row["final_exact_match"] = exact_match(
            row["final_prediction"],
            references,
        )
        row["final_token_f1"] = token_f1(
            row["final_prediction"],
            references,
        )
        row["final_semantic_similarity"] = (
            best_reference_semantic_similarity(
                prediction=row["final_prediction"],
                references=references,
                similarity_function=similarity_function,
            )
        )

        row["escalation_improved"] = (
            row["escalated"]
            and row["small_exact_match"] == 0
            and row["final_exact_match"] == 1
        )

        row["escalation_harmed"] = (
            row["escalated"]
            and row["small_exact_match"] == 1
            and row["final_exact_match"] == 0
        )

    return rows

def evaluate_examples(
    examples: list[dict],
    retriever: DenseRetriever,
    small_generator: AnswerGenerator,
    config: dict,
) -> list[dict]:
    rows: list[dict] = []

    retrieval_config = config["retrieval"]
    small_generation_config = config["generation"]["small_model"]
    monitoring_config = config["monitoring"]

    for example in tqdm(examples, desc="Evaluating"):
        retrieval_result = retriever.retrieve(
            example["question"],
            top_k=retrieval_config["top_k"],
        )

        raw_answers: list[str] = []
        cleaned_answers: list[str] = []

        for sample_index in range(small_generation_config["num_consistency_samples"]):
            temperature = small_generation_config["temperature"]
            if sample_index > 0 and temperature == 0:
                temperature = 0.7

            raw_answer, cleaned_answer = small_generator.generate(
                question=example["question"],
                contexts=retrieval_result.contexts,
                max_new_tokens=small_generation_config["max_new_tokens"],
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
        semantic_similarity = best_reference_semantic_similarity(
            prediction=prediction,
            references=references,
            similarity_function=retriever.semantic_similarity,
        )

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
            "semantic_similarity": semantic_similarity,
        })

    return rows

def generate_large_calibration_outputs(
    rows: list[dict],
    examples: list[dict],
    large_generator: AnswerGenerator,
    large_generation_config: dict,
) -> list[dict]:
    """
    Generate large-model outputs for every CTD calibration example.

    CTD needs both small- and large-model outcomes to learn when
    delegation is beneficial.
    """

    if len(rows) != len(examples):
        raise ValueError(
            "Rows and examples must have equal lengths."
        )

    for row, example in tqdm(
        zip(rows, examples),
        total=len(rows),
        desc="Generating CTD calibration outputs",
    ):
        large_raw, large_cleaned = large_generator.generate(
            question=example["question"],
            contexts=row["retrieved_contexts"],
            max_new_tokens=large_generation_config[
                "max_new_tokens"
            ],
            temperature=large_generation_config[
                "temperature"
            ],
        )

        references = row["references"]

        row["small_prediction"] = row["cleaned_prediction"]
        row["small_exact_match"] = exact_match(
            row["small_prediction"],
            references,
        )
        row["small_token_f1"] = token_f1(
            row["small_prediction"],
            references,
        )

        row["large_raw_prediction"] = large_raw
        row["large_prediction"] = large_cleaned
        row["large_exact_match"] = exact_match(
            large_cleaned,
            references,
        )
        row["large_token_f1"] = token_f1(
            large_cleaned,
            references,
        )

        row["delegation_beneficial"] = (
            row["small_exact_match"] == 0
            and row["large_exact_match"] == 1
        )

        row["delegation_harmful"] = (
            row["small_exact_match"] == 1
            and row["large_exact_match"] == 0
        )

        row["delegation_em_gain"] = (
            float(row["large_exact_match"])
            - float(row["small_exact_match"])
        )

    return rows

def aggregate_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {}

    accepted = [row for row in rows if row.get("decision") == "accept"]
    retrieved = [row for row in rows if row["gold_context_retrieved"]]
    not_retrieved = [row for row in rows if not row["gold_context_retrieved"]]

    escalated = [
    row for row in rows
    if row.get("escalated", False)
    ]

    improved = [
        row for row in escalated
        if row["small_exact_match"] == 0
        and row["final_exact_match"] == 1
    ]

    harmed = [
        row for row in escalated
        if row["small_exact_match"] == 1
        and row["final_exact_match"] == 0
    ]

    return {
        "n_examples": len(rows),
        "exact_match": float(np.mean([row["exact_match"] for row in rows])),
        "token_f1": float(np.mean([row["token_f1"] for row in rows])),
        "semantic_similarity": float(
                        np.mean([
                            row["semantic_similarity"]
                            for row in rows
                        ])
                    ),
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
        "cascade": {
            "escalation_rate": len(escalated) / len(rows),

            "small_model_exact_match": float(
                np.mean([row["small_exact_match"] for row in rows])
            ),
            "small_model_token_f1": float(
                np.mean([row["small_token_f1"] for row in rows])
            ),

            "large_exact_match_on_escalated": (
                float(np.mean([
                    row["large_exact_match"]
                    for row in escalated
                ]))
                if escalated
                else None
            ),

            "large_token_f1_on_escalated": (
                float(np.mean([
                    row["large_token_f1"]
                    for row in escalated
                ]))
                if escalated
                else None
            ),

            "final_exact_match": float(
                np.mean([row["final_exact_match"] for row in rows])
            ),

            "final_token_f1": float(
                np.mean([row["final_token_f1"] for row in rows])
            ),

            "improved_count": len(improved),
            "harmed_count": len(harmed),

            # NEW
            "net_improvement_count": len(improved) - len(harmed),

            "usage": {
                "small_model_calls": len(rows),
                "inference_large_model_calls": len(escalated),
                "inference_large_model_call_rate": len(escalated) / len(rows),
            },

            "final_em_gain_over_small": (
                float(np.mean([row["final_exact_match"] for row in rows]))
                - float(np.mean([row["small_exact_match"] for row in rows]))
            ),
            "small_model_semantic_similarity": float(
                np.mean([
                    row["small_semantic_similarity"]
                    for row in rows
                ])
            ),

            "large_semantic_similarity_on_escalated": (
                float(np.mean([
                    row["large_semantic_similarity"]
                    for row in escalated
                ]))
                if escalated
                else None
            ),

            "final_semantic_similarity": float(
                np.mean([
                    row["final_semantic_similarity"]
                    for row in rows
                ])
            ),

            "semantic_similarity_gain_over_small": (
                float(np.mean([
                    row["final_semantic_similarity"]
                    for row in rows
                ]))
                - float(np.mean([
                    row["small_semantic_similarity"]
                    for row in rows
                ]))
            ),
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
    small_model_config = config["generation"]["small_model"]
    large_model_config = config["generation"]["large_model"]

    small_generator = AnswerGenerator(
        model_name=small_model_config["model_name"],
        device=small_model_config["device"],
    )

    large_generator = AnswerGenerator(
        model_name=large_model_config["model_name"],
        device=large_model_config["device"],
    )

    calibration_rows = evaluate_examples(
        calibration_examples,
        retriever,
        small_generator,
        config,
    )

    policy_name = config["delegation"]["policy"]
    ctd_policy = None

    if policy_name == "ctd":
        calibration_rows = (
            generate_large_calibration_outputs(
                rows=calibration_rows,
                examples=calibration_examples,
                large_generator=large_generator,
                large_generation_config=large_model_config,
            )
        )

        ctd_policy = fit_ctd_policy(
            calibration_rows=calibration_rows,
            target_delegation_rate=config[
                "delegation"
            ]["target_delegation_rate"],
            seed=config["run"]["seed"],
        )

    # Learn the thresholds from the small-model calibration results.
    thresholds = calibrate_thresholds(
        calibration_rows,
        target_risk=config["monitoring"]["target_risk"],
        flag_margin=config["monitoring"]["flag_margin"],
    )

    # Apply decisions and call the large model for escalated calibration examples.
    calibration_rows = apply_cascade(
        rows=calibration_rows,
        examples=calibration_examples,
        large_generator=large_generator,
        large_generation_config=large_model_config,
        thresholds=thresholds,
        policy_name=policy_name,
        similarity_function=retriever.semantic_similarity,
        ctd_policy=ctd_policy,
    )

    test_rows = evaluate_examples(
        test_examples,
        retriever,
        small_generator,
        config,
    )

    # Use the calibration thresholds on the unseen test examples.
    test_rows = apply_cascade(
        rows=test_rows,
        examples=test_examples,
        large_generator=large_generator,
        large_generation_config=large_model_config,
        thresholds=thresholds,
        policy_name=policy_name,
        similarity_function=retriever.semantic_similarity,
        ctd_policy=ctd_policy,
    )

    threshold_payload = thresholds.to_dict()

    metrics = {
        "model_configuration": {
            "small_model": small_model_config["model_name"],
            "large_model": large_model_config["model_name"],
        },
        "signal_configuration": {
            "enabled_signals": config["monitoring"]["enabled_signals"],
            "score_weights": config["monitoring"]["score_weights"],
        },
        "delegation_configuration": {
        "policy": policy_name,
        "target_delegation_rate": (
            config["delegation"].get("target_delegation_rate")
        ),
        "ctd_threshold": (
            ctd_policy.threshold
            if ctd_policy is not None
            else None
        ),
        "ctd_features": (
            ctd_policy.feature_names
            if ctd_policy is not None
            else None
        ),

        # NEW
        "ctd_calibration_large_model_calls": (
            len(calibration_rows)
            if policy_name == "ctd"
            else 0
        ),
    },
        "calibration": aggregate_metrics(calibration_rows),
        "test": aggregate_metrics(test_rows),
        "calibration_status": threshold_payload,
    }

    save_jsonl(
        run_dir / "calibration_predictions.jsonl",
        calibration_rows,
    )
    save_jsonl(
        run_dir / "test_predictions.jsonl",
        test_rows,
    )
    save_json(
        run_dir / "thresholds.json",
        threshold_payload,
    )
    save_json(
        run_dir / "metrics.json",
        metrics,
    )
    save_summary_csv(
        run_dir / "summary.csv",
        calibration_rows,
        test_rows,
    )

    build_risk_coverage_table(calibration_rows).to_csv(
        run_dir / "risk_coverage_calibration.csv",
        index=False,
    )
    build_risk_coverage_table(test_rows).to_csv(
        run_dir / "risk_coverage_test.csv",
        index=False,
    )

    print(f"\nRun completed: {run_dir}")
    print(metrics)

if __name__ == "__main__":
    main()
