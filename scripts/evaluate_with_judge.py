from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_monitoring.judge import JudgeEvaluator


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from error

    return rows


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
        )


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def judge_prediction(
    judge: JudgeEvaluator,
    row: dict,
    prediction_field: str,
    prefix: str,
    max_new_tokens: int,
) -> dict:
    prediction = row.get(prediction_field)

    if prediction is None:
        return {
            f"{prefix}_judge_faithfulness": None,
            f"{prefix}_judge_answer_relevance": None,
            f"{prefix}_judge_reasoning": None,
            f"{prefix}_judge_raw_response": None,
            f"{prefix}_judge_parse_success": False,
        }

    result = judge.evaluate(
        question=row["question"],
        contexts=row["retrieved_contexts"],
        prediction=prediction,
        max_new_tokens=max_new_tokens,
    )

    return {
        f"{prefix}_judge_faithfulness": result.faithfulness,
        f"{prefix}_judge_answer_relevance": result.answer_relevance,
        f"{prefix}_judge_reasoning": result.reasoning,
        f"{prefix}_judge_raw_response": result.raw_response,
        f"{prefix}_judge_parse_success": result.parse_success,
    }


def valid_scores(
    rows: list[dict],
    score_field: str,
    parse_success_field: str,
) -> list[float]:
    scores: list[float] = []

    for row in rows:
        if not row.get(parse_success_field, False):
            continue

        score = row.get(score_field)

        if score is not None:
            scores.append(float(score))

    return scores


def mean_or_none(
    values: list[float],
) -> float | None:
    if not values:
        return None

    return float(np.mean(values))


def aggregate_judge_metrics(
    rows: list[dict],
    prefix: str,
) -> dict:
    faithfulness = valid_scores(
        rows,
        f"{prefix}_judge_faithfulness",
        f"{prefix}_judge_parse_success",
    )

    relevance = valid_scores(
        rows,
        f"{prefix}_judge_answer_relevance",
        f"{prefix}_judge_parse_success",
    )

    successful = sum(
        bool(
            row.get(
                f"{prefix}_judge_parse_success",
                False,
            )
        )
        for row in rows
    )

    return {
        "n_examples": len(rows),
        "successful_judgements": successful,
        "failed_judgements": len(rows) - successful,
        "parse_success_rate": (
            successful / len(rows)
            if rows
            else None
        ),
        "mean_faithfulness": mean_or_none(
            faithfulness
        ),
        "mean_answer_relevance": mean_or_none(
            relevance
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to test_predictions.jsonl",
    )

    parser.add_argument(
        "--judge-model",
        default="mistralai/Mistral-7B-Instruct-v0.3",
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=160,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of examples for a smoke test.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory. Defaults to the directory containing "
            "the predictions file."
        ),
    )

    args = parser.parse_args()

    predictions_path = Path(
        args.predictions
    )

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {predictions_path}"
        )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else predictions_path.parent
    )

    rows = load_jsonl(
        predictions_path
    )

    if args.limit is not None:
        rows = rows[: args.limit]

    if not rows:
        raise ValueError(
            "No prediction rows were loaded."
        )

    judge = JudgeEvaluator(
        model_name=args.judge_model,
        device=args.device,
    )

    judged_rows: list[dict] = []

    for row in tqdm(
        rows,
        desc="LLM-as-a-Judge evaluation",
    ):
        judged_row = dict(row)

        small_scores = judge_prediction(
            judge=judge,
            row=row,
            prediction_field="small_prediction",
            prefix="small",
            max_new_tokens=args.max_new_tokens,
        )

        final_scores = judge_prediction(
            judge=judge,
            row=row,
            prediction_field="final_prediction",
            prefix="final",
            max_new_tokens=args.max_new_tokens,
        )

        judged_row.update(
            small_scores
        )

        judged_row.update(
            final_scores
        )

        judged_rows.append(
            judged_row
        )

    metrics = {
        "judge_configuration": {
            "judge_model": args.judge_model,
            "device": args.device,
            "max_new_tokens": args.max_new_tokens,
            "source_predictions": str(
                predictions_path
            ),
            "evaluated_examples": len(
                judged_rows
            ),
            "criteria": [
                "faithfulness",
                "answer_relevance",
            ],
        },

        "small_prediction": (
            aggregate_judge_metrics(
                judged_rows,
                prefix="small",
            )
        ),

        "final_prediction": (
            aggregate_judge_metrics(
                judged_rows,
                prefix="final",
            )
        ),
    }

    small_faithfulness = metrics[
        "small_prediction"
    ]["mean_faithfulness"]

    final_faithfulness = metrics[
        "final_prediction"
    ]["mean_faithfulness"]

    small_relevance = metrics[
        "small_prediction"
    ]["mean_answer_relevance"]

    final_relevance = metrics[
        "final_prediction"
    ]["mean_answer_relevance"]

    metrics[
        "judge_faithfulness_gain_over_small"
    ] = (
        final_faithfulness
        - small_faithfulness
        if (
            final_faithfulness is not None
            and small_faithfulness is not None
        )
        else None
    )

    metrics[
        "judge_answer_relevance_gain_over_small"
    ] = (
        final_relevance
        - small_relevance
        if (
            final_relevance is not None
            and small_relevance is not None
        )
        else None
    )

    save_jsonl(
        output_dir
        / "judge_results.jsonl",
        judged_rows,
    )

    save_json(
        output_dir
        / "judge_metrics.json",
        metrics,
    )

    print(
        f"Judge evaluation completed: {output_dir}"
    )

    print(
        json.dumps(
            metrics,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()