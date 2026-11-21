
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.answer_evaluator import AnswerEvaluator
from src.evaluation.rag_runner import RAGEvaluationRunner
from src.evaluation.evaluation_report import save_evaluation_report


def main():
    dataset_path = (
        PROJECT_ROOT / "data" / "evaluation" / "qa_dataset.json"
    )

    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    with dataset_path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list):
        raise ValueError("qa_dataset.json must contain a JSON list")

    print("\n========== ADVANCED RAG ANSWER EVALUATION ==========\n")
    print(f"Dataset: {dataset_path}")
    print(f"Questions: {len(dataset)}")

    runner = RAGEvaluationRunner()

    report = AnswerEvaluator().evaluate(
        dataset,
        runner.run
    )

    print("\n========== ANSWER EVALUATION RESULTS ==========")

    for key in (
        "total_questions",
        "successful_questions",
        "failed_questions",
        "answerable_questions",
        "unanswerable_questions"
    ):
        print(f"{key:25s}: {report[key]}")

    print("\nAverage metrics:")

    for name, value in report["average_metrics"].items():
        print(f"{name:30s}: {value:.4f}")

    json_path, csv_path = save_evaluation_report(report)

    print(f"\nJSON: {json_path}")
    print(f"CSV : {csv_path}")

    if report["failures"]:
        print("\nEvaluation failures:")

        for failure in report["failures"]:
            print(failure)


if __name__ == "__main__":
    main()
