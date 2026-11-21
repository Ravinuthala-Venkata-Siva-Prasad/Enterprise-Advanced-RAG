from src.generation.rag_chain import EnterpriseRAGChain

from src.evaluation.evaluation_runner import EvaluationRunner
from src.evaluation.report_generator import EvaluationReportGenerator


def main():

    rag = EnterpriseRAGChain()

    evaluator = EvaluationRunner(rag_chain=rag)

    results = evaluator.run(
        dataset_path="data/evaluation/qa_dataset.json",
        k=5
    )

    summary = EvaluationReportGenerator.save(
        results,
        output_dir="reports/evaluation"
    )

    print("\n========== EVALUATION SUMMARY ==========")

    for metric, value in summary.items():
        print(f"{metric}: {value}")

    print("\nReports saved in reports/evaluation/")


if __name__ == "__main__":
    main()