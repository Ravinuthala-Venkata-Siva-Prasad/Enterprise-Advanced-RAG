import argparse
import json
from pathlib import Path

from src.embeddings.embedding_manager import EmbeddingManager
from src.embeddings.embedding_evaluator import EmbeddingEvaluator
from src.utils.logger import logger


def main():

    parser = argparse.ArgumentParser(description="Generate and evaluate RAG embeddings")
    parser.add_argument("--input",default="data/processed/chunks.jsonl")
    parser.add_argument("--output",default="data/embeddings")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32
    )

    args = parser.parse_args()

    try:

        logger.info("Starting embedding generation")

        manager = EmbeddingManager()

        chunks, vectors = manager.process(
            input_path=args.input,
            output_dir=args.output,
            batch_size=args.batch_size
        )

        evaluator = EmbeddingEvaluator()

        report = evaluator.evaluate(vectors)

        report["model_name"] = manager.embedding_model.model_name

        latency = evaluator.measure_query_latency(
            manager.embedding_model,
            "How does the company manage security?"
        )

        report["query_latency"] = latency

        report_path = Path("reports/evaluation")
        report_path.mkdir(parents=True, exist_ok=True)

        with (report_path / "embedding_report.json").open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(report, file, indent=4)

        logger.info(f"Processed chunks: {len(chunks)}")
        logger.info(f"Embedding matrix shape: {vectors.shape}")
        logger.info(f"Evaluation report: {report}")

    except Exception:

        logger.exception("Embedding generation failed")
        raise


if __name__ == "__main__":
    main()