import json
import time
from pathlib import Path

from src.evaluation.retrieval_metrics import RetrievalMetrics
from src.evaluation.answer_metrics import AnswerMetrics


class EvaluationRunner:

    def __init__(self, rag_chain, retrieval_pipeline=None):

        self.rag_chain = rag_chain
        self.retrieval_pipeline = (
            retrieval_pipeline or rag_chain.retriever
        )

    @staticmethod
    def load_dataset(path):

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(path)

        with path.open("r", encoding="utf-8") as file:
            dataset = json.load(file)

        if not isinstance(dataset, list):
            raise ValueError("Evaluation dataset must be a JSON list")

        return dataset

    @staticmethod
    def get_source_name(document):

        metadata = document.get("metadata", {})

        source = metadata.get("source") or metadata.get("file_name")

        if not source:
            return None

        return Path(source).name

    def evaluate_question(self, item, k=5):

        question = item["question"]

        relevant_sources = set(
            item.get("relevant_sources", [])
        )

        answerable = item["answerable"]

        start = time.perf_counter()

        retrieval_results = self.retrieval_pipeline.retrieve(
            question
        )

        retrieval_latency = (
            time.perf_counter() - start
        ) * 1000

        retrieved_sources = [
            self.get_source_name(result["document"])
            for result in retrieval_results
        ]

        retrieved_sources = [
            source for source in retrieved_sources
            if source is not None
        ]

        if answerable and not relevant_sources:
            raise ValueError(
                f"{item['id']} has no relevant_sources"
            )

        retrieval_metrics = RetrievalMetrics.evaluate(
            retrieved_sources,
            relevant_sources,
            k=k
        )

        start = time.perf_counter()

        result = self.rag_chain.ask(question)

        answer_latency = (
            time.perf_counter() - start
        ) * 1000

        answer_metrics = AnswerMetrics.evaluate(
            prediction=result["answer"],
            reference=item.get("reference_answer", ""),
            answerable=answerable
        )

        return {
            "id": item["id"],
            "question": question,
            "answer": result["answer"],
            "retrieved_sources": retrieved_sources,
            "relevant_sources": list(relevant_sources),
            "retrieval_metrics": retrieval_metrics,
            "answer_metrics": answer_metrics,
            "retrieval_latency_ms": retrieval_latency,
            "answer_latency_ms": answer_latency
        }

    def run(self, dataset_path, k=5):

        dataset = self.load_dataset(dataset_path)

        results = []

        for item in dataset:

            print(f"Evaluating {item['id']}: {item['question']}")

            result = self.evaluate_question(item, k=k)

            results.append(result)

        return results