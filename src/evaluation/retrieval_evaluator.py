
from __future__ import annotations

import math
from typing import Any


# ============================================================
# DOCUMENT METADATA HELPERS
# ============================================================

def _get_metadata(document: dict) -> dict:
    """Support flat and nested metadata."""
    metadata = document.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _get_chunk_id(result: dict) -> str | None:
    """Extract chunk ID from a retrieval result."""
    document = result.get("document", result)

    if not isinstance(document, dict):
        return None

    metadata = _get_metadata(document)
    value = metadata.get("chunk_id", document.get("chunk_id"))

    return str(value) if value is not None else None


def _get_source(result: dict) -> str | None:
    """Extract and normalize source path."""
    document = result.get("document", result)

    if not isinstance(document, dict):
        return None

    metadata = _get_metadata(document)
    value = metadata.get("source", document.get("source"))

    if value is None:
        return None

    return str(value).replace("\\", "/").lower()


def _normalize_source_name(source: str) -> str:
    """Normalize source path to filename."""
    return (
        str(source)
        .replace("\\", "/")
        .lower()
        .split("/")[-1]
    )


# ============================================================
# RETRIEVAL METRICS
# ============================================================

def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int
) -> float:
    """Calculate Precision@K."""

    if k <= 0:
        raise ValueError("k must be greater than zero")

    top_k = retrieved_ids[:k]

    if not top_k:
        return 0.0

    return sum(
        item in relevant_ids for item in top_k
    ) / k


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int
) -> float:
    """Calculate Recall@K."""

    if k <= 0:
        raise ValueError("k must be greater than zero")

    if not relevant_ids:
        return 0.0

    found = len(
        set(retrieved_ids[:k]) & relevant_ids
    )

    return found / len(relevant_ids)


def hit_rate_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int
) -> float:
    """Calculate Hit Rate@K."""

    if k <= 0:
        raise ValueError("k must be greater than zero")

    return float(
        any(
            item in relevant_ids
            for item in retrieved_ids[:k]
        )
    )


def reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: set[str]
) -> float:
    """Calculate Mean Reciprocal Rank for one query."""

    for rank, item in enumerate(retrieved_ids, start=1):
        if item in relevant_ids:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int
) -> float:
    """Calculate Normalized Discounted Cumulative Gain@K."""

    if k <= 0:
        raise ValueError("k must be greater than zero")

    dcg = 0.0

    for rank, item in enumerate(
        retrieved_ids[:k],
        start=1
    ):
        relevance = 1.0 if item in relevant_ids else 0.0

        dcg += relevance / math.log2(rank + 1)

    ideal_hits = min(len(relevant_ids), k)

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return dcg / idcg if idcg else 0.0


# ============================================================
# RETRIEVAL EVALUATOR
# ============================================================

class RetrievalEvaluator:
    """
    Evaluates a retrieval pipeline using ground-truth chunk IDs
    or source filenames.

    Answerable question:

    {
        "question": "What is the MFA policy?",
        "answerable": true,
        "relevant_sources": ["security.pdf"]
    }

    Chunk-based question:

    {
        "question": "What is the MFA policy?",
        "is_answerable": true,
        "relevant_chunk_ids": ["chunk-001"]
    }

    Unanswerable question:

    {
        "question": "What is the employee bonus percentage?",
        "is_answerable": false,
        "relevant_sources": [],
        "relevant_chunk_ids": []
    }

    Chunk IDs take precedence over source labels when both
    are supplied.

    Note:
    Source-based evaluation considers a retrieved chunk relevant
    when its filename matches a ground-truth source filename.
    Duplicate chunks from one source can affect precision.
    """

    def __init__(
        self,
        retriever: Any,
        k_values: tuple[int, ...] = (1, 3, 5, 10)
    ):
        if not k_values or any(k <= 0 for k in k_values):
            raise ValueError(
                "k_values must contain positive integers"
            )

        self.retriever = retriever
        self.k_values = tuple(
            sorted(set(k_values))
        )

    # --------------------------------------------------------
    # SOURCE NORMALIZATION
    # --------------------------------------------------------

    @staticmethod
    def _normalize_sources(
        sources: list[str]
    ) -> set[str]:

        return {
            _normalize_source_name(source)
            for source in sources
            if str(source).strip()
        }

    # --------------------------------------------------------
    # SINGLE QUESTION EVALUATION
    # --------------------------------------------------------

    def evaluate_one(self, item: dict) -> dict:

        question = (
            item.get("question")
            or item.get("query")
        )

        if (
            not isinstance(question, str)
            or not question.strip()
        ):
            raise ValueError(
                "Each evaluation item needs a non-empty question"
            )

        # Support both dataset field names.
        is_answerable = item.get(
            "is_answerable",
            item.get("answerable", True)
        )

        if not isinstance(is_answerable, bool):
            raise ValueError(
                f"answerable/is_answerable must be boolean: {question}"
            )

        # Execute retrieval.
        results = self.retriever.retrieve(question)

        if not isinstance(results, list):
            raise TypeError(
                "Retriever must return a list of results"
            )

        retrieved_chunk_ids = [
            _get_chunk_id(result)
            for result in results
        ]

        retrieved_sources = [
            _get_source(result)
            for result in results
        ]

        base_result = {
            "question": question,
            "is_answerable": is_answerable,
            "retrieved_count": len(results),
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "retrieved_sources": retrieved_sources
        }

        # ----------------------------------------------------
        # UNANSWERABLE QUESTIONS
        # ----------------------------------------------------

        if not is_answerable:

            return {
                **base_result,
                "status": "unanswerable",
                "relevance_type": None,
                "relevance_flags": [],
                "metrics": {}
            }

        # ----------------------------------------------------
        # LOAD GROUND-TRUTH LABELS
        # ----------------------------------------------------

        relevant_ids = {
            str(value)
            for value in item.get(
                "relevant_chunk_ids",
                item.get("relevant_ids", [])
            )
            if value is not None
        }

        raw_sources = item.get(
            "relevant_sources",
            item.get("relevant_documents", [])
        )

        if isinstance(raw_sources, str):
            raw_sources = [raw_sources]

        if not isinstance(raw_sources, list):
            raise ValueError(
                "relevant_sources must be a list or string"
            )

        relevant_sources = self._normalize_sources(
            raw_sources
        )

        # ----------------------------------------------------
        # SELECT RELEVANCE STRATEGY
        # ----------------------------------------------------

        if relevant_ids:

            relevance_type = "chunk_id"

            retrieved_ids = [
                chunk_id
                if chunk_id is not None
                else f"unknown-{index}"
                for index, chunk_id in enumerate(
                    retrieved_chunk_ids
                )
            ]

            effective_relevant = relevant_ids

        elif relevant_sources:

            relevance_type = "source"

            retrieved_ids = [
                _normalize_source_name(source)
                if source
                else f"unknown-{index}"
                for index, source in enumerate(
                    retrieved_sources
                )
            ]

            effective_relevant = relevant_sources

        else:

            raise ValueError(
                "Answerable question has no ground-truth labels: "
                f"{question}"
            )

        # ----------------------------------------------------
        # CALCULATE METRICS
        # ----------------------------------------------------

        metrics = {}

        for k in self.k_values:

            metrics[f"precision@{k}"] = precision_at_k(
                retrieved_ids,
                effective_relevant,
                k
            )

            metrics[f"recall@{k}"] = recall_at_k(
                retrieved_ids,
                effective_relevant,
                k
            )

            metrics[f"hit_rate@{k}"] = hit_rate_at_k(
                retrieved_ids,
                effective_relevant,
                k
            )

            metrics[f"ndcg@{k}"] = ndcg_at_k(
                retrieved_ids,
                effective_relevant,
                k
            )

        metrics["mrr"] = reciprocal_rank(
            retrieved_ids,
            effective_relevant
        )

        # ----------------------------------------------------
        # RELEVANCE FLAGS
        # ----------------------------------------------------

        relevance_flags = [
            value in effective_relevant
            for value in retrieved_ids
        ]

        return {
            **base_result,
            "status": "evaluated",
            "relevance_type": relevance_type,
            "relevance_flags": relevance_flags,
            "metrics": metrics
        }

    # --------------------------------------------------------
    # COMPLETE DATASET EVALUATION
    # --------------------------------------------------------

    def evaluate(
        self,
        dataset: list[dict]
    ) -> dict:

        if not dataset:
            raise ValueError(
                "Evaluation dataset is empty"
            )

        results = []
        errors = []

        for index, item in enumerate(dataset):

            try:
                results.append(
                    self.evaluate_one(item)
                )

            except Exception as exc:

                errors.append({
                    "index": index,
                    "question": item.get(
                        "question",
                        item.get("query", "")
                    ),
                    "error": str(exc)
                })

        # ----------------------------------------------------
        # SEPARATE QUESTION CATEGORIES
        # ----------------------------------------------------

        evaluated_results = [
            row
            for row in results
            if row["status"] == "evaluated"
        ]

        unanswerable_results = [
            row
            for row in results
            if row["status"] == "unanswerable"
        ]

        # ----------------------------------------------------
        # CALCULATE AVERAGES
        # ----------------------------------------------------

        averages = {}

        if evaluated_results:

            metric_names = (
                evaluated_results[0]["metrics"].keys()
            )

            averages = {
                name: sum(
                    row["metrics"][name]
                    for row in evaluated_results
                ) / len(evaluated_results)

                for name in metric_names
            }

        # ----------------------------------------------------
        # FINAL REPORT
        # ----------------------------------------------------

        return {
            "total_questions": len(dataset),

            "successful_questions": len(results),

            "answerable_questions": len(
                evaluated_results
            ),

            "unanswerable_questions": len(
                unanswerable_results
            ),

            "failed_questions": len(errors),

            "average_metrics": averages,

            "results": results,

            "errors": errors
        }
