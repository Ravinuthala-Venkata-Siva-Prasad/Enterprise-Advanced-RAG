
import pytest

from src.evaluation.retrieval_evaluator import (
    RetrievalEvaluator,
    precision_at_k,
    recall_at_k,
    hit_rate_at_k,
    reciprocal_rank,
    ndcg_at_k,
)


def test_precision_at_k():
    retrieved = ["A", "B", "C"]
    relevant = {"A", "C"}

    assert precision_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)


def test_recall_at_k():
    retrieved = ["A", "B", "C"]
    relevant = {"A", "C"}

    assert recall_at_k(retrieved, relevant, 2) == pytest.approx(0.5)


def test_hit_rate_at_k():
    retrieved = ["A", "B", "C"]
    relevant = {"C"}

    assert hit_rate_at_k(retrieved, relevant, 3) == 1.0


def test_reciprocal_rank():
    retrieved = ["A", "B", "C"]
    relevant = {"C"}

    assert reciprocal_rank(retrieved, relevant) == pytest.approx(1 / 3)


def test_ndcg_at_k():
    retrieved = ["A", "B", "C"]
    relevant = {"A", "C"}

    assert 0.0 <= ndcg_at_k(retrieved, relevant, 3) <= 1.0


def test_retrieval_evaluator_with_source_labels():
    class FakeRetriever:
        def retrieve(self, question):
            return [
                {
                    "document": {
                        "metadata": {
                            "source": "policy.pdf"
                        }
                    }
                }
            ]

    evaluator = RetrievalEvaluator(FakeRetriever())

    result = evaluator.evaluate_one({
        "question": "What is the policy?",
        "is_answerable": True,
        "relevant_sources": ["policy.pdf"]
    })

    assert result["status"] == "evaluated"
    assert result["metrics"]["precision@1"] == 1.0


def test_missing_ground_truth_is_reported():
    class FakeRetriever:
        def retrieve(self, question):
            return []

    evaluator = RetrievalEvaluator(FakeRetriever())

    report = evaluator.evaluate([
        {
            "question": "What is the policy?",
            "is_answerable": True,
            "relevant_sources": []
        }
    ])

    assert report["failed_questions"] == 1


def test_unanswerable_question_is_not_failed():
    class FakeRetriever:
        def retrieve(self, question):
            return []

    evaluator = RetrievalEvaluator(FakeRetriever())

    result = evaluator.evaluate_one({
        "question": "What is the employee bonus percentage?",
        "is_answerable": False,
        "relevant_chunk_ids": [],
        "relevant_sources": []
    })

    assert result["status"] == "unanswerable"
    assert result["metrics"] == {}


def test_unanswerable_question_count():
    class FakeRetriever:
        def retrieve(self, question):
            return []

    evaluator = RetrievalEvaluator(FakeRetriever())

    dataset = [
        {
            "question": "What is MFA?",
            "is_answerable": True,
            "relevant_sources": ["security.pdf"]
        },
        {
            "question": "What is the employee bonus percentage?",
            "is_answerable": False,
            "relevant_sources": [],
            "relevant_chunk_ids": []
        }
    ]

    report = evaluator.evaluate(dataset)

    assert report["total_questions"] == 2
    assert report["successful_questions"] == 2
    assert report["answerable_questions"] == 1
    assert report["unanswerable_questions"] == 1
    assert report["failed_questions"] == 0


def test_answerable_question_without_labels_fails():
    class FakeRetriever:
        def retrieve(self, question):
            return []

    evaluator = RetrievalEvaluator(FakeRetriever())

    with pytest.raises(ValueError, match="no ground-truth labels"):
        evaluator.evaluate_one({
            "question": "What is the bonus percentage?",
            "is_answerable": True
        })
