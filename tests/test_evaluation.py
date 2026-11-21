import pytest

from src.evaluation.retrieval_metrics import RetrievalMetrics
from src.evaluation.answer_metrics import AnswerMetrics


def test_recall_at_k():

    retrieved = ["A", "B", "C"]
    relevant = ["A", "C", "D"]

    assert RetrievalMetrics.recall_at_k(
        retrieved, relevant, 3
    ) == pytest.approx(2 / 3)


def test_precision_at_k():

    retrieved = ["A", "B", "C"]
    relevant = ["A", "C"]

    assert RetrievalMetrics.precision_at_k(
        retrieved, relevant, 3
    ) == pytest.approx(2 / 3)


def test_mrr():

    retrieved = ["A", "B", "C"]
    relevant = ["C"]

    assert RetrievalMetrics.reciprocal_rank(
        retrieved, relevant
    ) == pytest.approx(1 / 3)


def test_average_precision():

    retrieved = ["A", "B", "C", "D"]
    relevant = ["A", "C"]

    assert RetrievalMetrics.average_precision(
        retrieved, relevant
    ) == pytest.approx((1 + 2 / 3) / 2)


def test_ndcg():

    retrieved = ["A", "B", "C"]
    relevant = ["A", "C"]

    score = RetrievalMetrics.ndcg_at_k(
        retrieved, relevant, 3
    )

    assert 0 <= score <= 1


def test_exact_match():

    assert AnswerMetrics.exact_match(
        "MFA is required.",
        "MFA is required"
    ) == 1.0


def test_token_f1():

    score = AnswerMetrics.token_f1(
        "MFA is required for email",
        "MFA is required"
    )

    assert 0 < score <= 1


def test_correct_abstention():

    result = AnswerMetrics.evaluate(
        prediction=(
            "I could not find sufficient information "
            "in the provided documents."
        ),
        reference="",
        answerable=False
    )

    assert result["abstention_correct"] is True


def test_wrong_abstention():

    result = AnswerMetrics.evaluate(
        prediction="The bonus is 20 percent.",
        reference="",
        answerable=False
    )

    assert result["abstention_correct"] is False