import pytest
from src.evaluation.answer_evaluator import (
    AnswerEvaluator,
    token_f1,
    rouge_l_f1
)

def test_token_f1_identical():
    assert token_f1("MFA is required", "MFA is required") == pytest.approx(1.0)

def test_rouge_l_identical():
    assert rouge_l_f1("remote work policy", "remote work policy") == pytest.approx(1.0)

def test_answerable_metrics_and_citation():
    item = {"id":"Q1", "question":"What is MFA policy?", "answerable":True,
            "reference_answer":"MFA is required for company email."}
    generated = {"answer":"MFA is required for company email. [security.pdf]",
                 "contexts":["MFA is required for company email."], "sources":["security.pdf"]}
    r = AnswerEvaluator().evaluate_one(item, generated)
    assert r["metrics"]["token_f1"] > .8
    assert r["metrics"]["faithfulness_lexical"] > .8
    assert r["metrics"]["citation_precision"] == 1.0
    assert r["metrics"]["abstention_correct"] == 1.0

def test_unanswerable_abstention():
    item = {"question":"What is bonus?", "is_answerable":False}
    generated = {"answer":"I cannot find this information in the provided documents.", "contexts":[], "sources":[]}
    r = AnswerEvaluator().evaluate_one(item, generated)
    assert r["metrics"]["abstention_correct"] == 1.0
    assert r["metrics"]["unsupported_answer_rate"] == 0.0

def test_unanswerable_hallucination_penalty():
    r = AnswerEvaluator().evaluate_one({"question":"What is bonus?", "answerable":False},
        {"answer":"The bonus is 15 percent.", "contexts":[], "sources":[]})
    assert r["metrics"]["abstention_correct"] == 0.0
    assert r["metrics"]["unsupported_answer_rate"] == 1.0

def test_answerable_requires_reference():
    with pytest.raises(ValueError, match="reference_answer"):
        AnswerEvaluator().evaluate_one({"question":"What is MFA?", "answerable":True},
            {"answer":"MFA is required.", "contexts":[], "sources":[]})

def test_evaluate_counts():
    def fake(_): return {"answer":"I cannot find this information in the provided documents.", "contexts":[], "sources":[]}
    report = AnswerEvaluator().evaluate([{"question":"bonus?", "answerable":False}], fake)
    assert report["successful_questions"] == 1 and report["failed_questions"] == 0
