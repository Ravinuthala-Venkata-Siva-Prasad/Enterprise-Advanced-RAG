
from __future__ import annotations

import re
from collections import Counter
from typing import Callable


# ============================================================
# TEXT PROCESSING
# ============================================================

def _evaluation_tokens(text: str) -> list[str]:
    """Normalize text into lowercase tokens."""

    if not text:
        return []

    return re.findall(r"\b\w+\b", str(text).lower())


def _strip_citations(text: str) -> str:
    """Remove filename citations such as [security.pdf]."""

    if not text:
        return ""

    return re.sub(
        r"\[[^\]]+\.(?:pdf|docx?|txt|csv|html?|md)\]",
        "",
        str(text),
        flags=re.IGNORECASE
    ).strip()


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "be", "been", "being", "to", "of", "in", "on",
    "at", "for", "and", "or", "but", "with", "by",
    "as", "it", "this", "that", "these", "those",
    "from", "if", "then", "than", "their", "there",
    "they", "them", "we", "you", "your", "our",
    "i", "me", "my", "can", "could", "should",
    "would", "will", "shall", "may", "might",
    "must", "do", "does", "did", "have", "has",
    "had", "not", "no", "yes", "into", "about",
    "which", "who", "what", "when", "where",
    "how", "why", "also", "such"
}


def _content_tokens(text: str) -> list[str]:
    """Return tokens excluding common function words."""

    return [
        token
        for token in _evaluation_tokens(_strip_citations(text))
        if token not in STOPWORDS
    ]


# ============================================================
# TOKEN F1
# ============================================================

def token_f1(prediction: str, reference: str) -> float:
    """Calculate token-level F1."""

    pred_tokens = _evaluation_tokens(
        _strip_citations(prediction)
    )

    ref_tokens = _evaluation_tokens(
        _strip_citations(reference)
    )

    if not pred_tokens and not ref_tokens:
        return 1.0

    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)

    overlap = sum(common.values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)


# ============================================================
# ROUGE-L F1
# ============================================================

def rouge_l_f1(prediction: str, reference: str) -> float:
    """Calculate ROUGE-L F1 using longest common subsequence."""

    pred_tokens = _evaluation_tokens(
        _strip_citations(prediction)
    )

    ref_tokens = _evaluation_tokens(
        _strip_citations(reference)
    )

    if not pred_tokens and not ref_tokens:
        return 1.0

    if not pred_tokens or not ref_tokens:
        return 0.0

    rows = len(pred_tokens) + 1
    cols = len(ref_tokens) + 1

    dp = [[0] * cols for _ in range(rows)]

    for i in range(1, rows):
        for j in range(1, cols):

            if pred_tokens[i - 1] == ref_tokens[j - 1]:

                dp[i][j] = dp[i - 1][j - 1] + 1

            else:

                dp[i][j] = max(
                    dp[i - 1][j],
                    dp[i][j - 1]
                )

    lcs_length = dp[-1][-1]

    precision = lcs_length / len(pred_tokens)
    recall = lcs_length / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


# ============================================================
# ANSWER EVALUATOR
# ============================================================

class AnswerEvaluator:
    """
    Evaluate generated RAG answers.

    Metrics:
        1. Token F1
        2. ROUGE-L F1
        3. Lexical faithfulness
        4. Lexical answer relevancy
        5. Citation precision
        6. Citation recall
        7. Abstention correctness
        8. Unsupported answer rate
    """

    def __init__(self):

        self.abstention_phrases = [
            "i cannot find",
            "cannot find this information",
            "not available in the provided documents",
            "not mentioned in the documents",
            "information is not available",
            "insufficient information",
            "no relevant information",
            "i don't know",
            "not found in the provided context"
        ]

    # ========================================================
    # ABSTENTION DETECTION
    # ========================================================

    def _is_abstention(self, answer: str) -> bool:

        answer_lower = str(answer or "").lower()

        return any(
            phrase in answer_lower
            for phrase in self.abstention_phrases
        )

    # ========================================================
    # SOURCE NORMALIZATION
    # ========================================================

    @staticmethod
    def _normalize_source(source) -> str:

        if isinstance(source, dict):

            source = (
                source.get("source")
                or source.get("filename")
                or source.get("file_name")
                or ""
            )

        if not source:
            return ""

        return (
            str(source)
            .replace("\\", "/")
            .split("/")[-1]
            .lower()
        )

    def _get_expected_sources(self, item: dict) -> set[str]:

        sources = (
            item.get("relevant_sources")
            or item.get("relevant_documents")
            or item.get("expected_sources")
            or []
        )

        if isinstance(sources, str):
            sources = [sources]

        return {
            self._normalize_source(source)
            for source in sources
            if self._normalize_source(source)
        }

    def _get_generated_sources(self, generated: dict) -> set[str]:

        sources = generated.get("sources") or []

        normalized = {
            self._normalize_source(source)
            for source in sources
            if self._normalize_source(source)
        }

        answer = str(generated.get("answer", ""))

        citations = re.findall(
            r"\[([^\]]+\.(?:pdf|docx?|txt|csv|html?|md))\]",
            answer,
            flags=re.IGNORECASE
        )

        for citation in citations:
            normalized.add(
                self._normalize_source(citation)
            )

        return normalized

    # ========================================================
    # FAITHFULNESS
    # ========================================================

    def _faithfulness(
        self,
        answer: str,
        contexts: list
    ) -> float:
        """
        Estimate lexical context support.

        This is a lexical overlap metric, not semantic
        entailment or factual verification.
        """

        answer_tokens = _content_tokens(answer)

        if not answer_tokens:
            return 1.0

        context_text = " ".join(
            str(context)
            for context in contexts
        )

        context_tokens = set(
            _content_tokens(context_text)
        )

        if not context_tokens:
            return 0.0

        supported = sum(
            1
            for token in answer_tokens
            if token in context_tokens
        )

        return supported / len(answer_tokens)

    # ========================================================
    # ANSWER RELEVANCY
    # ========================================================

    def _answer_relevancy(
        self,
        question: str,
        answer: str
    ) -> float:
        """Estimate lexical overlap between question and answer."""

        question_tokens = set(
            _content_tokens(question)
        )

        answer_tokens = set(
            _content_tokens(answer)
        )

        if not answer_tokens or not question_tokens:
            return 0.0

        overlap = question_tokens.intersection(
            answer_tokens
        )

        return len(overlap) / len(answer_tokens)

    # ========================================================
    # CITATION EVALUATION
    # ========================================================

    def _citation_metrics(
        self,
        item: dict,
        generated: dict
    ) -> tuple[float, float]:

        predicted = self._get_generated_sources(generated)

        expected = self._get_expected_sources(item)

        # No generated citations.
        if not predicted:

            if not expected:
                return 1.0, 1.0

            return 0.0, 0.0

        # No ground-truth citation labels.
        if not expected:

            provided = generated.get("sources") or []

            if provided:

                provided_names = {
                    self._normalize_source(source)
                    for source in provided
                    if self._normalize_source(source)
                }

                precision = (
                    len(predicted.intersection(provided_names))
                    / len(predicted)
                )

                return precision, 1.0

            return 1.0, 1.0

        correct = predicted.intersection(expected)

        precision = len(correct) / len(predicted)

        recall = len(correct) / len(expected)

        return precision, recall

    # ========================================================
    # EVALUATE ONE QUESTION
    # ========================================================

    def evaluate_one(
        self,
        item: dict,
        generated: dict
    ) -> dict:

        question = (
            item.get("question")
            or item.get("query")
            or ""
        )

        answer = str(
            generated.get("answer", "") or ""
        )

        contexts = generated.get("contexts") or []

        if isinstance(contexts, str):
            contexts = [contexts]

        answerable = item.get(
            "is_answerable",
            item.get("answerable", True)
        )

        abstained = self._is_abstention(answer)

        abstention_correct = (
            abstained == (not answerable)
        )

        unsupported_answer_rate = (
            1.0
            if not answerable and not abstained
            else 0.0
        )

        # Reference-answer metrics apply only to answerable items.
        if answerable:

            reference = (
                item.get("reference_answer")
                or item.get("ground_truth")
                or item.get("expected_answer")
            )

            if not reference:

                raise ValueError(
                    "Answerable evaluation item requires "
                    "'reference_answer'."
                )

            reference = str(reference)

            token_score = token_f1(
                answer,
                reference
            )

            rouge_score = rouge_l_f1(
                answer,
                reference
            )

        else:

            token_score = 0.0
            rouge_score = 0.0

        faithfulness = self._faithfulness(
            answer,
            contexts
        )

        relevancy = self._answer_relevancy(
            question,
            answer
        )

        citation_precision, citation_recall = (
            self._citation_metrics(
                item,
                generated
            )
        )

        metrics = {
            "token_f1": token_score,
            "rouge_l_f1": rouge_score,
            "faithfulness_lexical": faithfulness,
            "answer_relevancy_lexical": relevancy,
            "citation_precision": citation_precision,
            "citation_recall": citation_recall,
            "abstention_correct": float(abstention_correct),
            "unsupported_answer_rate": unsupported_answer_rate
        }

        return {
            "id": item.get("id"),
            "question": question,
            "answer": answer,
            "answerable": answerable,
            "contexts": contexts,
            "sources": generated.get("sources", []),
            "metrics": metrics
            }

    # ========================================================
    # EVALUATE COMPLETE DATASET
    # ========================================================

    def evaluate(
        self,
        dataset: list[dict],
        answer_generator: Callable[[str], dict]
    ) -> dict:

        results = []
        failures = []

        for index, item in enumerate(dataset):

            question = (
                item.get("question")
                or item.get("query")
                or ""
            )

            try:

                if not isinstance(question, str) or not question.strip():

                    raise ValueError(
                        "Evaluation question is empty"
                    )

                # IMPORTANT:
                # Pass a question string, not the full item dictionary.
                generated = answer_generator(question)

                result = self.evaluate_one(
                    item,
                    generated
                )

                results.append(result)

            except Exception as exc:

                failures.append({
                    "index": index,
                    "question": question,
                    "error": str(exc)
                })

        successful = len(results)
        failed = len(failures)

        answerable_count = sum(
            1
            for item in dataset
            if item.get(
                "is_answerable",
                item.get("answerable", True)
            )
        )

        unanswerable_count = (
            len(dataset) - answerable_count
        )

        metric_names = [
            "token_f1",
            "rouge_l_f1",
            "faithfulness_lexical",
            "answer_relevancy_lexical",
            "citation_precision",
            "citation_recall",
            "abstention_correct",
            "unsupported_answer_rate"
        ]

        average_metrics = {}

        for metric in metric_names:

            values = [
                result["metrics"][metric]
                for result in results
            ]

            average_metrics[metric] = (
                sum(values) / len(values)
                if values
                else 0.0
            )

        return {
            "total_questions": len(dataset),
            "successful_questions": successful,
            "failed_questions": failed,
            "answerable_questions": answerable_count,
            "unanswerable_questions": unanswerable_count,
            "average_metrics": average_metrics,
            "results": results,
            "failures": failures
        }
