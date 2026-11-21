import re
from collections import Counter


class AnswerMetrics:

    ABSTENTION_TEXT = (
        "I could not find sufficient information "
        "in the provided documents."
    )

    @staticmethod
    def normalize(text):

        text = text.lower()

        text = re.sub(r"[^a-z0-9\s]", " ", text)

        return " ".join(text.split())

    @classmethod
    def exact_match(cls, prediction, reference):

        return float(
            cls.normalize(prediction)
            == cls.normalize(reference)
        )

    @classmethod
    def token_f1(cls, prediction, reference):

        prediction_tokens = cls.normalize(
            prediction
        ).split()

        reference_tokens = cls.normalize(
            reference
        ).split()

        if not prediction_tokens or not reference_tokens:
            return 0.0

        common = (
            Counter(prediction_tokens)
            & Counter(reference_tokens)
        )

        overlap = sum(common.values())

        if overlap == 0:
            return 0.0

        precision = overlap / len(prediction_tokens)
        recall = overlap / len(reference_tokens)

        return (
            2 * precision * recall
            / (precision + recall)
        )

    @classmethod
    def abstained(cls, answer):

        normalized = cls.normalize(answer)

        return (
            "could not find sufficient information"
            in normalized
        )

    @classmethod
    def evaluate(cls, prediction, reference, answerable):

        is_abstained = cls.abstained(prediction)

        if not answerable:

            return {
                "answerable": False,
                "abstained": is_abstained,
                "abstention_correct": is_abstained,
                "exact_match": None,
                "token_f1": None
            }

        return {
            "answerable": True,
            "abstained": is_abstained,
            "abstention_correct": None,
            "exact_match": cls.exact_match(
                prediction,
                reference
            ),
            "token_f1": cls.token_f1(
                prediction,
                reference
            )
        }