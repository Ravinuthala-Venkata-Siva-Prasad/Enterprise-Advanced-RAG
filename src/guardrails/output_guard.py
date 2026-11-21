from src.guardrails.citation_validator import CitationValidator
from src.guardrails.pii_detector import PIIDetector


class OutputGuard:

    def __init__(self):

        self.citation_validator = CitationValidator()
        self.pii_detector = PIIDetector()

    def validate(self, answer, retrieved_documents):

        citation_result = self.citation_validator.validate(
            answer,
            retrieved_documents
        )

        pii_result = self.pii_detector.detect(answer)

        return {
            "citation_validation": citation_result,
            "pii_detection": pii_result,
            "passed": (
                citation_result["valid"]
                and not pii_result["contains_pii"]
            )
        }