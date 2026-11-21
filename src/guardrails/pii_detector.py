import re


class PIIDetector:

    PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "phone": r"\b(?:\+91[- ]?)?[6-9]\d{9}\b",
        "aadhaar_like": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
        "pan_like": r"\b[A-Z]{5}\d{4}[A-Z]\b"
    }

    def detect(self, text):

        findings = {}

        for pii_type, pattern in self.PATTERNS.items():

            matches = re.findall(
                pattern,
                text,
                flags=re.IGNORECASE
            )

            if matches:
                findings[pii_type] = len(matches)

        return {
            "contains_pii": bool(findings),
            "findings": findings
        }