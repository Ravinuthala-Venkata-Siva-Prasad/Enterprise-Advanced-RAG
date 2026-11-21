import re


class InjectionDetector:

    PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"ignore\s+(all\s+)?system\s+instructions",
        r"reveal\s+(your\s+)?system\s+prompt",
        r"show\s+your\s+hidden\s+instructions",
        r"act\s+as\s+if\s+you\s+have\s+no\s+restrictions",
        r"bypass\s+(all\s+)?safety\s+rules",
        r"developer\s+message\s+override"
    ]

    def detect(self, text):

        if not isinstance(text, str):
            raise TypeError("Input must be a string")

        matched_patterns = []

        for pattern in self.PATTERNS:

            if re.search(pattern, text, re.IGNORECASE):
                matched_patterns.append(pattern)

        return {
            "is_suspicious": bool(matched_patterns),
            "matched_patterns": matched_patterns
        }