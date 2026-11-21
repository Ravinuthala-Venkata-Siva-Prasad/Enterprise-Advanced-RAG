class RelevanceGuard:

    def __init__(self, minimum_score=0.15):

        if not 0 <= minimum_score <= 1:
            raise ValueError("minimum_score must be between 0 and 1")

        self.minimum_score = minimum_score

    def validate(self, results):

        if not results:
            return {
                "is_relevant": False,
                "reason": "No documents retrieved"
            }

        scores = [
            float(result.get("reranker_score", 0))
            for result in results
        ]

        best_score = max(scores)

        return {
            "is_relevant": best_score >= self.minimum_score,
            "best_score": best_score,
            "reason": (
                "Relevant evidence found"
                if best_score >= self.minimum_score
                else "Retrieval relevance below threshold"
            )
        }