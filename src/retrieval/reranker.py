from sentence_transformers import CrossEncoder


class CrossEncoderReranker:

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):

        self.model = CrossEncoder(model_name)

    def rerank(self, query, results, top_k=5):

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if not results:
            return []

        pairs = [
            (
                query,
                result["document"]["page_content"]
            )
            for result in results
        ]

        scores = self.model.predict(
            pairs,
            show_progress_bar=False
        )

        reranked = []

        for result, score in zip(results, scores):

            item = result.copy()

            item["reranker_score"] = float(score)

            reranked.append(item)

        reranked.sort(
            key=lambda item: item["reranker_score"],
            reverse=True
        )

        return reranked[:top_k]