from collections import defaultdict


class HybridRetriever:

    def __init__(
        self,
        dense_retriever,
        sparse_retriever,
        dense_weight=0.5,
        sparse_weight=0.5,
        rrf_k=60
    ):

        if dense_weight < 0 or sparse_weight < 0:
            raise ValueError("Retrieval weights cannot be negative")

        if dense_weight + sparse_weight == 0:
            raise ValueError("At least one retrieval weight must be positive")

        if rrf_k < 0:
            raise ValueError("rrf_k cannot be negative")

        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever

        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k

    @staticmethod
    def _document_id(document):

        metadata = document.get("metadata", {})

        chunk_id = metadata.get("chunk_id")

        if chunk_id:
            return str(chunk_id)

        return (
            metadata.get("source", ""),
            metadata.get("page_number", ""),
            document.get("page_content", "")
        )

    def retrieve(self, query, top_k=10):

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        dense_results = self.dense_retriever.retrieve(
            query,
            top_k=top_k
        )

        sparse_results = self.sparse_retriever.retrieve(
            query,
            top_k=top_k
        )

        fused = {}
        scores = defaultdict(float)

        for weight, results in [
            (self.dense_weight, dense_results),
            (self.sparse_weight, sparse_results)
        ]:

            for rank, result in enumerate(results, start=1):

                document = result["document"]
                doc_id = self._document_id(document)

                scores[doc_id] += weight / (self.rrf_k + rank)

                if doc_id not in fused:
                    fused[doc_id] = {
                        "document": document,
                        "retrieval_methods": []
                    }

                if result["retrieval_method"] not in fused[doc_id]["retrieval_methods"]:
                    fused[doc_id]["retrieval_methods"].append(
                        result["retrieval_method"]
                    )

        ranked = sorted(
            fused.items(),
            key=lambda item: scores[item[0]],
            reverse=True
        )

        return [
            {
                "document": fused[doc_id]["document"],
                "score": scores[doc_id],
                "retrieval_methods": fused[doc_id]["retrieval_methods"],
                "retrieval_method": "hybrid"
            }
            for doc_id, _ in ranked[:top_k]
        ]