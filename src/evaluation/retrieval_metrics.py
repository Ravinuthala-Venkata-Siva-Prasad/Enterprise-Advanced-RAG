import math


class RetrievalMetrics:

    @staticmethod
    def recall_at_k(retrieved, relevant, k):

        relevant = set(relevant)

        if not relevant:
            return None

        retrieved_k = retrieved[:k]

        hits = len(set(retrieved_k) & relevant)

        return hits / len(relevant)

    @staticmethod
    def precision_at_k(retrieved, relevant, k):

        if k <= 0:
            raise ValueError("k must be positive")

        relevant = set(relevant)
        retrieved_k = retrieved[:k]

        hits = len(set(retrieved_k) & relevant)

        return hits / k

    @staticmethod
    def reciprocal_rank(retrieved, relevant):

        relevant = set(relevant)

        if not relevant:
            return None

        for rank, item in enumerate(retrieved, start=1):

            if item in relevant:
                return 1 / rank

        return 0.0

    @staticmethod
    def average_precision(retrieved, relevant):

        relevant = set(relevant)

        if not relevant:
            return None

        hits = 0
        precision_sum = 0.0

        for rank, item in enumerate(retrieved, start=1):

            if item in relevant:
                hits += 1
                precision_sum += hits / rank

        return precision_sum / len(relevant)

    @staticmethod
    def ndcg_at_k(retrieved, relevant, k):

        if k <= 0:
            raise ValueError("k must be positive")

        relevant = set(relevant)

        if not relevant:
            return None

        dcg = sum(
            1 / math.log2(rank + 1)
            for rank, item in enumerate(retrieved[:k], start=1)
            if item in relevant
        )

        ideal_hits = min(len(relevant), k)

        idcg = sum(
            1 / math.log2(rank + 1)
            for rank in range(1, ideal_hits + 1)
        )

        return dcg / idcg if idcg else 0.0

    @classmethod
    def evaluate(cls, retrieved, relevant, k=5):

        return {
            f"recall@{k}": cls.recall_at_k(
                retrieved, relevant, k
            ),
            f"precision@{k}": cls.precision_at_k(
                retrieved, relevant, k
            ),
            "mrr": cls.reciprocal_rank(
                retrieved, relevant
            ),
            "map": cls.average_precision(
                retrieved, relevant
            ),
            f"ndcg@{k}": cls.ndcg_at_k(
                retrieved, relevant, k
            )
        }