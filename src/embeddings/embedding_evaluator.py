import time

import numpy as np


class EmbeddingEvaluator:

    @staticmethod
    def evaluate(vectors):

        if vectors.ndim != 2 or vectors.shape[0] == 0:
            raise ValueError("Expected a non-empty 2D embedding matrix")

        norms = np.linalg.norm(vectors, axis=1)

        return {
            "number_of_embeddings": int(vectors.shape[0]),
            "embedding_dimension": int(vectors.shape[1]),
            "average_vector_norm": float(np.mean(norms)),
            "minimum_vector_norm": float(np.min(norms)),
            "maximum_vector_norm": float(np.max(norms)),
            "all_finite": bool(np.isfinite(vectors).all()),
            "normalized": bool(
                np.allclose(norms, 1.0, atol=1e-3)
            )
        }

    @staticmethod
    def similarity(vector_a, vector_b):

        vector_a = np.asarray(vector_a)
        vector_b = np.asarray(vector_b)

        if vector_a.shape != vector_b.shape:
            raise ValueError("Vector dimensions must match")

        denominator = (
            np.linalg.norm(vector_a) *
            np.linalg.norm(vector_b)
        )

        if denominator == 0:
            raise ValueError("Cannot compare zero vectors")

        return float(
            np.dot(vector_a, vector_b) / denominator
        )

    @staticmethod
    def measure_query_latency(embedding_model, query, runs=5):

        if runs < 1:
            raise ValueError("runs must be at least 1")

        # Warm-up
        embedding_model.embed_query(query)

        timings = []

        for _ in range(runs):

            start = time.perf_counter()

            embedding_model.embed_query(query)

            timings.append(
                (time.perf_counter() - start) * 1000
            )

        return {
            "average_latency_ms": float(np.mean(timings)),
            "minimum_latency_ms": float(np.min(timings)),
            "maximum_latency_ms": float(np.max(timings))
        }