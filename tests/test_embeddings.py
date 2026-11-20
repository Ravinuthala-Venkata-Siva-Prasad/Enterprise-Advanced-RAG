import numpy as np
import pytest

from src.embeddings.embedding_evaluator import EmbeddingEvaluator


def test_embedding_dimensions():

    vectors = np.random.rand(10, 384).astype(np.float32)
    report = EmbeddingEvaluator.evaluate(vectors)

    assert report["number_of_embeddings"] == 10
    assert report["embedding_dimension"] == 384


def test_normalized_vectors():

    vectors = np.random.rand(10, 384).astype(np.float32)
    vectors = vectors / np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True
    )

    report = EmbeddingEvaluator.evaluate(vectors)
    assert report["normalized"] is True


def test_similarity():

    vector_a = np.array([1.0, 0.0, 0.0])
    vector_b = np.array([1.0, 0.0, 0.0])

    score = EmbeddingEvaluator.similarity(
        vector_a,
        vector_b
    )

    assert score == pytest.approx(1.0)


def test_invalid_dimensions():

    with pytest.raises(ValueError):
        EmbeddingEvaluator.evaluate(
            np.array([1.0, 2.0, 3.0])
        )


def test_non_finite_vectors():

    vectors = np.array([
        [1.0, 0.0],
        [float("nan"), 1.0]
    ])

    report = EmbeddingEvaluator.evaluate(vectors)

    assert report["all_finite"] is False