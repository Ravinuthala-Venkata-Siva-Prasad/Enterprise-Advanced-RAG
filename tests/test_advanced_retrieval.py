import pytest

from src.retrieval.hybrid_retriever import HybridRetriever


class MockRetriever:

    def __init__(self, results):
        self.results = results

    def retrieve(self, query, top_k=10):
        return self.results[:top_k]


def make_result(chunk_id, method):

    return {
        "document": {
            "page_content": f"Document {chunk_id}",
            "metadata": {"chunk_id": chunk_id}
        },
        "score": 0.8,
        "retrieval_method": method
    }


def test_hybrid_retrieval():

    dense = MockRetriever([
        make_result("A", "dense"),
        make_result("B", "dense")
    ])

    sparse = MockRetriever([
        make_result("B", "sparse"),
        make_result("C", "sparse")
    ])

    hybrid = HybridRetriever(
        dense,
        sparse,
        dense_weight=0.5,
        sparse_weight=0.5
    )

    results = hybrid.retrieve("security policy", top_k=3)

    assert len(results) == 3

    # B appears in both rankings and should receive a combined score.
    assert results[0]["document"]["metadata"]["chunk_id"] == "B"


def test_invalid_weights():

    with pytest.raises(ValueError):
        HybridRetriever(
            MockRetriever([]),
            MockRetriever([]),
            dense_weight=0,
            sparse_weight=0
        )


def test_empty_results():

    hybrid = HybridRetriever(
        MockRetriever([]),
        MockRetriever([])
    )

    assert hybrid.retrieve("test") == []


def test_duplicate_documents_are_fused():

    dense = MockRetriever([
        make_result("A", "dense")
    ])

    sparse = MockRetriever([
        make_result("A", "sparse")
    ])

    hybrid = HybridRetriever(dense, sparse)

    results = hybrid.retrieve("test")

    assert len(results) == 1

    assert set(
        results[0]["retrieval_methods"]
    ) == {"dense", "sparse"}