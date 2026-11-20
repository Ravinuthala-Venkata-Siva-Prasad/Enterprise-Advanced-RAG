import numpy as np
import pytest

from src.vectorstores.faiss_store import FAISSStore


@pytest.fixture
def sample_store():

    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ],
        dtype=np.float32
    )

    documents = [
        {"page_content": "Security policy", "metadata": {"id": "1"}},
        {"page_content": "Leave policy", "metadata": {"id": "2"}},
        {"page_content": "Expense policy", "metadata": {"id": "3"}}
    ]

    store = FAISSStore(dimension=3)
    store.add_documents(documents, vectors)

    return store


def test_index_count(sample_store):
    assert sample_store.index.ntotal == 3


def test_similarity_search(sample_store):

    results = sample_store.search(
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        top_k=2
    )

    assert len(results) == 2
    assert results[0]["document"]["metadata"]["id"] == "1"
    assert results[0]["score"] == pytest.approx(1.0)


def test_top_k_larger_than_index(sample_store):

    results = sample_store.search(
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        top_k=10
    )

    assert len(results) == 3


def test_invalid_dimension(sample_store):

    with pytest.raises(ValueError):
        sample_store.search(
            np.array([1.0, 0.0], dtype=np.float32),
            top_k=2
        )


def test_save_and_load(sample_store, tmp_path):

    sample_store.save(tmp_path)

    loaded_store = FAISSStore.load(tmp_path)

    assert loaded_store.index.ntotal == 3
    assert loaded_store.dimension == 3

    results = loaded_store.search(
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        top_k=1
    )

    assert results[0]["document"]["page_content"] == "Security policy"


def test_reject_unnormalized_embeddings():

    store = FAISSStore(dimension=3)

    with pytest.raises(ValueError):
        store.add_documents(
            [{"page_content": "test", "metadata": {}}],
            np.array([[2.0, 0.0, 0.0]], dtype=np.float32)
        )