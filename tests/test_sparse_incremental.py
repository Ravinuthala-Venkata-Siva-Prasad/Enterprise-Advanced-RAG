
import json

import pytest

from src.retrieval.sparse_retriever import SparseRetriever


def make_document(chunk_id, source, content):

    return {
        "page_content": content,
        "metadata": {
            "chunk_id": chunk_id,
            "source": source,
            "file_type": "txt"
        }
    }


@pytest.fixture
def retriever(tmp_path):

    chunks_path = tmp_path / "chunks.jsonl"

    documents = [
        make_document(
            "chunk-1",
            "policy-a.txt",
            "Multi factor authentication is required."
        ),
        make_document(
            "chunk-2",
            "policy-b.txt",
            "Employees must use a company VPN."
        )
    ]

    with chunks_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        for document in documents:
            file.write(
                json.dumps(document) + "\n"
            )

    return SparseRetriever(
        chunks_path=str(chunks_path)
    )


def test_initial_index(retriever):

    assert len(retriever.documents) == 2
    assert retriever.bm25 is not None


def test_incremental_addition(retriever):

    document = make_document(
        "chunk-3",
        "policy-c.txt",
        "Cloud security requires encryption."
    )

    added = retriever.add_documents([document])

    assert added == 1
    assert len(retriever.documents) == 3


def test_duplicate_chunk_rejected(retriever):

    duplicate = make_document(
        "chunk-1",
        "another-policy.txt",
        "Different document content."
    )

    with pytest.raises(
        ValueError,
        match="already exist"
    ):
        retriever.add_documents([duplicate])

    assert len(retriever.documents) == 2


def test_remove_source(retriever):

    removed = retriever.remove_by_source(
        "policy-a.txt"
    )

    assert removed == 1
    assert len(retriever.documents) == 1

    assert all(
        document["metadata"]["source"] != "policy-a.txt"
        for document in retriever.documents
    )


def test_persistence_and_refresh(retriever):

    document = make_document(
        "chunk-3",
        "policy-c.txt",
        "Cloud security requires encryption."
    )

    retriever.add_documents([document])

    new_retriever = SparseRetriever(
        chunks_path=str(retriever.chunks_path)
    )

    assert len(new_retriever.documents) == 3

    refreshed_count = retriever.refresh()

    assert refreshed_count == 3


def test_retrieve_results(retriever):

    results = retriever.retrieve(
        "multi factor authentication",
        top_k=2
    )

    assert len(results) > 0
    assert results[0]["retrieval_method"] == "sparse"


def test_empty_corpus_after_removal(retriever):

    retriever.remove_by_source("policy-a.txt")
    retriever.remove_by_source("policy-b.txt")

    assert len(retriever.documents) == 0
    assert retriever.bm25 is None

    assert retriever.retrieve(
        "authentication",
        top_k=5
    ) == []
