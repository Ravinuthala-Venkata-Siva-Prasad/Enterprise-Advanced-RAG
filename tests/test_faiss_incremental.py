
import json

import numpy as np
import pytest

from src.vectorstores.faiss_store import FAISSStore


def normalized(vectors):

    vectors = np.asarray(
        vectors,
        dtype=np.float32
    )

    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True
    )

    return vectors / norms


def make_document(chunk_id, source, content):

    return {
        "page_content": content,
        "metadata": {
            "chunk_id": chunk_id,
            "source": source,
            "file_type": "txt"
        }
    }


def test_add_documents():

    store = FAISSStore(dimension=3)

    documents = [
        make_document(
            "chunk-1",
            "policy.txt",
            "Multi-factor authentication is required."
        ),
        make_document(
            "chunk-2",
            "policy.txt",
            "Employees must use a secure password."
        )
    ]

    embeddings = normalized([
        [1, 0, 0],
        [0, 1, 0]
    ])

    added = store.add_documents(
        documents,
        embeddings
    )

    assert added == 2
    assert store.index.ntotal == 2
    assert len(store.documents) == 2


def test_add_incremental_documents():

    store = FAISSStore(dimension=3)

    store.add_documents(
        [
            make_document(
                "chunk-1",
                "a.txt",
                "First document content."
            )
        ],
        normalized([[1, 0, 0]])
    )

    store.add_documents(
        [
            make_document(
                "chunk-2",
                "b.txt",
                "Second document content."
            )
        ],
        normalized([[0, 1, 0]])
    )

    assert store.index.ntotal == 2
    assert len(store.documents) == 2


def test_reject_duplicate_chunk_id():

    store = FAISSStore(dimension=3)

    store.add_documents(
        [
            make_document(
                "chunk-1",
                "a.txt",
                "Original document content."
            )
        ],
        normalized([[1, 0, 0]])
    )

    duplicate_document = make_document(
        "chunk-1",
        "another.txt",
        "Different document content."
    )

    with pytest.raises(
        ValueError,
        match="already exist"
    ):

        store.add_documents(
            [duplicate_document],
            normalized([[0, 1, 0]])
        )

    assert store.index.ntotal == 1


def test_save_and_reload(tmp_path):

    store = FAISSStore(dimension=3)

    documents = [
        make_document(
            "chunk-1",
            "a.txt",
            "First document."
        ),
        make_document(
            "chunk-2",
            "b.txt",
            "Second document."
        )
    ]

    store.add_documents(
        documents,
        normalized([
            [1, 0, 0],
            [0, 1, 0]
        ])
    )

    store.save(tmp_path)

    loaded_store = FAISSStore.load(tmp_path)

    assert loaded_store.index.ntotal == 2
    assert len(loaded_store.documents) == 2
    assert loaded_store.dimension == 3

    assert (
        loaded_store.documents[0]["metadata"]["chunk_id"]
        == "chunk-1"
    )


def test_remove_by_source():

    store = FAISSStore(dimension=3)

    documents = [
        make_document(
            "chunk-1",
            "a.txt",
            "First document."
        ),
        make_document(
            "chunk-2",
            "a.txt",
            "Second document."
        ),
        make_document(
            "chunk-3",
            "b.txt",
            "Third document."
        )
    ]

    store.add_documents(
        documents,
        normalized([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
    )

    removed = store.remove_by_source("a.txt")

    assert removed == 2
    assert store.index.ntotal == 1
    assert len(store.documents) == 1

    assert (
        store.documents[0]["metadata"]["source"]
        == "b.txt"
    )

    assert store.remove_by_source("missing.txt") == 0
