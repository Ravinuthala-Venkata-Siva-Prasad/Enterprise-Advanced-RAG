import pytest
from langchain_core.documents import Document
from src.parsing.chunking import RecursiveChunker
from src.parsing.chunk_manager import ChunkManager
from src.parsing.text_cleaner import TextCleaner


def test_text_cleaner():
    text = "Hello   World\r\n\r\n\r\nPython"
    cleaned = TextCleaner.clean(text)
    assert cleaned == "Hello World\n\nPython"


def test_empty_text():
    assert TextCleaner.clean("") == ""

def test_chunking():

    documents = [Document(page_content="This is a test sentence. " * 100,metadata={"source": "test.txt"})]

    chunker = RecursiveChunker(
        chunk_size=200,
        chunk_overlap=30
    )

    chunks = chunker.split_documents(documents)
    assert len(chunks) > 1

    assert all(
        len(chunk.page_content) <= 200
        for chunk in chunks
    )


def test_metadata_preservation():

    document = Document(page_content="Enterprise RAG metadata testing. " * 20,
        metadata={
            "source": "policy.pdf",
            "page_number": 1
        }
    )

    manager = ChunkManager(
        chunk_size=100,
        chunk_overlap=20
    )

    chunks = manager.process([document])

    assert len(chunks) > 0

    for chunk in chunks:
        assert chunk.metadata["source"] == "policy.pdf"
        assert chunk.metadata["page_number"] == 1
        assert "chunk_id" in chunk.metadata


def test_invalid_chunk_size():

    with pytest.raises(ValueError):
        RecursiveChunker(
            chunk_size=100,
            chunk_overlap=100
        )


def test_empty_document_is_skipped():

    document = Document(
        page_content="   ",
        metadata={"source": "empty.txt"}
    )

    chunker = RecursiveChunker()

    chunks = chunker.split_documents([document])

    assert chunks == []