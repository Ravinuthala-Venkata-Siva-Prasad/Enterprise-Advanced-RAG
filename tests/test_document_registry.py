
import pytest

from src.document_management.document_registry import (
    DocumentRegistry
)


@pytest.fixture
def registry(tmp_path):

    return DocumentRegistry(
        registry_path=tmp_path / "registry.json"
    )


def sample_document():

    return {
        "document_id": "DOC-001",
        "filename": "company_policy.txt",
        "stored_filename": "abc123.txt",
        "source": "data/uploads/abc123.txt",
        "file_hash": "abc123",
        "file_size": 1024,
        "total_chunks": 5,
        "status": "indexed"
    }


def test_add_document(registry):

    result = registry.add_document(
        sample_document()
    )

    assert result["document_id"] == "DOC-001"
    assert result["status"] == "indexed"


def test_get_document(registry):

    registry.add_document(
        sample_document()
    )

    result = registry.get_document("DOC-001")

    assert result is not None
    assert result["filename"] == "company_policy.txt"


def test_get_all_documents(registry):

    registry.add_document(
        sample_document()
    )

    assert len(registry.get_all_documents()) == 1


def test_duplicate_document_id_rejected(registry):

    registry.add_document(
        sample_document()
    )

    with pytest.raises(ValueError):

        registry.add_document(
            sample_document()
        )


def test_remove_document(registry):

    registry.add_document(
        sample_document()
    )

    result = registry.remove_document("DOC-001")

    assert result["document_id"] == "DOC-001"
    assert registry.get_all_documents() == []


def test_registry_persistence(tmp_path):

    path = tmp_path / "registry.json"

    first = DocumentRegistry(path)

    first.add_document(
        sample_document()
    )

    second = DocumentRegistry(path)

    assert len(second.get_all_documents()) == 1


def test_registry_statistics(registry):

    registry.add_document(
        sample_document()
    )

    stats = registry.get_stats()

    assert stats["total_documents"] == 1
    assert stats["indexed_documents"] == 1
    assert stats["total_chunks"] == 5
