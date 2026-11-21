from fastapi.testclient import TestClient

from api.dependencies import ApplicationServices
from api.main import create_app


class FakeRAG:
    def ask(self, question, session_id):
        return {"answer": "mock", "sources": []}

    def clear_session(self, session_id):
        return None


class FakeValidator:
    max_file_size_bytes = 1024


class FakeDocumentManager:
    validator = FakeValidator()

    def __init__(self):
        self.docs = [{
            "document_id": "doc-1", "filename": "a.txt",
            "stored_filename": "a.txt", "source": "data/uploads/a.txt",
            "file_hash": "hash", "file_size": 3,
            "total_chunks": 1, "status": "indexed",
            "uploaded_at": "2026-01-01T00:00:00+00:00"
        }]

    def get_document_history(self):
        return list(self.docs)

    def get_document(self, document_id):
        return next((d for d in self.docs if d["document_id"] == document_id), None)

    def delete_document(self, document_id):
        item = self.get_document(document_id)
        if item is None:
            return None
        self.docs.remove(item)
        return {"status": "success", "document_id": document_id, "filename": item["filename"], "removed_chunks": 1, "faiss_vectors": 0, "bm25_documents": 0}

    def get_registry_stats(self):
        return {"total_documents": len(self.docs), "indexed_documents": len(self.docs), "total_chunks": len(self.docs)}

    class Indexer:
        class Store:
            class Index:
                ntotal = 0
            index = Index()
        faiss_store = Store()
        class Sparse:
            documents = []
        sparse_retriever = Sparse()
    indexing_service = Indexer()


def make_client():
    manager = FakeDocumentManager()
    services = ApplicationServices(FakeRAG(), document_manager=manager)
    return TestClient(create_app(services)), manager


def test_document_list_and_detail():
    client, _ = make_client()
    assert client.get("/api/v1/documents").status_code == 200
    assert client.get("/api/v1/documents/doc-1").json()["filename"] == "a.txt"


def test_document_not_found():
    client, _ = make_client()
    assert client.get("/api/v1/documents/missing").status_code == 404


def test_delete_document():
    client, manager = make_client()
    response = client.delete("/api/v1/documents/doc-1")
    assert response.status_code == 200
    assert response.json()["removed_chunks"] == 1
    assert manager.docs == []


def test_delete_missing_document():
    client, _ = make_client()
    assert client.delete("/api/v1/documents/missing").status_code == 404
