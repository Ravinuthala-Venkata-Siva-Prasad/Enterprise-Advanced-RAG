"""Register legacy files in data/uploads only when their source chunks exist in both indexes.

Run from the project root after stopping the API:
    python scripts/migrate_upload_registry.py
"""
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.document_management.document_registry import DocumentRegistry
from src.vectorstores.faiss_store import FAISSStore
from src.retrieval.sparse_retriever import SparseRetriever

UPLOAD_DIR = (ROOT / "data/uploads").resolve()
REGISTRY_PATH = ROOT / "data/document_registry.json"
VECTORSTORE_DIR = ROOT / "data/vectorstore"
CHUNKS_PATH = ROOT / "data/embeddings/chunks.jsonl"
ALLOWED = {".pdf", ".docx", ".txt", ".csv", ".html", ".htm"}


def norm(value):
    return str(Path(value).resolve()).casefold()


def main():
    if not UPLOAD_DIR.exists():
        print(f"Upload directory not found: {UPLOAD_DIR}")
        return

    faiss_store = FAISSStore.load(VECTORSTORE_DIR)
    sparse = SparseRetriever(chunks_path=CHUNKS_PATH)

    faiss_sources = {}
    for doc in faiss_store.documents:
        source = (doc.get("metadata") or {}).get("source") or doc.get("source")
        if source:
            faiss_sources.setdefault(norm(source), []).append(doc)

    sparse_sources = {}
    for doc in sparse.documents:
        source = (doc.get("metadata") or {}).get("source") or doc.get("source")
        if source:
            sparse_sources.setdefault(norm(source), []).append(doc)

    registry = DocumentRegistry(REGISTRY_PATH)
    known_hashes = {item["file_hash"] for item in registry.get_all_documents()}
    added = 0

    for path in sorted(UPLOAD_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() not in ALLOWED:
            continue
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest in known_hashes:
            print(f"SKIP already registered: {path.name}")
            continue

        key = norm(path)
        faiss_docs = faiss_sources.get(key, [])
        sparse_docs = sparse_sources.get(key, [])
        faiss_ids = {
            (d.get("metadata") or {}).get("chunk_id") or d.get("chunk_id")
            for d in faiss_docs
        }
        sparse_ids = {
            (d.get("metadata") or {}).get("chunk_id") or d.get("chunk_id")
            for d in sparse_docs
        }
        if not faiss_ids or faiss_ids != sparse_ids:
            print(f"SKIP no matching synchronized FAISS/BM25 chunks: {path.name}")
            continue

        record = {
            "document_id": digest[:16],
            "filename": path.name,
            "stored_filename": path.name,
            "source": str(path.resolve()),
            "file_hash": digest,
            "file_size": len(content),
            "total_chunks": len(faiss_ids),
            "status": "indexed",
            "uploaded_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            "migrated": True,
        }
        registry.add_document(record)
        known_hashes.add(digest)
        added += 1
        print(f"REGISTERED: {path.name} ({len(faiss_ids)} chunks)")

    print(f"\nMigration complete. Added {added} record(s).")
    print(f"Registry: {REGISTRY_PATH}")


if __name__ == "__main__":
    main()
