import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock


class DocumentRegistry:
    """Small JSON-backed registry for uploaded and indexed documents."""

    REQUIRED_FIELDS = (
        "document_id", "filename", "stored_filename", "source",
        "file_hash", "file_size", "total_chunks", "status"
    )

    def __init__(self, registry_path="data/document_registry.json"):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()
        if not self.registry_path.exists():
            self._write_registry([])
        self.documents = self._load_registry()

    def _load_registry(self):
        try:
            with self.registry_path.open("r", encoding="utf-8") as file:
                documents = json.load(file)
        except json.JSONDecodeError as exc:
            raise ValueError("Document registry contains invalid JSON") from exc
        if not isinstance(documents, list):
            raise ValueError("Document registry must contain a JSON list")
        for item in documents:
            if not isinstance(item, dict) or any(k not in item for k in self.REQUIRED_FIELDS):
                raise ValueError("Document registry contains an invalid record")
        return documents

    def _write_registry(self, documents):
        temp = self.registry_path.with_suffix(self.registry_path.suffix + ".tmp")
        try:
            with temp.open("w", encoding="utf-8") as file:
                json.dump(documents, file, indent=2, ensure_ascii=False)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temp, self.registry_path)
        finally:
            if temp.exists():
                temp.unlink()

    def add_document(self, document):
        missing = [field for field in self.REQUIRED_FIELDS if field not in document]
        if missing:
            raise ValueError(f"Missing registry fields: {', '.join(missing)}")
        with self.lock:
            if any(d["document_id"] == document["document_id"] for d in self.documents):
                raise ValueError("Document ID already exists")
            if any(d["file_hash"] == document["file_hash"] for d in self.documents):
                raise ValueError("Document hash already exists")
            record = dict(document)
            record.setdefault("uploaded_at", datetime.now(timezone.utc).isoformat())
            updated = self.documents + [record]
            self._write_registry(updated)
            self.documents = updated
            return dict(record)

    def get_all_documents(self):
        with self.lock:
            return [dict(d) for d in self.documents]

    def get_document(self, document_id):
        with self.lock:
            item = next((d for d in self.documents if d["document_id"] == document_id), None)
            return dict(item) if item else None

    def get_document_by_source(self, source):
        with self.lock:
            item = next((d for d in self.documents if d["source"] == source), None)
            return dict(item) if item else None

    def get_document_by_hash(self, file_hash):
        with self.lock:
            item = next((d for d in self.documents if d["file_hash"] == file_hash), None)
            return dict(item) if item else None

    def remove_document(self, document_id):
        with self.lock:
            removed = next((d for d in self.documents if d["document_id"] == document_id), None)
            if removed is None:
                return None
            updated = [d for d in self.documents if d["document_id"] != document_id]
            self._write_registry(updated)
            self.documents = updated
            return dict(removed)

    def get_stats(self):
        with self.lock:
            return {
                "total_documents": len(self.documents),
                "indexed_documents": sum(d.get("status") == "indexed" for d in self.documents),
                "total_chunks": sum(int(d.get("total_chunks", 0)) for d in self.documents),
            }
