import hashlib
from pathlib import Path

from src.document_management.document_registry import DocumentRegistry
from src.document_management.document_validator import DocumentValidator


class DocumentManager:
    def __init__(self, ingestion_pipeline, indexing_service,
                 upload_directory="data/uploads", max_file_size_mb=20,
                 registry_path="data/document_registry.json"):
        self.ingestion_pipeline = ingestion_pipeline
        self.indexing_service = indexing_service
        self.upload_directory = Path(upload_directory).resolve()
        self.upload_directory.mkdir(parents=True, exist_ok=True)
        self.validator = DocumentValidator(max_file_size_mb=max_file_size_mb)
        self.registry = DocumentRegistry(registry_path=registry_path)

    def upload_bytes(self, filename, file_bytes):
        if not filename or not isinstance(filename, str):
            raise ValueError("A valid filename is required")
        if not isinstance(file_bytes, bytes):
            raise TypeError("file_bytes must be bytes")
        if not file_bytes:
            raise ValueError("Uploaded file is empty")
        if len(file_bytes) > self.validator.max_file_size_bytes:
            raise ValueError("Uploaded file exceeds the maximum size")

        safe_filename = Path(filename).name
        extension = Path(safe_filename).suffix.lower()
        if extension not in self.validator.ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file extension: {extension}")

        file_hash = hashlib.sha256(file_bytes).hexdigest()
        document_id = file_hash[:16]
        stored_filename = file_hash + extension
        destination = (self.upload_directory / stored_filename).resolve()
        source = str(destination)

        if self.registry.get_document_by_hash(file_hash):
            raise ValueError("This exact file has already been uploaded")
        if destination.exists():
            raise ValueError("This file already exists in uploads but is not registered. Run the upload registry migration or reconcile it before retrying.")

        temporary_path = destination.with_name(destination.name + ".tmp")
        try:
            with temporary_path.open("xb") as file:
                file.write(file_bytes)
            temporary_path.replace(destination)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

        indexed = False
        registered = False
        try:
            self.validator.validate(destination)
            processed = self.ingestion_pipeline.process_file(destination)
            chunks = processed.get("chunks", [])
            embeddings = processed.get("embeddings")
            if not chunks:
                raise ValueError("Document produced no chunks")
            indexing_result = self.indexing_service.add_documents(chunks, embeddings)
            indexed = True

            record = self.registry.add_document({
                "document_id": document_id,
                "filename": safe_filename,
                "stored_filename": stored_filename,
                "source": source,
                "file_hash": file_hash,
                "file_size": len(file_bytes),
                "total_chunks": len(chunks),
                "status": "indexed",
            })
            registered = True
            return {
                "status": "success",
                "document_id": document_id,
                "filename": safe_filename,
                "stored_filename": stored_filename,
                "source": source,
                "file_hash": file_hash,
                "file_size": len(file_bytes),
                "total_chunks": len(chunks),
                "uploaded_at": record["uploaded_at"],
                "indexing": indexing_result,
            }
        except Exception:
            # Compensating rollback; this is not crash-atomic across disk/index/registry.
            if registered:
                self.registry.remove_document(document_id)
            if indexed:
                try:
                    self.indexing_service.remove_documents_by_source(source)
                except Exception as rollback_error:
                    raise RuntimeError(
                        "Upload failed and index rollback also failed: "
                        f"{rollback_error}. Reconcile indexes before further writes."
                    )
            if destination.exists():
                destination.unlink()
            raise

    def get_document_history(self):
        return self.registry.get_all_documents()

    def get_document(self, document_id):
        return self.registry.get_document(document_id)

    def get_registry_stats(self):
        return self.registry.get_stats()

    def delete_document(self, document_id):
        record = self.registry.get_document(document_id)
        if record is None:
            return None

        source_path = Path(record["source"]).resolve()
        # Never allow a registry record to delete files outside upload_directory.
        try:
            source_path.relative_to(self.upload_directory)
        except ValueError as exc:
            raise ValueError("Registered source is outside the upload directory; deletion refused") from exc

        if source_path.name != record["stored_filename"]:
            raise ValueError("Stored filename does not match registry record; deletion refused")

        # Remove indexed chunks first. If this fails, leave registry and file intact.
        index_result = self.indexing_service.remove_documents_by_source(str(source_path))
        if index_result.get("status") == "not_found":
            raise RuntimeError("No indexed chunks found for this document. Registry and file were left unchanged for reconciliation.")

        # Remove registry record only after both indexes have been updated.
        removed_record = self.registry.remove_document(document_id)
        if removed_record is None:
            raise RuntimeError("Index was updated, but registry record disappeared. Reconciliation is required.")

        # Physical file removal is last; report a clear error if it fails.
        try:
            source_path.unlink(missing_ok=True)
        except OSError as exc:
            raise RuntimeError(
                "Indexes and registry were updated, but the uploaded file could not be deleted: "
                f"{exc}"
            ) from exc

        return {
            "status": "success",
            "document_id": document_id,
            "filename": removed_record["filename"],
            "removed_chunks": index_result["removed_chunks"],
            "faiss_vectors": index_result["faiss_vectors"],
            "bm25_documents": index_result["bm25_documents"],
        }
