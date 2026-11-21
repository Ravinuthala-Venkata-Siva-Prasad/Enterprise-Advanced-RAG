from src.pipeline.conversational_rag import ConversationalRAG


class ApplicationServices:
    def __init__(self, conversational_rag: ConversationalRAG,
                 input_guard=None, document_manager=None):
        self.conversational_rag = conversational_rag
        self.input_guard = input_guard
        self.document_manager = document_manager

    def validate_question(self, question: str):
        if self.input_guard is None:
            return
        result = self.input_guard.validate(question)
        if result is False:
            raise ValueError("Question rejected by input guardrail.")

    def upload_document(self, filename: str, file_bytes: bytes):
        if self.document_manager is None:
            raise RuntimeError("Document management service is not configured.")
        return self.document_manager.upload_bytes(filename, file_bytes)

    def document_stats(self):
        manager = self._require_document_manager()
        indexer = manager.indexing_service
        registry_stats = manager.get_registry_stats()
        return {
            "faiss_vectors": int(indexer.faiss_store.index.ntotal),
            "bm25_documents": len(indexer.sparse_retriever.documents),
            "uploaded_files": registry_stats["total_documents"],
            "indexed_documents": registry_stats["indexed_documents"],
            "registered_chunks": registry_stats["total_chunks"],
        }

    def _require_document_manager(self):
        if self.document_manager is None:
            raise RuntimeError("Document management service is not configured.")
        return self.document_manager

    def document_history(self):
        return self._require_document_manager().get_document_history()

    def get_document(self, document_id):
        return self._require_document_manager().get_document(document_id)

    def delete_document(self, document_id):
        return self._require_document_manager().delete_document(document_id)
