
import faiss
import numpy as np


class IndexingService:

    def __init__(
        self,
        faiss_store,
        sparse_retriever,
        vectorstore_path,
        hybrid_retriever=None
    ):
        self.faiss_store = faiss_store
        self.sparse_retriever = sparse_retriever
        self.vectorstore_path = vectorstore_path
        self.hybrid_retriever = hybrid_retriever

    # --------------------------------------------------
    # HELPERS
    # --------------------------------------------------

    @staticmethod
    def _chunk_id(document):
        metadata = document.get("metadata", {})
        return metadata.get(
            "chunk_id",
            document.get("chunk_id")
        )

    def _validate_index_alignment(self):

        faiss_ids = {
            self._chunk_id(doc)
            for doc in self.faiss_store.documents
        }

        sparse_ids = {
            self._chunk_id(doc)
            for doc in self.sparse_retriever.documents
        }

        if faiss_ids != sparse_ids:
            raise ValueError(
                "FAISS and BM25 are out of sync."
            )

        if self.faiss_store.index.ntotal != len(
            self.faiss_store.documents
        ):
            raise ValueError(
                "FAISS vector and document counts differ."
            )

        return faiss_ids

    def _refresh_retrievers(self):

        if self.hybrid_retriever is None:
            return None

        # Hybrid refresh updates its dense and sparse retrievers.
        result = self.hybrid_retriever.refresh()

        # Dense refresh may replace the FAISSStore object.
        self.faiss_store = (
            self.hybrid_retriever.dense_retriever.vectorstore
        )

        return result

    def _snapshot(self):

        return {
            "faiss_index": faiss.clone_index(
                self.faiss_store.index
            ),
            "faiss_documents": list(
                self.faiss_store.documents
            ),
            "sparse_documents": list(
                self.sparse_retriever.documents
            )
        }

    def _restore(self, snapshot, original_faiss_store):

        # Restore the original FAISS store.
        original_faiss_store.index = snapshot[
            "faiss_index"
        ]

        original_faiss_store.documents = snapshot[
            "faiss_documents"
        ]

        # Restore BM25 documents and persisted chunks.
        self.sparse_retriever.documents = snapshot[
            "sparse_documents"
        ]

        self.sparse_retriever._save_documents(
            snapshot["sparse_documents"]
        )

        self.sparse_retriever._build_index()

        # Restore persisted FAISS.
        original_faiss_store.save(
            self.vectorstore_path
        )

        self.faiss_store = original_faiss_store

        # Refresh live retrievers from restored indexes.
        if self.hybrid_retriever is not None:
            try:
                self._refresh_retrievers()
            except Exception:
                # Keep the restored original store reference.
                self.faiss_store = original_faiss_store
                raise

    # --------------------------------------------------
    # ADD DOCUMENTS
    # --------------------------------------------------

    def add_documents(self, chunks, embeddings):

        chunks = list(chunks)

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        if not chunks:
            raise ValueError("No chunks to index")

        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2D matrix"
            )

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Chunk and embedding counts differ"
            )

        incoming_ids = [
            self._chunk_id(document)
            for document in chunks
        ]

        if any(not chunk_id for chunk_id in incoming_ids):
            raise ValueError(
                "Every chunk needs a chunk_id"
            )

        if len(incoming_ids) != len(set(incoming_ids)):
            raise ValueError(
                "Duplicate incoming chunk IDs"
            )

        existing_ids = self._validate_index_alignment()

        duplicates = existing_ids.intersection(
            incoming_ids
        )

        if duplicates:
            raise ValueError(
                f"Chunks already indexed: {list(duplicates)[:5]}"
            )

        if embeddings.shape[1] != self.faiss_store.dimension:
            raise ValueError(
                "Embedding dimension mismatch"
            )

        if not np.isfinite(embeddings).all():
            raise ValueError(
                "Embedding values must be finite"
            )

        norms = np.linalg.norm(
            embeddings,
            axis=1
        )

        if not np.allclose(
            norms,
            1.0,
            atol=1e-3
        ):
            raise ValueError(
                "Embeddings must be normalized"
            )

        original_faiss_store = self.faiss_store
        snapshot = self._snapshot()

        try:

            # Add to BM25 and persist chunks.
            self.sparse_retriever.add_documents(
                chunks,
                persist=True
            )

            # Add to FAISS.
            original_faiss_store.add_documents(
                chunks,
                embeddings
            )

            # Persist FAISS.
            original_faiss_store.save(
                self.vectorstore_path
            )

            # Refresh live retrievers once.
            refresh_result = self._refresh_retrievers()

        except Exception:

            self._restore(
                snapshot,
                original_faiss_store
            )

            raise

        return {
            "status": "success",
            "added_chunks": len(chunks),
            "faiss_vectors": self.faiss_store.index.ntotal,
            "bm25_documents": len(
                self.sparse_retriever.documents
            ),
            "retrievers_refreshed": (
                self.hybrid_retriever is not None
            ),
            "refresh_result": refresh_result
        }

    # --------------------------------------------------
    # REMOVE DOCUMENT BY SOURCE
    # --------------------------------------------------

    def remove_documents_by_source(self, source):

        if not isinstance(source, str) or not source.strip():
            raise ValueError(
                "A valid document source is required"
            )

        self._validate_index_alignment()

        # Check whether source exists before changing indexes.
        faiss_matches = sum(
            1
            for document in self.faiss_store.documents
            if self.faiss_store._get_source(document) == source
        )

        sparse_matches = sum(
            1
            for document in self.sparse_retriever.documents
            if self.sparse_retriever._get_source(document) == source
        )

        if faiss_matches != sparse_matches:
            raise ValueError(
                "FAISS and BM25 source counts differ. "
                "Deletion cancelled."
            )

        if faiss_matches == 0:
            return {
                "status": "not_found",
                "source": source,
                "removed_chunks": 0
            }

        original_faiss_store = self.faiss_store
        snapshot = self._snapshot()

        try:

            # Remove from FAISS.
            faiss_removed = (
                original_faiss_store.remove_by_source(source)
            )

            # Remove from BM25 and persist.
            sparse_removed = (
                self.sparse_retriever.remove_by_source(
                    source,
                    persist=True
                )
            )

            if faiss_removed != sparse_removed:
                raise ValueError(
                    "FAISS and BM25 removed different "
                    "numbers of chunks."
                )

            # Persist updated FAISS.
            original_faiss_store.save(
                self.vectorstore_path
            )

            # Refresh live retrievers once.
            refresh_result = self._refresh_retrievers()

        except Exception:

            self._restore(
                snapshot,
                original_faiss_store
            )

            raise

        return {
            "status": "success",
            "source": source,
            "removed_chunks": faiss_removed,
            "faiss_vectors": self.faiss_store.index.ntotal,
            "bm25_documents": len(
                self.sparse_retriever.documents
            ),
            "retrievers_refreshed": (
                self.hybrid_retriever is not None
            ),
            "refresh_result": refresh_result
        }
