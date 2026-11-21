
import json
from pathlib import Path

import faiss
import numpy as np


class FAISSStore:

    def __init__(self, dimension: int):

        if dimension <= 0:
            raise ValueError("Embedding dimension must be positive")

        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.documents = []

    # --------------------------------------------------
    # METADATA HELPERS
    # --------------------------------------------------

    @staticmethod
    def _get_metadata(document):

        metadata = document.get("metadata", {})

        if not isinstance(metadata, dict):
            raise ValueError("Document metadata must be a dictionary")

        return metadata

    @classmethod
    def _get_chunk_id(cls, document):

        metadata = cls._get_metadata(document)

        return metadata.get(
            "chunk_id",
            document.get("chunk_id")
        )

    @classmethod
    def _get_source(cls, document):

        metadata = cls._get_metadata(document)

        return metadata.get(
            "source",
            document.get("source")
        )

    # --------------------------------------------------
    # DOCUMENT VALIDATION
    # --------------------------------------------------

    @classmethod
    def _validate_documents(cls, documents):

        for document in documents:

            if not isinstance(document, dict):
                raise ValueError(
                    "Each document must be a dictionary"
                )

            page_content = document.get("page_content")

            if not isinstance(page_content, str) or not page_content.strip():
                raise ValueError(
                    "Document page_content cannot be empty"
                )

            cls._get_metadata(document)

    # --------------------------------------------------
    # ADD DOCUMENTS
    # --------------------------------------------------

    def add_documents(self, documents, embeddings):

        documents = list(documents)

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2D matrix"
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                "Embedding dimension mismatch"
            )

        if len(documents) != len(embeddings):
            raise ValueError(
                "Document and embedding counts differ"
            )

        if len(embeddings) == 0:
            raise ValueError(
                "No embeddings provided"
            )

        if not np.isfinite(embeddings).all():
            raise ValueError(
                "Embeddings contain invalid values"
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
                "Expected normalized embeddings"
            )

        self._validate_documents(documents)

        # ----------------------------------------------
        # DUPLICATE CHUNK VALIDATION
        # ----------------------------------------------

        existing_ids = {
            self._get_chunk_id(document)
            for document in self.documents
            if self._get_chunk_id(document) is not None
        }

        incoming_ids = [
            self._get_chunk_id(document)
            for document in documents
        ]

        # Only enforce uniqueness when chunk IDs exist.
        incoming_ids = [
            chunk_id
            for chunk_id in incoming_ids
            if chunk_id is not None
        ]

        if len(incoming_ids) != len(set(incoming_ids)):
            raise ValueError(
                "Duplicate chunk IDs in incoming documents"
            )

        duplicates = existing_ids.intersection(
            incoming_ids
        )

        if duplicates:
            raise ValueError(
                "Chunk IDs already exist in FAISS: "
                f"{list(duplicates)[:5]}"
            )

        # ----------------------------------------------
        # ADD VECTORS
        # ----------------------------------------------

        self.index.add(
            np.ascontiguousarray(embeddings)
        )

        self.documents.extend(documents)

        return len(documents)

    # --------------------------------------------------
    # REMOVE DOCUMENTS BY SOURCE
    # --------------------------------------------------

    def remove_by_source(self, source: str):

        if not isinstance(source, str) or not source.strip():
            raise ValueError(
                "Source cannot be empty"
            )

        if self.index.ntotal != len(self.documents):
            raise ValueError(
                "FAISS index and document metadata are inconsistent"
            )

        # Find documents to retain.
        keep_indices = [
            i
            for i, document in enumerate(self.documents)
            if self._get_source(document) != source
        ]

        removed_count = (
            len(self.documents) - len(keep_indices)
        )

        if removed_count == 0:
            return 0

        # Reconstruct vectors belonging to retained documents.
        if keep_indices:

            remaining_vectors = np.vstack([
                self.index.reconstruct(i)
                for i in keep_indices
            ]).astype(np.float32)

            remaining_documents = [
                self.documents[i]
                for i in keep_indices
            ]

        else:

            remaining_vectors = np.empty(
                (0, self.dimension),
                dtype=np.float32
            )

            remaining_documents = []

        # Create a new FAISS index.
        new_index = faiss.IndexFlatIP(
            self.dimension
        )

        if len(remaining_vectors) > 0:

            new_index.add(
                np.ascontiguousarray(
                    remaining_vectors
                )
            )

        # Replace old index and metadata.
        self.index = new_index
        self.documents = remaining_documents

        return removed_count

    # --------------------------------------------------
    # SEARCH
    # --------------------------------------------------

    def search(self, query_embedding, top_k=5):

        if top_k <= 0:
            raise ValueError(
                "top_k must be positive"
            )

        if self.index.ntotal != len(self.documents):
            raise ValueError(
                "FAISS index and document metadata are inconsistent"
            )

        if self.index.ntotal == 0:
            return []

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        ).reshape(1, -1)

        if query_embedding.shape[1] != self.dimension:
            raise ValueError(
                "Query embedding dimension mismatch"
            )

        if not np.isfinite(query_embedding).all():
            raise ValueError(
                "Query embedding contains invalid values"
            )

        norm = np.linalg.norm(
            query_embedding
        )

        if not np.isclose(
            norm,
            1.0,
            atol=1e-3
        ):
            raise ValueError(
                "Query embedding must be normalized"
            )

        actual_k = min(
            top_k,
            self.index.ntotal
        )

        scores, indices = self.index.search(
            np.ascontiguousarray(query_embedding),
            actual_k
        )

        results = []

        for score, index_id in zip(
            scores[0],
            indices[0]
        ):

            if index_id < 0:
                continue

            results.append({
                "score": float(score),
                "document": self.documents[int(index_id)],
                "index": int(index_id)
            })

        return results

    # --------------------------------------------------
    # SAVE INDEX
    # --------------------------------------------------

    def save(self, directory):

        directory = Path(directory)

        directory.mkdir(
            parents=True,
            exist_ok=True
        )

        if self.index.ntotal != len(self.documents):
            raise ValueError(
                "FAISS index and document metadata counts differ"
            )

        # Save FAISS index.
        faiss.write_index(
            self.index,
            str(directory / "index.faiss")
        )

        # Save document metadata.
        documents_path = directory / "documents.jsonl"

        with documents_path.open(
            "w",
            encoding="utf-8"
        ) as file:

            for document in self.documents:

                file.write(
                    json.dumps(
                        document,
                        ensure_ascii=False
                    ) + "\n"
                )

        # Save index metadata.
        metadata = {
            "dimension": self.dimension,
            "total_vectors": self.index.ntotal,
            "index_type": "IndexFlatIP",
            "similarity_metric": "cosine_similarity",
            "normalized_embeddings": True
        }

        metadata_path = directory / "metadata.json"

        with metadata_path.open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                metadata,
                file,
                indent=4
            )

    # --------------------------------------------------
    # LOAD INDEX
    # --------------------------------------------------

    @classmethod
    def load(cls, directory):

        directory = Path(directory)

        index_path = directory / "index.faiss"
        documents_path = directory / "documents.jsonl"
        metadata_path = directory / "metadata.json"

        # Check required files.
        for path in [
            index_path,
            documents_path,
            metadata_path
        ]:

            if not path.exists():
                raise FileNotFoundError(path)

        # Load metadata.
        with metadata_path.open(
            "r",
            encoding="utf-8"
        ) as file:

            metadata = json.load(file)

        if metadata.get("index_type") != "IndexFlatIP":
            raise ValueError(
                "Unsupported FAISS index type"
            )

        if metadata.get("normalized_embeddings") is not True:
            raise ValueError(
                "Saved index does not declare normalized embeddings"
            )

        # Initialize store.
        store = cls(
            dimension=metadata["dimension"]
        )

        # Load FAISS index.
        store.index = faiss.read_index(
            str(index_path)
        )

        # Load documents.
        with documents_path.open(
            "r",
            encoding="utf-8"
        ) as file:

            store.documents = [
                json.loads(line)
                for line in file
                if line.strip()
            ]

        # Validate documents.
        store._validate_documents(
            store.documents
        )

        # Validate index consistency.
        if store.index.ntotal != len(store.documents):
            raise ValueError(
                "FAISS index and document metadata counts do not match"
            )

        if store.index.d != store.dimension:
            raise ValueError(
                "Saved FAISS index dimension mismatch"
            )

        if store.index.ntotal != metadata["total_vectors"]:
            raise ValueError(
                "FAISS metadata vector count mismatch"
            )

        return store
