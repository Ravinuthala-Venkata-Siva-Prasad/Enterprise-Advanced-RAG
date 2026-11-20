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

    def add_documents(self, documents, embeddings):

        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D matrix")

        if embeddings.shape[1] != self.dimension:
            raise ValueError("Embedding dimension mismatch")

        if len(documents) != len(embeddings):
            raise ValueError("Document and embedding counts differ")

        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings contain NaN or infinite values")

        if len(embeddings) == 0:
            raise ValueError("No embeddings provided")

        norms = np.linalg.norm(embeddings, axis=1)

        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError(
                "Expected normalized embeddings for cosine similarity"
            )

        self.index.add(np.ascontiguousarray(embeddings))
        self.documents.extend(documents)

    def search(self, query_embedding, top_k=5):

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if self.index.ntotal == 0:
            return []

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        ).reshape(1, -1)

        if query_embedding.shape[1] != self.dimension:
            raise ValueError("Query embedding dimension mismatch")

        if not np.isfinite(query_embedding).all():
            raise ValueError("Query embedding contains invalid values")

        norm = np.linalg.norm(query_embedding)

        if not np.isclose(norm, 1.0, atol=1e-3):
            raise ValueError("Query embedding must be normalized")

        actual_k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(
            np.ascontiguousarray(query_embedding),
            actual_k
        )

        results = []

        for score, index_id in zip(scores[0], indices[0]):

            if index_id < 0:
                continue

            results.append({
                "score": float(score),
                "document": self.documents[int(index_id)],
                "index": int(index_id)
            })

        return results

    def save(self, directory):

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(
            self.index,
            str(directory / "index.faiss")
        )

        with (directory / "documents.jsonl").open(
            "w",
            encoding="utf-8"
        ) as file:

            for document in self.documents:
                file.write(
                    json.dumps(document, ensure_ascii=False) + "\n"
                )

        metadata = {
            "dimension": self.dimension,
            "total_vectors": self.index.ntotal,
            "index_type": "IndexFlatIP",
            "similarity_metric": "cosine_similarity",
            "normalized_embeddings": True
        }

        with (directory / "metadata.json").open(
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(metadata, file, indent=4)

    @classmethod
    def load(cls, directory):

        directory = Path(directory)

        index_path = directory / "index.faiss"
        documents_path = directory / "documents.jsonl"
        metadata_path = directory / "metadata.json"

        for path in [index_path, documents_path, metadata_path]:
            if not path.exists():
                raise FileNotFoundError(path)

        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        store = cls(dimension=metadata["dimension"])

        store.index = faiss.read_index(str(index_path))

        with documents_path.open("r", encoding="utf-8") as file:
            store.documents = [
                json.loads(line)
                for line in file
                if line.strip()
            ]

        if store.index.ntotal != len(store.documents):
            raise ValueError(
                "FAISS index and document metadata counts do not match"
            )

        if store.index.d != store.dimension:
            raise ValueError("Saved FAISS index dimension mismatch")

        return store