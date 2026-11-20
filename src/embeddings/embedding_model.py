from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel:

    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.model_name = model_name

        self.model = SentenceTransformer(model_name)

        self.dimension = self.model.get_embedding_dimension()

        if self.dimension is None:
            raise ValueError(
                "Unable to determine embedding dimension"
            )

    def embed_documents(self, texts, batch_size=32):

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if not texts:
            return np.empty(
                (0, self.dimension),
                dtype=np.float32
            )

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return np.asarray(
            embeddings,
            dtype=np.float32
        )

    def embed_query(self, query):

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query cannot be empty")

        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return np.asarray(
            embedding,
            dtype=np.float32
        )