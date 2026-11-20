from src.embeddings.embedding_model import EmbeddingModel
from src.vectorstores.faiss_store import FAISSStore


class DenseRetriever:

    def __init__(
        self,
        vectorstore_path="data/vectorstore",
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ):

        self.embedding_model = EmbeddingModel(model_name)

        self.vectorstore = FAISSStore.load(vectorstore_path)

        if self.embedding_model.dimension != self.vectorstore.dimension:
            raise ValueError("Embedding dimensions do not match")

    def retrieve(self, query, top_k=10):

        query_vector = self.embedding_model.embed_query(query)

        results = self.vectorstore.search(
            query_embedding=query_vector,
            top_k=top_k
        )

        return [
            {
                "document": result["document"],
                "score": result["score"],
                "retrieval_method": "dense"
            }
            for result in results
        ]