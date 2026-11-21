
from pathlib import Path

import numpy as np


class IngestionPipeline:

    def __init__(
        self,
        load_documents,
        chunk_manager,
        embedding_model
    ):

        if not callable(load_documents):
            raise TypeError(
                "load_documents must be callable"
            )

        self.load_documents = load_documents
        self.chunk_manager = chunk_manager
        self.embedding_model = embedding_model

    def process_file(self, file_path):

        file_path = Path(file_path).resolve()

        if not file_path.is_file():
            raise FileNotFoundError(file_path)

        # 1. Load the document using your existing loader.
        documents = self.load_documents(str(file_path))

        if not documents:
            raise ValueError(
                "No documents extracted from uploaded file"
            )

        # 2. Add source information to document metadata.
        for document in documents:

            document.metadata = dict(
                document.metadata or {}
            )

            document.metadata["source"] = str(file_path)

        # 3. Chunk using your existing ChunkManager.
        langchain_chunks = self.chunk_manager.process(
            documents
        )

        if not langchain_chunks:
            raise ValueError(
                "No chunks generated from uploaded document"
            )

        # 4. Convert LangChain Documents into FAISS/BM25 format.
        chunks = []

        for chunk in langchain_chunks:

            metadata = dict(chunk.metadata)

            metadata["source"] = str(file_path)

            chunk_id = metadata.get("chunk_id")

            if not chunk_id:
                raise ValueError(
                    "ChunkManager did not generate a chunk_id"
                )

            chunks.append({
                "page_content": chunk.page_content,
                "metadata": metadata
            })

        # 5. Validate unique chunk IDs.
        chunk_ids = [
            chunk["metadata"]["chunk_id"]
            for chunk in chunks
        ]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError(
                "Duplicate chunk IDs generated"
            )

        # 6. Generate normalized embeddings.
        texts = [
            chunk["page_content"]
            for chunk in chunks
        ]

        embeddings = np.asarray(
            self.embedding_model.embed_documents(texts),
            dtype=np.float32
        )

        # 7. Validate embeddings.
        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2D matrix"
            )

        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                "Embedding count does not match chunk count"
            )

        if embeddings.shape[1] != self.embedding_model.dimension:
            raise ValueError(
                "Embedding dimension mismatch"
            )

        if not np.isfinite(embeddings).all():
            raise ValueError(
                "Embeddings contain invalid values"
            )

        norms = np.linalg.norm(
            embeddings,
            axis=1
        )

        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError(
                "Expected normalized embeddings"
            )

        return {
            "source": str(file_path),
            "chunks": chunks,
            "embeddings": embeddings,
            "total_chunks": len(chunks)
        }
