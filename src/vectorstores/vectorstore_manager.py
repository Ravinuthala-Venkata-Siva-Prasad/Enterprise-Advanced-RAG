import json
from pathlib import Path

import numpy as np

from src.vectorstores.faiss_store import FAISSStore


class VectorStoreManager:

    def build(
        self,
        embeddings_path,
        chunks_path,
        output_dir
    ):

        embeddings_path = Path(embeddings_path)
        chunks_path = Path(chunks_path)

        if not embeddings_path.exists():
            raise FileNotFoundError(embeddings_path)

        if not chunks_path.exists():
            raise FileNotFoundError(chunks_path)

        embeddings = np.load(
            embeddings_path,
            allow_pickle=False
        )

        with chunks_path.open("r", encoding="utf-8") as file:
            chunks = [
                json.loads(line)
                for line in file
                if line.strip()
            ]

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks does not match number of embeddings"
            )

        if len(chunks) == 0:
            raise ValueError("No chunks available to index")

        dimension = embeddings.shape[1]

        store = FAISSStore(dimension=dimension)

        store.add_documents(chunks, embeddings)

        store.save(output_dir)

        return store