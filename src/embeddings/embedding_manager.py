import json
from pathlib import Path

import numpy as np

from src.embeddings.embedding_model import EmbeddingModel


class EmbeddingManager:

    def __init__(self, model_name=None):

        self.embedding_model = EmbeddingModel(
            model_name=model_name
            or "sentence-transformers/all-MiniLM-L6-v2"
        )

    def load_chunks(self, file_path):

        file_path = Path(file_path)

        if not file_path.is_file():
            raise FileNotFoundError(
                f"Chunk file not found: {file_path}"
            )

        chunks = []

        with file_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):

                if not line.strip():
                    continue

                chunk = json.loads(line)

                if not isinstance(chunk, dict):
                    raise ValueError(
                        f"Invalid chunk at line {line_number}"
                    )

                if not isinstance(chunk.get("page_content"), str):
                    raise ValueError(
                        f"Missing or invalid page_content at line {line_number}"
                    )

                chunks.append(chunk)

        return chunks

    def generate(self, chunks, batch_size=32):

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        texts = [
            chunk["page_content"]
            for chunk in chunks
        ]

        vectors = self.embedding_model.embed_documents(
            texts,
            batch_size=batch_size
        )

        vectors = np.asarray(vectors, dtype=np.float32)

        if vectors.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2D array"
            )

        if vectors.shape[0] != len(chunks):
            raise ValueError(
                "Embedding and chunk counts do not match"
            )

        if vectors.shape[1] == 0:
            raise ValueError("Embedding dimension cannot be zero")

        if not np.isfinite(vectors).all():
            raise ValueError(
                "Embeddings contain NaN or infinite values"
            )

        return vectors

    def save(self, chunks, vectors, output_dir):

        output_dir = Path(output_dir)

        vectors = np.asarray(vectors, dtype=np.float32)

        if vectors.ndim != 2 or vectors.shape[0] == 0:
            raise ValueError("No valid embedding vectors to save")

        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                "Chunk and embedding counts do not match"
            )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        np.save(
            output_dir / "embeddings.npy",
            vectors
        )

        with (output_dir / "chunks.jsonl").open(
            "w",
            encoding="utf-8"
        ) as file:

            for chunk in chunks:
                file.write(
                    json.dumps(
                        chunk,
                        ensure_ascii=False
                    ) + "\n"
                )

        metadata = {
            "model_name": self.embedding_model.model_name,
            "embedding_dimension": int(vectors.shape[1]),
            "number_of_chunks": len(chunks),
            "normalized": None
        }

        with (output_dir / "embedding_metadata.json").open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                metadata,
                file,
                indent=4
            )

    def process(
        self,
        input_path,
        output_dir,
        batch_size=32
    ):

        chunks = self.load_chunks(input_path)

        if not chunks:
            raise ValueError(
                "No chunks found in input file"
            )

        vectors = self.generate(
            chunks,
            batch_size=batch_size
        )

        self.save(
            chunks,
            vectors,
            output_dir
        )

        return chunks, vectors