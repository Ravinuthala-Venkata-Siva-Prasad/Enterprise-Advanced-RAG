from typing import List
from uuid import uuid4

from langchain_core.documents import Document
from src.parsing.chunking import RecursiveChunker


class ChunkManager:

    def __init__(self,chunk_size: int = 800,chunk_overlap: int = 120):
        self.chunker = RecursiveChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def process(self,documents: List[Document]) -> List[Document]:

        chunks = self.chunker.split_documents(documents)

        for chunk in chunks:
            chunk.metadata["chunk_id"] = str(uuid4())

        return chunks