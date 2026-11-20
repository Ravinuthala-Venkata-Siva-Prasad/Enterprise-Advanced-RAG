from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.parsing.text_cleaner import TextCleaner

class RecursiveChunker:

    def __init__(self,chunk_size: int = 800, chunk_overlap: int = 120):

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "? ",
                "! ",
                "; ",
                ", ",
                " ",
                ""
            ],
            keep_separator=True
        )

    def split_documents(self,documents:List[Document])-> List[Document]:
        cleaned_documents = []

        for document in documents:
            cleaned_text = TextCleaner.clean(document.page_content)

            if not cleaned_text:
                continue

            cleaned_documents.append(Document(page_content=cleaned_text,
                                              metadata = document.metadata.copy()))
        chunks = self.splitter.split_documents(cleaned_documents)

        for index,chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = index
            chunk.metadata["chunk_size"] = len(chunk.page_content)
            chunk.metadata["chunking_method"] = "recursive"

        return chunks
