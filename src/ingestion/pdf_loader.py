from pathlib import Path
from typing import List

from pypdf import PdfReader
from langchain_core.documents import Document

from src.ingestion.base_loader import BaseLoader
from src.utils.logger import logger

class PDFLoader(BaseLoader):

    def load(self, file_path: str | Path)-> List[Document]:
        path = self.validate_file(file_path)
        documents = []

        try:
            reader = PdfReader(str(path))

            if reader.is_encrypted:
                raise ValueError("Encrypted PDF is not supported")

            for page_number,page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""

                if not text.strip():
                    continue

                metadata = {
                    "source": str(path.resolve()),
                    "file_name": path.name,
                    "file_type": "pdf",
                    "page_number": page_number,
                    "total_pages": len(reader.pages)
                }

                documents.append(
                    Document(page_content=text.strip(), metadata=metadata)
                )
            if not documents:
                raise ValueError(f"No extractable test found in PDF: {path.name}")

            logger.info(f"PDF loader: {path.name},"
                        f"pages extracted: {len(documents)}")

            return documents
        except Exception:
            logger.exception(f"PDF loading failed: {path.name}")

            raise