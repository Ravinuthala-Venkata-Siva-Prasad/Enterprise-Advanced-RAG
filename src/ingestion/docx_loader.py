from pathlib import Path
from typing import List
from docx import Document as DocxDocuments
from langchain_core.documents import Document
from src.ingestion.base_loader import BaseLoader
from src.utils.logger import logger

class DOCXLoader(BaseLoader):
    def load(self,file_path:str | Path)-> List[Document]:
        path = self.validate_file(file_path)

        try:

            word_document = DocxDocuments(str(path))
            content= []

            for paragraph in word_document.paragraphs:
                text = paragraph.text.strip()

                if text:
                    content.append(text)

            for table_index,table in enumerate(word_document.tables, start=1):
                content.append(f"\nTable {table_index}:")

                for row in table.rows:

                    row_values = [cell.text.strip() for cell in row.cells]
                    content.append(" | ".join(row_values))

            text = "\n".join(content)

            if not text.strip():
                raise ValueError(f"No readable content in DOCX: {path.name}")

            metadata = {
                "source": str(path.resolve()),
                "file_name": path.name,
                "file_type": "docx"
            }

            logger.info( f"DOCX loader: {path.name}")

            return [Document(page_content=text, metadata=metadata)]

        except Exception:
            logger.exception(f"DOCX loading failed: {path.name}")
            raise