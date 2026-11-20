from pathlib import Path
from typing import List

from langchain_core.documents import Document
from src.ingestion.base_loader import BaseLoader
from src.utils.logger import logger

class TXTLoader(BaseLoader):

    def load(self, file_path:str | Path) -> List[Document]:
        path = self.validate_file(file_path)

        try:

            encodings = ["utf-8","utf-8-sig","latin-1"]
            text = None

            for encoding in encodings:

                try:

                    text = path.read_text(encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue

            if  text is None or not text.strip():

                raise ValueError(f"No readable text in file: {path.name}")

            metadata = {
                "source": str(path.resolve()),
                "file_name": path.name,
                "file_type": "txt"
            }

            logger.info(f"TXT loader: {path.name}")

            return[
                Document(page_content=text.strip(), metadata=metadata)
            ]

        except Exception:

            logger.exception(f"TXT loading failed: {path.name}")

            raise

            