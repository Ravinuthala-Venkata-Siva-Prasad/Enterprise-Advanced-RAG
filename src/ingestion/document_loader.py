from pathlib import Path
from typing import List

from langchain_core.documents import Document

from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.docx_loader import DOCXLoader
from src.ingestion.txt_loader import TXTLoader
from src.ingestion.csv_loader import CSVLoader
from src.ingestion.html_loader import HTMLLoader

from src.utils.logger import logger


class DocumentLoader:

    def __init__(self):

        self.loaders = {
            ".pdf": PDFLoader(),
            ".docx": DOCXLoader(),
            ".txt": TXTLoader(),
            ".csv": CSVLoader(),
            ".html": HTMLLoader(),
            ".htm": HTMLLoader()
        }

    def load_file(self,file_path: str | Path) -> List[Document]:

        path = Path(file_path)
        extension = path.suffix.lower()
        loader = self.loaders.get(extension)

        if loader is None:

            raise ValueError(f"Unsupported file format: {extension}")

        logger.info(f"Processing document: {path.name}")

        return loader.load(path)

    def load_directory(self,directory: str | Path,recursive: bool = True,continue_on_error: bool = False) -> List[Document]:

        directory = Path(directory)

        if not directory.exists():

            raise FileNotFoundError(f"Directory not found: {directory}")

        if not directory.is_dir():

            raise NotADirectoryError(f"Not a directory: {directory}")

        pattern = "**/*" if recursive else "*"

        files = sorted(
            file for file in directory.glob(pattern)
            if file.is_file()
            and file.suffix.lower() in self.loaders
        )

        all_documents = []

        for file_path in files:

            try:

                documents = self.load_file(file_path)
                all_documents.extend(documents)

            except Exception:

                logger.exception(f"Skipping failed document: {file_path}")

                if not continue_on_error:
                    raise

        logger.info(
            f"Ingestion completed. "
            f"Files discovered: {len(files)}, "
            f"Documents extracted: {len(all_documents)}"
        )

        return all_documents