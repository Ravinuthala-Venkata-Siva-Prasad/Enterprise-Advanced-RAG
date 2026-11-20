from pathlib import Path
from typing import List

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from src.ingestion.base_loader import BaseLoader
from src.utils.logger import logger


class HTMLLoader(BaseLoader):

    def load(self, file_path: str | Path) -> List[Document]:

        path = self.validate_file(file_path)

        try:

            html_content = path.read_text(encoding="utf-8")

            soup = BeautifulSoup(html_content,"html.parser")

            # Remove non-content elements
            for element in soup(["script", "style", "noscript"]):

                element.decompose()

            title = (
                soup.title.get_text(strip=True)
                if soup.title else path.stem
            )

            text = soup.get_text(separator="\n",strip=True)

            if not text.strip():

                raise ValueError(f"No readable HTML content: {path.name}")

            metadata = {
                "source": str(path.resolve()),
                "file_name": path.name,
                "file_type": "html",
                "title": title
            }

            logger.info(f"HTML loaded: {path.name}")

            return [
                Document(page_content=text,metadata=metadata)]

        except Exception:

            logger.exception(f"HTML loading failed: {path.name}")

            raise