from pathlib import Path
from typing import List

import pandas as pd
from langchain_core.documents import Document
from src.ingestion.base_loader import BaseLoader
from src.utils.logger import logger


class CSVLoader(BaseLoader):

    def load(self, file_path: str | Path) -> List[Document]:

        path = self.validate_file(file_path)

        try:

            dataframe = pd.read_csv(path,encoding="utf-8",keep_default_na=False)

            if dataframe.empty:

                raise ValueError(f"CSV contains no records: {path.name}")

            documents = []

            for index, row in dataframe.iterrows():

                row_content = "\n".join(
                    f"{column}: {value}"
                    for column, value in row.items()
                )

                metadata = {
                    "source": str(path.resolve()),
                    "file_name": path.name,
                    "file_type": "csv",
                    "row_number": int(index) + 2
                }

                documents.append(
                    Document(page_content=row_content,metadata=metadata))

            logger.info(
                f"CSV loaded: {path.name}, "
                f"rows extracted: {len(documents)}"
            )

            return documents

        except Exception:

            logger.exception(f"CSV loading failed: {path.name}")

            raise