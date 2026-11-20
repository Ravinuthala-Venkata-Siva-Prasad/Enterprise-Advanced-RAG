from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from langchain_core.documents import Document

class BaseLoader(ABC):

    @abstractmethod
    def load(self, file_path:str | Path) -> List[Document]:
        pass

    @staticmethod
    def validate_file(file_path:str | Path) -> Path:

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"file not found: {path}")

        if not path.is_file():
            raise ValueError(f"Expected a file,received: {path}")

        if path.stat().st_size == 0:
            raise ValueError(f"file is empty {path}")

        return path