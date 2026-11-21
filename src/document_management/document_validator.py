
from pathlib import Path


class DocumentValidator:

    ALLOWED_EXTENSIONS = {
        ".pdf",
        ".docx",
        ".txt",
        ".csv",
        ".html",
        ".htm"
    }

    def __init__(self, max_file_size_mb=20):

        if max_file_size_mb <= 0:
            raise ValueError(
                "Maximum file size must be positive"
            )

        self.max_file_size_bytes = (
            max_file_size_mb * 1024 * 1024
        )

    def validate(self, file_path):

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(file_path)

        if not file_path.is_file():
            raise ValueError("Uploaded path must be a file")

        if file_path.suffix.lower() not in self.ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {file_path.suffix}"
            )

        if file_path.stat().st_size == 0:
            raise ValueError("Uploaded file is empty")

        if file_path.stat().st_size > self.max_file_size_bytes:
            raise ValueError(
                "File exceeds the maximum allowed size"
            )

        return True
