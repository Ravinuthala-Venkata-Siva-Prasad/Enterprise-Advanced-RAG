import json
import argparse

from pathlib import Path
from datetime import datetime, timezone

from src.ingestion.document_loader import DocumentLoader
from src.utils.logger import logger


def save_documents(documents, output_path):

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True,exist_ok=True)

    with output_path.open("w",encoding="utf-8") as file:

        for document in documents:

            record = {"page_content": document.page_content,"metadata": document.metadata}
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(documents)} documents to {output_path}")


def main():

    parser = argparse.ArgumentParser(description="Enterprise RAG Document Ingestion")

    parser.add_argument(
        "--input",
        default="data/raw",
        help="Input document directory"
    )

    parser.add_argument(
        "--output",
        default="data/processed/ingested_documents.jsonl",
        help="Output JSONL path"
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Skip failed files and continue ingestion"
    )

    args = parser.parse_args()
    start_time = datetime.now(timezone.utc)
    logger.info("Starting document ingestion")
    loader = DocumentLoader()

    documents = loader.load_directory(directory=args.input,continue_on_error=args.continue_on_error)

    save_documents(documents,args.output)

    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds()

    logger.info("========== INGESTION REPORT ==========")
    logger.info(f"Documents extracted: {len(documents)}")
    logger.info(f"Execution time: {duration:.2f} seconds")
    logger.info("Ingestion completed successfully")


if __name__ == "__main__":
    main()