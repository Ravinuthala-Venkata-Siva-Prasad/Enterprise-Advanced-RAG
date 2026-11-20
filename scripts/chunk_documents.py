import argparse
import json
from pathlib import Path

from langchain_core.documents import Document

from src.parsing.chunk_manager import ChunkManager
from src.utils.logger import logger


def load_jsonl(file_path: Path):

    documents = []

    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            item = json.loads(line)

            documents.append(
                Document(
                    page_content=item["page_content"],
                    metadata=item.get("metadata", {})
                )
            )

    return documents


def save_jsonl(documents, output_path: Path):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        for document in documents:

            record = {
                "page_content": document.page_content,
                "metadata": document.metadata
            }

            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def main():

    parser = argparse.ArgumentParser(
        description="Enterprise RAG document chunking"
    )

    parser.add_argument(
        "--input",
        default="data/processed/ingested_documents.jsonl"
    )

    parser.add_argument(
        "--output",
        default="data/processed/chunks.jsonl"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=120
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    try:
        logger.info("Starting document chunking")

        documents = load_jsonl(input_path)

        logger.info(f"Loaded documents: {len(documents)}")

        manager = ChunkManager(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )

        chunks = manager.process(documents)

        save_jsonl(chunks, output_path)

        logger.info(f"Generated chunks: {len(chunks)}")
        logger.info(f"Saved chunks to: {output_path}")

    except Exception:
        logger.exception("Document chunking failed")
        raise


if __name__ == "__main__":
    main()