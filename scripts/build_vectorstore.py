import argparse

from src.utils.logger import logger
from src.vectorstores.vectorstore_manager import VectorStoreManager


def main():

    parser = argparse.ArgumentParser(
        description="Build Enterprise RAG FAISS vector store"
    )

    parser.add_argument(
        "--embeddings",
        default="data/embeddings/embeddings.npy"
    )

    parser.add_argument(
        "--chunks",
        default="data/embeddings/chunks.jsonl"
    )

    parser.add_argument(
        "--output",
        default="data/vectorstore"
    )

    args = parser.parse_args()

    try:

        logger.info("Starting FAISS index construction")

        manager = VectorStoreManager()

        store = manager.build(
            embeddings_path=args.embeddings,
            chunks_path=args.chunks,
            output_dir=args.output
        )

        logger.info(
            f"FAISS index created successfully: {store.index.ntotal} vectors"
        )

        logger.info(
            f"Embedding dimension: {store.dimension}"
        )

        logger.info(
            f"Index saved at: {args.output}"
        )

    except Exception:

        logger.exception("FAISS index construction failed")
        raise


if __name__ == "__main__":
    main()