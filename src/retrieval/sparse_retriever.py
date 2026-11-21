
import json
import os
import re

from pathlib import Path
from rank_bm25 import BM25Okapi


class SparseRetriever:

    def __init__(
        self,
        chunks_path="data/embeddings/chunks.jsonl"
    ):

        self.chunks_path = Path(chunks_path)
        self.documents = self._load_documents(self.chunks_path)

        # An empty corpus is allowed after document deletion.
        self.bm25 = None
        self._build_index()

    # --------------------------------------------------
    # TOKENIZATION
    # --------------------------------------------------

    @staticmethod
    def _tokenize(text):

        if not isinstance(text, str):
            return []

        return re.findall(
            r"[a-zA-Z0-9_]+",
            text.lower()
        )

    # --------------------------------------------------
    # METADATA HELPERS
    # --------------------------------------------------

    @staticmethod
    def _get_metadata(document):

        metadata = document.get("metadata", {})

        if not isinstance(metadata, dict):
            raise ValueError(
                "Document metadata must be a dictionary"
            )

        return metadata

    @classmethod
    def _get_chunk_id(cls, document):

        metadata = cls._get_metadata(document)

        return metadata.get(
            "chunk_id",
            document.get("chunk_id")
        )

    @classmethod
    def _get_source(cls, document):

        metadata = cls._get_metadata(document)

        return metadata.get(
            "source",
            document.get("source")
        )

    # --------------------------------------------------
    # DOCUMENT VALIDATION
    # --------------------------------------------------

    @classmethod
    def _validate_documents(cls, documents):

        chunk_ids = []

        for document in documents:

            if not isinstance(document, dict):
                raise ValueError(
                    "Each document must be a dictionary"
                )

            page_content = document.get("page_content")

            if (
                not isinstance(page_content, str)
                or not page_content.strip()
            ):
                raise ValueError(
                    "Document page_content cannot be empty"
                )

            cls._get_metadata(document)

            chunk_id = cls._get_chunk_id(document)

            if not chunk_id:
                raise ValueError(
                    "Every document must have a chunk_id"
                )

            chunk_ids.append(str(chunk_id))

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError(
                "Duplicate chunk IDs in documents"
            )

    # --------------------------------------------------
    # LOAD DOCUMENTS
    # --------------------------------------------------

    @staticmethod
    def _load_documents(path):

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(path)

        documents = []

        with path.open(
            "r",
            encoding="utf-8"
        ) as file:

            for line_number, line in enumerate(file, start=1):

                if not line.strip():
                    continue

                try:
                    document = json.loads(line)

                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at line {line_number} "
                        f"in {path}"
                    ) from exc

                documents.append(document)

        SparseRetriever._validate_documents(documents)

        return documents

    # --------------------------------------------------
    # BUILD BM25 INDEX
    # --------------------------------------------------

    def _build_index(self):

        if not self.documents:
            self.bm25 = None
            return

        tokenized_corpus = [
            self._tokenize(document["page_content"])
            for document in self.documents
        ]

        # Avoid building BM25 with an entirely empty corpus.
        if not any(tokenized_corpus):
            self.bm25 = None
            return

        self.bm25 = BM25Okapi(tokenized_corpus)

    # --------------------------------------------------
    # SAVE DOCUMENTS
    # --------------------------------------------------

    def _save_documents(self, documents=None):

        documents = (
            self.documents if documents is None else documents
        )

        self.chunks_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        temp_path = self.chunks_path.with_suffix(
            self.chunks_path.suffix + ".tmp"
        )

        try:

            with temp_path.open(
                "w",
                encoding="utf-8"
            ) as file:

                for document in documents:

                    file.write(
                        json.dumps(
                            document,
                            ensure_ascii=False
                        ) + "\n"
                    )

            os.replace(
                temp_path,
                self.chunks_path
            )

        finally:

            if temp_path.exists():
                temp_path.unlink()

    # --------------------------------------------------
    # ADD DOCUMENTS
    # --------------------------------------------------

    def add_documents(
        self,
        new_documents,
        persist=True
    ):

        new_documents = list(new_documents)

        if not new_documents:
            return 0

        self._validate_documents(new_documents)

        existing_ids = {
            str(self._get_chunk_id(document))
            for document in self.documents
        }

        incoming_ids = {
            str(self._get_chunk_id(document))
            for document in new_documents
        }

        duplicates = existing_ids.intersection(
            incoming_ids
        )

        if duplicates:
            raise ValueError(
                "Chunk IDs already exist: "
                f"{list(duplicates)[:5]}"
            )

        updated_documents = (
            self.documents + new_documents
        )

        # Persist before changing the current in-memory state.
        if persist:
            self._save_documents(updated_documents)

        self.documents = updated_documents

        self._build_index()

        return len(new_documents)

    # --------------------------------------------------
    # REMOVE DOCUMENTS BY SOURCE
    # --------------------------------------------------

    def remove_by_source(
        self,
        source,
        persist=True
    ):

        if not isinstance(source, str) or not source.strip():
            raise ValueError(
                "Source cannot be empty"
            )

        updated_documents = [
            document
            for document in self.documents
            if self._get_source(document) != source
        ]

        removed_count = (
            len(self.documents) - len(updated_documents)
        )

        if removed_count == 0:
            return 0

        if persist:
            self._save_documents(updated_documents)

        self.documents = updated_documents

        self._build_index()

        return removed_count

    # --------------------------------------------------
    # REFRESH FROM DISK
    # --------------------------------------------------

    def refresh(self):

        updated_documents = self._load_documents(
            self.chunks_path
        )

        self.documents = updated_documents

        self._build_index()

        return len(self.documents)

    # --------------------------------------------------
    # RETRIEVE DOCUMENTS
    # --------------------------------------------------

    def retrieve(
        self,
        query,
        top_k=10
    ):

        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                "Query cannot be empty"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be positive"
            )

        if self.bm25 is None:
            return []

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True
        )

        results = []

        for index in ranked_indices[:top_k]:

            results.append({
                "document": self.documents[index],
                "score": float(scores[index]),
                "retrieval_method": "sparse"
            })

        return results

    # --------------------------------------------------
    # STATISTICS
    # --------------------------------------------------

    def get_stats(self):

        return {
            "total_documents": len(self.documents),
            "bm25_initialized": self.bm25 is not None,
            "chunks_path": str(self.chunks_path)
        }
