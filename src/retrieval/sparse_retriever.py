import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


class SparseRetriever:

    def __init__(self, chunks_path="data/embeddings/chunks.jsonl"):

        self.documents = self._load_documents(chunks_path)

        if not self.documents:
            raise ValueError("No documents available for BM25")

        tokenized_corpus = [
            self._tokenize(doc["page_content"])
            for doc in self.documents
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)

    @staticmethod
    def _tokenize(text):

        return re.findall(
            r"[a-zA-Z0-9_]+",
            text.lower()
        )

    @staticmethod
    def _load_documents(path):

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(path)

        documents = []

        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    documents.append(json.loads(line))

        return documents

    def retrieve(self, query, top_k=10):

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

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