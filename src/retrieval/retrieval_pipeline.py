
import re
from difflib import SequenceMatcher
from pathlib import Path

import yaml

from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker


class AdvancedRetrievalPipeline:

    def __init__(self, config_path="config/retrieval_config.yaml"):

        with Path(config_path).open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)["retrieval"]

        self.config = config

        self.dense = DenseRetriever()
        self.sparse = SparseRetriever()

        self.hybrid = HybridRetriever(
            dense_retriever=self.dense,
            sparse_retriever=self.sparse,
            dense_weight=config["hybrid"]["dense_weight"],
            sparse_weight=config["hybrid"]["sparse_weight"],
            rrf_k=config["hybrid"]["rrf_k"]
        )

        self.reranker = CrossEncoderReranker(
            config["reranker"]["model_name"]
        )

        # Optional settings; defaults preserve compatibility
        dedup_config = config.get("deduplication", {})

        self.deduplication_enabled = dedup_config.get(
            "enabled", True
        )

        self.similarity_threshold = dedup_config.get(
            "similarity_threshold", 0.92
        )

        if not 0 < self.similarity_threshold <= 1:
            raise ValueError(
                "Deduplication similarity_threshold must be between 0 and 1"
            )

    @staticmethod
    def _normalize_content(content):
        """
        Normalize text for duplicate comparison.

        Does not modify the original document content.
        """
        if not isinstance(content, str):
            return ""

        content = content.lower()

        # Normalize common Unicode dash characters
        content = re.sub(r"[\u2010-\u2015\u2212]", "-", content)

        # Normalize whitespace
        content = re.sub(r"\s+", " ", content)

        # Remove punctuation differences
        content = re.sub(r"[^\w\s]", "", content)

        return content.strip()

    @classmethod
    def _similarity(cls, first, second):
        first = cls._normalize_content(first)
        second = cls._normalize_content(second)

        if not first or not second:
            return 0.0

        if first == second:
            return 1.0

        return SequenceMatcher(
            None,
            first,
            second,
            autojunk=False
        ).ratio()

    @staticmethod
    def _get_content(result):
        document = result.get("document", {})

        if isinstance(document, dict):
            return document.get("page_content", "")

        return getattr(document, "page_content", "")

    def deduplicate_results(self, results):
        """
        Keep the highest-ranked version of duplicate passages.

        Results must already be sorted by reranker relevance.
        Original result dictionaries and metadata are preserved.
        """
        if not self.deduplication_enabled:
            return results

        unique_results = []
        seen_contents = []

        for result in results:
            content = self._get_content(result)
            normalized = self._normalize_content(content)

            if not normalized:
                unique_results.append(result)
                continue

            duplicate = False

            for previous_content in seen_contents:
                similarity = self._similarity(
                    normalized,
                    previous_content
                )

                if similarity >= self.similarity_threshold:
                    duplicate = True
                    break

            if not duplicate:
                unique_results.append(result)
                seen_contents.append(normalized)

        return unique_results

    def retrieve(self, query):

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query cannot be empty")

        hybrid_config = self.config["hybrid"]
        reranker_config = self.config["reranker"]

        candidates = self.hybrid.retrieve(
            query,
            top_k=hybrid_config["top_k"]
        )

        # First rerank a wider candidate set.
        # Deduplication happens after relevance ordering.
        rerank_candidates = self.reranker.rerank(
            query,
            candidates,
            top_k=hybrid_config["top_k"]
        )

        # Remove duplicate passages while preserving ranking.
        unique_results = self.deduplicate_results(
            rerank_candidates
        )

        # Return the requested number of unique results.
        return unique_results[:reranker_config["top_k"]]

    def refresh(self):

        return self.hybrid.refresh()
