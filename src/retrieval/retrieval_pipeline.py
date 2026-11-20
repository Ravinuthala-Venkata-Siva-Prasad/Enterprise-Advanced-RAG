import yaml
from pathlib import Path

from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker


class AdvancedRetrievalPipeline:

    def __init__(
        self,
        config_path="config/retrieval_config.yaml"
    ):

        with Path(config_path).open(
            "r",
            encoding="utf-8"
        ) as file:

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

    def retrieve(self, query):

        hybrid_config = self.config["hybrid"]
        reranker_config = self.config["reranker"]

        candidates = self.hybrid.retrieve(
            query,
            top_k=hybrid_config["top_k"]
        )

        results = self.reranker.rerank(
            query,
            candidates,
            top_k=reranker_config["top_k"]
        )

        return results