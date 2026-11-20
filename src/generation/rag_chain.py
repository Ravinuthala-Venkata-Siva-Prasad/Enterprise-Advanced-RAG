import yaml
from pathlib import Path

from src.retrieval.retrieval_pipeline import AdvancedRetrievalPipeline
from src.generation.answer_generator import AnswerGenerator


class EnterpriseRAGChain:

    def __init__(
        self,
        config_path="config/generation_config.yaml"
    ):

        with Path(config_path).open(
            "r",
            encoding="utf-8"
        ) as file:

            config = yaml.safe_load(file)["generation"]

        self.retriever = AdvancedRetrievalPipeline()

        self.generator = AnswerGenerator(
            model_name=config["model_name"],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            max_chunks=config["context"]["max_chunks"],
            max_context_characters=config["context"][
                "max_context_characters"
            ]
        )

    def ask(self, question):

        retrieved_documents = self.retriever.retrieve(question)

        result = self.generator.generate(
            question=question,
            retrieved_documents=retrieved_documents
        )

        result["question"] = question

        result["retrieved_documents"] = [
            item["document"]
            for item in retrieved_documents
        ]

        return result