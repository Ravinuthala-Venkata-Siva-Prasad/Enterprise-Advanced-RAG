from src.generation.llm_client import GroqLLMClient 
from src.generation.context_builder import ContextBuilder
from src.generation.prompt_templates import build_rag_prompt


class AnswerGenerator:

    def __init__(
        self,
        model_name="openai/gpt-oss-120b",
        temperature=0.1,
        max_tokens=1500,
        max_chunks=5,
        max_context_characters=12000
    ):

        self.llm_client = GroqLLMClient(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )

        self.context_builder = ContextBuilder(
            max_chunks=max_chunks,
            max_context_characters=max_context_characters
        )

    def generate(self, question, retrieved_documents):

        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        if not retrieved_documents:
            return {
                "answer": (
                    "I could not find sufficient information "
                    "in the provided documents."
                ),
                "sources": [],
                "retrieved_count": 0
            }

        context = self.context_builder.build(
            retrieved_documents
        )

        if not context.strip():
            return {
                "answer": (
                    "I could not find sufficient information "
                    "in the provided documents."
                ),
                "sources": [],
                "retrieved_count": 0
            }

        prompt = build_rag_prompt(
            question=question,
            context=context
        )

        answer = self.llm_client.generate(prompt)

        sources = []

        for result in retrieved_documents:

            metadata = result["document"].get("metadata", {})

            source = metadata.get(
                "source",
                metadata.get("file_name")
            )

            if source and source not in sources:
                sources.append(source)

        return {"answer": answer,
                "retrieved_sources": sources,
                "retrieved_count": len(retrieved_documents)}