import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq


class GroqLLMClient:

    def __init__(
        self,
        model_name="openai/gpt-oss-120b",
        temperature=0.1,
        max_tokens=1500
    ):

        load_dotenv()

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. Add it to your .env file."
            )

        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key
        )

    def generate(self, prompt):

        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        response = self.llm.invoke(prompt)

        return response.content