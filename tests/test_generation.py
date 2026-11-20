from src.generation.context_builder import ContextBuilder
from src.generation.prompt_templates import build_rag_prompt


def test_context_builder():

    results = [
        {
            "document": {
                "page_content": "MFA is required.",
                "metadata": {
                    "source": "security.pdf",
                    "chunk_id": "chunk-1",
                    "page_number": 2
                }
            }
        }
    ]

    builder = ContextBuilder()

    context = builder.build(results)

    assert "MFA is required" in context
    assert "security.pdf" in context
    assert "chunk-1" in context


def test_context_limit():

    results = [
        {
            "document": {
                "page_content": "A" * 1000,
                "metadata": {"source": "test.txt"}
            }
        }
    ]

    builder = ContextBuilder(
        max_chunks=5,
        max_context_characters=100
    )

    context = builder.build(results)

    assert len(context) <= 100


def test_prompt_contains_question():

    prompt = build_rag_prompt(
        "What is MFA?",
        "MFA is multi-factor authentication."
    )

    assert "What is MFA?" in prompt
    assert "MFA is multi-factor authentication." in prompt


def test_empty_context():

    builder = ContextBuilder()

    assert builder.build([]) == ""