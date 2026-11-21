
import sys
from pathlib import Path

# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.retrieval_pipeline import AdvancedRetrievalPipeline


def make_result(content, score=1.0, source="document.txt"):
    return {
        "document": {
            "page_content": content,
            "metadata": {"source": source}
        },
        "reranker_score": score
    }


def create_pipeline():
    pipeline = AdvancedRetrievalPipeline.__new__(
        AdvancedRetrievalPipeline
    )
    pipeline.deduplication_enabled = True
    pipeline.similarity_threshold = 0.92
    return pipeline


def test_exact_duplicates_are_removed():
    results = [
        make_result("Enable MFA for email and cloud consoles.", 0.9),
        make_result("Enable MFA for email and cloud consoles.", 0.8),
    ]

    output = create_pipeline().deduplicate_results(results)

    assert len(output) == 1
    assert output[0]["reranker_score"] == 0.9


def test_whitespace_duplicates_are_removed():
    results = [
        make_result("Enable MFA for email and cloud consoles."),
        make_result("Enable MFA for email\nand cloud consoles."),
    ]

    output = create_pipeline().deduplicate_results(results)

    assert len(output) == 1


def test_different_documents_are_preserved():
    results = [
        make_result("Enable MFA for email and cloud consoles."),
        make_result("Employees may work remotely three days per week."),
    ]

    output = create_pipeline().deduplicate_results(results)

    assert len(output) == 2


def test_disabled_deduplication_preserves_all_results():
    results = [
        make_result("Same content."),
        make_result("Same content."),
    ]

    pipeline = create_pipeline()
    pipeline.deduplication_enabled = False

    output = pipeline.deduplicate_results(results)

    assert len(output) == 2


def test_original_ranking_is_preserved():
    results = [
        make_result("First unique passage.", 0.95),
        make_result("Second unique passage.", 0.80),
        make_result("Third unique passage.", 0.60),
    ]

    output = create_pipeline().deduplicate_results(results)

    scores = [item["reranker_score"] for item in output]

    assert scores == [0.95, 0.80, 0.60]
