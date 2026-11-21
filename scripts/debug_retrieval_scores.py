
import sys
from pathlib import Path

# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.retrieval_pipeline import AdvancedRetrievalPipeline


def main():
    pipeline = AdvancedRetrievalPipeline()

    questions = [
        "What is the company security policy for MFA?",
        "What is the remote work policy?",
        "What are the business expense reimbursement rules?"
    ]

    for question in questions:
        print("\n" + "=" * 80)
        print("QUESTION:", question)
        print("=" * 80)

        hybrid_config = pipeline.config["hybrid"]
        reranker_config = pipeline.config["reranker"]

        candidates = pipeline.hybrid.retrieve(
            question,
            top_k=hybrid_config["top_k"]
        )

        reranked = pipeline.reranker.rerank(
            question,
            candidates,
            top_k=hybrid_config["top_k"]
        )

        unique_results = pipeline.deduplicate_results(reranked)

        for rank, item in enumerate(unique_results, start=1):
            document = item["document"]
            metadata = document.get("metadata", {})

            print(f"\nRank: {rank}")
            print("Source:", metadata.get("source"))
            print("Reranker score:", item.get("reranker_score"))
            print("Content:", document.get("page_content", "")[:250])

        final_results = unique_results[:reranker_config["top_k"]]

        print("\nFinal returned documents:", len(final_results))


if __name__ == "__main__":
    main()
