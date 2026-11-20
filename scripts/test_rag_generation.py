from src.generation.rag_chain import EnterpriseRAGChain


def main():

    rag = EnterpriseRAGChain()

    questions = [
        "What is the company security policy for MFA?",
        "What is the company's remote work policy?",
        "How does the company handle expense reimbursement?",
        "What is the employee bonus percentage?"
    ]

    for question in questions:

        print("\n" + "=" * 70)
        print("QUESTION:", question)

        result = rag.ask(question)

        print("\nANSWER:")
        print(result["answer"])

        print("\nSOURCES:")
        for source in result["sources"]:
            print("-", source)

        print("\nRETRIEVED DOCUMENTS:")
        print(result["retrieved_count"])


if __name__ == "__main__":
    main()