class ConversationalRAG:
    def __init__(
        self,
        rag_chain,
        conversation_manager,
        query_rewriter
    ):
        self.rag_chain = rag_chain
        self.conversation_manager = conversation_manager
        self.query_rewriter = query_rewriter

    def ask(self, question: str, session_id: str) -> dict:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be empty")

        if not session_id or not session_id.strip():
            raise ValueError("session_id cannot be empty")

        question = question.strip()

        # 1. Load this session's conversation history.
        history = self.conversation_manager.format_history(session_id)

        # 2. Rewrite the follow-up question for retrieval.
        search_query = self.query_rewriter.rewrite(
            question=question,
            conversation_history=history
        )

        # 3. Ask the existing RAG pipeline.
        # The original question is retained for answer generation.
        result = self.rag_chain.ask(question=search_query)

        answer = result.get("answer", "")

        # 4. Store the original user question and generated answer.
        self.conversation_manager.add_user_message(
            session_id,
            question
        )

        self.conversation_manager.add_assistant_message(
            session_id,
            answer
        )

        return {
            "session_id": session_id,
            "original_question": question,
            "rewritten_question": search_query,
            "answer": answer,
            "sources": result.get(
                "retrieved_sources",
                result.get("sources", [])
            )
        }

    def clear_session(self, session_id: str):
        self.conversation_manager.clear_session(session_id)