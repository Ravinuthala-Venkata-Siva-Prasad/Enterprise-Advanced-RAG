import pytest

from src.memory.conversation_store import ConversationStore
from src.memory.conversation_manager import ConversationManager
from src.memory.query_rewriter import QueryRewriter
from src.pipeline.conversational_rag import ConversationalRAG


@pytest.fixture
def manager(tmp_path):
    db_path = str(tmp_path / "test_chat.db")
    store = ConversationStore(db_path=db_path)
    return ConversationManager(store=store, max_turns=2)


def test_store_and_retrieve_messages(manager):
    manager.add_user_message("session-1", "What is MFA?")
    manager.add_assistant_message("session-1", "MFA is required.")

    history = manager.get_history("session-1")

    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_session_isolation(manager):
    manager.add_user_message("session-A", "Question A")
    manager.add_user_message("session-B", "Question B")

    history_a = manager.get_history("session-A")
    history_b = manager.get_history("session-B")

    assert len(history_a) == 1
    assert len(history_b) == 1
    assert history_a[0]["content"] == "Question A"
    assert history_b[0]["content"] == "Question B"


def test_history_limit(manager):
    manager.add_user_message("session-1", "Question 1")
    manager.add_assistant_message("session-1", "Answer 1")
    manager.add_user_message("session-1", "Question 2")
    manager.add_assistant_message("session-1", "Answer 2")
    manager.add_user_message("session-1", "Question 3")

    history = manager.get_history("session-1")

    assert len(history) == 4
    assert history[0]["content"] == "Answer 1"
    assert history[-1]["content"] == "Question 3"


def test_clear_session(manager):
    manager.add_user_message("session-1", "Hello")
    manager.clear_session("session-1")

    assert manager.get_history("session-1") == []


def test_invalid_role(manager):
    with pytest.raises(ValueError):
        manager.store.add_message("session-1", "system", "Invalid role")


def test_empty_session_id(manager):
    with pytest.raises(ValueError):
        manager.add_user_message("", "Hello")


class FakeLLM:
    def invoke(self, messages):
        class Response:
            content = "What are the requirements for enabling MFA?"
        return Response()


def test_query_rewriter():
    rewriter = QueryRewriter(llm=FakeLLM())

    result = rewriter.rewrite(
        question="What about its requirements?",
        conversation_history=(
            "USER: What is the company's MFA policy?\n"
            "ASSISTANT: MFA is required for company email."
        )
    )

    assert result == "What are the requirements for enabling MFA?"


class FakeRAGChain:
    def ask(self, question):
        return {
            "answer": f"Test answer for: {question}",
            "retrieved_sources": ["security_policy.txt"]
        }


class FakeQueryRewriter:
    def rewrite(self, question, conversation_history):
        return "What are the MFA requirements?"


def test_conversational_rag(manager):
    rag = ConversationalRAG(
        rag_chain=FakeRAGChain(),
        conversation_manager=manager,
        query_rewriter=FakeQueryRewriter()
    )

    result = rag.ask(
        question="What about its requirements?",
        session_id="session-1"
    )

    assert result["rewritten_question"] == "What are the MFA requirements?"
    assert "Test answer" in result["answer"]
    assert len(manager.get_history("session-1")) == 2