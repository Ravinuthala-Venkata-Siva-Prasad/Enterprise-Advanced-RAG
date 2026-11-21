from fastapi.testclient import TestClient

from api.main import create_app
from api.dependencies import ApplicationServices


class FakeConversationalRAG:

    def __init__(self):
        self.sessions = {}

    def ask(self, question: str, session_id: str):
        self.sessions.setdefault(session_id, []).append(question)

        return {
            "answer": f"Mock answer for: {question}",
            "rewritten_question": question,
            "sources": ["security_policy.txt"]
        }

    def clear_session(self, session_id: str):
        self.sessions.pop(session_id, None)


def create_test_client():
    fake_rag = FakeConversationalRAG()

    services = ApplicationServices(
        conversational_rag=fake_rag
    )

    app = create_app(services)

    return TestClient(app), fake_rag


def test_health_endpoint():
    client, _ = create_test_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_chat_endpoint():
    client, _ = create_test_client()

    response = client.post(
        "/api/v1/chat",
        json={
            "question": "What is the MFA policy?"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == "Mock answer for: What is the MFA policy?"
    assert data["session_id"]
    assert data["sources"] == ["security_policy.txt"]


def test_existing_session():
    client, fake_rag = create_test_client()

    first = client.post(
        "/api/v1/chat",
        json={
            "question": "What is MFA?",
            "session_id": "test-session"
        }
    )

    second = client.post(
        "/api/v1/chat",
        json={
            "question": "What about remote access?",
            "session_id": "test-session"
        }
    )

    assert first.status_code == 200
    assert second.status_code == 200

    assert len(fake_rag.sessions["test-session"]) == 2


def test_empty_question_rejected():
    client, _ = create_test_client()

    response = client.post(
        "/api/v1/chat",
        json={"question": ""}
    )

    assert response.status_code == 422


def test_clear_session():
    client, fake_rag = create_test_client()

    client.post(
        "/api/v1/chat",
        json={
            "question": "What is MFA?",
            "session_id": "test-session"
        }
    )

    response = client.delete(
        "/api/v1/sessions/test-session"
    )

    assert response.status_code == 200
    assert "test-session" not in fake_rag.sessions