import requests

from .config import (
    CHAT_ENDPOINT,
    HEALTH_ENDPOINT,
    API_BASE_URL,
)
from frontend.config import (
    CHAT_ENDPOINT,
    HEALTH_ENDPOINT,
    API_BASE_URL,
)

class RAGAPIClient:
    def __init__(self, timeout: int = 120):
        self.timeout = timeout

    def health_check(self):
        response = requests.get(HEALTH_ENDPOINT, timeout=10)
        response.raise_for_status()
        return response.json()

    def ask_question(self, question: str, session_id: str):
        response = requests.post(
            CHAT_ENDPOINT,
            json={"question": question, "session_id": session_id},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def clear_session(self, session_id: str):
        response = requests.delete(
            f"{API_BASE_URL}/api/v1/sessions/{session_id}", timeout=15
        )
        response.raise_for_status()
        return response.json()

    def upload_document(self, uploaded_file):
        response = requests.post(
            f"{API_BASE_URL}/api/v1/documents/upload",
            files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "application/octet-stream")},
            timeout=180,
        )
        response.raise_for_status()
        return response.json()

    def get_document_stats(self):
        response = requests.get(f"{API_BASE_URL}/api/v1/documents/stats", timeout=15)
        response.raise_for_status()
        return response.json()

    def get_documents(self):
        response = requests.get(f"{API_BASE_URL}/api/v1/documents", timeout=20)
        response.raise_for_status()
        return response.json().get("documents", [])

    def get_document(self, document_id: str):
        response = requests.get(
            f"{API_BASE_URL}/api/v1/documents/{document_id}", timeout=15
        )
        response.raise_for_status()
        return response.json()

    def delete_document(self, document_id: str):
        response = requests.delete(
            f"{API_BASE_URL}/api/v1/documents/{document_id}", timeout=60
        )
        response.raise_for_status()
        return response.json()
