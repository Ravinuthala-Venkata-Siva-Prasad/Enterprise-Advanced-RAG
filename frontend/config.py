import os

API_BASE_URL = os.getenv(
    "RAG_API_URL",
    "http://api:8000"
)

CHAT_ENDPOINT = f"{API_BASE_URL}/api/v1/chat"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"