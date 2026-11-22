import os

API_BASE_URL = os.getenv(
    "RAG_API_URL",
    "http://3.27.213.85"
).rstrip("/")

CHAT_ENDPOINT = f"{API_BASE_URL}/api/v1/chat"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"