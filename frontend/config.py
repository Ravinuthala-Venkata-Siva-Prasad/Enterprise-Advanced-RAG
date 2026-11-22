import os

# Read the API URL from Streamlit Cloud environment/secrets.
# Local development uses localhost.
API_BASE_URL = os.getenv(
    "RAG_API_URL",
    "http://127.0.0.1:8000"
).rstrip("/")

# API endpoints
CHAT_ENDPOINT = f"{API_BASE_URL}/api/v1/chat"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"