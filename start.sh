#!/bin/bash

set -e

cd /app

export PYTHONPATH=/app
export RAG_API_URL=http://127.0.0.1:8000

echo "=========================================="
echo " ENTERPRISE ADVANCED RAG - STARTING"
echo "=========================================="

# Create required data directories
mkdir -p \
    data/vectorstore \
    data/conversations \
    data/uploads \
    data/raw \
    data/processed

# Start FastAPI in background
echo "[1/3] Starting FastAPI..."

uvicorn api.main:build_production_app \
    --factory \
    --host 127.0.0.1 \
    --port 8000 &

API_PID=$!

# Wait for FastAPI health check
echo "[2/3] Waiting for FastAPI..."

python - <<'PY'
import time
import urllib.request

health_url = "http://127.0.0.1:8000/health"

for attempt in range(120):
    try:
        response = urllib.request.urlopen(
            health_url,
            timeout=5
        )

        if response.status == 200:
            print("FastAPI is healthy.")
            break

    except Exception:
        time.sleep(2)

else:
    raise SystemExit("FastAPI failed to start.")
PY

# Stop container if API exits
if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "FastAPI process stopped unexpectedly."
    exit 1
fi

# Start Streamlit in foreground
echo "[3/3] Starting Streamlit..."

exec streamlit run frontend/app.py \
    --server.address 0.0.0.0 \
    --server.port 7860 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection true