
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=300
ENV PIP_RETRIES=10
ENV RAG_API_URL=http://127.0.0.1:8000

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install \
    --no-cache-dir \
    --retries 10 \
    --timeout 300 \
    -r requirements.txt

COPY . .

RUN mkdir -p \
    /app/data/vectorstore \
    /app/data/conversations \
    /app/data/uploads \
    /app/data/raw \
    /app/data/processed

EXPOSE 7860

CMD ["/bin/bash", "/app/start.sh"]
