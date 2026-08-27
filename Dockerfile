# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm

LABEL description="All-in-one: http.server (PDFs) + LangChain/Chroma ingest + search API"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY ingest.py serve_pdfs.py search_api.py ./

RUN mkdir -p /data/pdfs /data/chroma /data/hf_cache

ENV PDF_DIR=/data/pdfs \
    CHROMA_DIR=/data/chroma \
    COLLECTION_NAME=db_providers \
    EMBED_MODEL=all-MiniLM-L6-v2 \
    CHUNK_SIZE=800 \
    CHUNK_OVERLAP=150 \
    HOST=0.0.0.0 \
    PDF_PORT=8080 \
    PORT=8000 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/hf_cache \
    TRANSFORMERS_CACHE=/data/hf_cache

CMD ["sleep", "infinity"]
