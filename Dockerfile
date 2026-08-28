# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm

LABEL description="RAG service: http.server + LangChain + fastembed → remote ChromaDB"

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip

COPY app.py serve_pdfs.py ./

RUN mkdir -p /data/pdfs /data/hf_cache

ENV PDF_DIR=/data/pdfs \
    CHROMA_HOST=chromadb \
    CHROMA_PORT=8000 \
    COLLECTION_NAME=db_providers \
    EMBED_MODEL=BAAI/bge-small-en-v1.5 \
    CHUNK_SIZE=800 \
    CHUNK_OVERLAP=150 \
    HOST=0.0.0.0 \
    PORT=8000 \
    PDF_PORT=8080 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/hf_cache \
    TRANSFORMERS_CACHE=/data/hf_cache

# Start both PDF server and FastAPI in one container via a small entrypoint
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

EXPOSE 8000 8080
CMD ["./entrypoint.sh"]
