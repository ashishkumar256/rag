FROM python:3.11-slim

# build-essential: chromadb's hnswlib dependency sometimes needs to compile
# from source if no prebuilt wheel matches the platform (e.g. on arm64).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first, separately from app code, so Docker's layer cache
# skips this slow step (torch download included) on every code change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model at build time instead of on first run.
# Adds a few hundred MB to the image but means `docker run ... ask ...`
# doesn't silently pause to download a model the first time it's used.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('BAAI/bge-small-en-v1.5')"

COPY rag_pipeline.py generate_test_pdfs.py ./

# Mount points (created here so bind mounts have somewhere to land):
#   /app/pdfs      -> your source PDFs (read-only is fine)
#   /app/chroma_db -> persisted vector store + incremental-index cache
RUN mkdir -p /app/pdfs /app/chroma_db

ENTRYPOINT ["python", "rag_pipeline.py"]
CMD ["ask", "--help"]
