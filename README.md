# All-in-one RAG stack for Database Provider PDFs

Single `docker-compose.yaml` that brings up:

| Service       | Port  | Role                                      |
|---------------|-------|-------------------------------------------|
| `pdf-server`  | 8080  | Python **stdlib http.server** serving PDFs |
| `ingest`      | –     | LangChain → chunk → embed → **ChromaDB** (one-shot) |
| `search`      | 8000  | Lightweight search API over the vectors   |

---

## Start everything

```bash
cd rag_simple
docker compose up --build -d
```

What happens:

1. `pdf-server` starts immediately → http://localhost:8080/
2. `ingest` runs once (downloads embedding model the first time, parses all 20 PDFs, stores vectors)
3. `search` starts after ingest finishes → http://localhost:8000/

Follow progress:

```bash
docker compose logs -f
```

---

## Endpoints

**PDF server (http.server)**
- http://localhost:8080/
- http://localhost:8080/01_aws_rds_aurora.pdf
- http://localhost:8080/04_mongodb_atlas.pdf
- …

**Search API**
```bash
# Health
curl http://localhost:8000/health | jq

# Search
curl "http://localhost:8000/search?q=MongoDB%20Atlas%20SLA%20encryption&k=3" | jq

# POST
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "customer managed encryption keys", "k": 5}' | jq
```

---

## Re-ingest / force rebuild

```bash
# Normal (skips files already present)
docker compose run --rm ingest

# Wipe collection and re-ingest everything
docker compose run --rm ingest python ingest.py --force
```

---

## Local (without Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

export PDF_DIR=../db_proposals
export CHROMA_DIR=./chroma_data

# Terminal 1
python serve_pdfs.py --port 8080

# Terminal 2
python ingest.py

# Terminal 3
python search_api.py
```

---

## Pipeline details

| Step              | Tool |
|-------------------|------|
| Load PDF pages    | `langchain_community.document_loaders.PyPDFLoader` |
| Chunk             | `RecursiveCharacterTextSplitter` (800 / 150) |
| Embed             | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector store      | `langchain_chroma.Chroma` → persistent ChromaDB |
| Static file serve | Python stdlib `http.server` |
