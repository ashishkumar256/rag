# Two-container RAG stack

| Container    | Role                                              | Ports        |
|--------------|---------------------------------------------------|--------------|
| **chromadb** | Official ChromaDB vector database                 | 8001 → 8000  |
| **rag**      | http.server (PDFs) + LangChain index + `/ask` API | 8080, 8000   |

---

## Config (`.env` → injected by Docker Compose `env_file`)

Edit `rag_simple/.env` to change defaults:

```env
DEFAULT_TOP_K=5
CHUNK_SIZE=800
CHUNK_OVERLAP=150
EMBED_MODEL=BAAI/bge-small-en-v1.5
COLLECTION_NAME=db_providers
```

`k` in `/ask` falls back to `DEFAULT_TOP_K` when omitted.

---

## Start

```bash
cd rag_simple
mkdir -p chroma_data hf_cache
docker compose up --build -d
```

---

## 1. Index PDFs

```bash
# Start indexing
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"force": false}'

# Force full rebuild
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"force": true}'

# Status (works for 20k+ files)
curl http://localhost:8000/index/status | jq
```

---

## 2. Ask (natural language)

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the SLA for MongoDB Atlas?"}'
```

Optional top-k:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Compare encryption at rest across providers", "k": 8}'
```

**Response shape** (retrieval only — ready for LLM later):

```json
{
  "question": "What is the SLA for MongoDB Atlas?",
  "k": 5,
  "contexts": [
    {
      "rank": 1,
      "score": 0.87,
      "text": "...",
      "source_file": "04_mongodb_atlas.pdf",
      "company_slug": "mongodb_atlas",
      "page": 2,
      "chunk_index": 3
    }
  ],
  "answer": null,
  "note": "Retrieval only. LLM answer integration can be added later."
}
```

---

## Other

```bash
curl http://localhost:8000/health | jq
curl http://localhost:8000/documents | jq
curl -I http://localhost:8080/01_aws_rds_aurora.pdf
```
