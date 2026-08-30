# Two-container RAG stack

| Container    | Role                                              | Ports        |
|--------------|---------------------------------------------------|--------------|
| **chromadb** | Official ChromaDB vector database                 | 8001 → 8000  |
| **rag**      | http.server (PDFs) + LangChain index + search API | 8080, 8000   |

---

## Start

```bash
cd rag_simple
docker compose up --build -d
```

---

## Index PDFs into ChromaDB

```bash
# Start indexing (background)
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"force": false}'

# Force full re-index (drops collection first)
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"force": true}'

# Poll status
curl http://localhost:8000/index/status | jq
```

Example `/index/status` response while running:

```json
{
  "status": "running",
  "message": "Indexing started",
  "started_at": "2026-08-29T05:40:00Z",
  "finished_at": null,
  "force": false,
  "total_pdfs": 20,
  "docs_processed": 7,
  "docs_skipped": 0,
  "chunks_created": 142,
  "current_file": "08_cockroachdb_cloud.pdf",
  "errors": [],
  "vectors_in_collection": 142
}
```

When finished: `"status": "done"`.

---

## Other endpoints

```bash
# Health
curl http://localhost:8000/health | jq

# Search
curl "http://localhost:8000/search?q=MongoDB%20Atlas%20SLA%20encryption&k=3" | jq

# List indexed documents
curl http://localhost:8000/documents | jq

# PDFs (static)
curl -I http://localhost:8080/01_aws_rds_aurora.pdf
```

---

## Architecture

```
Browser / curl
      │
      ├─ :8080  →  rag (Python http.server)  →  PDFs
      │
      └─ :8000  →  rag (FastAPI)
                      │  LangChain parse / chunk / embed
                      ▼
                 chromadb :8000  (persisted volume)
```
