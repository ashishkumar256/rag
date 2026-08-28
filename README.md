# Two-container RAG stack

| Container   | Image / Role                                      | Ports      |
|-------------|---------------------------------------------------|------------|
| **chromadb**| Official ChromaDB vector database                 | 8001→8000  |
| **rag**     | http.server (PDFs) + LangChain parse/chunk/embed + search API | 8080 (PDFs), 8000 (API) |

---

## Start

```bash
cd rag_simple
docker compose up --build -d
```

- PDFs:     http://localhost:8080/
- API:      http://localhost:8000/
- Chroma:   http://localhost:8001/

On first start the `rag` container auto-ingests all PDFs into Chroma.

---

## API

```bash
# Health
curl http://localhost:8000/health | jq

# Search
curl "http://localhost:8000/search?q=MongoDB%20Atlas%20SLA&k=3" | jq

# Force re-ingest
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"force": true}' | jq

# List ingested documents
curl http://localhost:8000/documents | jq
```

---

## Architecture

```
┌─────────────┐       HTTP        ┌──────────────────┐
│  chromadb   │ ◄──────────────── │       rag        │
│  (vector)   │                   │  - http.server   │
│  :8000      │                   │  - LangChain     │
└─────────────┘                   │  - fastembed     │
                                  │  - FastAPI       │
                                  └────────┬─────────┘
                                           │ volume
                                  ../db_proposals (PDFs)
```

PDFs are mounted read-only from `../db_proposals`.
