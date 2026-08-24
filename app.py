"""
HTTP server wrapping rag_pipeline.py.

Endpoints:
  GET  /health          -> liveness check
  POST /upload           -> multipart file upload, saves a PDF into the shared pdfs folder
  POST /index            -> {"folder": "/app/pdfs"} (optional, defaults to PDFS_DIR) starts indexing in the background
  GET  /index/status     -> current indexing state
  POST /ask              -> {"question": "...", "top_k": 5} -> {"answer": ..., "sources": [...]}

Run directly:      uvicorn app:app --host 0.0.0.0 --port 8000
Run via Docker:    see README.md
"""

import os
import shutil
import threading
from pathlib import Path

from fastapi import FastAPI, BackgroundTasks, HTTPException, Header, UploadFile, File
from pydantic import BaseModel

import rag_pipeline as rag

app = FastAPI(title="RAG Pipeline API")

# Where uploaded PDFs land. Matches the volume mount in docker-compose.yml.
PDFS_DIR = Path(os.environ.get("PDFS_DIR", "/app/pdfs"))
PDFS_DIR.mkdir(parents=True, exist_ok=True)

# Optional simple auth: if API_KEY is set in the environment, every request
# must include a matching header. Without this, anyone who can reach the
# server can trigger paid LLM calls (/ask) or a long indexing job (/index).
API_KEY = os.environ.get("API_KEY")

def check_auth(x_api_key: str | None):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")


# ---------------------------------------------------------------------------
# In-memory indexing status. Fine for a single-instance deployment; if you
# ever run multiple replicas behind a load balancer, this needs to move to
# shared storage (e.g. Redis) since each replica would otherwise have its
# own view of indexing progress.
# ---------------------------------------------------------------------------
_index_status = {"state": "idle", "detail": ""}
_lock = threading.Lock()


class AskRequest(BaseModel):
    question: str
    top_k: int = rag.TOP_K


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]


class IndexRequest(BaseModel):
    folder: str = str(PDFS_DIR)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), x_api_key: str | None = Header(default=None)):
    check_auth(x_api_key)

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted")

    dest = PDFS_DIR / file.filename
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    return {"status": "uploaded", "filename": file.filename, "path": str(dest)}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, x_api_key: str | None = Header(default=None)):
    check_auth(x_api_key)

    chunks = rag.retrieve(req.question, top_k=req.top_k)
    if not chunks:
        raise HTTPException(status_code=404, detail="No indexed content found. Call /index first.")

    answer_text = rag.answer_question(req.question, top_k=req.top_k)
    sources = [{"file": Path(c["source"]).name, "page": c["page"]} for c in chunks]
    return {"answer": answer_text, "sources": sources}


def _run_index(folder: str):
    with _lock:
        _index_status["state"] = "running"
        _index_status["detail"] = f"Indexing {folder}"
    try:
        rag.index_folder(folder)
        with _lock:
            _index_status["state"] = "done"
            _index_status["detail"] = f"Finished indexing {folder}"
    except Exception as e:
        with _lock:
            _index_status["state"] = "error"
            _index_status["detail"] = str(e)


@app.post("/index")
def index(background_tasks: BackgroundTasks, req: IndexRequest = IndexRequest(),
          x_api_key: str | None = Header(default=None)):
    check_auth(x_api_key)

    if not Path(req.folder).exists():
        raise HTTPException(status_code=400,
                             detail=f"Folder {req.folder} not found inside the container. "
                                    f"Did you mount it as a volume?")

    with _lock:
        if _index_status["state"] == "running":
            raise HTTPException(status_code=409, detail="Indexing already in progress")

    background_tasks.add_task(_run_index, req.folder)
    return {"status": "started", "folder": req.folder}


@app.get("/index/status")
def index_status():
    return _index_status
