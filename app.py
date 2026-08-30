#!/usr/bin/env python3
"""
RAG service:
  - LangChain: parse PDFs → chunk → embed (fastembed) → store in ChromaDB
  - POST /index          – start indexing
  - GET  /index/status   – indexing progress (handles 20k+ files)
  - POST /ask            – natural language question → top matching chunks
  - GET  /health
  - GET  /documents
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import chromadb
import uvicorn

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PDF_DIR = Path(os.getenv("PDF_DIR", "/data/pdfs"))
CHROMA_HOST = os.getenv("CHROMA_HOST", "chromadb")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "db_providers")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("rag")

app = FastAPI(title="DB Providers RAG", version="2.2.0")

_embedder: Optional["FastEmbedEmbeddings"] = None
_vs: Optional[Chroma] = None
_index_lock = threading.Lock()

index_status: Dict[str, Any] = {
    "status": "idle",
    "message": "No indexing has been started yet",
    "started_at": None,
    "finished_at": None,
    "force": False,
    "total_pdfs": 0,
    "docs_processed": 0,
    "docs_skipped": 0,
    "chunks_created": 0,
    "current_file": None,
    "errors": [],
}


class FastEmbedEmbeddings(Embeddings):
    def __init__(self, model_name: str = EMBED_MODEL):
        from fastembed import TextEmbedding
        log.info("Loading fastembed model: %s", model_name)
        self.model = TextEmbedding(model_name=model_name)
        log.info("Embedding model ready")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [e.tolist() for e in self.model.embed(texts)]

    def embed_query(self, text: str) -> List[float]:
        return list(self.model.embed([text]))[0].tolist()


def get_chroma_client() -> chromadb.HttpClient:
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


def get_vectorstore() -> Chroma:
    global _vs, _embedder
    if _vs is None:
        _embedder = FastEmbedEmbeddings()
        client = get_chroma_client()
        _vs = Chroma(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding_function=_embedder,
        )
    return _vs


def file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def load_and_chunk(pdf_path: Path, splitter: RecursiveCharacterTextSplitter) -> List[Document]:
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    stem = pdf_path.stem
    company_slug = stem.split("_", 1)[1] if "_" in stem else stem
    fp = file_hash(pdf_path)

    for doc in pages:
        doc.metadata["source_file"] = pdf_path.name
        doc.metadata["company_slug"] = company_slug
        doc.metadata["file_hash"] = fp

    chunks = splitter.split_documents(pages)
    for i, c in enumerate(chunks):
        c.metadata["chunk_index"] = i
        c.metadata["chunk_id"] = f"{stem}__{fp}__{i}"
    return chunks


def run_index(force: bool = False) -> None:
    global index_status, _vs

    if not _index_lock.acquire(blocking=False):
        log.warning("Index already running – ignoring duplicate request")
        return

    try:
        index_status = {
            "status": "running",
            "message": "Indexing started",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "finished_at": None,
            "force": force,
            "total_pdfs": 0,
            "docs_processed": 0,
            "docs_skipped": 0,
            "chunks_created": 0,
            "current_file": None,
            "errors": [],
        }

        if not PDF_DIR.exists():
            index_status.update({
                "status": "error",
                "message": f"PDF_DIR does not exist: {PDF_DIR}",
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
            return

        pdfs = sorted(PDF_DIR.glob("*.pdf"))
        index_status["total_pdfs"] = len(pdfs)

        if not pdfs:
            index_status.update({
                "status": "error",
                "message": "No PDF files found",
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
            return

        if force:
            try:
                client = get_chroma_client()
                client.delete_collection(COLLECTION_NAME)
                log.info("Force=true → deleted collection '%s'", COLLECTION_NAME)
                _vs = None
            except Exception as e:
                log.warning("Could not delete collection: %s", e)

        vs = get_vectorstore()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        existing = set()
        if not force:
            try:
                sample = vs.get(include=["metadatas"], limit=50000)
                for m in sample.get("metadatas") or []:
                    if m and "source_file" in m:
                        existing.add(m["source_file"])
            except Exception:
                pass

        for pdf in pdfs:
            index_status["current_file"] = pdf.name

            if pdf.name in existing and not force:
                log.info("Skipping already-indexed %s", pdf.name)
                index_status["docs_skipped"] += 1
                continue

            try:
                log.info("Indexing %s ...", pdf.name)
                chunks = load_and_chunk(pdf, splitter)
                if not chunks:
                    index_status["errors"].append(f"{pdf.name}: no text extracted")
                    continue

                vs.add_documents(chunks)
                index_status["docs_processed"] += 1
                index_status["chunks_created"] += len(chunks)
                log.info("  → %d chunks stored", len(chunks))
            except Exception as e:
                msg = f"{pdf.name}: {e}"
                log.exception(msg)
                index_status["errors"].append(msg)

        index_status["current_file"] = None
        index_status["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        index_status["status"] = "done"
        index_status["message"] = (
            f"Indexed {index_status['docs_processed']} file(s), "
            f"{index_status['chunks_created']} chunk(s); "
            f"skipped {index_status['docs_skipped']}"
        )
        log.info(index_status["message"])

    finally:
        _index_lock.release()


# ---------- Models ----------
class IndexRequest(BaseModel):
    force: bool = Field(False, description="If true, drop collection and re-index everything")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural language question")
    k: int = Field(DEFAULT_TOP_K, ge=1, le=30, description="Number of chunks to retrieve")


# ---------- Routes ----------
@app.get("/health")
def health():
    try:
        client = get_chroma_client()
        hb = client.heartbeat()
        vs = get_vectorstore()
        count = vs._collection.count()
        pdf_count = len(list(PDF_DIR.glob("*.pdf"))) if PDF_DIR.exists() else 0
        return {
            "status": "ok",
            "chroma_host": f"{CHROMA_HOST}:{CHROMA_PORT}",
            "chroma_heartbeat": hb,
            "collection": COLLECTION_NAME,
            "vectors": count,
            "pdf_dir": str(PDF_DIR),
            "pdf_count": pdf_count,
            "embed_model": EMBED_MODEL,
            "index_status": index_status["status"],
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/index")
def start_index(body: IndexRequest = IndexRequest(), background_tasks: BackgroundTasks = None):
    """Start indexing all PDFs into ChromaDB (runs in background)."""
    if index_status["status"] == "running":
        return {
            "accepted": False,
            "message": "Indexing is already running",
            "status": index_status,
        }

    background_tasks.add_task(run_index, body.force)
    return {
        "accepted": True,
        "message": "Indexing started in background",
        "force": body.force,
        "status_url": "/index/status",
    }


@app.get("/index/status")
def get_index_status():
    """Return current indexing status and progress counters."""
    vectors = None
    try:
        vs = get_vectorstore()
        vectors = vs._collection.count()
    except Exception:
        pass

    return {
        **index_status,
        "vectors_in_collection": vectors,
    }


@app.post("/ask")
def ask(body: AskRequest):
    """
    Natural language question → retrieve top matching chunks from ChromaDB.
    Ready for later LLM integration (chunks are returned as context).
    """
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    try:
        vs = get_vectorstore()
        results = vs.similarity_search_with_score(question, k=body.k)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Search failed: {e}")

    contexts = []
    for rank, (doc, dist) in enumerate(results, 1):
        meta = doc.metadata or {}
        contexts.append({
            "rank": rank,
            "score": round(float(1.0 - dist) if dist <= 1 else float(dist), 4),
            "text": doc.page_content,
            "source_file": meta.get("source_file"),
            "company_slug": meta.get("company_slug"),
            "page": meta.get("page"),
            "chunk_index": meta.get("chunk_index"),
        })

    return {
        "question": question,
        "k": body.k,
        "contexts": contexts,
        # Placeholder for future LLM answer
        "answer": None,
        "note": "Retrieval only. LLM answer integration can be added later.",
    }


@app.get("/documents")
def documents():
    vs = get_vectorstore()
    sample = vs.get(include=["metadatas"], limit=50000)
    files: Dict[str, Any] = {}
    for m in sample.get("metadatas") or []:
        if not m:
            continue
        src = m.get("source_file")
        if src:
            files.setdefault(src, {
                "source_file": src,
                "company_slug": m.get("company_slug"),
                "chunks": 0,
            })
            files[src]["chunks"] += 1
    return {"documents": list(files.values()), "total_files": len(files)}


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
